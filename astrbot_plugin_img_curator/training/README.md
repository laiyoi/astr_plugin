# Windows 训练 → Ubuntu CPU 部署

训练和部署严格分离：

- **Windows**：PyTorch + CUDA + RTX 5070 12GB，只负责训练、校准、评估、导出。
- **Ubuntu/N100**：只装 `onnxruntime`、`numpy`、`Pillow`，只加载 `.onnx`，不装 PyTorch、不跑训练。

> 以下命令是待执行脚本，尚未在 Windows/RTX 5070 上实际跑过。第一次运行请先确认 CUDA 可用。

---

## 1. Windows 环境

RTX 5070 是 Blackwell（sm_120），需要 CUDA 12.8+ 的 PyTorch。建议 Python 3.11/3.12。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 先装 Blackwell 可用的 torch/torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 再装训练依赖
pip install -r training\requirements.txt

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果 `torch.cuda.is_available()` 是 False：

- 更新 NVIDIA 驱动到支持 CUDA 12.8 的版本；
- 确认安装的是 cu128 轮子，而不是 CPU 轮子；
- 不要用 WSL 里的旧 CUDA。

---

## 2. 数据来源与目录

插件的合并记录采集会输出：

```text
training_data/forwarded/images/*.jpg|png|webp|gif
training_data/forwarded/videos/*.mp4
training_data/forwarded/<batch_id>/metadata.jsonl
```

把它们按标签放进原始目录：

```text
astrbot_plugin_img_curator/training/
  dataset_raw/
    collect/       # 愿意收藏：图片、GIF、视频
    suspicious/    # 可疑，交给发布者历史判定
    skip/          # 不收藏：图片、GIF、视频
  dataset/         # preprocess.py 生成，不要手改
    train/collect/
    train/suspicious/
    train/skip/
    val/collect/
    val/suspicious/
    val/skip/
    test/collect/
    test/suspicious/
    test/skip/
```

`dataset_raw` 的空目录带 `.gitkeep`，可以直接保留。`dataset` 是预处理产物，不放 `.gitkeep`，由 `preprocess.py` 自动创建。

---

## 3. 预处理

视频必须先在预处理阶段抽帧，使用和检测完全相同的高变化/均匀覆盖算法；训练机需要安装 `ffmpeg` 并加入 PATH。

参数全部在 `training/config.json` 的 `preprocess` 段，直接运行：

```powershell
python training\preprocess.py
```

默认：

```json
{
  "raw_root": "dataset_raw",
  "output_root": "dataset",
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "split_seed": 42,
  "video_max_frames": 8,
  "video_max_seconds": 15,
  "video_fps": 1.0,
  "resize_size": 256,
  "jpeg_quality": 95,
  "clean": false
}
```

处理规则：

- 图片/GIF：先缩放再保存为 JPEG（默认最长边 256，质量 95），GIF 只取首帧用于训练；
- 视频：`core.video.extract_frames()` 抽帧，帧同样缩放为 256 内的 JPEG；
- 同一视频的所有帧分到同一个 split，避免泄漏；
- 重新预处理时会先删除该源文件在旧 split/class 下的副本，再写到新 split；
- 如果视频换过编码/字节不同，旧副本不会被识别为同一源，建议改 `clean: true` 全量重建；
- 按 `collect` / `suspicious` / `skip` 分层划分 train/val/test，默认 70/15/15；
- 输出 `training/dataset/preprocess_report.json`；
- `clean: true` 会先删除整个输出目录，慎用；
- 超大/损坏图片会按 `max_image_pixels`（默认 4000 万像素）跳过并打印路径；
- 预处理产物统一为小尺寸 JPEG，DataLoader 不再解码原始大图。

处理完成后目录就是训练脚本需要的 `train/val/test/{collect,suspicious,skip}` 结构。

---

## 4. 训练

参数全部在 `training/config.json` 的 `train` 段，直接运行：

```powershell
python training\train.py
```

默认：

```json
{
  "data_root": "dataset",
  "output_dir": "runs/taste_dinov2_v1",
  "checkpoint_dir": "ckpt",
  "save_every_epochs": 20,
  "max_image_pixels": 40000000,
  "num_workers": 4
}
```

- 不设置总 epoch 数，按验证集 early stopping 停止；
- 每 `save_every_epochs` 个 epoch 自动保存 `ckpt/epoch_XXXX.pt`，默认 20；
- 同时保存 `ckpt/last.pt` 和 `ckpt/best.pt`；
- `checkpoint_dir` 相对 `output_dir`，也可以写绝对路径；
- 超大/损坏图片在扫描阶段会跳过，不会进入 DataLoader；
- Windows 如果仍报 `MemoryError`，先把 `num_workers` 改成 `0`，并把 `max_image_pixels` 降到 `20000000`。

TensorBoard 默认开启，事件文件直接写到 checkpoint 目录：

```text
<output-dir>/ckpt/
```

启动：

```powershell
tensorboard --logdir D:\runs\taste_dinov2_v1\ckpt
```

需要关闭时把配置里的 `tensorboard` 改为 `false`，或设置 `tensorboard_dir`。

默认：

- DINOv2 ViT-S/14，ImageNet 预训练，三分类 `skip / suspicious / collect`；
- 冻结视觉主干，只训练 LayerNorm + Dropout + Linear 分类头；
- 混合精度；
- class weight 处理三分类不平衡；
- 按验证 loss 保存 `ckpt/best.pt`；
- 输出 `model_config.json`（阈值先设为 0.5）。

效果不足时再解冻最后几个 block：

效果不足时，在 `training/config.json` 里把：

```json
"unfreeze_last_blocks": 2,
"lr_backbone": 0.0001
```

12GB 显存下 `"batch_size": 32` 通常够用；OOM 就降到 16。不要一开始 `"full_finetune": true`，小数据容易过拟合。

如果 Windows DataLoader 报错，把 `"num_workers": 0` 先跑通。

---

## 4.1 suspicious 的保存策略

模型只负责三分类，`suspicious` 不直接保存。发布者的“男娘置信度”用该发布者已经保存的图片数量近似：

```text
publisher_saved_count < cache_below_saved_count  -> cache
cache_below_saved_count <= count < save_at_saved_count -> skip
count >= save_at_saved_count -> save
```

默认在 `model_config.json`：

```json
"suspicious_policy": {
  "save_at_saved_count": 5,
  "cache_below_saved_count": 3
}
```

`inference.py` 提供 `decide_save(label, publisher_saved_count, policy)`；模型本身不计算这个数量。插件侧可以用 `Store.count_saved_by_sender(sender_id)` 取这个计数。

`config.json` 还包含：

```json
"calibrate": {
  "split": "val",
  "target_precision": 0.95,
  "batch_size": 32,
  "num_workers": 0,
  "device": "cuda",
  "dedupe_threshold": 6,
  "update_model_config": true,
  "max_image_pixels": 40000000
},
"evaluate": {
  "split": "test",
  "batch_size": 32,
  "num_workers": 0,
  "device": "cuda",
  "dedupe_threshold": 6,
  "max_image_pixels": 40000000
},
"export": {"opset": 17, "device": "cuda"},
"test_onnx": {"num_samples": 8, "device": "cuda"}
```

## 5. 校准阈值

参数在 `training/config.json` 的 `calibrate` 段，直接运行：

```powershell
python training\calibrate_threshold.py
# 指定第 20 个 epoch 的 checkpoint
python training\calibrate_threshold.py --ckpt 20
```

默认读取：

```text
runs/taste_dinov2_v1/ckpt/best.pt
runs/taste_dinov2_v1/model_config.json
training/dataset
```

在验证集上选择 precision ≥ target 的前提下 recall 最大的阈值；结果写 `runs/taste_dinov2_v1/threshold.json`，并自动更新 `model_config.json`。使用 `--ckpt 20` 时读取 `ckpt/epoch_0020.pt`，结果写 `threshold_epoch0020.json`，不覆盖 `model_config.json`。**不要对 test 调阈值。**

---

## 6. 评估

参数在 `training/config.json` 的 `evaluate` 段，直接运行：

```powershell
python training\evaluate.py
# 指定第 20 个 epoch
python training\evaluate.py --ckpt 20
```

使用 `--ckpt 20` 时读取 `ckpt/epoch_0020.pt`，优先使用 `threshold_epoch0020.json`；输出文件带 `_epoch0020` 后缀。

输出：

- `evaluation_test.json`：三分类 confusion matrix、per-class precision/recall/F1、macro F1/accuracy，以及 collect-vs-rest 的阈值指标；
- `evaluation_test.csv`：每张图的真实类别、三分类预测、collect 概率和阈值；
- `false_positives`、`false_negatives` 列表。

---

## 7. 导出 ONNX

参数在 `training/config.json` 的 `export` 段，直接运行：

```powershell
python training\export_onnx.py
```

生成：

```text
runs/taste_dinov2_v1/deploy/
  taste_classifier.onnx
  model_config.json
```

`model_config.json` 包含输入尺寸、RGB 转换、均值和标准差、类别映射、正向类别索引、阈值、版本、ONNX 输入/输出名和 suspicious 策略。

---

## 8. Windows 上验证 PyTorch / ONNX 一致性

参数在 `training/config.json` 的 `test_onnx` 段，直接运行：

```powershell
python training\test_onnx.py
```

它会用部署时完全相同的预处理跑 test 图片，比较 PyTorch 与 ONNX logits：

- 输出 `max_abs_diff` 和 `allclose`；
- 默认要求 `atol=1e-4`；
- 不通过就不要复制到 Ubuntu。

---

## 9. 复制到 Ubuntu 的文件

只需要：

```text
taste_classifier.onnx
model_config.json
inference.py
```

可选：

```text
requirements-deploy.txt
```

把它们放到例如：

```text
/home/laiyoi/astr_plugin/models/taste/
  taste_classifier.onnx
  model_config.json
  inference.py
```

**不要复制**：`best.pt`、`last.pt`、训练图片、`torch` 相关文件。

---

## 10. Ubuntu / N100 CPU 测试

Ubuntu 上不安装 PyTorch。建议单独 venv：

```bash
cd /home/laiyoi/astr_plugin/models/taste
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-deploy.txt

python inference.py \
  --onnx taste_classifier.onnx \
  --config model_config.json \
  --image /path/to/test.jpg \
  --json
```

输出：

```json
{
  "label": "collect",
  "collect": true,
  "score": 0.93,
  "threshold": 0.91,
  "class_scores": {"skip": 0.07, "suspicious": 0.02, "collect": 0.91},
  "version": "1.0.0"
}
```

也可以直接在 AstrBot 插件里调用：

```python
from inference import TasteClassifier

classifier = TasteClassifier(
    "/home/laiyoi/astr_plugin/models/taste/taste_classifier.onnx",
    "/home/laiyoi/astr_plugin/models/taste/model_config.json",
)
result = classifier.predict(pil_image)
if result["collect"]:
    ...
```

前提是 `inference.py` 在该 Python 的 import 路径中。插件侧不要 import `training.*`。

---

## 11. 预处理一致性

训练、评估、导出、Ubuntu 推理必须完全一致：

```text
RGB
→ resize shortest edge 256, bicubic
→ center crop 224×224
→ /255
→ ImageNet mean [0.485,0.456,0.406]
→ std [0.229,0.224,0.225]
→ NCHW float32
```

`inference.py` 已按这个顺序实现；`test_onnx.py` 会用同一套 preprocessing 验证。不要只改其中一处。

---

## 12. 性能与验收

- 先看 `test_onnx.py` 的 parity，再看 `evaluate.py` 的 precision/recall；
- 在 Ubuntu 上用 `time` 或插件日志实测单张 CPU 延迟；
- 记录 N100 上的均值/P95、峰值 RSS、模型体积；
- 没有实际跑过的数字不要写进最终报告；
- 如果 ONNX Runtime 版本不同导致 logits 有微小差异，先检查预处理和输入 dtype，再考虑 atol。

## 发布/更新 Hugging Face 模型

模型仓库（private）：https://huggingface.co/laiyoi/femboydetector

重新训练后要更新线上模型，按这个顺序跑，**每一步都要看输出**再往下走：

```powershell
cd astrbot_plugin_img_curator

# 1. 导出 ONNX（读 training/config.json 的 export 段）
python training\export_onnx.py

# 2. 校验 PyTorch 与 ONNX 一致；不通过就不要发布
python training\test_onnx.py

# 3. 生成清洗过的发布目录（默认 training/hf_release/）
python training\stage_hf_release.py

# 4. 上传（文件夹整体推，等于一次 commit）
$env:HF_HUB_DISABLE_XET='1'
hf upload laiyoi/femboydetector .\training\hf_release . --type model
```

要点：

- **必须先做第 2 步**：`test_onnx.py` 输出 `allclose=true` 才能发布，否则 Ubuntu 上的 CPU 推理和训练时的 PyTorch 会不一致；
- 第 3 步的 `stage_hf_release.py` 负责清洗：把 `model_config.json`、`evaluation_test.*`、`threshold*.json` 里的本机绝对路径改写成相对形式，只放行模型卡、`best.pt`、`taste_classifier.onnx`、`inference.py` 和指标文件；
- 本机（Windows + 沙箱）不走 `HF_HUB_DISABLE_XET=1` 会报 `OSError: I/O error: 拒绝访问 (os error 5)`；
- 目标仓库已存在时 `hf upload` 直接追加 commit；仓库不存在时加 `--private` 会新建一个私有仓库；
- `training/` 目录不可写时（例如在受限沙箱里），用 `HF_RELEASE_DIR` 把发布目录指到别处：`$env:HF_RELEASE_DIR='D:\hf_release'`；
- **不要上传**训练图片、`training/dataset/`、`training/dataset_raw/`、任何 QQ 号/群号/昵称，也不要上传 `HUGGINGFACE_HANDOFF.md`、`DEPLOY_HANDOFF.md` 这类内部交接文档——仓库里的 `README.md` 是给使用者看的模型卡。

只改模型卡（不动权重）时，单独推模型卡文件就够，不用重传 80MB。注意推的是 `training/hf_release/README.md`（由脚本生成的**模型卡**），不是仓库根目录这份项目 README：

```powershell
$env:HF_HUB_DISABLE_XET='1'
hf upload laiyoi/femboydetector .\training\hf_release\README.md README.md --type model
```

模型卡正文由 `training/stage_hf_release.py` 里的 `MODEL_CARD` 生成，改文案要改那里再重跑第 3 步，否则下次发布会把旧卡片覆盖回去。

发布后确认（私有仓库必须带 token，未登录会 401，那是正常现象）：

```powershell
python -c "from huggingface_hub import HfApi; a=HfApi(); i=a.repo_info('laiyoi/femboydetector'); print('private:', i.private); print(a.list_repo_files('laiyoi/femboydetector'))"
```

每次上传后都看一眼 `private:` 是不是 `True`；万一被翻成 public，改回来：

```powershell
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('laiyoi/femboydetector', private=True)"
```

已知坑：

- `hf repos move` 在本机会报 `SSL: UNEXPECTED_EOF_WHILE_READING`，改仓库名得用 `HfApi().move_repo(from_id=..., to_id=..., repo_type="model")`；
- 训练产出的 `evaluation_test.csv` 带 UTF-8 BOM，自己写清洗脚本要用 `utf-8-sig` 读，否则路径清洗会静默失效；
- `best.pt` 内部的 `training.args` 保留训练机路径（checkpoint 原样未改），其中不含隐私字段。

更细的产物清单和字段说明见 [`astrbot_plugin_img_curator/HUGGINGFACE_HANDOFF.md`](astrbot_plugin_img_curator/HUGGINGFACE_HANDOFF.md)。

---

## 13. 按文件哈希去重

`training/dedupe_files.py` 按 sha256 内容哈希找完全相同的文件，同一组里**保留创建日期最新的**，删掉创建时间较早的副本。默认只预览，加 `--apply` 才真删。

```powershell
# 预览（不动文件）
python training\dedupe_files.py D:\pics

# 真正删除创建时间较早的重复文件
python training\dedupe_files.py D:\pics --apply

# 改成按修改时间判断新旧
python training\dedupe_files.py D:\pics --by modified --apply

# 不删，移动到回收目录
python training\dedupe_files.py D:\pics --apply --trash D:\dup_backup

# 只处理指定后缀，并输出 JSON 报告
python training\dedupe_files.py D:\pics --ext .jpg .png --apply --json-out dedupe.json
```

规则：

- 默认看**创建时间**：Windows 取 `st_ctime`（文件的“创建日期”），macOS 取 `st_birthtime`；Linux 没有创建时间，会退回修改时间（`st_ctime` 在 Linux 上是 inode 变更时间，不适合代表创建）；
- 想按修改时间就去掉歧义，显式加 `--by modified`；
- 先按文件大小分组，只有大小相同的才计算哈希，避免全盘哈希；
- 同组内按选定的时间排序，最新的留下；时间相同按路径字典序；
- 0 字节文件直接跳过（哈希全一样，删了没意义）；
- 支持一次传多个目录，同一文件被重复传入只算一次；
- `--trash` 模式下重名文件会自动加序号，不会覆盖；
- JSON 报告记录每个被删文件的原路径、时间、保留文件、sha256 和字节数，并标注 `order_by`。
