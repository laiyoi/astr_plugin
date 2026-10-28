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

空目录带 `.gitkeep`，可以直接保留。`dataset_raw` 是人工放素材的地方，`dataset` 是预处理产物。

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
  "clean": false
}
```

处理规则：

- 图片/GIF：原文件按内容哈希复制，不抽帧、不转码；
- 视频：`core.video.extract_frames()` 抽帧，帧保存为 JPEG；
- 同一视频的所有帧分到同一个 split，避免泄漏；
- 按 `collect` / `suspicious` / `skip` 分层划分 train/val/test，默认 70/15/15；
- 输出 `training/dataset/preprocess_report.json`；
- `clean: true` 会先删除整个输出目录，慎用；
- 超大/损坏图片会按 `max_image_pixels`（默认 4000 万像素）跳过并打印路径。

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

TensorBoard 默认开启，日志写到：

```text
<output-dir>\tensorboard
```

启动：

```powershell
tensorboard --logdir D:\runs\taste_dinov2_v1\tensorboard
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

## 5. 在验证集校准阈值（优先降低误收藏）

```powershell
python training\calibrate_threshold.py `
  --checkpoint D:\runs\taste_dinov2_v1\ckpt\best.pt `
  --data-root training\dataset `
  --config D:\runs\taste_dinov2_v1\model_config.json `
  --target-precision 0.95 `
  --update-config
```

策略：在验证集上选择 **precision ≥ target** 的前提下 recall 最大的阈值；如果达不到目标，会选验证集上 precision 最高的点并打印 WARNING。

**不要对 test 调阈值。** `calibrate_threshold.py` 只读取 val。

---

## 6. 评估（只在 test 上报告）

```powershell
python training\evaluate.py `
  --checkpoint D:\runs\taste_dinov2_v1\ckpt\best.pt `
  --data-root training\dataset `
  --config D:\runs\taste_dinov2_v1\model_config.json `
  --split test `
  --device cuda
```

输出：

- `evaluation_test.json`：Precision、Recall、F1、混淆矩阵；
- `evaluation_test.csv`：每张图的 label/score/prediction；
- `false_positives`、`false_negatives` 列表。

---

## 7. 导出 ONNX

```powershell
python training\export_onnx.py `
  --checkpoint D:\runs\taste_dinov2_v1\ckpt\best.pt `
  --config D:\runs\taste_dinov2_v1\model_config.json `
  --output-dir D:\runs\taste_dinov2_v1\deploy `
  --opset 17 `
  --device cuda
```

生成：

```text
deploy\
  taste_classifier.onnx
  model_config.json
```

`model_config.json` 包含：输入尺寸、RGB 转换、均值和标准差、类别映射、正向类别索引、阈值、模型版本、训练配置、ONNX 输入/输出名。

---

## 8. Windows 上验证 PyTorch / ONNX 一致性

```powershell
python training\test_onnx.py `
  --checkpoint D:\runs\taste_dinov2_v1\ckpt\best.pt `
  --onnx D:\runs\taste_dinov2_v1\deploy\taste_classifier.onnx `
  --config D:\runs\taste_dinov2_v1\deploy\model_config.json `
  --data-root training\dataset `
  --device cuda
```

它会用部署时完全相同的预处理跑 test 图片，比较 PyTorch 与 ONNX 的 logits：

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
