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

可以用插件的合并记录采集作为原始来源：在群里发送 `/imgfilter forward 10`，再发合并聊天记录；**所有图片会平铺保存到 `training_data/forwarded/images/`**，每批的 `metadata.jsonl` 仍放在 `training_data/forwarded/<batch_id>/`。之后按 `collect` / `skip` 分好类即可。

必须人工标注成 `collect` / `skip` 两类。现在支持两种目录：

**A. 已手动分好 train/val/test**

```text
D:\data\taste\
  train\
    collect\   # 愿意收藏
    skip\      # 不收藏
  val\
    collect\
    skip\
  test\
    collect\
    skip\
```

**B. 只分 collect/skip，训练时自动划分**

```text
D:\data\taste\
  collect\
  skip\
```

自动划分是分层抽样，按类别分别切分，默认：

```text
train 70% / val 15% / test 15%
```

并先做去重再划分，避免同一张图跨 split。

要求：

- `collect` 为正类，`skip` 为负类；
- 测试集不参与训练、早停、阈值选择；
- 视频帧要按视频分组，同一视频的帧不要跨 split；
- 同一人物/同一组连拍尽量放在同一侧；
- 脚本会做 SHA-256 精确去重 + dHash 近似去重（默认汉明距离 ≤6），并写 `dedupe_report.json`；
- 自动划分模式会先对全部数据去重，再按类别分层切分；
- 手动划分模式跨 split 重复时优先保留 train，删除 val/test 的重复项；
- **测试集被去重删除的图会记录在报告里，但不会用于调参。**

---

## 3. 训练

```powershell
python training\train.py `
  --data-root D:\data\taste `
  --output-dir D:\runs\taste_dinov2_v1 `
  --auto-split `
  --epochs 30 `
  --batch-size 32 `
  --image-size 224 `
  --unfreeze-last-blocks 0 `
  --amp `
  --num-workers 4
```

如果目录已经是 `train/val/test`，去掉 `--auto-split` 即可。自动划分的比例可调：

```powershell
--val-ratio 0.15 --test-ratio 0.15 --split-seed 42
```

TensorBoard 默认开启，日志写到：

```text
<output-dir>\tensorboard
```

启动：

```powershell
tensorboard --logdir D:\runs\taste_dinov2_v1\tensorboard
```

不需要时加 `--no-tensorboard`，或指定目录 `--tensorboard-dir D:\runs\tb`。

默认：

- DINOv2 ViT-S/14，ImageNet 预训练；
- 冻结视觉主干，只训练 LayerNorm + Dropout + Linear 分类头；
- 混合精度；
- class weight 处理 collect/skip 不平衡；
- 按验证 loss 保存 `best.pt`；
- 输出 `model_config.json`（阈值先设为 0.5）。

效果不足时再解冻最后几个 block：

```powershell
python training\train.py ... --unfreeze-last-blocks 2 --lr-backbone 1e-4
```

12GB 显存下 `--batch-size 32` 通常够用；OOM 就降到 16。不要一开始 `--full-finetune`，小数据容易过拟合。

如果 Windows DataLoader 报错，把 `--num-workers` 改成 0 先跑通。

---

## 4. 在验证集校准阈值（优先降低误收藏）

```powershell
python training\calibrate_threshold.py `
  --checkpoint D:\runs\taste_dinov2_v1\best.pt `
  --data-root D:\data\taste `
  --config D:\runs\taste_dinov2_v1\model_config.json `
  --target-precision 0.95 `
  --update-config
```

策略：在验证集上选择 **precision ≥ target** 的前提下 recall 最大的阈值；如果达不到目标，会选验证集上 precision 最高的点并打印 WARNING。

**不要对 test 调阈值。** `calibrate_threshold.py` 只读取 val。

---

## 5. 评估（只在 test 上报告）

```powershell
python training\evaluate.py `
  --checkpoint D:\runs\taste_dinov2_v1\best.pt `
  --data-root D:\data\taste `
  --config D:\runs\taste_dinov2_v1\model_config.json `
  --split test `
  --device cuda
```

输出：

- `evaluation_test.json`：Precision、Recall、F1、混淆矩阵；
- `evaluation_test.csv`：每张图的 label/score/prediction；
- `false_positives`、`false_negatives` 列表。

---

## 6. 导出 ONNX

```powershell
python training\export_onnx.py `
  --checkpoint D:\runs\taste_dinov2_v1\best.pt `
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

## 7. Windows 上验证 PyTorch / ONNX 一致性

```powershell
python training\test_onnx.py `
  --checkpoint D:\runs\taste_dinov2_v1\best.pt `
  --onnx D:\runs\taste_dinov2_v1\deploy\taste_classifier.onnx `
  --config D:\runs\taste_dinov2_v1\deploy\model_config.json `
  --data-root D:\data\taste `
  --device cuda
```

它会用部署时完全相同的预处理跑 test 图片，比较 PyTorch 与 ONNX 的 logits：

- 输出 `max_abs_diff` 和 `allclose`；
- 默认要求 `atol=1e-4`；
- 不通过就不要复制到 Ubuntu。

---

## 8. 复制到 Ubuntu 的文件

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

## 9. Ubuntu / N100 CPU 测试

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
  "class_scores": {"skip": 0.07, "collect": 0.93},
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

## 10. 预处理一致性

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

## 11. 性能与验收

- 先看 `test_onnx.py` 的 parity，再看 `evaluate.py` 的 precision/recall；
- 在 Ubuntu 上用 `time` 或插件日志实测单张 CPU 延迟；
- 记录 N100 上的均值/P95、峰值 RSS、模型体积；
- 没有实际跑过的数字不要写进最终报告；
- 如果 ONNX Runtime 版本不同导致 logits 有微小差异，先检查预处理和输入 dtype，再考虑 atol。
