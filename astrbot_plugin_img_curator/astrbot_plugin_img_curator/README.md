# Femboy Detector 收藏插件

使用 Hugging Face 模型 [`laiyoi/femboydetector`](https://huggingface.co/laiyoi/femboydetector) 的 DINOv2 三分类 ONNX 分类器，在 CPU 上收藏图片/短视频。

不再使用 CLIP、不再有文本提示词、不需要 torch。

## 分类

```text
skip        不收藏
suspicious  缓存，按发布者已保存数量决定
collect     收藏
```

`suspicious` 默认策略：

```text
< 3 张已保存 -> cache
3-4          -> skip
>= 5         -> save
```

阈值可在插件配置中调整。

## 功能

- 群图片/短视频自动识别；
- **合并聊天记录自动解析**，包括嵌套的 `message` 数组；
- 私聊合并记录也处理；
- `collect` 保存到 `images/`，`suspicious` 缓存到 `pending/`，`skip` 只记 SHA；
- 图片 UUID 命名，SQLite + CSV 元数据；
- 本地网页审核：
  - 浏览保存/缓存图片；
  - 纠正标签；
  - 按用户、标签筛选；
  - 导出 zip 训练集；
  - 发布者按已保存数量/平均分排行。

## 模型

```text
HF repo: laiyoi/femboydetector
文件: taste_classifier.onnx + model_config.json
输入: [1,3,224,224] float32
输出: logits [1,3]
类别: ["skip","suspicious","collect"]
```

公共仓库会自动下载模型到插件数据目录的 `models/femboydetector/`。

## 网页

在 AstrBot 管理界面 → 插件 → 本插件的 **可视化视图** 中打开。

不再单独启动 HTTP 服务；页面通过 AstrBot 插件页 bridge 调用 API。

## 部署

插件运行时只依赖：

```text
onnxruntime
numpy
Pillow
aiohttp
huggingface_hub
```

更新插件代码：

```bash
./install.sh
```

训练/模型发布说明见仓库根目录的 `HUGGINGFACE_HANDOFF.md`。
