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

- 群图片/短视频自动识别；视频抽帧只用于检测，原视频会保存到 `images/`，并生成海报帧用于网页缩略图；
- **合并聊天记录自动解析**，包括嵌套的 `message` 数组；
- 私聊的图片/视频和合并记录都处理；
- `collect` 保存到 `images/`，`suspicious` 缓存到 `pending/`；
- **模型自动判为 `skip` 时不保存文件**，只保留 SHA 去重记录；
- **网页人工改成 `skip` 时保留文件**并移入 `skip/`，供数据集导出；改成 `collect` 时会把文件移入 `images/`；
- 图片 UUID 命名，SQLite + CSV 元数据；
- 本地网页审核：
  - 浏览保存/缓存图片；
  - 纠正标签；
  - 按用户、标签筛选；
  - 导出 zip 训练集；
  - 发布者按已保存数量/平均分排行；
- 在「QQ 映射」页手动设置昵称与真实 QQ 的对应，立即回填该昵称下所有 QQ 未知的历史图片；
- 保存用户所在群、群昵称（群名片）和基础昵称，映射页可查看；
- 批量选择图片后可以手动指定发送者 QQ 和昵称，解决 NapCat 返回“QQ用户”的匿名记录；
- 排行榜显示 QQ 头像（通过 qlogo 获取）；
- 支持单张、批量删除图片，以及一键删除全部 skip。

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
