# 本地图片筛选收藏（AstrBot 插件）

在 **CPU** 上监听 QQ 群图片/短视频，用本地 ONNX 视觉模型筛选“女装 / 反串风格”的真人照片和二次元男娘图片。高置信度图片自动收藏，不确定图片进入待确认目录，明显不符合的图片只留下 SHA-256 记录。短视频会先用 ffmpeg 抽帧，只保存评分最高的一帧，不归档原视频。

- 无云端视觉 API、无 GPU、无 Ollama、无 torch。
- 无群号白名单：默认处理所有群。
- 图片统一保存为 `<UUID>.<jpg|png|webp>`，**不按日期、不按群号分目录**。
- 发送者昵称、QQ 号、群名称、群号等写入 CSV；去重与计数在 SQLite。
- **默认不下载、不加载视觉模型**：当前阶段只接收合并记录并保存训练集。只有把 `recognition_enabled` 设为 `true` 后，才会按配置下载/加载 ONNX 模型。

> 只根据图片内容、服装和视觉风格筛选，不推断真人的性别认同或真实性别。真人图片请遵守发送者同意与群规；不要把 CSV 或图片目录公开传播。

---

## 1. 模型选型结论

**默认：OpenCLIP ViT-B/32 的 ONNX fp32 版本（约 608 MB 权重）。**

你提供的 116 张验收集（`real_crossdress` 61、`anime_crossdress` 17、`negative` 38）实测：

| 模型 | 权重 | 模型加载 | 单张均值/P95 | 峰值 RSS | AUC | 推荐阈值 | precision | recall | 待确认（正/负） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT-B/32 fp32（默认）** | 579.8 MiB | 7.0 s | 291.7 / 344.3 ms | 1389.9 MiB | **0.913** | 0.011 | **0.963（52/54）** | **0.667（52/78）** | 13 / 4 |
| ViT-B/32 int8（省内存） | 148.6 MiB | 4.3 s | 192.3 / 234.8 ms | 868.4 MiB | 0.871 | 0.010 | 0.951（39/41） | 0.500（39/78） | 12 / 5 |
| MobileCLIP-S0 fp32 | 207.5 MiB | 4.0 s | 163.1 / 188.0 ms | 597.0 MiB | 0.823* | 0.040 | 1.000（2/2）* | 0.250（2/8）* | 11* |
| MobileCLIP-S0 int8 | 54.2 MiB | 3.4 s | 667.2 / 940.3 ms | 345.3 MiB | 0.504* | 0.040 | —（0/0）* | 0.000（0/8）* | 9* |

`*` 这四行 MobileCLIP 是早期 37 张公开冒烟集的数字，不是 116 张验收集；保留它们是为了说明为什么淘汰 MobileCLIP。116 张上的 ViT fp32 / int8 是同一批用户数据。

另有 1 个短视频（`real_crossdress`）：当时的抽帧逻辑抽了 7 帧，最佳帧约 2.0 s，判为 `saved`，margin `+0.0116`。现在已改为“等时间窗选变化最大帧”，下次 benchmark 的帧数/时间会不同。把它计入后是 117 个样本：precision `0.964`、recall `0.671`；媒体总数里包含 1 个视频。

5 折交叉验证（每折只用训练折选阈值，目标 precision ≥ 0.95，再在留出折评估）：

- fp32：precision **0.946**，recall **0.679**（TP 53 / FP 3 / FN 25）；
- int8：precision **0.947**，recall **0.462**（TP 36 / FP 2 / FN 42）。

全量 116 张、fp32 阈值 0.011 的 bootstrap 95% 区间：precision 0.906–1.000，recall 0.558–0.767。样本量仍然不大，区间很宽。

### 为什么默认从 int8 改成 fp32？

在同样的约 95% precision 下，fp32 的 recall 比 int8 高约 17 个百分点（全量）或 22 个百分点（5 折 CV）。代价是：

- 权重 580 MiB，峰值 RSS 1.39 GiB；int8 分别是 149 MiB、868 MiB。
- 单张 292 ms vs 192 ms；这是本机无 AVX 虚拟机的数字，N100 原生会更快，但**未实测**。

6 GB 内存下 fp32 仍能跑；如果还要同时跑较多服务、或实际内存紧张，切回 `int8`，并把阈值改成 0.010 左右。

### 为什么不是 MobileCLIP-S0？

MobileCLIP-S0 更小、fp32 单张更快，但在早期 37 张冒烟集上 AUC 只有 0.823，安全阈值下只收藏 2/8 个目标；它的 int8 版本在这颗无 AVX 的虚拟 CPU 上退化到 AUC 0.504、667 ms/张，完全不可用。MobileCLIP 仍作为可选配置，但不是默认。

### 重要限制

- CLIP 余弦相似度**不是概率**。`0.011` 是在你这 116 张数据上选的，不是通用常数；换群、换图源后需要重新校准。
- 零样本 CLIP 无法可靠判断“衣服下面的人的真实性别/性别认同”。它只能学到“裙子、化妆、女性化服装、二次元画风”等视觉线索；普通女性、二次元女性角色也可能得分很高。默认策略仍是**宁可进待确认，不自动收藏**。
- 116 张里 `real_crossdress` 61 张、`anime_crossdress` 17 张、`negative` 38 张，正负比较不平衡；标签由你提供，我没有逐张复核。
- 本机测试环境是 PVE 虚拟机，CPU 型号 `QEMU Virtual CPU`，没有 AVX/AVX2 flags；因此 292 ms 是偏悲观数字。**N100 原生性能没有实测**，不能给出加速倍数。把 PVE CPU 类型改成 `host` 后可用 benchmark 复测。

早期 37 张公开冒烟集的逐图原始数据、复现命令和图片来源见 [`benchmark/RESULTS.md`](benchmark/RESULTS.md) 与 [`benchmark/PROVENANCE.md`](benchmark/PROVENANCE.md)。

---

## 2. 处理流程

1. 监听所有群的图片消息（`aiocqhttp` / OneBot v11）。
2. 从 `Image.url`、`Image.path`、`Image.file` 或 `base64://` 取图；视频从 `Video` 组件取 `file/url/path`。HTTP 流式下载，边下边限制体积。
3. 图片用 Pillow 校验格式、像素数、动画帧；只接受 JPEG/PNG/WebP。视频交给 ffmpeg，限制前 15 秒；先按约 4 fps 抽候选帧，再在每个等长时间窗里选变化最大的一帧，最多 8 帧。
4. 计算 SHA-256；开启去重时，已见过的图片/视频直接跳过。
5. 图片直接编码；视频对候选帧逐帧编码，选 margin 最高的一帧。24 条文本提示词的向量在加载时只算一次。
6. 计算 `margin = max(正向提示词) - max(负向提示词)`：
   - `margin >= high_threshold` 且正分足够：保存；
   - `pending_threshold <= margin < high_threshold`：进入待确认；
   - 其它：删除临时文件，只记录哈希。
7. 图片/视频最佳帧都用 UUID 重命名，写入 `images/` 或 `pending/`；SQLite 记录 `source_type=image|video_frame`、`frame_time`、哈希、分数、耗时；CSV 由 SQLite 原子重写。
8. 网络/模型/格式异常一律**不保存**，只增加错误计数。
9. 管理员用 `/imgfilter forward [分钟]` 开启采集后，当前群或私聊里的合并聊天记录会被单独解析；图片保存到 `training_data/forwarded/<batch>/`，附发送者/群/嵌套路径元数据，不进入正式收藏。

---

## 3. 目录结构

```text
astrbot_plugin_img_curator/
├── __init__.py
├── main.py                  # 监听器、命令、生命周期
├── metadata.yaml
├── _conf_schema.json
├── config.example.json
├── requirements.txt
├── README.md
├── THIRD_PARTY_MODELS.md
├── core/
│   ├── config.py            # 配置与路径
│   ├── model.py             # ONNX 会话、下载、预处理、推理
│   ├── pipeline.py          # 队列、去重、保存、审核
│   ├── scoring.py           # 24 条提示词与阈值判定
│   ├── sources.py           # 图片/视频获取与格式校验
│   ├── video.py             # ffmpeg 抽帧 + 等窗变化选帧
│   ├── forward_collector.py # 合并聊天记录 → 训练集
│   └── store.py             # SQLite + CSV
├── tests/test_core.py
└── benchmark/
    ├── benchmark.py
    ├── RESULTS.md
    └── results/*.json
```

插件数据目录（AstrBot 默认 `data/plugin_data/astrbot_plugin_img_curator/`）：

```text
images/<uuid>.jpg          # 自动收藏，平铺，不按日期/群号
pending/<uuid>.jpg         # 待人工确认
metadata.csv               # 每张已保存/待确认图片一行
state.sqlite3              # 去重哈希、状态、计数、耗时
models/clip_vit_b32_fp32/  # 权重（首次下载；int8 时为 clip_vit_b32_int8）
tmp/                       # 下载/校验临时文件
```

---

## 4. 安装

### 4.1 复制插件

把整个 `astrbot_plugin_img_curator/` 放到 AstrBot 的插件目录：

```bash
# 以本机 compose 的挂载为例
cp -r astrbot_plugin_img_curator /path/to/astrbot/data/plugins/
```

### 4.2 安装依赖

在 AstrBot 容器内安装（不要装 torch）：

```bash
docker exec -it astrbot python -m pip install \
  -r /AstrBot/data/plugins/astrbot_plugin_img_curator/requirements.txt
```

依赖：`onnxruntime`、`tokenizers`、`Pillow`、`numpy`、`aiohttp`。AstrBot 镜像已带 Pillow/numpy/aiohttp，实际新增的主要是 `onnxruntime` 和 `tokenizers`。

### 4.3 启用

在 AstrBot WebUI 的插件页重载插件，或重启容器。默认 `recognition_enabled=false`，**不会读取模型文件**；先把合并记录训练集收好。要正式识别时再打开 `recognition_enabled`，并配置模型/自动下载。

检查：

```text
/imgfilter model
/imgfilter status
```

`model` 第一次可能显示 `loading`；等日志出现“模型就绪”即可。也可以手动触发：

```text
/imgfilter model load
```

### 4.4 网络受限时手工放置模型

默认从 Hugging Face 下载：

- `Xenova/clip-vit-base-patch32`，commit `d15189d7028b43f1d3e65039190477f6af591c2a`
  - fp32：`onnx/vision_model.onnx`、`onnx/text_model.onnx`
  - int8：`onnx/vision_model_quantized.onnx`、`onnx/text_model_quantized.onnx`
  - `tokenizer.json`

fp32 放到：

```text
<plugin_data>/models/clip_vit_b32_fp32/
  vision_model.onnx
  text_model.onnx
  tokenizer.json
```

int8 放到 `models/clip_vit_b32_int8/`，文件名同样是上面三个。

文件名必须改成上面三个；插件会校验固定 SHA-256，校验失败会删除重下。校验值见 `core/model.py` 的 `_SPECS`。

---

## 5. 配置

所有配置在 AstrBot 插件配置页修改；[`config.example.json`](config.example.json) 是一份可读的默认值示例（AstrBot 实际使用 `_conf_schema.json` 生成配置）。配置文件里没有密钥，也不需要任何视觉 API。

| 配置 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 插件总开关；关闭后合并记录也不接收 |
| `recognition_enabled` | `false` | 图片/视频识别开关；关闭时**不加载模型**，只做训练集采集 |
| `save_dir` | 空 → `images/` | 收藏目录；统一 UUID 文件名，不分子目录 |
| `pending_dir` | 空 → `pending/` | 待确认目录 |
| `save_pending` | `true` | 不确定图片是否落盘 |
| `strip_metadata` | `true` | 保存前移除 EXIF/GPS（重新编码） |
| `dedupe` | `true` | 相同 SHA-256 只识别一次 |
| `max_file_mb` | `12` | 单张图片最大体积 |
| `max_video_mb` | `50` | 单个视频最大体积 |
| `forward_dir` | 空 → `training_data/forwarded/` | 合并记录训练图片目录 |
| `forward_max_minutes` | `60` | 单次采集最长分钟数 |
| `forward_max_images` | `500` | 单批最多保存图片数 |
| `forward_max_nodes` | `2000` | 单批最多节点数 |
| `forward_max_depth` | `8` | 合并记录最大嵌套深度 |
| `video_max_seconds` | `15` | 视频最多解码前 N 秒 |
| `video_max_frames` | `8` | 每个视频最多抽帧数 |
| `video_fps` | `1.0` | 输出抽帧频率下限；内部候选帧约 4 fps |
| `video_extract_timeout_sec` | `60` | 抽帧超时 |
| `model_name` | `CLIP-ViT-B-32` | 可选 `MobileCLIP-S0`（不推荐默认） |
| `model_variant` | `fp32` | `fp32` / `int8` |
| `model_dir` | 空 → `models/` | 模型根目录 |
| `auto_download_model` | `false` | 识别启用时是否自动下载模型；默认关闭 |
| `high_threshold` | `0.011` | 自动收藏的 margin 阈值 |
| `pending_threshold` | `0.005` | 进入待确认的 margin 下限 |
| `min_positive_score` | `0.240` | 自动收藏的最低正分 |
| `min_pending_score` | `0.210` | 待确认的最低正分 |
| `cpu_threads` | `4` | ONNX intra-op 线程数 |
| `queue_size` | `64` | 有界队列；满则丢弃并计数 |
| `rate_limit_count` | `20` | 每群每窗口最多处理张数 |
| `rate_limit_window_sec` | `60` | 限流窗口 |
| `max_pending_files` | `500` | 待确认上限；达到后丢弃新候选 |
| `max_image_pixels_million` | `40` | 解码像素上限 |
| `min_image_side` | `64` | 最小边长 |
| `image_download_timeout_sec` | `30` | 图片下载超时 |
| `download_timeout_sec` | `600` | 模型下载超时 |
| `csv_path` | 空 → `metadata.csv` | CSV 路径 |

### 切换到 int8 / MobileCLIP-S0

省内存（约 868 MiB 峰值 RSS）时：

```json
{
  "model_name": "CLIP-ViT-B-32",
  "model_variant": "int8",
  "high_threshold": 0.010
}
```

切换 MobileCLIP-S0（不推荐作为默认）：

```json
{
  "model_name": "MobileCLIP-S0",
  "model_variant": "fp32",
  "high_threshold": 0.040,
  "pending_threshold": 0.010,
  "min_positive_score": 0.115,
  "min_pending_score": 0.095
}
```

切换后要重新下载权重并重新校准；两套模型的分数尺度不同，不能直接复用阈值。

---

## 6. 命令

全部命令仅管理员可用（AstrBot 管理员判定）。命令前缀以你的 AstrBot 配置为准，下面用 `/` 举例。

| 命令 | 作用 |
|---|---|
| `/imgfilter status` | 运行状态、模型状态、队列、识别/收藏/待确认计数 |
| `/imgfilter model [load]` | 查看模型；`load` 主动下载并加载 |
| `/imgfilter stats` | 累计数量、平均/P95/最大推理耗时 |
| `/imgfilter pending` | 最近 10 张待确认图片的 ID、分数、时间 |
| `/imgfilter pause` | 暂停识别（持久化，重启仍暂停） |
| `/imgfilter resume` | 恢复识别 |
| `/imgfilter test` | 对当前消息附带/引用的图片或短视频分类一次，不保存、不去重；视频会显示抽帧数与最佳帧时间 |
| `/imgfilter forward [分钟]` | 在当前群或私聊开启合并聊天记录采集，默认 10 分钟 |
| `/imgfilter forward_status` | 查看采集目录、队列、批次和已保存图片数 |
| `/imgfilter forward_cancel` | 停止当前会话采集 |
| `/imgfilter confirm <ID前缀>` | 待确认 → 收藏；文件名保持 UUID 不变 |
| `/imgfilter reject <ID前缀>` | 删除待确认文件，标记负样本 |
| `/imgfilter help` | 帮助 |

`test` 的图片附件不会被监听器同时收藏：监听器会跳过 `imgfilter` 命令消息。

## 6.1 合并聊天记录 → 训练集

```text
/imgfilter forward 10
（接下来 10 分钟，在当前群或私聊发送合并转发）
/imgfilter forward_status
/imgfilter forward_cancel
```

行为：

- 只接受管理员开启后、当前群或当前私聊、指定时间窗内的合并记录；
- 私聊只处理合并记录，不处理普通私聊图片；
- 自动处理 `Nodes`/`Node`，遇到只有 `Forward(id)` 的段会调用 OneBot `get_forward_msg` 拉取；
- 支持嵌套转发，深度上限 `forward_max_depth`；
- 图片保存为 `training_data/forwarded/<batch_id>/images/<uuid>.<ext>`；
- 每批写 `metadata.jsonl`（发送者 QQ/昵称、群号/群名、节点发送者、时间、嵌套路径、SHA-256、尺寸）；
- 写 `batch.json` 汇总和全局 `index.jsonl` 做跨批次 SHA 去重；
- 图片标签为 `unlabeled`，不会自动进入 `collect`/`skip`；
- 不调用识别模型，使用独立队列，避免和正式收藏抢模型。

元数据示例字段：`batch_id, image_id, image_index, status, label, file, sha256, group_id, group_name, sender_id, sender_name, node_sender_id, node_sender_name, node_time, depth, forward_path, collected_at`。

示例：

```text
/imgfilter forward 10
/imgfilter test
/imgfilter pending
/imgfilter confirm 0b743f29
/imgfilter reject 0b743f29
```

---

## 7. 数据与隐私

### 7.1 图片

- 文件名：`uuid4().hex` 风格的完整 UUID + 扩展名，例如 `037f2a70-2903-4718-b9d1-628e721c39da.jpg`。
- 所有收藏图片平铺在 `save_dir`，不按日期/群号建目录。
- 待确认图片同样用 UUID，确认后移动到 `save_dir`，文件名不变。
- 不保存 GIF/动画；WebP 只接受静态帧。
- 默认 `strip_metadata=true`：保存前重新编码，去掉 EXIF/GPS/设备信息；会轻微改变 JPEG/WebP 画质。若必须保留原始字节，可关闭，但请自行评估隐私风险。

### 7.2 CSV 元数据

`metadata.csv` 为 UTF-8 BOM，Excel 可直接打开；每次保存/审核后从 SQLite 原子重写。列：

```text
image_id,file_name,status,source_type,frame_time,review_label,score,margin,
top_group,top_prompt,sender_id,sender_name,group_id,group_name,message_id,
sha256,mime,width,height,size_bytes,model,latency_ms,created_at,reviewed_at,
stored_path
```

`source_type=video_frame` 时，`frame_time` 是最佳帧在视频中的大致秒数，`sha256` 是原视频的哈希；视频本身不保存。

- `sender_id` 是 QQ 号，`sender_name` 是昵称，`group_id` 是群号，`group_name` 是群名。
- `status`：`saved` / `pending`；审核后 `review_label` 为 `positive` / `negative`。
- 以 `=`、`+`、`-`、`@` 开头的昵称/群名会被加一个前导 `'`，避免 CSV 公式注入。
- CSV 与 SQLite 权限为 `0600`，数据目录 `0700`；但**仍建议整个目录只给管理员读取**。
- 不收藏的图片不会写 CSV，也不会写发送者信息；SQLite 的 `seen` 表只保留哈希、状态、分数、耗时。

### 7.3 日志

日志只记录群号、UUID、分数、耗时和错误类型；不打印昵称、QQ、图片 URL 或 base64。审核命令的回复包含 UUID 和文件名。

---

## 8. 性能设计

- **单个常驻模型**：模型只在首次加载/切换后留在内存，不按图片重载。
- **文本向量缓存**：24 条提示词只在加载时编码一次（约 2–4 秒）；之后每张图只跑视觉编码。
- **视频抽帧**：候选帧约 4 fps，按等长时间窗选变化最大的一帧，最多 8 帧/15 秒/1280 宽；视频只处理一次，最佳帧入库，原视频删除。
- **合并记录采集**：独立队列/worker，只下载图片并写训练元数据，不经过 CLIP；自动限制批次图片数、节点数和嵌套深度。
- **单 worker 串行推理**：避免多张图同时抢占 4 个 vCPU；队列有界，满时丢弃并计数。
- **异步**：图片下载、Pillow 校验、ONNX 推理、CSV 重写都在线程/异步中执行，不阻塞 AstrBot 事件循环。
- **去重先于推理**：`SHA-256` 命中直接跳过。
- **限流**：每群 60 秒内最多 20 张（默认）。
- **保守判定**：网络失败、模型异常、格式错误都不会默认保存。
- **超时**：图片下载 30 秒、视频抽帧 60 秒、模型下载 600 秒；超大媒体在流式下载阶段截断。

实测（116 张用户数据，独立进程，不含 AstrBot 自身）：

- 默认 fp32：292 ms 均值 / 344 ms P95，峰值 RSS 1.39 GiB，模型加载约 7 秒；
- int8：192 ms 均值 / 235 ms P95，峰值 RSS 868 MiB，模型加载约 4 秒。

首张图片会有一次模型加载 + 24 条提示词编码 + warmup 开销。

---

## 9. 测试、验收与你需要提供的测试集

### 9.1 单元测试

在 AstrBot Python 环境、插件目录下：

```bash
python -m unittest discover -s tests -v
```

覆盖：阈值判定、配置默认值、SQLite 去重/CSV、图片格式校验、队列保存/去重/CSV。

### 9.2 benchmark

数据集两种组织方式：

只需要三个目录，**不需要** `negative_*` 这种通配符目录：

```text
dataset/
  real_crossdress/*.jpg|*.mp4
  anime_crossdress/*.jpg
  negative/*.jpg
```

```bash
mkdir -p dataset/real_crossdress dataset/anime_crossdress dataset/negative
```

`negative/` 里可以混放普通男性、女性、二次元、风景、宠物、梗图；benchmark
只按目录名判断正负样本，默认 `positive-labels=real_crossdress,anime_crossdress`，
其它目录一律算负样本。

或使用 JSONL：

```jsonl
{"path": "samples/a.jpg", "label": "real_crossdress"}
{"path": "samples/b.jpg", "label": "negative"}
```

运行：

```bash
python benchmark/benchmark.py \
  --dataset /path/to/dataset \
  --model-root /path/to/plugin_data/models \
  --model CLIP-ViT-B-32 --variant fp32 --threads 4 \
  --high 0.011 --pending 0.005 --download \
  --video-frames 8 --video-seconds 15 \
  --json-out benchmark/results.json \
  --csv-out benchmark/results.csv
```

首次运行可加 `--download` 让脚本下载并校验权重（默认不下载）。输出包含：AUC、自动收藏的 TP/FP/FN/TN、precision/recall、待确认数、平均/P95/最大推理耗时、加载耗时、峰值 RSS、逐图分数。视频会先抽帧再按最佳帧计分，JSON/CSV 会记录 `media_type=video`、`frames`、`frame_time`。脚本不会保存图片/视频。

### 9.3 当前验收集与后续校准

你已经提供了 116 张（61 张真人女装、17 张二次元、38 张负样本），本次结果见下一节和 [`benchmark/RESULTS.md`](benchmark/RESULTS.md)。这比公开冒烟集可靠，但仍建议：

- 保留一批**没有参与选阈值**的图片做最终验收，避免“同一批数据选阈值又报指标”的乐观偏差；
- 如果实际群里的图片风格变化较大，再补 50–100 张新样本，重点补负样本（普通男性、普通女性、二次元女性、截图）；
- 所有真人图片继续遵守发送者同意和群规，不要公开传播；
- 不要用未成年人或非自愿内容做测试。

当前默认阈值 `high=0.011` 是按“自动收藏 precision ≈ 0.95”选的；5 折 CV 估计为 precision 0.946、recall 0.679。若你更看重召回，可以把 `high` 降到 0.008–0.010，但务必让 `save_pending=true` 并人工审核。

---

## 10. 故障排查

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: onnxruntime` | 在 AstrBot 的 Python 环境执行 `pip install -r requirements.txt`；确认是 x86_64 轮子 |
| `model files missing ...` | 开 `auto_download_model`，或 `/imgfilter model load`，或按 4.4 手工放置 |
| `sha256 mismatch` | 删除 `models/<key>/` 后重新下载；不要使用被截断的权重 |
| 下载很慢/403 | 检查到 `huggingface.co` 的网络；可手工放置并核对哈希 |
| `no usable image URL` | NapCat 可能只给了裸文件名或 URL 已过期；查看原始消息结构，确认 `url/path/file` 至少有一个可用 |
| 群名为空 | `get_group_info` 失败或机器人无权限；`group_id` 仍会记录，群名留空 |
| 命令无反应 | 只有 AstrBot 管理员可用；确认前缀和权限 |
| 首张很慢 | 模型加载 + 文本编码 + warmup，约 5–10 秒；之后恢复 |
| 视频不处理 | 确认 AstrBot 镜像内有 `ffmpeg`（官方镜像已有）；查看日志中的“视频抽帧失败” |
| 视频太慢 | 降低 `video_max_frames`/`video_max_seconds`，或把 `video_fps` 调到 0.5 |
| 合并记录没反应 | 确认已用 `/imgfilter forward` 开群、未超过时长、发送的是合并转发而不是普通图片 |
| 合并记录图片为 0 | 查看 `/imgfilter forward_status` 的“最近错误”；NapCat 需要能响应 `get_forward_msg` |
| `retcode=1200 ... 内层消息` | 这是 NapCat 拒绝按内层 forward id 拉取。插件会依次尝试 Segment id、外层 message_id、`get_forward_msg`、`get_msg` 和原始事件里的内联 content；如果都拿不到，说明 NapCat 没有提供可读取的内容，需要换适配器/让发送者重新转发 |
| 内存紧张 | 用 `int8`、减少 `cpu_threads`、降低 `queue_size`、关闭 `save_pending` |
| 待确认目录暴涨 | 降低 `high_threshold` 不会减少待确认；提高 `pending_threshold` 或降低 `max_pending_files` |
| 自动收藏太少 | 先看 benchmark 的 margin 分布，再小幅降低 `high_threshold`；不要一次降很多 |

---

## 11. 已知限制与后续

- 零样本 CLIP 对“男娘/女装”的语义是近似的，尤其对二次元中性角色；默认高精度、低召回是刻意选择。
- 真人图片的性别表达不应由模型推断；本插件只做视觉风格筛选，最终由管理员审核。
- v1 不训练新模型；`confirm/reject` 会把正/负标签写入 SQLite 和 CSV，可作为后续轻量分类器或 k-NN 原型的样本。
- 没有群号白名单（按需求）；如果以后需要，可在配置层增加，不影响当前存储格式。
- 没有按日期/群号归档（按需求）；所有图片平铺，靠 CSV 的 `group_id/group_name/created_at` 检索。
- 没有自动清理策略；`pending/` 和 `images/` 会持续增长，建议配合外部定时任务或后续版本增加保留天数。

---

## 12. 参考

- MobileCLIP: <https://github.com/apple/ml-mobileclip>，论文 <https://arxiv.org/abs/2311.17049>
- OpenCLIP: <https://github.com/mlfoundations/open_clip>
- ONNX 权重：<https://huggingface.co/Xenova/clip-vit-base-patch32>、<https://huggingface.co/Xenova/mobileclip_s0>
- 第三方模型与许可：[`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md)
- ONNX Runtime: <https://onnxruntime.ai/>
- AstrBot 插件开发：<https://docs.astrbot.app/>


---

## 13. DINOv2 收藏品味模型（Windows 训练 / Ubuntu 推理）

- 训练与导出（独立于插件，不随插件安装）：[`../training/README.md`](../training/README.md)
- Windows 训练脚本：`../training/train.py`
- 验证集阈值：`../training/calibrate_threshold.py`
- 测试集评估：`../training/evaluate.py`
- ONNX 导出：`../training/export_onnx.py`
- PyTorch/ONNX 一致性：`../training/test_onnx.py`
- Ubuntu 纯 CPU 推理：`inference.py`
- 部署依赖：`requirements-deploy.txt`

训练只在 Windows + CUDA 上运行；Ubuntu/AstrBot 只加载 `.onnx`，不安装 PyTorch。
