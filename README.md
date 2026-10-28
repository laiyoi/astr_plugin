# AstrBot Img Curator

一个本地运行的 AstrBot 图片/视频收藏插件，以及完全独立的 PyTorch 训练目录。

## 目录

```text
astrbot_plugin_img_curator/
  astrbot_plugin_img_curator/ # 插件本体：安装到 AstrBot 只需要这层
  training/                   # Windows + CUDA 训练、评估、ONNX 导出
  tests/
.vscode/                      # 本地调试 AstrBot 的配置
install.sh                    # 只复制插件源码到 ~/astrbot/data/plugins
pyproject.toml                # 本地开发环境（uv），不含 torch
```

插件运行时只依赖：

```text
onnxruntime
numpy
Pillow
```

安装插件不需要 PyTorch、CUDA、torchvision 或训练脚本。训练相关依赖全部在 `astrbot_plugin_img_curator/training/requirements.txt`，只在 Windows 训练机安装。

## 插件安装

运行：

```bash
./install.sh
```

默认复制到：

```text
~/astrbot/data/plugins/astrbot_plugin_img_curator
```

可用 `ASTRBOT_DATA_DIR` 覆盖 AstrBot 数据目录。脚本只复制内层插件源码，不会带上 `training/`、`tests/`。

当前阶段默认：

```json
{
  "enabled": true,
  "recognition_enabled": false,
  "auto_download_model": false
}
```

即只做合并记录训练集采集，不加载视觉模型。

## 本地开发

```bash
./uv sync
./uv run python -m unittest discover -s astrbot_plugin_img_curator/tests -v
```

`./uv` 是本地包装脚本，cache 放在工作区。当前 DSH 沙箱里如果 `uv sync` 报 `Invalid cross-device link`，换普通终端运行。

## 训练

训练与插件完全分开：

- 训练：Windows + PyTorch + CUDA，见 [`astrbot_plugin_img_curator/training/README.md`](astrbot_plugin_img_curator/training/README.md)
- 部署：Ubuntu/N100，只把导出的 `.onnx` 放进插件模型目录

不要把 `training/` 复制进插件目录，也不要让插件导入 torch。

## 本地 AstrBot 调试

`.vscode/` 已配置成从当前工作区的 `AstrBot/` 源码启动，而不是 Docker。

`AstrBot/` 是本地调试用的 AstrBot 源码 checkout，已加入 `.gitignore`，不进入仓库。
没有这个目录时，插件依然可以单独安装和运行。

## 数据与隐私

`dataset/`、`benchmark_results/`、插件运行数据、模型权重和 `apikey` 都已忽略，不会提交。
