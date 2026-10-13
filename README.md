# RevealLayer / FLUX.2 Klein 训练计划

目标：基于 `FLUX.2-klein-base-4B`，使用 RevealLayer-100K 训练图像分层能力，输入 RGB 图像及目标框，输出背景和 RGBA 前景层。

当前仓库包含实施计划、依赖清单、环境采集和小模型检查工具，尚未包含训练器、模型权重或数据集。

## 文档

- [训练实施计划](训练实施计划.md)：技术路线、分阶段实现与验收。
- [离线 A100 操作方案](离线A100操作方案.md)：联网 Windows 与四卡 A100 服务器之间的工作安排。
- [第一步操作说明](RUN_FIRST_STEP.txt)：环境采集与结果返回。
- [环境检查结论与第二步](env/README.md)：已确认环境、版本选择和验证范围。

## 两端协作

Windows 开发端负责修改代码并推送到远程仓库。A100 服务器已确认可以拉取仓库，但不能推送；服务器执行检查与训练即可，无需提交报告或日志。

公开仓库：https://github.com/littlesun0727/reveallayer-flux2-plan

首次使用：

```bash
git clone https://github.com/littlesun0727/reveallayer-flux2-plan.git
cd reveallayer-flux2-plan
```

后续更新在仓库目录运行：

```bash
git pull --ff-only
```

## 第一步：采集服务器环境

先激活准备训练的 Python / Conda 环境。如果使用容器，请在实际训练容器内运行。脚本需要 Python 3.8 及以上，基础采集仅使用标准库，不安装软件，不访问网络。

在仓库目录执行下列命令，将 `/data` 替换为服务器上实际存在的训练数据目录：

```bash
python collect_machine_info.py --probe-torch --disk-path /data --output checks/a100_environment.json
```

`--probe-torch` 会额外尝试导入当前环境的 PyTorch 和 Klein 相关类。即使缺少依赖，也应保留生成的报告，作为后续准备离线依赖包的依据。

将 `checks/a100_environment.json` 通过可用的文件传输方式带回 Windows 同名目录；也可以在终端显示后复制完整 JSON：

```bash
cat checks/a100_environment.json
```

无需在服务器上执行 `git add`、`git commit` 或 `git push`。`checks/`、日志和训练产物不纳入版本控制。若采集失败且未生成报告，返回完整终端报错。

这份报告只盘点环境；模型加载、GPU 训练与四卡通信需要在后续步骤单独验证。

## 第二步：安装依赖并检查 Klein

服务器已确认可以正常 `pip install`。第一阶段使用 [requirements.txt](requirements.txt)，保留现有 PyTorch 2.7.1 / CUDA 12.6 构建，新增 Diffusers 0.38.0，更新 PEFT 和 Safetensors 以满足其要求。

建议先复制现有环境，避免影响原 RevealLayer 推理环境（只需执行一次）：

```bash
conda create -n reveallayer_flux2 --clone reveallayer
conda activate reveallayer_flux2
```

在仓库目录执行：

```bash
git pull --ff-only
python -m pip install -r requirements.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python check_klein_environment.py --output checks/klein_environment.json
```

当前暂定可用卡号为 `0,1,2,3`，后续可能变化。上面的小检查只使用其中的 GPU 0；换卡时修改 `CUDA_VISIBLE_DEVICES`。若调度系统已设置该变量，请保留其分配，直接运行 Python 命令，脚本默认使用第一张逻辑 GPU。后续四卡启动器再使用分配的四张卡。

检查不下载权重，包含 fp32/bf16 矩阵及卷积反向传播、Klein 类导入、32/128 通道 patchify 往返，以及随机初始化的微型 Flux2 Transformer 的 LoRA 优化器更新。输出 JSON 中 `passed: true` 才代表本次检查通过。它不代表 4B 预训练模型加载、VAE BN 归一化、NCCL 或分层质量已验证。

把 `checks/klein_environment.json` 的内容发回；如果安装或 `pip check` 报错，请保留完整输出，先解决错误再执行后续步骤。

## 后续顺序

确认服务器环境后，固定依赖和上游版本，准备模型及首个数据分片，检查完整场景的数据配对。先验证 Klein VAE 到透明解码器的重建，再实现单卡最小分解训练，最后扩展四卡训练。

## 第三步：下载模型和第一份训练数据

确认第二步报告为 `passed: true` 后执行。服务器已确认可以直接下载 Hugging Face 文件。[下载清单](configs/assets.lock.json) 固定了三个仓库的 commit、文件大小和上游哈希；总计约 27.19 GB：完整 Diffusers 格式的 Klein base 4B、XVAE 检查点，以及数据说明、元数据和首个训练 tar。

选择有足够空间的资源根目录，以下以 `/data/reveallayer_flux2_assets` 为例，请替换成自己有写权限的路径：

```bash
git pull --ff-only
python download_assets.py --root /data/reveallayer_flux2_assets --verify-hashes
python inspect_first_shard.py --root /data/reveallayer_flux2_assets
```

下载支持中断后重跑，已完成的文件可复用。`--verify-hashes` 会额外顺序读取下载文件校验哈希，所需时间取决于磁盘吞吐。可先加 `--dry-run` 查看清单；也可用 `--component data`、`--component klein` 或 `--component xvae` 分开下载。已下载后可用 `--verify-only` 完全离线校验。

本批不包含用途重复的 Klein 单文件权重，也不包含原 FLUX.1 的 LoRA / OGA。文件保存在：

```text
资源根目录/
  models/klein_base_4b/
  models/xvae_original/xvae/transparent_decoder_ckpt.pth
  data/raw/train/trainMetaData.json
  data/raw/train/train-000000.tar
  manifests/assets_all.json
```

初查脚本直接读取 tar，不解压整个分片。它统计能完整配对的场景，并检查前 5 个完整场景的图像模式、尺寸、alpha 和框，生成 `checks/first_shard.json`。不在首分片内的元数据场景会统计为 absent，不能因此判定数据集损坏。该检查尚未验证层序、重合成或完成计划要求的 100 场景可视化审计。

返回 `checks/first_shard.json` 后，继续实现正式场景读取与可视化审计，然后用单卡验证真实 Klein VAE 编码/解码和透明解码器适配。
