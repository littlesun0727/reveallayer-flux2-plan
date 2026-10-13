# RevealLayer / FLUX.2 Klein 训练计划

目标：基于 `FLUX.2-klein-base-4B`，使用 RevealLayer-100K 训练图像分层能力，输入 RGB 图像及目标框，输出背景和 RGBA 前景层。

当前仓库包含实施计划和环境采集工具，尚未包含训练器、模型权重或数据集。

## 文档

- [训练实施计划](训练实施计划.md)：技术路线、分阶段实现与验收。
- [离线 A100 操作方案](离线A100操作方案.md)：联网 Windows 与四卡 A100 服务器之间的工作安排。
- [第一步操作说明](RUN_FIRST_STEP.txt)：环境采集与结果返回。

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

## 后续顺序

确认服务器环境后，固定依赖和上游版本，准备模型及首个数据分片，检查完整场景的数据配对。先验证 Klein VAE 到透明解码器的重建，再实现单卡最小分解训练，最后扩展四卡训练。
