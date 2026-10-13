# 环境结论与依赖基线

依据：2026-10-13 返回的服务器环境报告。公开仓库仅记录下表摘要，不包含报告中的机器路径。

| 项目 | 已确认情况 | 决定 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04、x86_64、glibc 2.35 | 使用 Linux x86_64 包 |
| Python | 3.10.21 | 保持 3.10 |
| PyTorch | 2.7.1+cu126，CUDA 可被识别 | 先复用，做实际前向/反向检查 |
| 驱动 / nvcc | 550.120 / 12.4 | 暂不改动，首阶段不编译 CUDA 扩展 |
| GPU | 系统和 PyTorch 均可见 8 张 A100 80GB PCIe | 项目暂用 0、1、2、3，具体分配可变；初次检查只用一张 |
| 磁盘 | 所查文件系统空闲约 7.80 TiB | 容量盘点完成；实际占用随数据与缓存测量 |
| 网络 | Git 可拉取、不可推送；pip 安装可用 | 通过 Git 下发代码，直接 pip 安装依赖；模型与数据通路后续验证 |
| Klein | 缺少 diffusers，尚不能导入 | 安装 requirements.txt 后验证 |

## 本次固定的直接依赖

- 保留报告中的 `torch==2.7.1`、`torchvision==0.22.1`、`transformers==4.57.1`、`accelerate==1.6.0`、`huggingface-hub==0.36.0`、`numpy==2.2.6`、`Pillow==10.3.0`、`einops==0.8.0`。
- 新增 `diffusers==0.38.0`，该版本包含 Klein 管线。
- 将 `peft` 从 0.15.2 更新至 0.17.1；Diffusers 0.38.0 的依赖检查要求 PEFT 至少为 0.17.0。
- 将 `safetensors` 从 0.4.3 更新至 0.8.0，满足 Diffusers 0.38.0 的要求。
- 固定 `tokenizers==0.22.1`，位于 Transformers 4.57.1 声明的支持范围内。
- 首阶段不需要安装 FlashAttention、xFormers 或 Prodigy。数据读取和 XVAE 适配所需的额外依赖，在实现对应代码时再补充。

`torch==2.7.1` 的版本条件能匹配已安装的 `2.7.1+cu126`，正常安装无需替换该构建。清单面向已有基础环境，尚不是独立环境的完整传递依赖锁；在 A100 验证通过后再记录完整冻结版本。

## 验证边界

Windows 侧已针对 Python 3.10、Linux x86_64/glibc 2.35、CUDA 12.6 解析依赖；这只能证明解析器能找到满足声明约束的版本。GPU 运行结果必须由服务器上的 `check_klein_environment.py` 返回。

驱动 550 与 cu126 的组合不能只凭版本不同判定失败。NVIDIA 为 CUDA 12.x 提供较旧驱动上的有限小版本兼容，但部分新特性和 PTX JIT 有限制。先验证本项目实际使用的运算，后续若需要编译或 JIT，再检查对应要求。

参考上游：

- [Diffusers 0.38.0 依赖表](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/dependency_versions_table.py)
- [Diffusers 0.38.0 Klein 管线](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [Transformers 4.57.1 依赖声明](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)
- [NVIDIA CUDA 小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
