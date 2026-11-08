# AirSim 安装与运行检查

日期：2026-11-04。结论：安装成功，场景启动和非图像 RPC 正常；当前 CPU 软件渲染环境取图失败，尚不能用于视觉数据采集。

## 实际环境与安装

- 本次命令实际运行于 Ubuntu 20.04 x86_64 容器，没有安装到 Windows 桌面。
- `nvidia-smi` 返回 `No devices were found`；没有 `/dev/dri` 或可用的 NVIDIA GPU 设备。
- 官方 Linux Blocks v1.8.1 安装于 `/opt/airsim/Blocks-v1.8.1`，包含 Unreal Engine 4.27.2 运行时，无须安装完整编辑器。
- 官方压缩包大小 142533463 字节；下载后通过 ZIP CRC 校验再解压。来源与下载 SHA256 见 `download_source.json`。SHA256 是本地计算值，官方资产元数据未提供 digest。
- Python 3.8 独立环境位于工作区 `envs/airsim`；AirSim Python 包版本 1.8.1；完整依赖见 `python_packages.txt`。`pip check` 通过。
- 安装了 Xvfb、xauth、Mesa Vulkan 驱动和 vulkan-tools 及其依赖。软件渲染设备为 llvmpipe / Mesa 21.2.6 / LLVM 12。
- 新建系统账号 `airsim`，主目录 `/opt/airsim/home`，运行时目录 `/opt/airsim/runtime`，用于非 root 运行 Unreal。
- 安装目录约 271 MB，Python 环境约 283 MB；原始下载包不保留。

## 实测结果

| 检查 | 结果 |
|---|---|
| Python 导入、依赖一致性 | 通过 |
| Unreal 启动并加载 Blocks | 通过 |
| ComputerVision 模式、RPC ping | 通过 |
| 客户端/服务端协议版本 | 均为 1；这是 RPC 版本，不是发行版本 |
| 场景对象查询 | 返回 209 个对象 |
| 位姿控制 | 请求 NED 米制位置 `[3, 0, -4]`，读回 `[3.0, 0.0, -4.0]` |
| 相机信息 | 成功返回；FOV 约 89.9036° |
| 320×240 RGB、深度、分割联合取图 | Vulkan 断言后崩溃，无有效图像 |
| 320×240 低画质、单独 RGB 取图 | 同样崩溃 |
| 64×48 低画质、32 软件线程、禁用 RHI 线程 | 同样崩溃 |

三次取图尝试均在 `VulkanCommandBuffer.cpp:474` 出现 `Assertion failed: bSuccess`，随后 SIGSEGV；启动时另有 `VulkanTexture.cpp:427` 的纹理标志 ensure。证据见三个 `blocks_*failure.log`。这些记录支持当前软件渲染路径无法完成取图，尚不足以确定驱动内部的唯一根因，也不证明启用 GPU 后必然成功。

非图像接口的独立验证结果见 `metadata_check.json`。没有成功生成 RGB、深度或分割样本，也没有进行飞行物理、多旋翼控制或长期稳定性测试。

测试结束后已停止本次启动的场景和虚拟显示服务，保留安装、配置、脚本及日志。

## 重新启动与检查

以下命令适用于本次安装所在的 Linux 容器，从工作区根目录执行。脚本需要 root 调用，内部将 Unreal 切换到 `airsim` 账号。Windows PowerShell 不能直接执行这些命令。

```bash
bash tools/run_airsim_blocks.sh
```

上述默认软件模式仅供复现当前故障。在另一个终端运行：

```bash
envs/airsim/bin/python tools/smoke_test_airsim.py --output results/airsim_retest
```

下一步应在具有可用 NVIDIA GPU、图形驱动和 Vulkan ICD 的环境重新测试。容器需要暴露 GPU 与图形驱动能力；仅安装 CUDA 或仅存在 `nvidia-smi` 命令不足以满足渲染条件。确认 GPU 渲染设备可被 Vulkan 枚举后运行：

```bash
AIRSIM_RENDERER=gpu bash tools/run_airsim_blocks.sh
```

GPU 分支不强制 Mesa 软件 ICD，尚未在本环境验证。已有外部 Vulkan 环境变量也会影响设备选择。

启动配置为 `configs/airsim_blocks_cv.json`：ComputerVision、320×240、RPC 仅监听 127.0.0.1:41451。启动器将配置复制到服务账号的独立配置目录，不覆盖其他用户的 AirSim 配置。

`tools/smoke_test_airsim.py` 会尝试两处相机位姿并保存 RGB、浮点深度、分割图及报告。它仅在所有检查通过后写出 `status: passed`；异常退出不能解释为通过。每次复测使用新的输出目录，避免混淆旧结果。分割颜色为默认 mesh ID 映射，并非已核验的语义类别或唯一实例标签。

重新建立 Python 3.8 环境时，先安装 NumPy、OpenCV、msgpack-rpc-python，再安装 AirSim；其旧式安装脚本会在构建阶段导入这些模块。锁定版本见 `tools/airsim-requirements.txt`。
