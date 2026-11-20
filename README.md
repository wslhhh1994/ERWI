# ERWI：融合 SeismicMoE 的弹性反射波形反演

基于声–弹耦合波动方程（AECE）的二维弹性反射波形反演研究代码，使用 CuPy 进行 GPU 波场计算、MPI 分配炮集任务，并结合 PyTorch 混合专家网络（SeismicMoE）更新纵波和横波速度模型。

本仓库提供反演主程序与网络实现。运行完整反演需要另外准备模型和观测炮集；数据生成程序、数据文件及预训练权重未包含在本仓库中。

## 主要功能

- **交替反演**：内层 ELSRTM 更新 `δ(Vp²)`、`δ(Vs²)`，外层 ERWI 更新背景 `Vp`、`Vs`；默认每轮包含 3 次内层与 5 次外层迭代。
- **GPU 与 MPI**：基于交错网格有限差分、PML 吸收边界及边界波场存储/重建，按炮分配计算任务。
- **梯度处理**：包含深度加权、梯度平滑与截断、共轭梯度方向及偏移距渐进策略。
- **SeismicMoE**：软门控融合 FNO、双向 GRU 和 CNN 专家，结合二维旋转位置编码、残差瓶颈、跨道注意力和频率感知解码器。
- **结构与测井约束**：支持结构引导平滑及可选测井约束；主程序默认 `USE_WELL = False`。

## 文件结构

```text
ERWI/
├── nnc_aece_rwi.py   # AECE 正/反传播、ELSRTM/ERWI 交替反演与 MoE 训练
├── seismic_moe.py    # SeismicMoE 网络、训练辅助函数与独立自检
├── requirements.txt # 通用 Python 依赖；CUDA/CuPy 按环境单独安装
├── .gitignore       # 排除缓存、模型数据、权重和计算结果
└── README.md
```

## 运行环境与安装

建议使用 Python 3.10 或以上版本，以及具备 NVIDIA GPU 的 CUDA 计算环境。完整反演依赖 NVIDIA 驱动、可用的 CUDA 版 PyTorch、CuPy 和 MPI 实现（例如 Open MPI、MPICH 或 Windows 上的 MS-MPI）。代码未锁定依赖版本，具体组合需要在目标计算环境验证。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 则使用：.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

确保安装的 PyTorch 支持目标 GPU。随后按 CUDA 环境安装一种对应的 CuPy 发行包，例如 CUDA 12.x：

```bash
python -m pip install cupy-cuda12x
```

CuPy 的包名必须与所选环境匹配，不要同时安装多个 CuPy 发行包。`mpi4py` 还需要可用的 MPI 运行时；仅安装 Python 依赖不能代替驱动、CUDA 和 MPI 环境配置。

## 数据准备

请从仓库根目录运行主程序，因为所有输入、输出路径均相对于当前工作目录。

### 1. 模型文件

准备以下两个无文件头的 NumPy `float32` 二进制文件：

| 文件 | 用途 | 读取形状 |
| --- | --- | --- |
| `overthrust_450_3_175_marine_iso_3.bin` | 参考模型及密度 | `(450, 3, 175)` |
| `overthrust_450_3_175_marine_linear.bin` | 初始模型输入 | `(450, 3, 175)` |

数组维度顺序为 `(x, parameter, z)`，参数通道依次为 `Vp`、`Vs`、密度，采用 NumPy 默认 C 顺序存储。速度按 m/s 使用；密度应与数据生成过程保持一致。每个模型包含 236250 个数值，即 945000 字节。

**实际初始化行为**：程序保留浅层初始值，将 `bound` 以下的初始 `Vp`、`Vs` 替换为参考模型经 `sigma=10` 高斯平滑后的结果，并直接使用参考模型密度。因此，当前示例配置需要参考模型，不能直接当作仅依赖未知实测模型的通用入口。

### 2. 观测炮集

默认数据目录为：

```text
ae_record_overthrust_50_90_f8_dx25_Tn3000_P_iso_cut/
├── 0_pwave_cut.bin
├── 1_pwave_cut.bin
├── ...
├── 49_pwave_cut.bin
├── 0_vx_cut.bin       # 可选速度分量
├── 0_vz_cut.bin       # 可选速度分量
└── ...
```

每个文件按 `float32` 读取为 `(NR, Tn) = (90, 3000)`，即每炮每分量 1080000 字节；文件不含头信息。速度分量存在时程序会尝试使用；缺失时会关闭该炮的速度数据项。

**必须提供有效压力炮集**：当前代码在压力文件缺失时会使用模拟压力作为替代，从而可能产生零残差。程序继续运行并不代表数据已正确加载，请运行前检查文件是否齐全、格式是否匹配。

观测系统须与代码一致：50 炮、每炮 90 道，炮间距为 7 个网格，近偏移距为 10 个网格；接收器位于震源左侧，位置为 `rx[shot][i] = sx[shot] - 10 - i`。源和接收器深度均为 `pml + 1`，源横向位置为 `pml + 100 + shot * 7`（含 PML 的网格坐标）。

## 配置与运行

当前版本直接在 `nnc_aece_rwi.py` 中设置参数，不提供命令行参数接口。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `xn`, `zn` | `450`, `175` | 不含 PML 的模型网格数 |
| `dx`, `dz` | `25`, `25` | 网格间距，单位 m |
| `dt`, `Tn` | `0.002`, `3000` | 时间采样间隔（s）及采样数 |
| `FM` | `8` | 震源主频，单位 Hz |
| `pml` | `30` | PML 厚度，单位网格 |
| `N_INNER_PER_CYCLE` | `3` | 每轮 ELSRTM 迭代次数 |
| `N_OUTER_PER_CYCLE` | `5` | 每轮 ERWI 迭代次数 |
| `USE_WELL` | `False` | 是否启用测井约束 |
| `X_start` | `70` | MoE 更新区域的横向起始索引 |
| `DATA_DIR` | 见数据目录 | 炮集输入路径 |
| `save_dir` | `nn_erwi_iso_result_over_pro2_sm/` | 结果输出路径 |

主循环默认为 `range(0, 100)`。修改网格、采样或观测系统时，必须同步调整数据格式和模型尺寸。

### GPU 映射

原始配置在脚本内写入 `CUDA_VISIBLE_DEVICES = "0,1,3,4,5"`，并设置 `processes_per_gpu = 5`。设备映射使用 `rank // processes_per_gpu`：每连续 5 个 MPI 进程共用一个可见 GPU；rank 0 同时执行 MoE 训练。

运行前请按机器配置修改这两项。可见 GPU 数应至少为 `ceil(MPI进程数 / processes_per_gpu)`；同卡并发数还应根据显存调整。该环境变量目前位于 CuPy 导入之后，为确保可见性设置在 CUDA 初始化前生效，建议同时在启动环境中设置相同值，或将该设置移动到所有 GPU 库导入之前。完整反演没有 CPU 回退路径。

以下以将脚本内可见设备修改为 `"0"` 后、单 GPU 单 MPI 进程为例：

```bash
# Linux shell
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python nnc_aece_rwi.py
```

Windows PowerShell 对应运行方式：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
mpiexec -n 1 python nnc_aece_rwi.py
```

确认单进程环境和输入正确后，可增加 `-n` 并同步检查 GPU 映射。完整反演计算量和显存占用较大。

### 单独检查神经网络

无需反演输入数据即可运行网络自检（需要 PyTorch；无 CUDA 时该自检可使用 CPU）：

```bash
python seismic_moe.py
```

自检包含有/无测井信息的前向传播、输出形状断言及训练步骤。网络输入形状为 `(n_x, 2, 5, n_z)`，输出为 `(n_x, 2, 1, n_z)`；两个通道分别对应纵波与横波相关量，5 为相邻道窗口大小。

## 输出结果

结果写入 `nn_erwi_iso_result_over_pro2_sm/`：

- `models/`：速度模型 `vp_<iter>.bin`、`vs_<iter>.bin`，扰动 `vp_p_<iter>.bin`、`vs_p_<iter>.bin`，外层保存的密度 `den_<iter>.bin`，目标函数历史 `obj_f.txt`，以及触发 MoE 训练后保存的 `moe_weights.pth`。
- `grads/`：背景与扰动的梯度/搜索方向文件。
- `residuals/`：各炮压力残差 `<shot>_p.bin`；文件名随迭代复用。

模型与梯度二进制数组不含 PML，按 `float32`、形状 `(450, 175)` 读取。`vp_p`、`vs_p` 表示平方速度扰动，不能直接解释为速度增量。内层保存的 `vp`、`vs` 为背景与平方速度扰动组合后开平方得到的模型，外层会保存背景更新后的模型。权重文件名会被后续训练覆盖。

```python
import numpy as np

vp = np.fromfile(
    "nn_erwi_iso_result_over_pro2_sm/models/vp_0.bin",
    dtype=np.float32,
).reshape(450, 175)
```

## 使用范围

本项目是带有固定实验配置的研究代码。完整结果依赖外部数据、GPU/MPI 环境和参数一致性；仓库不提供预计算结果或精度保证。源文件中的实验配置与数据路径应根据实际任务调整。

本仓库暂未指定开源许可证；公开可见不等于授予任意再分发或商业使用许可。如需复用，请与仓库所有者确认授权。
