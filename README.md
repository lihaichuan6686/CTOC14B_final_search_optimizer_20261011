# CTOC14B：平面引导束搜索与多脉冲轨迹优化

针对 CTOC14 乙题的35目标飞越任务，先用平面带时窗和多圈 Lambert 束搜索产生访问链，再联合优化访问时刻、初始轨道、自由脉冲位置及访问顺序。目标是最小化实际脉冲总和，不含比赛提交时间奖励系数。

当前版本采用科研程序写法：**在主程序直接改参数 → 读取数组 → 调用数值算法 → 输出轨迹**。搜索运行 `search.py`，优化运行 `optimize.py`，没有命令行参数、启动器或服务器提交系统。

**完整方法、公式、各阶段参数和实现边界见 [METHOD.md](METHOD.md)。** README 负责运行和阅读导航。

## 运行

已验算环境为 Python 3.12，数值库版本固定在 `requirements.txt`。已有这些依赖的解释器可直接使用；首次建立环境可执行：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe search.py
.\venv\Scripts\python.exe optimize.py
```

Linux 对应解释器路径为 `venv/bin/python`。在 VS Code 选择安装了上述依赖的解释器，打开入口直接运行也可以。虚拟环境在本机建立，不上传 Git。

| 输入文件 | 内容 |
|---|---|
| `data/states.npy` | 35个目标的 ICRF 位置、速度；形状 `(35,14401,6)`，间隔60 s |
| `data/calibration.json` | 固定极轴 J₂ 代理模型的标定矩阵及原标定记录 |
| `data/example_route.json` | 一条未深度优化的完整二体初解，供单条试跑 |

纪元为 **2035-01-01 12:00:00 UTC**，任务长864000 s；计算使用 km、s、km/s。目标内部编号0–34。

## 在哪里改参数、优化哪份结果

`search.py` 开头：

```python
OUTPUT_DIRECTORY = PROJECT / "results/search"
PLANE_COUNT = 128
THREAD_COUNT = 8
BEAM_WIDTHS = (96, 192, 192, 288)
```

平面容差、首访相位数、时窗取点、局部细化也在该文件的 `PARAMETERS` 中列出。默认逐轮换平面，保存每轮全部完整链到 `results/search/run_000001.jsonl.gz` 等。

`optimize.py` 开头：

```python
SEARCH_DIRECTORY = PROJECT / "results/search"
TRAJECTORY_FILES = []
OUTPUT_DIRECTORY = PROJECT / "results/optimization"
WORKERS = 4
BEAM_THREADS = 8
INITIAL_COUNT = 8
```

- `TRAJECTORY_FILES` 为空：只读 `SEARCH_DIRECTORY` 直接包含的 `run_*.jsonl.gz`，不递归查子目录。
- `TRAJECTORY_FILES` 非空：只读其中明确列出的完整轨迹 JSON，忽略搜索目录。
- 各阶段秒数、迭代上限、换位轮数和精英数量都在同一入口顶部。

先试一条示例的基本联调，可改为：

```python
TRAJECTORY_FILES = [PROJECT / "data/example_route.json"]
RELOCATION_ROUNDS = 0
ROTATION_PARENTS = 0
DEEP_BEAM_PARENTS = 0
```

这会运行二体三阶段联调与 J₂ SCP，跳过换位、脉冲增删、旋转重分配和末级重搜；完整流程需恢复原默认值。时间预算是迭代间检查的软上限，首次编译及单次求解可能超出设置。

| 优化输出 | 含义 |
|---|---|
| `candidate_0001.json` 等 | 通过完整35目标连续传播复验的候选，包括非当前最优的多样性候选 |
| `BEST_J2.json` | 本次运行截至当前最低实际 Δv 轨迹 |
| `BEST_J2_validation.json` | 对应连续传播、飞越距离、初轨、高度及脉冲明细 |

**不同实验请分别改 `OUTPUT_DIRECTORY`。** 重跑同目录会覆盖同名文件，但不清除上次多出来的文件；程序也不会自动读回该目录接续。

## 并行设置

- 初搜：单进程，Numba 默认8线程，平面依次运行、束内候选并行；BLAS 限1线程。逐轮输出不代表只用一核。
- 优化：默认4个工作进程，每进程 Numba/BLAS 均为1线程。
- 末级重搜：进程池结束后用8条 Numba 线程，BLAS 为1线程。

这些是手动设置的本地默认值，不会自动变成96核，也不按剩余内存自动降档。修改并发需要考虑求解器内存和星历传递开销。

## 阅读顺序

| 研究步骤 | 文件与主要函数 |
|---|---|
| 整体搜索 | [search.py](search.py)：`main` |
| 首访、束扩展、前瞻、多样性 | [beam_search.py](beam_search.py)：`continuous_beam_run`、`seed_routes`、`expand`、`select_records` |
| 平面带时窗与二体转移 | [plane_windows.py](plane_windows.py)、[lambert.py](lambert.py) |
| 整体优化 | [optimize.py](optimize.py)：`main`、`initial_refinement` |
| 候选池及家族选择 | [candidate_selection.py](candidate_selection.py) |
| 二体全链 SLSQP、分支动态规划 | [two_body_optimization.py](two_body_optimization.py) |
| J₂、STM与端点打靶 | [orbit_dynamics.py](orbit_dynamics.py) |
| 脉冲导数、SCP与弹性修复 | [j2_optimization.py](j2_optimization.py)：`pulse_model`、`refine`、`repair` |
| 目标释放、换位、整体重分配 | [target_relocation.py](target_relocation.py) |
| 长弧细分、自由节点增删 | [impulse_nodes.py](impulse_nodes.py)、[primer_insertion.py](primer_insertion.py) |
| 带中途脉冲的精英重搜 | [midcourse_search.py](midcourse_search.py)、[midcourse_transfers.py](midcourse_transfers.py) |
| 独立连续传播复验 | [trajectory_evaluation.py](trajectory_evaluation.py)：`evaluate` |

共两个入口、13个计算文件。旧版运行框架不保留在当前目录，旧代码仍可从 Git 历史查看。

## 结果应如何理解

这是一套可批量执行的多起点搜索与连续/结构优化方法。首轮二体费用只是候选指标，低二体费用、低松弛费用与低 J₂有效费用并不等价。固定平面和有限束宽会剪掉可能性；默认预算不保证复现历史约6.4 km/s，也不能保证算力增加就找到全局最优。

最终复验是**标定固定极轴 J₂ 模型下的独立 DOP853 连续传播**，不等同于官方 ATK 判分。此前讨论的“初筛前给全部候选做三脉冲潜力估计”尚未加入；中途脉冲搜索位于优化末级。

当前版改变了候选池随机抽样和部分计时边界，不保证与旧版整批随机实验路径一致。详细边界、2026-10-13的重构核对记录，以及2026-10-20的发布验算结果见 [METHOD.md](METHOD.md)。
