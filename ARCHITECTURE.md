# PCT Planner — 架构与实现笔记

本文档说明 PCT Planner 的**功能、导航表示、处理流程、代码地图与学习路线**。
偏"读代码/上手"视角，参数含义请看 [PARAMETERS.md](PARAMETERS.md)，安装请看 [SETUP.md](SETUP.md)。

论文：*Efficient Global Navigational Planning in 3-D Structures Based on Point Cloud Tomography* (TMECH 2024)

---

## 1. 一句话定位

**把一个 3D 点云地图"切片"成少数几张 2.5D 可通行性栅格图（tomogram），在其上做全局路径搜索 + 轨迹优化**，让地面机器人在多层 3D 结构（楼梯、坡道、天桥、多层室内）中高效地规划出一条平滑、可行、且满足机器人尺寸约束的轨迹。

**为什么不用现成方案：**

| 方案 | 能否表达多层 | 搜索复杂度 | 本项目 |
|---|---|---|---|
| 2D 占据栅格 | ✗ | O(N²) | ✗ |
| 高程图 (Elevation Map) | 勉强（只存一张地表） | O(N²) | 部分（每层存地面/天花板） |
| **tomogram（本项目）** | ✓ | O(N_layers · N²) | ✓ |
| 3D 体素 | ✓ | O(N³) | ✗ |

关键取舍：**分层 2.5D**。层数会被大幅简化（clinic 场景从初始几十层压到 **4 层**，见运行日志 `max layers: 4`），因此 A\* 只需 ~0.3 ms。

---

## 2. 导航依赖的表示：tomogram（断层图）

**不是纯 2D 栅格图，也不是单层高程图，而是"多层栅格 + 高程"的组合。**

tomogram 是一个 5 通道张量，存为 `rsc/tomogram/<scene>.pickle`（`data` 以 **float16** 存储）：

| 通道 | 代码名 | 含义 |
|------|--------|------|
| `[0]` | `trav` (`layers_t`) | **可通行代价**栅格。值越大越难走；`cost_barrier`(50) = 不可通行 |
| `[1]` `[2]` | `trav_gx`, `trav_gy` | 代价在 x / y 方向的**梯度**（供优化器构造解析梯度因子） |
| `[3]` | `elev_g` (`layers_g`) | **地面高程** ground height（无地面处为 NaN） |
| `[4]` | `elev_c` (`layers_c`) | **天花板高程** ceiling height（无遮挡处为 NaN） |

pickle 的其它 key：`resolution`(0.1 m)、`center`、`slice_h0`、`slice_dh`(0.5 m)。

**结构要点：**

- **每个高度切片是一张完整的 2D 栅格图**。
- **每格不存"占据/空闲"**，而存 **地面 g、天花板 c、代价 cost**；派生量 `interval = c - g` 表示该格的**净空高度**。
- 同一 `(x, y)` 在不同切片对应不同的物理面（地面、楼梯平台、天桥底面……），所以查询必须带高度提示消歧义 → C++ 侧 `GetValueBilinearSafe(layer, x, y, height_hint)`。
- C++ 对应容器是 **`DenseElevationMap`**（`planner/lib/src/map_manager/dense_elevation_map.h`），把 5 个通道拍平成 `(n_layers * max_y, max_x)` 的 `Eigen::MatrixXd`，支持双线性插值取代价/高度，以及 `UpdateLayer*` 在(x,y)处推断当前真实层号。

### ⚠️ z 坐标约定（易踩坑）

想清楚"某个像素的高度是多少"是读懂本项目可视化与调试的关键：

- **点云的地面高度存放在 `elev_g` 里，它本身就是 z。** 项目里 RViz 发布（`tomography.py:publishTomogram`、`run_ros2_interactive.py:_publish_tomo`）、planner 的高度图（`Astar` 的 `Node::height`）、轨迹输出（`transTrajGrid2Map` 的 `+0.5`）**全都直接用 `elev_g` 当 z**，而不是 `slice_h0 + s*slice_dh`。
- **导出的层是"简化后"的层**（`idx_simp`），**原始切片下标没有存进 pickle**；所以 `slice_h0 + s*slice_dh` 只能用于描述初始均匀切片，**不能用来索引导出后的层**。
- **`scene_*.py` 里的 `ground_h` 必须与 PCD 的实际 z 下限一致**——它会被写进 `points_min[-1]` 并决定 `slice_h0`。clinic 场景的 PCD 实际 z ∈ [-1.05, 4.2]，但配置里 `ground_h = -13.1` 是**过期值**，导致 `slice_h0 = -12.6` 完全脱离数据（最低那层因此没有任何地面格子）。改动 PCD 后务必同步 `ground_h`。
- 最低层没有地面格子的后果：planner 读到 `elev_g` 的 `-100` 哨兵值，**以该层为起终点会规划出 z ≈ -99.5 的"悬浮"路径**。`visualize_tomogram.py` 会检测并警告这种情况。

---

## 3. 处理流程

```
PCD 点云
   │  【阶段一 · 离线 · GPU/CuPy】tomography/scripts/
   ├─ point2map()                          tomogram.py:94
   │    ① tomographyKernel   点 → 各切片的地面 g / 天花板 c   kernels.py:99
   │    ② 地表粗糙度 grad_mag_sq / grad_mag_max               tomogram.py:110
   │    ③ travKernel         → 可通行代价 trav                kernels.py:133
   │    ④ inflationKernel    → 障碍膨胀 inflated_cost         kernels.py:201
   │    ⑤ 层简化 idx_simp    → 丢弃冗余切片                   tomogram.py:153
   ▼
tomogram (5, N_layer, H, W)  →  rsc/tomogram/<scene>.pickle
   │  【阶段二 · 在线 · C++/pybind11 + GTSAM】planner/
   ├─ TomogramPlanner.loadTomogram()       planner_wrapper.py
   ├─ initPlanner(): 算 gateway 层间通道标记  planner_wrapper.py:58
   ├─ A* 全局搜索（每层 2D + gateway 控层间跳转）a_star_search.cc
   ├─ GPMP 轨迹优化（GTSAM 因子图）          trajectory_optimization/
   ▼
trajectory (x, y, z)  →  rsc/<scene>_traj.npy  /  /pct_path (ROS2)
```

### 3.1 阶段一：点云 → tomogram

入口 `tomography/scripts/run_standalone.py --scene Clinic` → `tomography.py` → `tomogram.py::point2map()`。

**① `tomographyKernel`（kernels.py:99）—— "断层"操作的核心**

对每个点 `(px,py,pz)`，遍历所有切片高度 `slice`：

```
if pz <= slice:  layers_g[cell, s] = max(layers_g, pz)   # 切面以下最高的地面
else:            layers_c[cell, s] = min(layers_c, pz)   # 切面以上最低的天花板
```

用 `atomicMaxFloat` / `atomicMinFloat` 做并发归约。于是每张切片就是一个**水平切面**：哪里有地面、哪里有天花板、中间有多少自由空间——这正是 "tomography / CT 扫描" 的类比来源。

随后计算地表**粗糙度**（在 `layers_g` 上）：
- `grad_mag_sq = max(|g[i]-g[i-1]|², |g[i]-g[i+1]|²) + 同式(y)` —— 总粗糙度
- `grad_mag_max = max(前后向, 左右向)` —— 最陡方向

**② `travKernel`（kernels.py:133）—— 代价函数（最值得精读的一段）**

```
interval < interval_min(0.5)      → cost = 50 (barrier)   机器人钻不过去
否则                               → cost += 20*(interval_free(0.65) - interval)   净空不理想，惩罚
grad_mag_sq <= step_stand²        → cost += 15*grad²/step_stand²                 局部平坦，便宜
否则若 grad_mag_max <= step_max²  → 在 (2*half_k+1)² 窗口数"可站立格"
                                      太少(< standable_ratio) → cost = 50 (站不住)
                                      否则 cost += 20*grad_max/step_cross²        台阶可跨
否则                               → cost = 50 (barrier)                           台阶太高
```

其中 `step_stand = 1.2 * resolution * tan(slope_max)`（坡度上限换算成相邻格高差），`step_cross = step_max`（台阶上限）。

**③ `inflationKernel`（kernels.py:201）—— 障碍膨胀 = 机器人半径**

```
inflated_cost[i] = max over 邻域( trav_cost[neighbor] * score_table[dist] )
```

膨胀半径 `(safe_margin + inflation) / resolution`，`score_table` 随距离从 1 线性衰减到 0（在 `tomogram.py` 中构建）。等价于：高代价区域向外扩散并衰减，保证路径与障碍保持 `safe_margin` 以上的距离。

**④ 层简化（tomogram.py:153）**

逐层比较，只保留"带来新信息"的切片——地面更高、或代价更低、且下一层地面继续上升（即存在真实的层间分隔）：

```python
mask_l_g = layers_g[m] - layers_g[l] > 0        # 地面更高
mask_l_t = inflated_cost[l] > inflated_cost[m]  # 代价更低
mask_u_g = diff_h[m] > 0                        # 下一层地面上升（有分隔）
mask_t   = inflated_cost[m] < cost_barrier      # 本层可通行
unique   = (mask_l_g | mask_l_t) & mask_u_g & mask_t
```

这一步把层数从"按高度均匀切"的几十层压到少数几层，**直接决定 A\* 的搜索规模**。

最后输出 `(layers_t, trav_gx, trav_gy, layers_g, layers_c)` 并 `np.stack` 成 `(5, N_layer, H, W)` 存 pickle。

### 3.2 阶段二：tomogram → 轨迹

**加载与建图**（`planner/scripts/planner_wrapper.py`）

- `loadTomogram`：NaN 填充——`elev_g` → `-100`，`elev_c` → `1e6`（表示"无地面/无天花板"，室外场景常用）。
- `initPlanner`（planner_wrapper.py:58）：计算 **gateway 层间通道标记**：相邻切片间若代价骤降 `diff_t < -8` 且地面连续 `|Δg| < 0.1`，说明存在"向上开口"（楼梯口/天井），记 `+2`；反向记 `-2`。该值即 A\* `Node::ele` 字段。
- `init_map(20, 15, 0.2, ...)`（planner_wrapper.py:79）：依次是 `a_star_cost_threshold=20`、`safe_cost_margin=15`、`step_cost_weight=0.2`。

**A\* 全局搜索**（`planner/lib/src/a_star/a_star_search.cc`）

- 节点 `idx = (z, row, col)`，其中 **`z = height / resolution`**，即量化后的真实高程（不是层号）。
- 邻居：**当前层上的 8 邻域**（2D）。
- **`DecideLayer()`（a_star_search.cc:197）—— 多层机制的核心**：在 `(row,col)` 处试探 `layer + {0, -1, +1}`（`search_layer_depth_ = 1`，见 a_star_search.cc:56），只接受 `|height - cur_height| < 0.2` 的同物理面；若该处 `ele > 0.5` → 跳到上一层，`ele < -0.5` → 跳到下一层。
  → **A\* 本质是"每层做 2D 搜索"，层间切换由 gateway 图驱动**，而不是全 3D 搜索。这是效率的关键。
- 阻挡与例外（a_star_search.cc:148）：`cost > cost_threshold_` 的格子视为阻挡，**但 gateway 格子（`|ele| ≥ 0.5`）例外**，只要高差 < 0.3 就能穿过 —— 这是爬楼梯的机制。
- 步代价：`sqrt(Δz² + Δrow² + Δcol²) + step_cost_weight * neighbor.cost`（`cost < 5` 的格子代价归零 = 自由空间）。注意 **Δz 项就是爬升代价**。
- 启发式：3D octile 距离（`kDiagonal`，`sqrt(3)/sqrt(2)` 加权）。
- 起终点所在切片：交互模式下 `_z_to_slice()`（run_ros2_interactive.py:308）遍历所有简化切片，选出**地面高程最接近点击 z** 的那一片作为起/终点层，从而保证在该楼层上规划。

**GPMP 轨迹优化**（`planner/lib/src/trajectory_optimization/`）

拿到 A\* 的 `path_` 后（`ele_planner/offline_ele_planner.cc:31`）做平滑：

- **GPMP**（Gaussian Process Motion Planning），底层是 **GTSAM + Levenberg-Marquardt**。
- 两个变体：`GPMPOptimizerWnoa`（常速先验）与 `GPMPOptimizer`（wnoj）。`use_quintic=True` 时走后者（offline_ele_planner.cc:32）。
- 因子里加入**避障代价**（用 `safe_cost_margin`）+ **朝向约束**（`max_heading_rate`），迭代上限 200。
- 通过 `GetValueBilinearSafe(layer, x, y, height_hint)` 查图取代价与高度。

**输出**：`traj_3d = (x, y, z)`（地图坐标系）→ `rsc/<scene>_traj.npy`；ROS2 下发布 `/pct_path`。

**第三方依赖**：GTSAM 4.1.1（优化）、OSQP（QP）、pybind11（Python 绑定）、CuPy（GPU kernel）、Open3D（读 PCD / 可视化）。

---

## 4. 代码地图

| 路径 | 职责 |
|------|------|
| `tomography/scripts/kernels.py` | **算法核心**：三个 CuPy kernel（断层 / 代价 / 膨胀） |
| `tomography/scripts/tomogram.py` | GPU 数据流编排、地表粗糙度、层简化、输出张量 |
| `tomography/scripts/tomography.py` | 读 PCD、初始化地图尺寸、导出 pickle、ROS1 可视化发布 |
| `tomography/scripts/run_standalone.py` | 无 ROS 入口（`--scene Clinic`） |
| `tomography/config/scene_*.py` | 场景参数（分辨率、坡度/台阶上限、机器人尺寸…） |
| `planner/scripts/planner_wrapper.py` | 读 pickle、算 gateway、调 C++ planner、把结果转成地图坐标 |
| `planner/scripts/plan_standalone.py` | 无 ROS 入口，命令行跑一次规划 |
| `planner/scripts/plan_ros2.py` | ROS2 入口，规划并发布 `/pct_path` |
| `planner/lib/src/map_manager/dense_elevation_map.*` | 多层高程/代价图的数据结构与插值查询 |
| `planner/lib/src/a_star/a_star_search.*` | **多层 A\***：`Search()` + `DecideLayer()` |
| `planner/lib/src/ele_planner/offline_ele_planner.*` | 顶层编排：`InitMap` → A\* → GPMP |
| `planner/lib/src/trajectory_optimization/` | GPMP 因子图优化（GTSAM） |
| `run_ros2_interactive.py` | 本 fork 新增：RViz2 点击起终点、在线规划 |

调用链（记住这条主线）：
```
planner_wrapper.plan()  (planner_wrapper.py:89)
  → OfflineElePlanner::Plan()            offline_ele_planner.cc:19
      → Astar::Search()                   a_star_search.cc:87
      → GPMP*.GenerateTrajectory()        offline_ele_planner.cc:33
```

---

## 5. 学习路线

**第 0 步 — 先跑通，建立直觉**

```bash
cd tomography/scripts && python3 run_standalone.py --scene Clinic
cd ../../planner/scripts && python3 plan_standalone.py --tomo clinic --start 0 0 --end 10 5
```

再用 `run_ros2_interactive.py` + RViz2 点击起终点，**肉眼观察"分层 + 路径"**。

**第 1 步 — 读论文**，理解 tomography 的动机与 `interval` 的几何含义。

**第 2 步 — 从数据反推，把每一层画出来**

```bash
cd tomography/scripts
python3 visualize_tomogram.py --tomo clinic            # 3D 视图（默认）
python3 visualize_tomogram.py --tomo clinic --2d       # 2D 每层面板
python3 visualize_tomogram.py --tomo clinic --pcd      # 另存彩色点云 PLY
```

该脚本（`tomography/scripts/visualize_tomogram.py`，不需要 ROS/GPU）：

- **默认 3D 视图**：把各层按 `elev_g` 铺在真实高度上、按代价着色（绿=自由，红=障碍），
  可用 `--elev/--azim/--z-scale` 调视角、`--stride` 降采样；若存在
  `rsc/<stem>_traj.npy` 会自动叠加轨迹（`--traj` 指定、`--no-traj` 关闭）。
- **`--2d`**：每层渲染 **可通行代价 / 地面高程 / 净空** 三个面板。
- 两个模式都会打印控制台统计：每层的**地面 z 范围**、代价范围、`barrier%`（不可通行占比）、
  地面/天花板覆盖率、净空中位数 —— 对照 RViz2 里的彩色层，就能把"栅格图长什么样"
  和"代价函数的含义"对上。

也可以直接用 Python 读：

```python
import pickle, numpy as np
d = pickle.load(open('rsc/tomogram/clinic.pickle', 'rb'))
print(d['data'].shape)   # (5, N_layer, H, W)
```

**第 3 步 — 精读 `kernels.py`**（约 200 行，全项目最值得读的地方）。搞清 `interval / grad_mag_sq / standable / score_table` 的物理含义。实验：把 `slope_max` 从 `0.4` 改到 `0.7`，重跑 tomography，看可通行区域与路径如何变化。

**第 4 步 — C++ 规划侧**，按依赖顺序读：

```
dense_elevation_map.h/.cc   ← 数据结构与双线性插值
a_star_search.h/.cc         ← Search() + DecideLayer()（多层机制核心）
trajectory_optimization/    ← 最后读；GTSAM 不熟可先跳过细节
```

**第 5 步 — 调参实验闭环**：改 `tomography/config/scene_clinic.py` 下的参数**必须重跑 tomography**（重新生成 pickle）；改 `planner/config/param.py` 与 `init_map` 的参数**立即生效**。参考 [PARAMETERS.md](PARAMETERS.md)。

**关键心智模型**（抓住这条就抓住了主线）：

> 点云 → **每层一张 2.5D 栅格（存地面/天花板/代价）** → 层简化 → **每层 2D A\* + gateway 控层间跳转** → GPMP 平滑

---

## 6. 环境陷阱（本 fork 踩过的坑）

这些**都不是算法问题**，是 ROS2 移植过程中的动态链接问题；已在本仓库修复，改动见 README 的 "Compatibility Fixes Applied"。

| 现象 | 原因 | 处理 |
|------|------|------|
| `libgtsam.so.4: undefined symbol: get_thread_reference_vertex` | `import open3d` 会先加载 open3d 自带的 oneTBB 2021.5（与所需 2022.0.0 同为 SONAME `libtbb.so.12`），glibc 按 SONAME 复用 → 缺符号 | 在 `import open3d` **之前**用 `ctypes` 预加载 `/usr/local/lib/libtbb.so.12`（RTLD_GLOBAL） |
| 优化器 `SIGSEGV`，栈里同时出现 `/opt/ros/humble/.../libgtsam.so.4` 与项目自带 gtsam | ROS Humble 自带 GTSAM **4.2.0**，项目自带 **4.1.1**，SONAME 相同；`source ROS` 后 `/opt/ros/humble/lib/x86_64-linux-gnu` 排在 RUNPATH 之前 → 两者被混着加载，ABI 不匹配 | 绝对路径 `ctypes.CDLL(..., RTLD_GLOBAL)` 预加载项目自带的 4.1.1，钉死加载版本 |
| Ctrl-C / SIGTERM 退出时报 `rcl_shutdown already called` | `rclpy.shutdown()` 重复调用 | 改用 `rclpy.try_shutdown()` 并捕获 `ExternalShutdownException` |
| `--2d`/3D 绘图报 `cannot import name 'docstring' from 'matplotlib'` | apt 的 matplotlib 3.5 装了 `/usr/lib/python3/dist-packages/mpl_toolkits.pth`，它在解释器启动时就把旧 `mpl_toolkits` 塞进 `sys.modules`；而 pip 的新 matplotlib 只提供命名空间版的 `mpl_toolkits`，永远抢不过普通包 | `visualize_tomogram.py` 内的 `_import_mpl3d()` 把 `mpl_toolkits` 指向与已加载 matplotlib 同目录的那份；彻底修法是 `sudo apt remove python3-matplotlib` |

**结论：任何 import 了 pybind11 模块的 Python 入口，都必须在导入前把 `planner/lib` 与项目自带的 GTSAM/libTBB 用绝对路径 + `RTLD_GLOBAL` 预加载。** 参考 `run_ros2_interactive.py`、`plan_standalone.py`、`plan_ros2.py` 顶部的预加载代码块。

---

## 7. 参考

- 论文：[Efficient Global Navigational Planning in 3-D Structures Based on Point Cloud Tomography](https://ieeexplore.ieee.org/document/10531813) (TMECH 2024)
- 演示：[pct_planner](https://byangw.github.io/projects/tmech2024/)
- 相关项目：[3D2M-planner](https://github.com/ZJU-FAST-Lab/3D2M-planner)（Spiral 场景来源）
