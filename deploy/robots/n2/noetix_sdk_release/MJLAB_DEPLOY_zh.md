# mjlab 速度策略上机指南（RK3588 NPU）

模型来源：`logs/rsl_rl/noetix_n2_velocity/2026-08-13_09-11-54/model_10000.pt`
→ `ning/policy_mjlab.rknn`（rk3588 / float16 / 65 → 18）

## 0. TL;DR

```bash
# 板子上（aarch64）
cd <sdk_root>
./build_mjlab.sh

# 1) 先自检，不动电机
./n2_rknn_check ning/policy_mjlab.rknn ning/mjlab_golden.bin     # 要 PASS

# 2) 确认横移摇杆是几号轴，填进 config/mjlab_policy.yaml
./n2_js_probe

# 3) 机器人吊装好，再跑控制程序
./n2_mjlab_ctrl
```

手柄：`[9]` 使能 → `[10]+[2]` 拉到默认姿态 → `[5]+[2]` 开始走 → `[11]` 停下 → `[9]` 断使能。
摇杆：`axes[1]` 前后，`axes[0]` 转向，横移轴见 §4.5。

## 1. 新增了什么，动了什么

**完全没有改动厂家的任何代码**。`usercontroller.cpp`、`controller_example.cpp`、
`ning_user.yaml`、`ning_ac.yaml` 一个字节都没动，原来的 `noetix_joint_controller`
（walk / run 策略）行为不变，出问题可以随时切回去。

新增：

| 文件 | 说明 |
|---|---|
| `include/mjlab_policy.h`, `src/mjlab_policy.cpp` | 策略封装：观测拼接 + RKNN 推理 + 动作映射 + 安全裁剪 |
| `src/mjlab_controller_main.cpp` | 独立控制程序（自带状态机），编成 `n2_mjlab_ctrl` |
| `config/mjlab_policy.yaml` | 本策略的全部参数（default_q / kp / kd / 限位 / 遥控映射） |
| `ning/policy_mjlab.rknn` | NPU 模型 |
| `ning/mjlab_golden.bin` | 上板自检向量（32 组输入 + float64 参考输出） |
| `ning/policy_mjlab_meta.json` | 完整元信息，含归一化 mean/std（供排查用，运行时不读） |
| `include/js_axis_reader.h`, `src/js_axis_reader.cpp` | 直读 `/dev/input/js0` 取横移轴（SDK 的 joydata 只有 2 轴） |
| `tools/n2_rknn_check.cpp` | 自检程序，编成 `n2_rknn_check` |
| `tools/n2_js_probe.cpp` | 手柄轴探测，编成 `n2_js_probe` |
| `include/rknpu2/*.h`, `lib/aarch64/librknnrt.so` | rknn runtime 2.1.0，随包自带 |
| `build_mjlab.sh` | 只编这三个新目标 |

`include/rknpu2/` 是独立子目录、只挂在新增的 target 上，**不进全局
include_directories**，所以厂家二进制用的还是系统里的 `rknn_api.h`，编译行为零变化。

`CMakeLists.txt` 是唯一被改的厂家文件：末尾新增三个 target
（`n2_mjlab_ctrl` / `n2_rknn_check` / `n2_js_probe`），原有 target 一行没动。

## 2. 和厂家 walk/run 策略的关键差异

改任何代码前先看清这四条，弄错哪一条机器人都会站不住：

| | 厂家 walk / run | 本策略（mjlab） |
|---|---|---|
| 控制关节数 | 10（只有腿），手臂用衰减公式单独保持 | **18（含手臂），全部由策略输出** |
| 观测维度 | 760 / 1280（76 或 128 维 × 堆 5~10 帧） | **65，不堆帧** |
| 观测缩放 | `ning_ac.yaml` 里的 `obs_scales`（dof_vel×0.05 等） | **没有 obs_scales，全是原始物理量** |
| 归一化 | 无（legged_gym 风格固定缩放） | **`(obs-mean)/(std+1e-2)` 已编进 .rknn 图** |

⚠️ **最容易踩的坑**：因为归一化在 NPU 图里面，`mjlab_policy.cpp` 喂进去的是**原始观测**。
如果你照抄 `computeObservation()` 里的 `* obsScales.dofVel` 之类的缩放，
或者自己再做一遍 `(x-mean)/std`，动作会缩到接近 0，机器人直接软腿。

另外 **踝关节增益不一样**：本策略训练用 `kp=20 / kd=2`，
而 `ning_ac.yaml` 的 `run` 段是 `15 / 1`。`config/mjlab_policy.yaml` 里已经填的是训练值。

## 3. 观测布局（65 维，顺序错一位就废）

```
[0:3]   base_ang_vel       IMU 陀螺，机体系，rad/s，无缩放
[3:6]   projected_gravity  R_wb^T * (0,0,-1)
[6:9]   command            vx (m/s), vy (m/s), wz (rad/s)
[9:11]  phase              [sin, cos](2*pi*t/0.6)，‖cmd‖<0.1 时置 0
[11:29] joint_pos_rel      q - default_q，rad
[29:47] joint_vel          dq，rad/s，无缩放
[47:65] last_action        上一步网络原始输出（未裁剪）
```

关节顺序就是 SDK 的固定顺序，`get_joint_state()` / `set_joint(motor_id)` 直接对应：

```
0-3   arm_l1..arm_l4    (肩 pitch/roll/yaw, 肘)
4-8   leg_l1..leg_l5    (髋 yaw/roll/pitch, 膝, 踝)
9-12  arm_r1..arm_r4
13-17 leg_r1..leg_r5
```

几个实现细节：

* **phase 的时间基准**是策略步数 × 0.02 s，进入 POLICY 模式时从 0 开始计。
  厂家 `computeObservation()` 用的是 `system_clock` 绝对时间（`phase_/0.64`），
  那样相位起点是随机的，本策略没有沿用。
* `projected_gravity` 用的是 SDK 里 `quatToZyx` + `getRotationMatrixFromZyxEulerAngles().inverse()`
  这条已在实机验证过的换算，没有另写一套，避免坐标约定不一致。
* `last_action` 存的是裁剪后实际使用的动作（训练时无裁剪，正常情况下两者相同）。

## 4. 状态机与手柄

```
IDLE ──[9]──> DAMPING ──[10]+[2]──> HOLD_DEFAULT ──[5]+[2]──> POLICY
 ^               ^                        ^                      |
 └────[9]────────┴────────────────────────┴────────[11]──────────┘
                 ^                                                |
                 └──────────── 任何异常自动回落 ──────────────────┘
```

| 状态 | 行为 |
|---|---|
| `IDLE` | 零力矩（kp=0, kd=0） |
| `DAMPING` | 阻尼保持（kp=0, kd=0.1），等同厂家 DEFAULT 模式 |
| `HOLD_DEFAULT` | 1 秒内把 18 个关节平滑拉到 `default_q`，kp 从 0 线性升到训练值 |
| `POLICY` | 50 Hz 跑 NPU 策略 |

**为什么一定要经过 `HOLD_DEFAULT`**：本策略的 `default_q` 里
`arm_l2 = +0.2`、`arm_r2 = -0.2`（肩外展），而厂家 STAND 姿态这两个关节是 0。
差 0.2 rad 对肩 roll 通道来说超过 3 个训练标准差，直接切进策略第一帧观测就跑偏了。
`[5]+[2]` 会检查斜坡是否走完，没走完会拒绝启动。

### 4.5 摇杆与横移（vy）

SDK 的 `legged::joydata` 在 `common.h` 里写死了 `axes[2]`（horz, vert），
`librlcontrol.a` 也只填这两个，pybind 绑定同样是 2 —— 也就是说
**`Controllerbase::get_jsdata()` 根本拿不到第二个摇杆**，横移指令没法从这条路来。

所以 `n2_mjlab_ctrl` 额外自己 `open("/dev/input/js0", O_RDONLY|O_NONBLOCK)`
读第三个轴。Linux 的 joydev 驱动允许同一个 js 设备被多个进程各自打开，
每个 fd 有独立事件队列，跟 SDK 自己那一路互不干扰。

| 指令 | 来源 |
|---|---|
| `vx`（前后） | SDK `joydata.axes[1]` |
| `wz`（转向） | SDK `joydata.axes[0]` |
| `vy`（横移） | 直读 js0 的第 `lateral_axis.index` 号轴 |

**轴号必须先确认**，不同手柄不一样：

```bash
./n2_js_probe          # 把每个摇杆推到底，看哪个轴号在动
```

把左右方向那个轴号填进 `config/mjlab_policy.yaml`：

```yaml
  command:
    max_lin_vel_y: 0.2          # 设成 0 则完全关闭横移，也不会去开 js0
    sign_lin_vel_y: 1.0         # 平移方向反了就改 -1.0
    lateral_axis:
      device: /dev/input/js0
      index: 2                  # <- 填 n2_js_probe 看到的轴号
      debug: false              # true 时控制程序会限频打印动过的轴号
```

如果 js0 打不开（权限/设备名不对），程序会打印一行提示并把 `vy` 按 0 处理，
不影响前后和转向 —— 不会因为这个起不来。

> 训练时 `vy` 的命令范围是 `(-0.5, 0.5)`，`max_lin_vel_y` 别超过 0.5。

自动回落到 `DAMPING` 的四种情况（都会打印原因）：

* `joint_disconnected` — 某个电机 `|pos| == 12.5`
* `non_finite_output` — 观测或网络输出出现 NaN/Inf
* `fallen` — `projected_gravity[2] > -0.5`（躯干倾斜超过约 60°）
* `rknn_error` — NPU 调用失败

## 5. 上机前必须确认的三件事

### 5.1 自检要 PASS

```bash
./n2_rknn_check ning/policy_mjlab.rknn ning/mjlab_golden.bin
```

判据 `max|dAction| < 1e-2`（PC 模拟器实测 0.0027）。同时会打印推理耗时，
50 Hz 的预算是 20 ms，这个网络应该远低于此。
误差如果是 0.1 以上量级，别往下走 —— 那不是精度问题，是模型或驱动不对。

### 5.2 控制频率必须是 50 Hz

`hwconfig.yaml` 里 `loop_frequency: 500`，所以 `mjlab_policy.yaml` 里
`runtime.decimation: 10`。程序启动后会**实测回调频率并打印**，如果算出来不是 50 Hz
它会直接告诉你 decimation 该填多少：

```
[mjlab] measured callback rate 500.2 Hz -> policy 50.0 Hz (decimation 10)
```

⚠️ 厂家 `ning_user.yaml` 里 `decimation: 5`（→100 Hz），那是给他们的策略用的，别照抄。

### 5.3 IMU 方向要对

这是我**没法在开发机上验证**的一项。第一次上机时：

1. 打开 `runtime.log_first_obs: true`（默认开），看启动第一帧观测。
2. 机器人直立时 `obs[3:6]`（projected_gravity）应该 ≈ `[0, 0, -1]`。
   如果 z 是 `+1` 或者 x/y 有大分量，说明 IMU 四元数的分量顺序或坐标系约定和
   `updateStateEstimation()` 的假设（`imu.ori` 为 x,y,z,w）不一致，先解决这个再谈别的。
3. 前后 / 转向 / 横移方向反了，分别改 `command.sign_lin_vel_x` /
   `sign_ang_vel_z` / `sign_lin_vel_y`。

## 6. 第一次上机流程

1. 机器人**吊装**，上电，等进入零力矩。
2. `./n2_rknn_check ...` PASS。
3. `./n2_mjlab_ctrl`，确认打印的 `default_q` / `kp` / `kd` / 回调频率都对。
4. 按 `[9]` 使能（进阻尼），确认关节能被手推动但有阻尼感。
5. 按 `[10]+[2]`，看机器人在 1 秒内平滑到微屈膝 + 肩略外展的姿态。
   **这一步如果动作猛或者姿态明显不对，立刻按 `[9]` 断掉。**
6. `mjlab_policy.yaml` 里 `max_lin_vel_x` 保持默认的 **0.3**（训练上限是 1.2，别一上来就拉满）。
7. 按 `[5]+[2]` 起步，摇杆先不要动 —— 站立指令下 phase 会置 0，应该只是原地站住。
8. 摇杆缓慢给前进量，观察。确认前后没问题后再试转向、最后试横移。
   任何异常按 `[11]` 或 `[9]`。
9. 按 `[11]` 退出后，`/tmp/mjlab_obs_act.csv` 会落盘（最近 60 s 的观测和动作），
   拿回开发机和 `deploy/rknn/` 里的 float64 参考实现对一遍。

## 7. 已验证 / 未验证

已在开发机上验证：

* `mjlab_policy.cpp` 的观测拼接与 Python 参考实现一致
  —— 3200 步真实 rollout（含前进/后退/转向/左右横移/复合/站立六类指令），
  逐元素最大偏差 `3.2e-07`（float32 存储精度），`target_q` 偏差 `0.0`，
  `projected_gravity` 偏差 `3.0e-08`（即 SDK 的四元数换算与 scipy 一致）。
  跑 `deploy/rknn/run_parity_test.sh` 可复现。
* 厂家那份手写 `float_to_half` 与 numpy float16 有 14/156000 个 1-ULP 差异
  （进位 tie 规则不同），无 NaN/Inf，可放心使用。
* RKNN 图 float16 在闭环里与 float64 参考轨迹一致（PC 模拟器，`deploy/rknn/` §5.3）。
* 五个新增 C++ 文件在 x86 上编译通过（`-std=c++14 -Wall`，无 warning）；
  CMake 三个 target 用桩依赖实测能配置、能构建，RUNPATH 正确带上 `$ORIGIN/lib/aarch64`。

**未验证**（只能你在板子上做）：

* 真机 NPU 的数值与驱动行为 → 靠 §5.1 自检闭合。
* aarch64 上的实际链接与运行（开发机没有 `lib/x86_64`，也没有交叉编译器）。
* IMU 四元数分量顺序与坐标系 → 靠 §5.3 确认。
* 横移摇杆是 js0 上的第几号轴、方向正负 → 靠 `./n2_js_probe` 确认（§4.5）。
* 电机侧的实际 PD 响应（`set_joint` 的 kp/kd 单位与 mjlab 的 `stiffness/damping` 是否同一量纲）。
  厂家 walk 策略用同样的量级（80/120/20）在跑，所以大概率一致，但值得在吊装状态下确认。
