# Noetix N2 策略部署指南（mjlab → 真机）

本文说明如何把在本仓库（`unitree_rl_mjlab` / mjlab）中训练并导出的 **Noetix N2 速度跟踪策略**（`policy.onnx`）部署到真机上查看效果。

训练与移植部分见 [robot_porting_guide_zh.md](robot_porting_guide_zh.md)；本文只覆盖 **仿真到实机（sim2real）** 环节。

> 严格区分：**能导出 ONNX ≠ 能上真机**。真机还需关节顺序、观测顺序/缩放、默认角、PD、控制频率、IMU 坐标系完全对齐。本文给出这些对齐细节与逐步验证流程。

---

## 参考资料

- Noetix N2 二次开发 SDK（DDS，含 `highcontroller.cpp` / `lowcontroller.cpp`）：<https://github.com/Noetix-Robotics/noetix_sdk_n2>
- Noetix N2 Isaac Gym 训练框架（含 `sim2sim` 与厂商策略格式）：<https://github.com/Noetix-Robotics/noetix_n2_gym>
- Noetix 官方文档（飞书，需登录）：
  - <https://noetixrobotics.feishu.cn/docx/PP2BdXkSUoQ2VOxctJKcEdHanAg>
  - <https://noetixrobotics.feishu.cn/docx/A8ogdPukYokkQNxrsgzcXrUcn8d>
  - <https://noetixrobotics.feishu.cn/docx/RPG9djsyToZN7txok8icA2hHnBf>
- 本仓库相关文件：
  - 机器人资产/常量：`src/assets/robots/noetix_n2/`
  - 速度任务配置：`src/tasks/velocity/config/noetix_n2/`
  - **部署形态 sim2sim 参考脚本（onnxruntime + 手搓观测）**：`scripts/n2_sim2sim.py`

---

## 0. 关键前提：mjlab 策略 ≠ Noetix gym/SDK 默认策略

本仓库导出的策略与 Noetix `noetix_n2_gym` / SDK 自带策略**观测/推理约定不同，不能直接互换**：

| | 本仓库 mjlab 策略 | Noetix gym / SDK 默认策略 |
|---|---|---|
| 模型格式 | `policy.onnx`（onnxruntime） | JIT `policy.pt` / `.rknn` |
| 观测维度 | **65，单帧** | 63 × 5 帧堆叠 = 315 |
| 观测顺序 | `ang_vel, proj_gravity, cmd, phase(2), q-def, dq, last_action` | `cmd·scale, ang_vel·scale, proj_gravity, q-def, dq·0.05, action` |
| 相位 phase | 有（`sin/cos(2π·t/0.6)`） | 无（或不同实现） |
| 归一化 | **烘焙进 ONNX**（EmpiricalNormalization），不额外乘 obs_scales | 外部显式 `obs_scales` |

因此**必须按本仓库的观测规格重写 SDK 的观测拼装**（见第 5 节），并把 `obs_scales` 全设 1、`stack_size=1`。

---

## 1. 策略接口规格（来自 ONNX 元数据 + 代码，务必逐项对齐）

- **控制频率**：50 Hz（`control_dt = 0.02 s`）。
- **关节顺序（18，与 SDK 电机顺序完全一致）**：

| idx | SDK 名称 | 部位 | mjlab 关节 | kp | kd | 默认角(rad) | 扭矩 |
|--|--|--|--|--|--|--|--|
|0|arm_l1|左肩 pitch|L_arm_shoulder_pitch|30|1|0.0|27|
|1|arm_l2|左肩 roll|L_arm_shoulder_roll|30|1|0.2|27|
|2|arm_l3|左大臂 yaw|L_arm_shoulder_yaw|30|1|0.0|27|
|3|arm_l4|左肘 pitch|L_arm_elbow|30|1|0.0|27|
|4|leg_l1|左髋 yaw|L_leg_hip_yaw|80|5|0.0|90|
|5|leg_l2|左髋 roll|L_leg_hip_roll|80|5|0.0|90|
|6|leg_l3|左大腿 pitch|L_leg_hip_pitch|120|5|-0.1495|150|
|7|leg_l4|左膝 pitch（连杆）|L_leg_knee|120|5|0.3215|150|
|8|leg_l5|左踝 pitch（连杆）|L_leg_ankle|20|2|-0.172|70|
|9|arm_r1|右肩 pitch|R_arm_shoulder_pitch|30|1|0.0|27|
|10|arm_r2|右肩 roll|R_arm_shoulder_roll|30|1|-0.2|27|
|11|arm_r3|右大臂 yaw|R_arm_shoulder_yaw|30|1|0.0|27|
|12|arm_r4|右肘 pitch|R_arm_elbow|30|1|0.0|27|
|13|leg_r1|右髋 yaw|R_leg_hip_yaw|80|5|0.0|90|
|14|leg_r2|右髋 roll|R_leg_hip_roll|80|5|0.0|90|
|15|leg_r3|右大腿 pitch|R_leg_hip_pitch|120|5|-0.1495|150|
|16|leg_r4|右膝 pitch（连杆）|R_leg_knee|120|5|0.3215|150|
|17|leg_r5|右踝 pitch（连杆）|R_leg_ankle|20|2|-0.172|70|

> `leg_*4`(膝) 与 `leg_*5`(踝) 是**连杆关节**，正负号最容易与仿真不一致，必须逐个校验（第 7 节）。

- **动作缩放**：`action_scale = 0.25`，`目标角 = 默认角 + 0.25 × action`。
- **观测（65）顺序拼接**：

```
base_ang_vel(3) | projected_gravity(3) | command[vx,vy,wz](3) | phase[sin,cos](2)
| (jointPos - 默认角)(18) | jointVel(18) | last_action(18)
```

  - `projected_gravity = R(imu_quat)^T · [0,0,-1]`（基座系重力单位向量，站立≈`[0,0,-1]`）。
  - `phase`：`t` 为启动后累计时间，`[sin(2π·t/0.6), cos(2π·t/0.6)]`；当 `|command| < 0.1` 时置 `[0,0]`。
  - 归一化已在 ONNX 内部，**喂原始值，不乘 obs_scales，不堆叠帧**。
- **ONNX I/O**：输入 `obs`[1,65] → 输出 `actions`[1,18]。

---

## 2. 三种部署路径

| 方案 | 说明 | 适用 |
|---|---|---|
| **A. mjlab 自带仿真** | `python scripts/play.py Noetix-N2-Flat --checkpoint_file <...>/model_XXXX.pt` | 最快看效果（训练机上） |
| **B. sim2sim（部署形态）** | `scripts/n2_sim2sim.py`：onnxruntime + 手搓观测，在纯 MuJoCo 中跑，是真机代码的可运行原型 | 上机前验证观测重建 |
| **C. 真机（DDS）** | DDS 客户端 `lowcontrol` 跑你的策略，经 Noetix SDK 控制 N2 | 真机查看效果，带安全回退 |
| **D. 真机（板载）** | 策略直接跑在 RK3588S（`noetix_sdk_release`，EtherCAT 直连），单板脱机 | 见第 4 节，脱机自主 |

真机（C）又分两种“单板”接法：
- **C-Jetson**：客户端跑在机器人二开算力板（Jetson Orin Nano Super，教育版特有）。
- **C-PC（推荐先用）**：客户端跑在你自己的笔记本/台式机（x86_64），机器人只用 RK3588S 跑原厂 DDS 服务端。**保留断连自动回遥控的安全回退**，调试期最安全，无需第二块板。代价：运行时需网线拴一台 PC（适合测试/演示，不适合脱机自主）。

下面给出 **方案 C-PC** 的完整流程。

---

## 3. 方案 C-PC 完整教程

### 3.0 前置条件与安全

- **RK3588S 必须已刷“支持 DDS 的服务端 SDK”**（出厂默认不支持 DDS，需按 Noetix “更换 N2 SDK 流程”一次性刷入）。之后你不再改运控板，策略全在 PC 上跑。
- 全程把 N2 **吊在保护架上**；手柄 `+` = 急停（任何时候按都会立即失能瘫倒）。
- 断开 PC 网线 → 机器人自动回遥控器控制（安全网）。
- **务必先做 3.7 的三级验证，再让它走路。**

### 3.1 PC 环境（x86_64, Ubuntu 22.04）

```bash
sudo apt update
sudo apt install -y build-essential git cmake libeigen3-dev libfmt-dev \
    libspdlog-dev libgoogle-glog-dev libyaml-cpp-dev libboost-all-dev \
    libglib2.0-dev pybind11-dev python3-dev

# onnxruntime C++（预编译包，别从源码编）
cd ~/Downloads
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar -zxf onnxruntime-linux-x64-1.22.0.tgz
sudo cp -r onnxruntime-linux-x64-1.22.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.22.0/lib/*     /usr/local/lib/
sudo ldconfig
```

### 3.2 下载 SDK + 取策略

```bash
cd ~
git clone https://github.com/Noetix-Robotics/noetix_sdk_n2
cd noetix_sdk_n2
mkdir -p policy
# 从训练机拷 policy.onnx（训练日志目录：logs/rsl_rl/noetix_n2_velocity/<时间戳>/policy.onnx）
scp <训练机用户>@<训练机IP>:/home/cjy/unitree_rl_mjlab/logs/rsl_rl/noetix_n2_velocity/<时间戳>/policy.onnx ./policy/n2_mjlab.onnx
```

### 3.3 网络 + `config/dds.xml`

```bash
sudo ip addr add 192.168.55.100/24 dev <你的网口名>   # 用 ip a 查网口
ping 192.168.55.102                                    # 能通=已连上 RK3588S
```

```xml
<CycloneDDS><Domain id="any">
  <General>
    <NetworkInterfaceAddress>192.168.55.100</NetworkInterfaceAddress>  <!-- 本机(PC)IP -->
    <Transport>udp</Transport>
    <AllowMulticast>false</AllowMulticast>
  </General>
  <Discovery>
    <ParticipantIndex>0</ParticipantIndex>
    <Peers><Peer address="192.168.55.102"/></Peers>   <!-- RK3588S IP -->
  </Discovery>
</Domain></CycloneDDS>
```

`config/hwconfig.yaml`：澳加狮手柄 `remote_port: /dev/input/js0`；Noetix 定制手柄改 `/dev/remote`。

### 3.4 `config/ning_user.yaml`

让“准备模式”站姿等于策略默认姿态，切走路不顿挫：

```yaml
init_state:
  default_joint_angle:
    arm_l1_joint: 0.0
    arm_l2_joint: 0.2
    arm_l3_joint: 0.0
    arm_l4_joint: 0.0
    leg_l1_joint: 0.0
    leg_l2_joint: 0.0
    leg_l3_joint: -0.1495
    leg_l4_joint: 0.3215
    leg_l5_joint: -0.172
    arm_r1_joint: 0.0
    arm_r2_joint: -0.2
    arm_r3_joint: 0.0
    arm_r4_joint: 0.0
    leg_r1_joint: 0.0
    leg_r2_joint: 0.0
    leg_r3_joint: -0.1495
    leg_r4_joint: 0.3215
    leg_r5_joint: -0.172
control:
  stiffness: { arm_l1_joint: 30, arm_l2_joint: 30, arm_l3_joint: 30, arm_l4_joint: 30,
               leg_l1_joint: 80, leg_l2_joint: 80, leg_l3_joint: 120, leg_l4_joint: 120, leg_l5_joint: 20,
               arm_r1_joint: 30, arm_r2_joint: 30, arm_r3_joint: 30, arm_r4_joint: 30,
               leg_r1_joint: 80, leg_r2_joint: 80, leg_r3_joint: 120, leg_r4_joint: 120, leg_r5_joint: 20 }
  damping:   { arm_l1_joint: 1,  arm_l2_joint: 1,  arm_l3_joint: 1,  arm_l4_joint: 1,
               leg_l1_joint: 5,  leg_l2_joint: 5,  leg_l3_joint: 5,  leg_l4_joint: 5,  leg_l5_joint: 2,
               arm_r1_joint: 1,  arm_r2_joint: 1,  arm_r3_joint: 1,  arm_r4_joint: 1,
               leg_r1_joint: 5,  leg_r2_joint: 5,  leg_r3_joint: 5,  leg_r4_joint: 5,  leg_r5_joint: 2 }
  action_scale: 0.25
  decimation: 10          # 策略 50Hz
  cycle_time: 0.6
normalization:
  obs_scales: { linVel: 1.0, angVel: 1.0, dofPos: 1.0, dofVel: 1.0, quat: 1.0 }   # 全 1（归一化在 onnx 内）
size:
  observations_size: 65
  actions_size: 18
  stack_size: 1
```

### 3.5 改代码：`lowcontroller.cpp`

下述代码自带 onnx 会话与全部参数，只用到 SDK 公开成员（`motor_state_buffer_ / imu_buffer_ / joy_buffer_` 的 `GetData()`、`set_joint(std::array<MotorCmd,18>)`）。

**(a) `lowcontroller.cpp` 顶部（`#include` 之后）添加：**

```cpp
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>

static const int   NJ = 18;
static const char* POLICY_PATH = "./policy/n2_mjlab.onnx";   // 相对可执行文件
static const std::array<float,NJ> DEFAULT_Q = {
  0.f,0.2f,0.f,0.f,  0.f,0.f,-0.1495f,0.3215f,-0.172f,
  0.f,-0.2f,0.f,0.f, 0.f,0.f,-0.1495f,0.3215f,-0.172f};
static const std::array<float,NJ> KP = {
  30,30,30,30, 80,80,120,120,20, 30,30,30,30, 80,80,120,120,20};
static const std::array<float,NJ> KD = {
  1,1,1,1, 5,5,5,5,2, 1,1,1,1, 5,5,5,5,2};
static const float ACTION_SCALE = 0.25f;
static const float PHASE_PERIOD = 0.6f;
static const float CONTROL_DT   = 0.02f;   // 50 Hz
static const float ACTION_CLIP  = 5.0f;

// 关节方向（robot→policy）。先全 +1，第 3.7 节逐个测完再改。
static std::array<float,NJ> DIR = {1,1,1,1, 1,1,1,1,1, 1,1,1,1, 1,1,1,1,1};

static const float CMD_VX = 0.8f, CMD_WZ = 1.0f;   // 手柄→指令幅度（正负第 3.7 节验）
#define IMU_QUAT_WXYZ 1     // 若上机发现 ori[] 是 (x,y,z,w)，改成 0

// 验证开关（跑策略前分别开一次）
// #define N2_DRY_RUN     // 只保持默认姿态并打印观测（验 IMU/关节零位）
// #define N2_DIR_TEST    // 单关节慢速摆动（验方向）
static int g_test_joint = 6;

static Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "n2");
static std::unique_ptr<Ort::Session> g_sess;
static std::array<float,NJ> g_last_action{};
static long g_step = 0;

static void n2_init() {
  Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
  g_sess = std::make_unique<Ort::Session>(g_env, POLICY_PATH, so);
  g_last_action.fill(0.f); g_step = 0;
}
```

**(b) 每控制周期调用的核心函数（放 `lowcontroller.cpp`）：**

```cpp
void LowController::n2_step() {
  auto ms  = motor_state_buffer_.GetData();   // shared_ptr<const array<MotorState,18>>
  auto imu = imu_buffer_.GetData();
  auto joy = joy_buffer_.GetData();
  if (!ms || !imu) return;

  // 手柄 → 指令（axes[0]=转向, axes[1]=前后；正负第 3.7 节验）
  float vx = 0, vy = 0, wz = 0;
  if (joy) { vx = (float)joy->axes[1] * CMD_VX; wz = (float)joy->axes[0] * CMD_WZ; }
  float cmd_norm = std::sqrt(vx*vx + vy*vy + wz*wz);

  // IMU 四元数 → 基座系重力投影（站立≈(0,0,-1)）
#if IMU_QUAT_WXYZ
  Eigen::Quaterniond q(imu->ori[0], imu->ori[1], imu->ori[2], imu->ori[3]);
#else
  Eigen::Quaterniond q(imu->ori[3], imu->ori[0], imu->ori[1], imu->ori[2]);
#endif
  q.normalize();
  Eigen::Vector3d gb = q.conjugate() * Eigen::Vector3d(0, 0, -1);

  // 拼 65 维观测
  std::vector<float> obs; obs.reserve(65);
  obs.push_back((float)imu->angular_vel[0]);
  obs.push_back((float)imu->angular_vel[1]);
  obs.push_back((float)imu->angular_vel[2]);
  obs.push_back((float)gb.x()); obs.push_back((float)gb.y()); obs.push_back((float)gb.z());
  obs.push_back(vx); obs.push_back(vy); obs.push_back(wz);
  float t  = g_step * CONTROL_DT;
  float gp = std::fmod(t, PHASE_PERIOD) / PHASE_PERIOD;
  if (cmd_norm < 0.1f) { obs.push_back(0.f); obs.push_back(0.f); }
  else { obs.push_back(std::sin(2*M_PI*gp)); obs.push_back(std::cos(2*M_PI*gp)); }
  for (int i=0;i<NJ;i++) obs.push_back(DIR[i]*((float)ms->at(i).pos - DEFAULT_Q[i]));
  for (int i=0;i<NJ;i++) obs.push_back(DIR[i]*(float)ms->at(i).vel);
  for (int i=0;i<NJ;i++) obs.push_back(g_last_action[i]);

#ifdef N2_DRY_RUN
  if (g_step % 25 == 0)
    printf("grav=%.2f %.2f %.2f | angvel=%.2f %.2f %.2f | q-def[6]=%.3f\n",
           gb.x(),gb.y(),gb.z(), imu->angular_vel[0],imu->angular_vel[1],imu->angular_vel[2],
           (double)(ms->at(6).pos-DEFAULT_Q[6]));
  std::array<MotorCmd,18> hold{};
  for (int i=0;i<NJ;i++){ hold[i].motor_id=i; hold[i].pos=DEFAULT_Q[i]; hold[i].kp=KP[i]*0.5f; hold[i].kd=KD[i]; }
  set_joint(hold); g_step++; return;
#endif
#ifdef N2_DIR_TEST
  std::array<MotorCmd,18> tc{};
  for (int i=0;i<NJ;i++){ tc[i].motor_id=i; tc[i].pos=DEFAULT_Q[i]; tc[i].kp=KP[i]*0.5f; tc[i].kd=KD[i]; }
  tc[g_test_joint].pos = DEFAULT_Q[g_test_joint] + 0.15f*std::sin(2*M_PI*t/4.0f);
  set_joint(tc); g_step++; return;
#endif

  // onnx 推理：输入 "obs"[1,65] → 输出 "actions"[1,18]
  std::array<int64_t,2> shp{1, 65};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), shp.data(), 2);
  const char* in_names[]  = {"obs"};
  const char* out_names[] = {"actions"};
  auto out = g_sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  float* a = out[0].GetTensorMutableData<float>();

  // 下发电机指令（位置 PD，驱动器 500Hz 闭环）
  std::array<MotorCmd,18> cmd{};
  for (int i=0;i<NJ;i++){
    g_last_action[i] = a[i];                              // 回喂观测用未限幅原始动作
    float ac = std::fmax(-ACTION_CLIP, std::fmin(ACTION_CLIP, a[i]));
    cmd[i].motor_id = i;
    cmd[i].pos = DEFAULT_Q[i] + DIR[i]*ACTION_SCALE*ac;   // 目标角（机器人系）
    cmd[i].vel = 0; cmd[i].tau = 0;
    cmd[i].kp  = KP[i]; cmd[i].kd = KD[i];
  }
  set_joint(cmd);
  g_step++;
}
```

**(c) `lowcontroller.h` 的 `LowController` 类 `public:` 里声明：**

```cpp
void n2_step();
```

**(d) 接线（两处）：**
- 在 `LowController::init(...)` 里 `loadModel(...)` 之后加 `n2_init();`（onnx 只初始化一次）。
- 在 `LowController::handleWalkMode()` 里，把原本算观测/动作/下发的几行**替换为** `n2_step();`（仅走路模式每周期调用，准备/失能模式不动，安全）。

> 只改这 4 处；DDS 连接、订阅回调、`send_thread` 发布、模式切换全部沿用 SDK 原实现。若实际成员名有出入，按上表语义微调即可。

### 3.6 编译

```bash
cd ~/noetix_sdk_n2
./build_release.sh        # 生成可执行文件 lowcontrol
```

若报 `libonnxruntime` 找不到：确认 3.1 的 `ldconfig` 已执行，或在 CMakeLists 的 x86_64 分支 `target_link_libraries(lowcontrol ...)` 里确保有 `onnxruntime`。

### 3.7 三级验证（**跑策略前必做，按顺序，吊保护架**）

先手柄 `LB & -` 进准备模式（机器人到默认姿态）。

**① 观测健全性（`#define N2_DRY_RUN` 后重编译运行）**
- 站直：`grav ≈ (0,0,-1)`、`angvel ≈ 0`；前倾时 grav.x 变化、侧倾时 grav.y 变化。若约定不对 → 改 `IMU_QUAT_WXYZ` 或调符号。
- `q-def[6]` 等应 ≈ 0（编码器零位与默认角对齐）。

**② 逐关节方向（`#define N2_DIR_TEST`，`g_test_joint` 从 0 改到 17，每个重编译看一次）**
被测关节 ±0.15rad 慢摆；同时开仿真对照：
```bash
python src/assets/robots/noetix_n2/noetix_n2_constants.py   # 在 MuJoCo 里把同一关节 +0.15 看朝向
```
方向一致 → `DIR[i]=+1`；相反 → 改 `-1`。重点核对膝/踝（连杆）。全部测完填好 `DIR[]`。

**③ 手柄指令方向**：正式模式下小推摇杆，确认前进/转向方向正确，不对则翻 `CMD_VX`/`CMD_WZ` 符号。

### 3.8 正式运行

去掉 `N2_DRY_RUN`/`N2_DIR_TEST` 重编译。**保护架上**：
1. 上电 → 手柄使能 → `LB & -` 进准备模式（到默认姿态）。
2. PC：`sudo ./lowcontrol`（会 `publishModeData(1)` 切到 DDS 控制）。
3. `LB & X` 进走路模式 → 策略开始跑。
4. 左摇杆前后、右摇杆转向；**先极小指令**观察，再逐步加。
5. 急停：手柄 `+`；或拔 PC 网线（回遥控器控制）。

### 3.9 排错速查

| 现象 | 先查 |
|---|---|
| 一进走路就抽搐/自打架 | 某关节 `DIR[i]` 反了（3.7②）；或 IMU 四元数顺序错 |
| 一直往一个方向倒 | grav 投影轴/符号错（3.7①）；或膝/踝方向反 |
| 站得住但不跟指令 | 手柄 axes 映射/符号错（3.7③）；`cmd_norm<0.1` 一直触发 |
| 动作过大/过冲 | 确认 `action_scale=0.25`、`obs_scales` 全 1、`stack_size=1`（无二次缩放/堆叠） |
| onnx 报输入维度错 | 观测须正好 65；`observations_size=65` |
| `libssl.so.1.1` 缺失 | 装 `libssl1.1`（deb 包） |
| 机器人不理 DDS 指令 | RK3588S 未刷 DDS 服务端 SDK（3.0）；`ping 192.168.55.102` 不通查网口/IP |

---

## 4. 方案 D：RK3588S 板载直跑（EtherCAT 直连，单板、可脱机自主）

在运控板（RK3588S）上用原厂 `noetix_sdk_release`（“底层控制开发”）直接跑策略：`Controllerbase::loadModel` 支持 ONNX，经 EtherCAT 以 500Hz 闭环控电机。**不需要 Jetson、不需要外接电脑、不需要 DDS**。

> ⚠️ 权衡（务必知悉）：① 官方“逐步对用户封闭、不推荐”；② 你的推理与 EtherCAT 500Hz 主循环**共用 CPU**，某帧超时会 `Cycle time exceeded` 导致瘫倒（小 MLP 通常没事，循环里别加重日志/阻塞）；③ **没有 DDS 断连自动回遥控的安全网**（你的进程就是控制器）；④ **运行时禁止再开 SSH / 传文件**到 RK3588S（会抢 EtherCAT 带宽导致瘫倒）——编辑/编译在“未运行控制程序”时用 SSH 没问题。你有天车保护，摔倒风险可控，但上述实时约束仍要遵守。

观测/参数/`DIR` 校准结果与方案 C 完全一致，可直接复用；差别只在**运行位置、读写 API、构建方式**。

### 4.1 登录板子 + 取策略

RK3588S：`192.168.55.102`，用户 `noetix`，密码 `n`。用一台电脑接入机器人内网（本机设 `192.168.55.100/24`）后：

```bash
# 把 policy.onnx 拷到板上（在你的电脑上执行）
scp policy.onnx noetix@192.168.55.102:/home/noetix/work/noetix_sdk_release/policy/n2_mjlab.onnx
# 登录板子（仅用于编辑/编译，控制程序未运行时）
ssh noetix@192.168.55.102        # 密码 n
cd /home/noetix/work/noetix_sdk_release
```

- **onnxruntime 已在板上**：原厂 SDK 已链接 aarch64 版 `onnxruntime`，`loadModel` 直接可用（无需自己装）。
- 板上**自带编译环境**（aarch64），无需额外配置。

### 4.2 配置

- `config/ning_user.yaml`：**填入方案 C 第 3.4 节完全相同的值**（默认角、kp/kd、`action_scale=0.25`、`cycle_time=0.6`、`obs_scales` 全 1、`observations_size=65`、`actions_size=18`、`stack_size=1`）。
- `config/hwconfig.yaml`：`net_card` 设为 **EtherCAT 网口**（原厂日志里是 `eth0`；用 `ip a` 确认）；`remote_port` 按手柄类型设（澳加狮 `/dev/input/js0`，Noetix 定制手柄 `/dev/remote`）。
- **不需要 `dds.xml`**。

### 4.3 改代码：`src/usercontroller.cpp`

先在入口 `src/controller_example.cpp` 里解除 `#define usermode` 注释（切到用户控制模式）。

**(a) `usercontroller.cpp` 顶部添加参数块（与方案 C 3.5(a) 完全相同）：**

```cpp
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>

static const int   NJ = 18;
static const char* POLICY_PATH = "./policy/n2_mjlab.onnx";
static const std::array<float,NJ> DEFAULT_Q = {
  0.f,0.2f,0.f,0.f,  0.f,0.f,-0.1495f,0.3215f,-0.172f,
  0.f,-0.2f,0.f,0.f, 0.f,0.f,-0.1495f,0.3215f,-0.172f};
static const std::array<float,NJ> KP = {
  30,30,30,30, 80,80,120,120,20, 30,30,30,30, 80,80,120,120,20};
static const std::array<float,NJ> KD = {
  1,1,1,1, 5,5,5,5,2, 1,1,1,1, 5,5,5,5,2};
static const float ACTION_SCALE = 0.25f;
static const float PHASE_PERIOD = 0.6f;
static const float CONTROL_DT   = 0.02f;   // 50 Hz
static const float ACTION_CLIP  = 5.0f;
static std::array<float,NJ> DIR = {1,1,1,1, 1,1,1,1,1, 1,1,1,1, 1,1,1,1,1}; // 第4.5节校准后填
static const float CMD_VX = 0.8f, CMD_WZ = 1.0f;
#define IMU_QUAT_WXYZ 1
// #define N2_DRY_RUN
// #define N2_DIR_TEST
static int g_test_joint = 6;

static Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "n2");
static std::unique_ptr<Ort::Session> g_sess;
static std::array<float,NJ> g_last_action{};
static long g_step = 0;

static void n2_init() {
  Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
  g_sess = std::make_unique<Ort::Session>(g_env, POLICY_PATH, so);
  g_last_action.fill(0.f); g_step = 0;
}
```

**(b) 核心步进函数（板载 API：`get_joint_state()`/`get_imu_data()`/`get_jsdata()` 读，`set_joint(MotorCmd)` 单个电机下发）：**

```cpp
// 若 UserController 继承 Controllerbase，直接用 this->get_*/set_joint；
// 若是 setcallback(StaticCallback) 风格，则用你的 controllerbase 实例调用。
void UserController::n2_step() {
  std::array<MotorState,18> ms = get_joint_state();
  NingImuData imu = get_imu_data();
  joydata joy = get_jsdata();

  float vx = (float)joy.axes[1] * CMD_VX;   // 前后（正负第4.5节验）
  float wz = (float)joy.axes[0] * CMD_WZ;   // 转向
  float vy = 0.f;
  float cmd_norm = std::sqrt(vx*vx + vy*vy + wz*wz);

#if IMU_QUAT_WXYZ
  Eigen::Quaterniond q(imu.ori[0], imu.ori[1], imu.ori[2], imu.ori[3]);
#else
  Eigen::Quaterniond q(imu.ori[3], imu.ori[0], imu.ori[1], imu.ori[2]);
#endif
  q.normalize();
  Eigen::Vector3d gb = q.conjugate() * Eigen::Vector3d(0, 0, -1);

  std::vector<float> obs; obs.reserve(65);
  obs.push_back((float)imu.angular_vel[0]);
  obs.push_back((float)imu.angular_vel[1]);
  obs.push_back((float)imu.angular_vel[2]);
  obs.push_back((float)gb.x()); obs.push_back((float)gb.y()); obs.push_back((float)gb.z());
  obs.push_back(vx); obs.push_back(vy); obs.push_back(wz);
  float t  = g_step * CONTROL_DT;
  float gp = std::fmod(t, PHASE_PERIOD) / PHASE_PERIOD;
  if (cmd_norm < 0.1f) { obs.push_back(0.f); obs.push_back(0.f); }
  else { obs.push_back(std::sin(2*M_PI*gp)); obs.push_back(std::cos(2*M_PI*gp)); }
  for (int i=0;i<NJ;i++) obs.push_back(DIR[i]*((float)ms[i].pos - DEFAULT_Q[i]));
  for (int i=0;i<NJ;i++) obs.push_back(DIR[i]*(float)ms[i].vel);
  for (int i=0;i<NJ;i++) obs.push_back(g_last_action[i]);

  auto send = [&](std::function<float(int)> pos_of, float kp_scale){
    for (int i=0;i<NJ;i++){
      MotorCmd c; c.motor_id=i; c.vel=0; c.tau=0;
      c.kp = KP[i]*kp_scale; c.kd = KD[i]; c.pos = pos_of(i);
      set_joint(c);
    }
  };

#ifdef N2_DRY_RUN
  if (g_step % 25 == 0)
    printf("grav=%.2f %.2f %.2f | angvel=%.2f %.2f %.2f | q-def[6]=%.3f\n",
           gb.x(),gb.y(),gb.z(), imu.angular_vel[0],imu.angular_vel[1],imu.angular_vel[2],
           (double)(ms[6].pos-DEFAULT_Q[6]));
  send([&](int i){ return DEFAULT_Q[i]; }, 0.5f); g_step++; return;
#endif
#ifdef N2_DIR_TEST
  send([&](int i){ return (i==g_test_joint)
        ? DEFAULT_Q[i] + 0.15f*std::sin(2*M_PI*t/4.0f) : DEFAULT_Q[i]; }, 0.5f);
  g_step++; return;
#endif

  std::array<int64_t,2> shp{1, 65};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), shp.data(), 2);
  const char* in_names[]  = {"obs"};
  const char* out_names[] = {"actions"};
  auto out = g_sess->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  float* a = out[0].GetTensorMutableData<float>();

  for (int i=0;i<NJ;i++) g_last_action[i] = a[i];   // 回喂未限幅原始动作
  send([&](int i){
        float ac = std::fmax(-ACTION_CLIP, std::fmin(ACTION_CLIP, a[i]));
        return DEFAULT_Q[i] + DIR[i]*ACTION_SCALE*ac; }, 1.0f);
  g_step++;
}
```

**(c) 接线：**
- `UserController::init(...)`（`ControlMode::USERMODE`）里 `loadModel(...)` 之后加 `n2_init();`。
- 在 `handleWalkMode()`（USERWALK 模式每周期调用）里把默认的观测/动作/下发替换成 `n2_step();`。
- `usercontroller.h` 的 `public:` 加声明 `void n2_step();`。

> 站立模式 `handleStandMode()` 会把关节 ramp 到 `ning_user.yaml` 的默认角（= 我们的 `DEFAULT_Q`），因此切到 USERWALK 时 `jointPos-default≈0`，不会顿挫。

### 4.4 编译（在板上）

```bash
cd /home/noetix/work/noetix_sdk_release
./build_release.sh     # 生成 noetix_joint_controller
```

### 4.5 三级验证（**跑策略前必做，天车挂好**）

流程与方案 C 第 3.7 节完全一致，只是运行的是 `sudo ./noetix_joint_controller`：
1. `#define N2_DRY_RUN` → 重编译运行，保持默认姿态并打印观测：站直时 `grav≈(0,0,-1)`、`angvel≈0`，前/侧倾时对应分量变化；`q-def≈0`。不对就调 `IMU_QUAT_WXYZ`/符号。
2. `#define N2_DIR_TEST` → `g_test_joint` 0→17 逐个测，对照 `python src/assets/robots/noetix_n2/noetix_n2_constants.py` 里同关节 +0.15 的朝向；相反则该 `DIR[i]=-1`（重点核对膝/踝连杆）。
3. 去掉两个 `#define`，小推摇杆验证前进/转向方向，不对翻 `CMD_VX`/`CMD_WZ` 符号。

### 4.6 运行

**天车挂好**，地面平坦不打滑：
1. 机器人上电、手柄使能、进准备模式（到默认姿态）。
2. 板上运行：`sudo ./noetix_joint_controller`（**启动后不要再开新的 SSH 会话/传文件**）。
3. 手柄进入用户走路模式（USERWALK，随 SDK 默认按键，一般 `LB & X`）→ 策略开始跑。
4. 左摇杆前后、右摇杆转向；**先极小指令**观察，再逐步加。
5. 急停：手柄 `+`。

### 4.7（可选）开机自启动

编辑 `/home/noetix/work/startup.sh`，让其启动你的 `noetix_joint_controller`，即可上电自动运行、**全程无需 SSH**（最符合实时约束）：

```bash
#!/bin/bash
sleep 5
sudo -s << EOF
cd /home/noetix/work/noetix_sdk_release
./noetix_joint_controller
EOF
wait
```

### 4.8 排错

除沿用 3.9 表外，板载特有：

| 现象 | 先查 |
|---|---|
| `Cycle time exceeded` / 突然瘫倒 | 控制循环耗时超标：去掉打印/阻塞；确认没在运行时开 SSH/传文件 |
| EtherCAT 初始化失败 | `hwconfig.yaml` 的 `net_card` 不是 EtherCAT 网口；网线/从站问题 |
| `loadModel` 失败 | `POLICY_PATH` 相对可执行文件的路径不对；onnx 未拷到板上 |
| 一进走路就乱动 | `DIR[]`/IMU 顺序未校准（先做 4.5 三级验证） |

---

## 附：真机前先跑仿真

上真机前，**务必先在电脑上跑纯软件版验证策略本身**（逻辑与上面 C++ 完全一致：onnxruntime + 手搓观测）：

```bash
python scripts/n2_sim2sim.py \
  --onnx logs/rsl_rl/noetix_n2_velocity/<时间戳>/policy.onnx \
  --command 0.4 0.0 0.0 --duration 8 --video /tmp/n2_sim2sim.mp4
```

或直接在 mjlab 中回放：

```bash
python scripts/play.py Noetix-N2-Flat \
  --checkpoint_file logs/rsl_rl/noetix_n2_velocity/<时间戳>/model_XXXX.pt
```

---

## 边界说明

- 当前 Flat 策略在**平地**训练，真机也请先在平坦不打滑地面测试；上下坡/台阶需 Rough 策略训练验证后再用。
- 若使用**基础版 N2（无 Jetson）**：要么方案 C-PC（外接电脑），要么方案 D（RK3588S 板载直跑，见第 4 节）。
