// mjlab / rsl_rl 训练出的 Noetix N2 速度策略（18 DoF，65 维观测）在 RK3588 NPU 上的推理封装。
//
// 与厂家自带的 walk/run 策略的区别（改代码前务必看清）：
//   * 18 个关节全部由策略控制（含手臂），不是只控 10 条腿关节。
//   * 观测 65 维、**不堆帧**（history_length = 1）。
//   * 观测**没有任何 obs_scales**，全是原始物理量（陀螺 rad/s、关节速度 rad/s）。
//   * 观测归一化 (obs-mean)/(std+1e-2) **已经编译进 .rknn 图里**，
//     所以这里喂进去的是原始观测，绝对不要再缩放或归一化一次。
//
// 观测布局（65）：
//   [0:3]   base_ang_vel        IMU 陀螺，机体系，rad/s
//   [3:6]   projected_gravity   R_wb^T * (0,0,-1)
//   [6:9]   command             vx (m/s), vy (m/s), wz (rad/s)
//   [9:11]  phase               [sin, cos](2*pi*t/0.6)；‖cmd‖<0.1 时置 0
//   [11:29] joint_pos_rel       q - default_q，rad
//   [29:47] joint_vel           dq，rad/s
//   [47:65] last_action         上一步网络原始输出
//
// 关节顺序与 SDK 的 get_joint_state()/set_joint() 完全一致：
//   arm_l1..l4, leg_l1..l5, arm_r1..r4, leg_r1..r5
#ifndef MJLAB_POLICY_H
#define MJLAB_POLICY_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "rknn_api.h"

namespace legged {

class MjlabPolicy {
 public:
  static constexpr int kNumJoints = 18;
  static constexpr int kObsDim = 65;
  static constexpr int kActDim = 18;
  static constexpr double kPolicyDt = 0.02;  // 训练时的 step_dt，phase 递推必须用这个

  struct Config {
    // --- 来自训练（不要随手改）---
    double defaultQ[kNumJoints];
    double kp[kNumJoints];
    double kd[kNumJoints];
    double actionScale[kNumJoints];
    double phasePeriod = 0.6;
    double phaseStandThreshold = 0.1;
    // --- 安全 ---
    double jointLimitLower[kNumJoints];
    double jointLimitUpper[kNumJoints];
    double clipActions = 10.0;      // 训练时是 null；这里只做异常兜底
    double fallGravityZ = -0.5;     // projected_gravity[2] 高于此值判定倾倒
    // --- 遥控器映射 ---
    double maxLinVelX = 0.8;
    double maxLinVelY = 0.0;    // 横移量程；轴来自 JsAxisReader（SDK 的 joydata 只有 2 轴）
    double constLinVelY = 0.0;  // 台架测试用的固定横移偏置，正常留 0
    double maxAngVelZ = 1.0;
    double axisDeadzone = 0.08;
    double signLinVelX = 1.0;
    double signLinVelY = 1.0;
    double signAngVelZ = 1.0;
    double commandLpfAlpha = 1.0;  // 1.0 = 不滤波（与训练一致）
    // 横移轴直读 js 设备（SDK 的 joydata 只暴露 2 个轴）
    std::string lateralDevice = "/dev/input/js0";
    int lateralAxisIndex = 2;
    bool lateralDebug = false;
    // --- 运行 ---
    int decimation = 10;           // 500 Hz 回调 / 10 = 50 Hz 策略
    int npuCoreMask = 0;           // 0 = NPU core0
    bool logFirstObs = true;
    int logCapacitySteps = 3000;   // 60 s @ 50 Hz，退出策略模式时落盘
    std::string logPath;           // 空 = 不记录
  };

  enum class Status : uint8_t {
    kOk = 0,
    kRknnError,
    kNonFiniteOutput,
    kJointDisconnected,
    kFallen,
  };

  MjlabPolicy() = default;
  ~MjlabPolicy();
  MjlabPolicy(const MjlabPolicy&) = delete;
  MjlabPolicy& operator=(const MjlabPolicy&) = delete;

  // yamlPath: config/mjlab_policy.yaml
  bool loadConfig(const std::string& yamlPath);
  // rknnPath: ning/policy_mjlab.rknn
  bool loadRknn(const std::string& rknnPath);

  // 进入策略模式前必须调用：清零 last_action、phase 计数、日志缓冲
  void reset();

  // 走一步策略。q / dq 按 SDK 关节顺序，长度 18。
  // gyro: 机体角速度 rad/s。quatXyzw: IMU 四元数（Eigen coeffs 顺序 x,y,z,w，与 SDK 一致）。
  // cmdAxes: 手柄原始轴，长度 3：
  //   [0] SDK joydata.axes[0]（横）-> wz
  //   [1] SDK joydata.axes[1]（纵）-> vx
  //   [2] 横移轴，来自 JsAxisReader；没有就传 0 -> vy
  // 内部做死区 + 缩放。
  // targetQOut: 输出的关节目标角（已按关节限位裁剪）。
  Status step(const double* q, const double* dq, const double gyro[3],
              const double quatXyzw[4], const double cmdAxes[3],
              double* targetQOut);

  const Config& config() const { return cfg_; }
  void setLogFirstObs(bool on) { cfg_.logFirstObs = on; }  // 测试用
  const float* lastObs() const { return obs_.data(); }
  const float* lastAction() const { return action_.data(); }
  const double* lastCommand() const { return command_; }
  int64_t policySteps() const { return policyStep_; }
  double lastInferMs() const { return lastInferMs_; }
  double worstInferMs() const { return worstInferMs_; }
  int limitClampCount() const { return limitClampCount_; }

  // 把缓存的 obs/action 落盘（退出策略模式时调用），返回写入行数
  int dumpLog();

  static const char* statusName(Status s);

 private:
  void buildObservation(const double* q, const double* dq, const double gyro[3],
                        const double quatXyzw[4], const double cmdAxes[3]);
  Status runInference();

  // fp16 转换（与厂家 usercontroller.cpp 里同一份实现，保持位级一致）
  static uint32_t asUint(float x);
  static float asFloat(uint32_t x);
  static uint16_t floatToHalf(float x);

  Config cfg_{};
  bool cfgLoaded_ = false;
  bool rknnLoaded_ = false;

  rknn_context ctx_ = 0;
  rknn_input_output_num ioNum_{};
  rknn_tensor_attr inAttr_{};
  rknn_tensor_attr outAttr_{};

  std::vector<float> obs_ = std::vector<float>(kObsDim, 0.0f);
  std::vector<uint16_t> obsFp16_ = std::vector<uint16_t>(kObsDim, 0);
  std::vector<float> action_ = std::vector<float>(kActDim, 0.0f);
  std::vector<float> lastAction_ = std::vector<float>(kActDim, 0.0f);

  double command_[3] = {0.0, 0.0, 0.0};
  double projGravityZ_ = -1.0;
  int64_t policyStep_ = 0;
  bool firstObsPrinted_ = false;
  double lastInferMs_ = 0.0;
  double worstInferMs_ = 0.0;
  int limitClampCount_ = 0;

  std::vector<float> log_;  // 每行 1 + kObsDim + kActDim 个 float
  int logRows_ = 0;
};

}  // namespace legged
#endif  // MJLAB_POLICY_H
