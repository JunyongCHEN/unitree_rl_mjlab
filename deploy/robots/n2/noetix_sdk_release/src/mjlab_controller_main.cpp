// mjlab 速度策略在 N2 实机上的控制程序（独立可执行文件 n2_mjlab_ctrl）。
//
// 刻意不改动 usercontroller.cpp / controller_example.cpp —— 厂家自带的 walk/run
// 走的还是原来的 noetix_joint_controller，互不影响，出问题可以随时切回去。
//
// 状态机（手柄按键沿用厂家习惯）：
//
//   IDLE ──button[9]──> DAMPING ──button[10]+[2]──> HOLD_DEFAULT ──button[5]+[2]──> POLICY
//    ^                     ^                              ^                            |
//    └──button[9]──────────┴──────────────────────────────┴────────button[11]──────────┘
//                          ^                                                            |
//                          └────────────────── 任何异常自动回落 ────────────────────────┘
//
//   IDLE          零力矩（kp=0, kd=0）
//   DAMPING       阻尼保持（kp=0, kd=0.1），等同厂家的 DEFAULT
//   HOLD_DEFAULT  1 秒内把 18 个关节平滑拉到策略的 default_q，kp 由 0 线性升到训练值
//   POLICY        50 Hz 跑 NPU 策略
//
// 用法：
//   cd <sdk_root> && ./n2_mjlab_ctrl
//   可选参数：./n2_mjlab_ctrl [config/mjlab_policy.yaml] [ning/policy_mjlab.rknn]
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#include "controllerbase.h"
#include "js_axis_reader.h"
#include "mjlab_policy.h"

using namespace legged;

namespace {

constexpr int kNumJoints = MjlabPolicy::kNumJoints;
constexpr double kTargetPolicyHz = 50.0;  // 训练时的控制频率，不可改

double nowSec() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

class MjlabController {
 public:
  enum class Mode { kIdle, kDamping, kHoldDefault, kPolicy };

  static MjlabController* instance;
  static void staticCallback() {
    if (instance != nullptr) instance->process();
  }

  bool init(const std::string& cfgPath, const std::string& rknnPath) {
    if (!policy_.loadConfig(cfgPath)) return false;
    if (!policy_.loadRknn(rknnPath)) return false;
    // 横移轴：SDK 的 joydata 只有 2 个轴，第三个轴自己从 js0 读
    latDevice_ = policy_.config().lateralDevice;
    latAxisIndex_ = policy_.config().lateralAxisIndex;
    latDebug_ = policy_.config().lateralDebug;
    if (!latDevice_.empty() && policy_.config().maxLinVelY != 0.0) {
      js_.open(latDevice_);
      if (js_.isOpen())
        printf("[mjlab] lateral axis: %s index %d (max_lin_vel_y %.2f)\n",
               latDevice_.c_str(), latAxisIndex_, policy_.config().maxLinVelY);
    } else {
      printf("[mjlab] lateral axis disabled (max_lin_vel_y = %.2f)\n",
             policy_.config().maxLinVelY);
    }
    rampSteps_ = 0;  // 在进入 HOLD_DEFAULT 时按实测回调频率确定
    instance = this;
    base_.setcallback(staticCallback);
    if (!base_.init(ControlMode::USERMODE)) {
      printf("[mjlab] Controllerbase::init(USERMODE) failed\n");
      return false;
    }
    ready_ = true;
    printf("\n[mjlab] ready. buttons: [9]=enable/disable  [10]+[2]=hold default pose  "
           "[5]+[2]=start policy  [11]=back to hold\n\n");
    return true;
  }

  void process() {
    if (!ready_) return;
    measureLoopRate();

    js_.poll();  // 排空 js0 事件队列（非阻塞，很便宜）
    if (latDebug_) reportMovedAxis();

    const joydata js = base_.get_jsdata();
    handleButtons(js);

    switch (mode_) {
      case Mode::kIdle:         applyIdle();                break;
      case Mode::kDamping:      applyDamping();             break;
      case Mode::kHoldDefault:  applyHoldDefault();         break;
      case Mode::kPolicy:       applyPolicy(js);            break;
    }
    ++callbackCount_;
  }

 private:
  void setMode(Mode m, const char* why) {
    if (mode_ == m) return;
    if (mode_ == Mode::kPolicy) {
      printf("[mjlab] policy stats: %lld steps, infer last %.3f ms worst %.3f ms, "
             "limit clamps %d\n",
             static_cast<long long>(policy_.policySteps()), policy_.lastInferMs(),
             policy_.worstInferMs(), policy_.limitClampCount());
      policy_.dumpLog();
    }
    mode_ = m;
    const char* name = "?";
    switch (m) {
      case Mode::kIdle: name = "IDLE"; break;
      case Mode::kDamping: name = "DAMPING"; break;
      case Mode::kHoldDefault: name = "HOLD_DEFAULT"; break;
      case Mode::kPolicy: name = "POLICY"; break;
    }
    printf("[mjlab] -> %s (%s)\n", name, why);
  }

  void measureLoopRate() {
    if (loopHz_ > 0.0) return;
    if (callbackCount_ == 0) {
      rateT0_ = nowSec();
      return;
    }
    if (callbackCount_ < 500) return;
    const double dt = nowSec() - rateT0_;
    if (dt <= 0.0) return;
    loopHz_ = callbackCount_ / dt;
    const double policyHz = loopHz_ / policy_.config().decimation;
    printf("[mjlab] measured callback rate %.1f Hz -> policy %.1f Hz (decimation %d)\n",
           loopHz_, policyHz, policy_.config().decimation);
    if (std::fabs(policyHz - kTargetPolicyHz) > 5.0) {
      printf("[mjlab] !! policy rate is %.1f Hz but the model was trained at %.0f Hz.\n"
             "[mjlab] !! set runtime.decimation = %d in mjlab_policy.yaml and restart.\n",
             policyHz, kTargetPolicyHz,
             static_cast<int>(std::lround(loopHz_ / kTargetPolicyHz)));
    }
  }

  void handleButtons(const joydata& js) {
    // button[9]：使能 / 全部关闭
    if (js.button[9] == 1 && keyflag_[9] == 0) {
      keyflag_[9] = 1;
      setMode(mode_ == Mode::kIdle ? Mode::kDamping : Mode::kIdle, "button 9");
    } else if (js.button[9] == 0) {
      keyflag_[9] = 0;
    }
    // button[10]+[2]：进入保持默认姿态
    if (js.button[10] == 1 && js.button[2] == 1 && keyflag_[10] == 0) {
      keyflag_[10] = 1;
      if (mode_ == Mode::kDamping || mode_ == Mode::kPolicy) enterHoldDefault("button 10+2");
    } else if (js.button[10] == 0) {
      keyflag_[10] = 0;
    }
    // button[5]+[2]：开始跑策略（必须先在 HOLD_DEFAULT 并且姿态已到位）
    if (js.button[5] == 1 && js.button[2] == 1 && keyflag_[5] == 0) {
      keyflag_[5] = 1;
      if (mode_ == Mode::kHoldDefault) {
        if (rampAlpha_ >= 1.0) {
          policy_.reset();
          decimCount_ = 0;
          haveTarget_ = false;
          setMode(Mode::kPolicy, "button 5+2");
        } else {
          printf("[mjlab] refusing to start: still ramping to default pose (%.0f%%)\n",
                 rampAlpha_ * 100.0);
        }
      }
    } else if (js.button[5] == 0) {
      keyflag_[5] = 0;
    }
    // button[11]：退出策略，回到保持默认姿态
    if (js.button[11] == 1 && keyflag_[11] == 0) {
      keyflag_[11] = 1;
      if (mode_ == Mode::kPolicy) enterHoldDefault("button 11");
    } else if (js.button[11] == 0) {
      keyflag_[11] = 0;
    }
  }

  // 找横移摇杆是几号轴：把手柄各摇杆都推一下，看这里打印的轴号
  void reportMovedAxis() {
    const int a = js_.takeMovedAxis();
    if (a < 0) return;
    if (callbackCount_ - lastAxisPrint_ < 250) return;  // 限频，约 0.5 s
    lastAxisPrint_ = callbackCount_;
    printf("[mjlab] js axis %d = %+.2f   <- 想用它做横移就把 index 填成 %d\n",
           a, js_.axis(a), a);
  }

  void enterHoldDefault(const char* why) {
    const std::array<MotorState, 18> st = base_.get_joint_state();
    for (int i = 0; i < kNumJoints; ++i) rampStartQ_[i] = st[i].pos;
    // 1 秒斜坡；回调频率还没测出来时按 500 Hz 保守估计
    const double hz = loopHz_ > 0.0 ? loopHz_ : 500.0;
    rampSteps_ = static_cast<int>(hz);
    rampCount_ = 0;
    rampAlpha_ = 0.0;
    setMode(Mode::kHoldDefault, why);
  }

  void sendJoint(int j, double pos, double kp, double kd) {
    MotorCmd cmd;
    cmd.pos = pos;
    cmd.vel = 0.0;
    cmd.tau = 0.0;
    cmd.kp = kp;
    cmd.kd = kd;
    cmd.motor_id = static_cast<uint16_t>(j);
    base_.set_joint(cmd);
  }

  void applyIdle() {
    for (int j = 0; j < kNumJoints; ++j) sendJoint(j, 0.0, 0.0, 0.0);
  }

  void applyDamping() {
    for (int j = 0; j < kNumJoints; ++j) sendJoint(j, 0.0, 0.0, 0.1);
  }

  void applyHoldDefault() {
    const MjlabPolicy::Config& c = policy_.config();
    if (rampSteps_ > 0 && rampCount_ < rampSteps_) {
      ++rampCount_;
      rampAlpha_ = static_cast<double>(rampCount_) / rampSteps_;
    } else {
      rampAlpha_ = 1.0;
    }
    for (int j = 0; j < kNumJoints; ++j) {
      const double pos = rampStartQ_[j] * (1.0 - rampAlpha_) + c.defaultQ[j] * rampAlpha_;
      // kp 跟着斜坡从 0 升到训练值，避免一上电就用 120 的刚度硬拽
      sendJoint(j, pos, c.kp[j] * rampAlpha_, c.kd[j]);
    }
  }

  void applyPolicy(const joydata& js) {
    const MjlabPolicy::Config& c = policy_.config();

    if (decimCount_ % c.decimation == 0) {
      decimCount_ = 0;
      const std::array<MotorState, 18> st = base_.get_joint_state();
      const NingImuData imu = base_.get_imu_data();
      double q[kNumJoints], dq[kNumJoints];
      for (int i = 0; i < kNumJoints; ++i) {
        q[i] = st[i].pos;
        dq[i] = st[i].vel;
      }
      const double gyro[3] = {imu.angular_vel[0], imu.angular_vel[1], imu.angular_vel[2]};
      // Eigen 四元数 coeffs 顺序是 (x,y,z,w)，与厂家 updateStateEstimation() 的用法一致
      const double quat[4] = {imu.ori[0], imu.ori[1], imu.ori[2], imu.ori[3]};
      const double axes[3] = {js.axes[0], js.axes[1], js_.axis(latAxisIndex_)};

      const MjlabPolicy::Status s = policy_.step(q, dq, gyro, quat, axes, targetQ_);
      if (s != MjlabPolicy::Status::kOk) {
        printf("[mjlab] FAULT: %s -> falling back to DAMPING\n", MjlabPolicy::statusName(s));
        setMode(Mode::kDamping, "fault");
        return;
      }
      if (policy_.lastInferMs() > 1000.0 / kTargetPolicyHz) {
        printf("[mjlab] WARNING: inference took %.2f ms (> %.1f ms budget)\n",
               policy_.lastInferMs(), 1000.0 / kTargetPolicyHz);
      }
      haveTarget_ = true;
    }
    ++decimCount_;

    if (!haveTarget_) return;  // 第一帧还没算出来就先什么都不发
    for (int j = 0; j < kNumJoints; ++j) sendJoint(j, targetQ_[j], c.kp[j], c.kd[j]);
  }

  Controllerbase base_;
  MjlabPolicy policy_;
  JsAxisReader js_;
  std::string latDevice_;
  int latAxisIndex_ = 2;
  bool latDebug_ = false;
  long long lastAxisPrint_ = -1000;
  bool ready_ = false;
  Mode mode_ = Mode::kIdle;
  int keyflag_[14] = {0};

  long long callbackCount_ = 0;
  double rateT0_ = 0.0;
  double loopHz_ = 0.0;

  int decimCount_ = 0;
  bool haveTarget_ = false;
  double targetQ_[kNumJoints] = {0.0};

  double rampStartQ_[kNumJoints] = {0.0};
  int rampSteps_ = 0;
  int rampCount_ = 0;
  double rampAlpha_ = 0.0;
};

MjlabController* MjlabController::instance = nullptr;

}  // namespace

int main(int argc, char** argv) {
  char buf[512];
  if (getcwd(buf, sizeof(buf)) == nullptr) {
    printf("getcwd failed\n");
    return 1;
  }
  const std::string root(buf);
  const std::string cfgPath = (argc > 1) ? argv[1] : root + "/config/mjlab_policy.yaml";
  const std::string rknnPath = (argc > 2) ? argv[2] : root + "/ning/policy_mjlab.rknn";
  printf("[mjlab] cwd    : %s\n", root.c_str());
  printf("[mjlab] config : %s\n", cfgPath.c_str());
  printf("[mjlab] model  : %s\n", rknnPath.c_str());

  static MjlabController controller;
  if (!controller.init(cfgPath, rknnPath)) {
    printf("[mjlab] init failed, exiting\n");
    return 1;
  }
  while (true) usleep(1000);
  return 0;
}
