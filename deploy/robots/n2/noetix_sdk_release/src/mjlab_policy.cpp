#include "mjlab_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "RotationTools.h"
#include "yaml-cpp/yaml.h"

namespace legged {
namespace {

// SDK 的固定关节顺序，与 get_joint_state()/set_joint(motor_id) 一一对应
const char* kJointNames[MjlabPolicy::kNumJoints] = {
    "arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint",
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
    "arm_r1_joint", "arm_r2_joint", "arm_r3_joint", "arm_r4_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"};

double nowMs() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// 读一个 map<关节名, double> 到长度 18 的数组，缺任何一个关节都算失败
bool readJointMap(const YAML::Node& node, const char* what, double* out) {
  if (!node || !node.IsMap()) {
    printf("[mjlab] config: missing or non-map section '%s'\n", what);
    return false;
  }
  for (int i = 0; i < MjlabPolicy::kNumJoints; ++i) {
    const YAML::Node v = node[kJointNames[i]];
    if (!v) {
      printf("[mjlab] config: '%s' missing joint '%s'\n", what, kJointNames[i]);
      return false;
    }
    out[i] = v.as<double>();
  }
  return true;
}

double applyDeadzone(double v, double dz) {
  if (v > -dz && v < dz) return 0.0;
  return v;
}

}  // namespace

MjlabPolicy::~MjlabPolicy() {
  if (ctx_ != 0) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
}

const char* MjlabPolicy::statusName(Status s) {
  switch (s) {
    case Status::kOk: return "ok";
    case Status::kRknnError: return "rknn_error";
    case Status::kNonFiniteOutput: return "non_finite_output";
    case Status::kJointDisconnected: return "joint_disconnected";
    case Status::kFallen: return "fallen";
  }
  return "unknown";
}

uint32_t MjlabPolicy::asUint(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

float MjlabPolicy::asFloat(uint32_t x) {
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

// 与厂家 usercontroller.cpp::float_to_half 完全同一套位运算
uint16_t MjlabPolicy::floatToHalf(float x) {
  const uint32_t b = asUint(x) + 0x00001000;
  const uint32_t e = (b & 0x7f800000) >> 23;
  const uint32_t m = b & 0x007fffff;
  return static_cast<uint16_t>((b & 0x80000000) >> 16 |
                               (e > 112) * ((((e - 112) << 10) & 0x7c00) | m >> 13) |
                               ((e < 113) & (e > 101)) * ((((0x007ff000 + m) >> (125 - e)) + 1) >> 1) |
                               (e > 143) * 0x7fff);
}

bool MjlabPolicy::loadConfig(const std::string& yamlPath) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(yamlPath);
  } catch (const std::exception& e) {
    printf("[mjlab] cannot load config '%s': %s\n", yamlPath.c_str(), e.what());
    return false;
  }
  const YAML::Node n = root["mjlab_policy"];
  if (!n) {
    printf("[mjlab] config: missing top-level key 'mjlab_policy'\n");
    return false;
  }

  if (!readJointMap(n["default_joint_angle"], "default_joint_angle", cfg_.defaultQ)) return false;
  if (!readJointMap(n["stiffness"], "stiffness", cfg_.kp)) return false;
  if (!readJointMap(n["damping"], "damping", cfg_.kd)) return false;
  if (!readJointMap(n["joint_limit_lower"], "joint_limit_lower", cfg_.jointLimitLower)) return false;
  if (!readJointMap(n["joint_limit_upper"], "joint_limit_upper", cfg_.jointLimitUpper)) return false;

  // action_scale 允许写成单个标量（训练里全关节都是 0.25）或按关节的 map
  const YAML::Node as = n["action_scale"];
  if (as && as.IsScalar()) {
    const double v = as.as<double>();
    for (int i = 0; i < kNumJoints; ++i) cfg_.actionScale[i] = v;
  } else if (!readJointMap(as, "action_scale", cfg_.actionScale)) {
    return false;
  }

  const YAML::Node o = n["observation"];
  if (o) {
    if (o["phase_period"]) cfg_.phasePeriod = o["phase_period"].as<double>();
    if (o["phase_stand_threshold"]) cfg_.phaseStandThreshold = o["phase_stand_threshold"].as<double>();
  }
  const YAML::Node s = n["safety"];
  if (s) {
    if (s["clip_actions"]) cfg_.clipActions = s["clip_actions"].as<double>();
    if (s["fall_gravity_z"]) cfg_.fallGravityZ = s["fall_gravity_z"].as<double>();
  }
  const YAML::Node c = n["command"];
  if (c) {
    if (c["max_lin_vel_x"]) cfg_.maxLinVelX = c["max_lin_vel_x"].as<double>();
    if (c["max_lin_vel_y"]) cfg_.maxLinVelY = c["max_lin_vel_y"].as<double>();
    if (c["const_lin_vel_y"]) cfg_.constLinVelY = c["const_lin_vel_y"].as<double>();
    if (c["sign_lin_vel_y"]) cfg_.signLinVelY = c["sign_lin_vel_y"].as<double>();
    if (c["max_ang_vel_z"]) cfg_.maxAngVelZ = c["max_ang_vel_z"].as<double>();
    if (c["axis_deadzone"]) cfg_.axisDeadzone = c["axis_deadzone"].as<double>();
    if (c["sign_lin_vel_x"]) cfg_.signLinVelX = c["sign_lin_vel_x"].as<double>();
    if (c["sign_ang_vel_z"]) cfg_.signAngVelZ = c["sign_ang_vel_z"].as<double>();
    if (c["lpf_alpha"]) cfg_.commandLpfAlpha = c["lpf_alpha"].as<double>();
    const YAML::Node lat = c["lateral_axis"];
    if (lat) {
      if (lat["device"]) cfg_.lateralDevice = lat["device"].as<std::string>();
      if (lat["index"]) cfg_.lateralAxisIndex = lat["index"].as<int>();
      if (lat["debug"]) cfg_.lateralDebug = lat["debug"].as<bool>();
    }
  }
  const YAML::Node r = n["runtime"];
  if (r) {
    if (r["decimation"]) cfg_.decimation = r["decimation"].as<int>();
    if (r["npu_core_mask"]) cfg_.npuCoreMask = r["npu_core_mask"].as<int>();
    if (r["log_first_obs"]) cfg_.logFirstObs = r["log_first_obs"].as<bool>();
    if (r["log_capacity_steps"]) cfg_.logCapacitySteps = r["log_capacity_steps"].as<int>();
    if (r["log_path"]) cfg_.logPath = r["log_path"].as<std::string>();
  }

  if (cfg_.decimation <= 0) {
    printf("[mjlab] config: decimation must be > 0\n");
    return false;
  }
  if (!cfg_.logPath.empty() && cfg_.logCapacitySteps > 0) {
    log_.assign(static_cast<size_t>(cfg_.logCapacitySteps) * (1 + kObsDim + kActDim), 0.0f);
  }

  printf("[mjlab] config loaded from %s\n", yamlPath.c_str());
  printf("[mjlab]   default_q  :");
  for (int i = 0; i < kNumJoints; ++i) printf(" %.4f", cfg_.defaultQ[i]);
  printf("\n[mjlab]   kp         :");
  for (int i = 0; i < kNumJoints; ++i) printf(" %.0f", cfg_.kp[i]);
  printf("\n[mjlab]   kd         :");
  for (int i = 0; i < kNumJoints; ++i) printf(" %.1f", cfg_.kd[i]);
  printf("\n[mjlab]   action_scale %.3f  decimation %d  phase_period %.3f\n",
         cfg_.actionScale[0], cfg_.decimation, cfg_.phasePeriod);
  printf("[mjlab]   cmd limits: vx %.2f  vy %.2f  wz %.2f  (deadzone %.3f)\n",
         cfg_.maxLinVelX, cfg_.maxLinVelY, cfg_.maxAngVelZ, cfg_.axisDeadzone);
  cfgLoaded_ = true;
  return true;
}

bool MjlabPolicy::loadRknn(const std::string& rknnPath) {
  FILE* fp = fopen(rknnPath.c_str(), "rb");
  if (fp == nullptr) {
    printf("[mjlab] cannot open rknn model '%s'\n", rknnPath.c_str());
    return false;
  }
  fseek(fp, 0, SEEK_END);
  const long len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  std::vector<unsigned char> blob(static_cast<size_t>(len));
  const size_t got = fread(blob.data(), 1, static_cast<size_t>(len), fp);
  fclose(fp);
  if (got != static_cast<size_t>(len)) {
    printf("[mjlab] short read on '%s'\n", rknnPath.c_str());
    return false;
  }

  int ret = rknn_init(&ctx_, blob.data(), static_cast<int>(len), 0, nullptr);
  if (ret < 0) {
    printf("[mjlab] rknn_init failed: %d\n", ret);
    return false;
  }

  // 策略网只有 4 层 Gemm，单核足够；多核对小模型只增加调度抖动
  rknn_core_mask mask = RKNN_NPU_CORE_0;
  if (cfg_.npuCoreMask == 1) mask = RKNN_NPU_CORE_1;
  else if (cfg_.npuCoreMask == 2) mask = RKNN_NPU_CORE_2;
  else if (cfg_.npuCoreMask == 3) mask = RKNN_NPU_CORE_0_1_2;
  ret = rknn_set_core_mask(ctx_, mask);
  if (ret < 0) printf("[mjlab] rknn_set_core_mask warning: %d\n", ret);

  rknn_sdk_version ver;
  if (rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0) {
    printf("[mjlab] rknn api %s / driver %s\n", ver.api_version, ver.drv_version);
  }

  if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_)) < 0) {
    printf("[mjlab] RKNN_QUERY_IN_OUT_NUM failed\n");
    return false;
  }
  if (ioNum_.n_input != 1 || ioNum_.n_output != 1) {
    printf("[mjlab] unexpected io num: in=%u out=%u (expected 1/1)\n",
           ioNum_.n_input, ioNum_.n_output);
    return false;
  }

  std::memset(&inAttr_, 0, sizeof(inAttr_));
  std::memset(&outAttr_, 0, sizeof(outAttr_));
  inAttr_.index = 0;
  outAttr_.index = 0;
  if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inAttr_, sizeof(inAttr_)) < 0 ||
      rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &outAttr_, sizeof(outAttr_)) < 0) {
    printf("[mjlab] tensor attr query failed\n");
    return false;
  }
  printf("[mjlab] input  n_elems=%u size=%u type=%s fmt=%s\n", inAttr_.n_elems,
         inAttr_.size, get_type_string(inAttr_.type), get_format_string(inAttr_.fmt));
  printf("[mjlab] output n_elems=%u size=%u type=%s fmt=%s\n", outAttr_.n_elems,
         outAttr_.size, get_type_string(outAttr_.type), get_format_string(outAttr_.fmt));

  // 形状不对就地失败，而不是等上机后走出诡异动作
  if (inAttr_.n_elems != static_cast<uint32_t>(kObsDim)) {
    printf("[mjlab] FATAL: model expects %u inputs, this code builds %d\n",
           inAttr_.n_elems, kObsDim);
    return false;
  }
  if (outAttr_.n_elems != static_cast<uint32_t>(kActDim)) {
    printf("[mjlab] FATAL: model outputs %u values, expected %d\n",
           outAttr_.n_elems, kActDim);
    return false;
  }
  if (inAttr_.type != RKNN_TENSOR_FLOAT16) {
    printf("[mjlab] WARNING: input dtype is %s, not float16 -- was the model quantized?\n",
           get_type_string(inAttr_.type));
  }

  rknnLoaded_ = true;
  printf("[mjlab] rknn model loaded: %s\n", rknnPath.c_str());
  return true;
}

void MjlabPolicy::reset() {
  std::fill(action_.begin(), action_.end(), 0.0f);
  std::fill(lastAction_.begin(), lastAction_.end(), 0.0f);
  std::fill(obs_.begin(), obs_.end(), 0.0f);
  command_[0] = command_[1] = command_[2] = 0.0;
  policyStep_ = 0;
  firstObsPrinted_ = false;
  lastInferMs_ = 0.0;
  worstInferMs_ = 0.0;
  limitClampCount_ = 0;
  logRows_ = 0;
}

void MjlabPolicy::buildObservation(const double* q, const double* dq, const double gyro[3],
                                   const double quatXyzw[4], const double cmdAxes[3]) {
  // --- 指令：手柄轴 -> [vx, vy, wz]。axes[1]=前后, axes[0]=转向（与厂家一致），
  //     axes[2]=横移（SDK 的 joydata 只有 2 轴，这一路由 JsAxisReader 直读 js0 补上）---
  const double ax0 = applyDeadzone(cmdAxes[0], cfg_.axisDeadzone);
  const double ax1 = applyDeadzone(cmdAxes[1], cfg_.axisDeadzone);
  const double ax2 = applyDeadzone(cmdAxes[2], cfg_.axisDeadzone);
  const double tgt[3] = {cfg_.signLinVelX * ax1 * cfg_.maxLinVelX,
                         cfg_.signLinVelY * ax2 * cfg_.maxLinVelY + cfg_.constLinVelY,
                         cfg_.signAngVelZ * ax0 * cfg_.maxAngVelZ};
  const double a = cfg_.commandLpfAlpha;
  for (int i = 0; i < 3; ++i) command_[i] = a * tgt[i] + (1.0 - a) * command_[i];

  // --- projected_gravity：沿用 SDK 里已在实机验证过的那套换算 ---
  Eigen::Quaternion<double> quat;
  quat.coeffs()(0) = quatXyzw[0];
  quat.coeffs()(1) = quatXyzw[1];
  quat.coeffs()(2) = quatXyzw[2];
  quat.coeffs()(3) = quatXyzw[3];
  const Eigen::Matrix<double, 3, 1> gravityWorld(0.0, 0.0, -1.0);
  const Eigen::Matrix<double, 3, 1> zyx = quatToZyx(quat);
  const Eigen::Matrix<double, 3, 3> invRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();
  const Eigen::Matrix<double, 3, 1> projGravity = invRot * gravityWorld;

  // --- phase：t 必须按策略步数 * 0.02 递推，和训练里的 episode_length_buf*step_dt 对齐 ---
  double phaseSin = 0.0, phaseCos = 0.0;
  const double cmdNorm = std::sqrt(command_[0] * command_[0] + command_[1] * command_[1] +
                                   command_[2] * command_[2]);
  if (cmdNorm >= cfg_.phaseStandThreshold) {
    const double t = static_cast<double>(policyStep_) * kPolicyDt;
    const double gp = std::fmod(t, cfg_.phasePeriod) / cfg_.phasePeriod;
    phaseSin = std::sin(2.0 * M_PI * gp);
    phaseCos = std::cos(2.0 * M_PI * gp);
  }

  int k = 0;
  for (int i = 0; i < 3; ++i) obs_[k++] = static_cast<float>(gyro[i]);            // 0..2
  for (int i = 0; i < 3; ++i) obs_[k++] = static_cast<float>(projGravity(i));     // 3..5
  for (int i = 0; i < 3; ++i) obs_[k++] = static_cast<float>(command_[i]);        // 6..8
  obs_[k++] = static_cast<float>(phaseSin);                                       // 9
  obs_[k++] = static_cast<float>(phaseCos);                                       // 10
  for (int i = 0; i < kNumJoints; ++i)                                            // 11..28
    obs_[k++] = static_cast<float>(q[i] - cfg_.defaultQ[i]);
  for (int i = 0; i < kNumJoints; ++i) obs_[k++] = static_cast<float>(dq[i]);     // 29..46
  for (int i = 0; i < kActDim; ++i) obs_[k++] = lastAction_[i];                   // 47..64

  // 注意：这里不做任何缩放，也不做归一化 —— 归一化在 .rknn 图内部。
  projGravityZ_ = projGravity(2);
}

MjlabPolicy::Status MjlabPolicy::runInference() {
  for (int i = 0; i < kObsDim; ++i) obsFp16_[i] = floatToHalf(obs_[i]);

  rknn_input in;
  std::memset(&in, 0, sizeof(in));
  in.index = 0;
  in.type = inAttr_.type;           // float16
  in.size = inAttr_.size;           // 65 * 2 bytes
  in.fmt = inAttr_.fmt;
  in.pass_through = 0;
  in.buf = obsFp16_.data();

  const double t0 = nowMs();
  if (rknn_inputs_set(ctx_, 1, &in) != 0) {
    printf("[mjlab] rknn_inputs_set failed\n");
    return Status::kRknnError;
  }
  if (rknn_run(ctx_, nullptr) != 0) {
    printf("[mjlab] rknn_run failed\n");
    return Status::kRknnError;
  }
  rknn_output out;
  std::memset(&out, 0, sizeof(out));
  out.index = 0;
  out.is_prealloc = 0;   // 沿用厂家已验证的用法；50 Hz 下这点分配开销无所谓
  out.want_float = 1;    // runtime 把 fp16 输出转回 float32，不要再自己 half_to_float
  if (rknn_outputs_get(ctx_, 1, &out, nullptr) != 0) {
    printf("[mjlab] rknn_outputs_get failed\n");
    return Status::kRknnError;
  }
  const float* raw = static_cast<const float*>(out.buf);
  bool finite = true;
  for (int i = 0; i < kActDim; ++i) {
    action_[i] = raw[i];
    if (!std::isfinite(raw[i])) finite = false;
  }
  rknn_outputs_release(ctx_, 1, &out);
  lastInferMs_ = nowMs() - t0;
  if (lastInferMs_ > worstInferMs_) worstInferMs_ = lastInferMs_;

  if (!finite) return Status::kNonFiniteOutput;
  return Status::kOk;
}

MjlabPolicy::Status MjlabPolicy::step(const double* q, const double* dq, const double gyro[3],
                                     const double quatXyzw[4], const double cmdAxes[3],
                                     double* targetQOut) {
  if (!cfgLoaded_ || !rknnLoaded_) return Status::kRknnError;

  for (int i = 0; i < kNumJoints; ++i) {
    // SDK 用 |pos| == 12.5 表示该电机掉线
    if (std::fabs(q[i]) == 12.5) {
      printf("[mjlab] joint %d (%s) disconnected\n", i, kJointNames[i]);
      return Status::kJointDisconnected;
    }
  }

  buildObservation(q, dq, gyro, quatXyzw, cmdAxes);
  for (int i = 0; i < kObsDim; ++i) {
    if (!std::isfinite(obs_[i])) {
      printf("[mjlab] non-finite obs at index %d\n", i);
      return Status::kNonFiniteOutput;
    }
  }
  // projected_gravity[2] 接近 -1 表示直立；高于阈值说明躯干倾倒
  if (projGravityZ_ > cfg_.fallGravityZ) return Status::kFallen;

  if (cfg_.logFirstObs && !firstObsPrinted_) {
    printf("[mjlab] first obs (65):\n ");
    for (int i = 0; i < kObsDim; ++i) printf(" %.4f", obs_[i]);
    printf("\n");
    firstObsPrinted_ = true;
  }

  const Status st = runInference();
  if (st != Status::kOk) return st;

  // 训练时 clip_actions = null；这里的裁剪只是异常兜底，正常步态里 |action| < ~4.2
  for (int i = 0; i < kActDim; ++i) {
    if (action_[i] > cfg_.clipActions) action_[i] = static_cast<float>(cfg_.clipActions);
    if (action_[i] < -cfg_.clipActions) action_[i] = static_cast<float>(-cfg_.clipActions);
  }

  for (int i = 0; i < kNumJoints; ++i) {
    double target = cfg_.defaultQ[i] + cfg_.actionScale[i] * static_cast<double>(action_[i]);
    if (target < cfg_.jointLimitLower[i]) {
      target = cfg_.jointLimitLower[i];
      ++limitClampCount_;
    } else if (target > cfg_.jointLimitUpper[i]) {
      target = cfg_.jointLimitUpper[i];
      ++limitClampCount_;
    }
    targetQOut[i] = target;
  }

  if (!log_.empty() && logRows_ < cfg_.logCapacitySteps) {
    const size_t stride = 1 + kObsDim + kActDim;
    float* row = &log_[static_cast<size_t>(logRows_) * stride];
    row[0] = static_cast<float>(policyStep_);
    std::memcpy(row + 1, obs_.data(), kObsDim * sizeof(float));
    std::memcpy(row + 1 + kObsDim, action_.data(), kActDim * sizeof(float));
    ++logRows_;
  }

  // last_action 是下一步观测的一部分：存**裁剪后实际使用的**动作，
  // 与训练里 last_action 取 action_manager 的动作一致（训练时无裁剪，两者等价）
  for (int i = 0; i < kActDim; ++i) lastAction_[i] = action_[i];
  ++policyStep_;
  return Status::kOk;
}

int MjlabPolicy::dumpLog() {
  if (log_.empty() || logRows_ == 0 || cfg_.logPath.empty()) return 0;
  FILE* f = fopen(cfg_.logPath.c_str(), "w");
  if (f == nullptr) {
    printf("[mjlab] cannot write log '%s'\n", cfg_.logPath.c_str());
    return 0;
  }
  fprintf(f, "step");
  for (int i = 0; i < kObsDim; ++i) fprintf(f, ",obs%d", i);
  for (int i = 0; i < kActDim; ++i) fprintf(f, ",act%d", i);
  fprintf(f, "\n");
  const size_t stride = 1 + kObsDim + kActDim;
  for (int r = 0; r < logRows_; ++r) {
    const float* row = &log_[static_cast<size_t>(r) * stride];
    fprintf(f, "%.0f", row[0]);
    for (size_t c = 1; c < stride; ++c) fprintf(f, ",%.7g", row[c]);
    fprintf(f, "\n");
  }
  fclose(f);
  printf("[mjlab] wrote %d rows to %s\n", logRows_, cfg_.logPath.c_str());
  const int n = logRows_;
  logRows_ = 0;
  return n;
}

}  // namespace legged
