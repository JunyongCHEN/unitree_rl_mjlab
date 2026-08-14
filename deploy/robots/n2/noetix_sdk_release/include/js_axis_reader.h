// 直接读 /dev/input/js0，补上 SDK 拿不到的摇杆轴。
//
// 为什么需要这个：SDK 的 legged::joydata 只有 axes[2]（horz, vert），
// 是在 common.h 里写死的，librlcontrol.a 也只填这两个，所以通过
// Controllerbase::get_jsdata() 拿不到第二个摇杆 —— 也就拿不到横移指令。
//
// Linux 的 joydev 驱动允许同一个 js 设备被多个进程各自 open，每个 fd 有独立的
// 事件队列，所以这里再开一个只读 fd 不会影响 SDK 自己那一路。
//
// 用 tools/n2_js_probe 先确认横移摇杆是第几号轴，再填到
// config/mjlab_policy.yaml 的 command.lateral_axis.index。
#ifndef JS_AXIS_READER_H
#define JS_AXIS_READER_H

#include <string>

namespace legged {

class JsAxisReader {
 public:
  static constexpr int kMaxAxes = 16;

  ~JsAxisReader();

  // 打不开也返回 false，但不算致命错误：调用方应当继续跑，横移量按 0 处理。
  bool open(const std::string& device);
  bool isOpen() const { return fd_ >= 0; }

  // 把队列里积压的事件全部读掉，更新内部轴状态。非阻塞，读不到就直接返回。
  void poll();

  // 归一化到 [-1, 1]；索引越界或没打开时返回 0
  double axis(int index) const;
  int numAxesSeen() const { return maxAxisSeen_ + 1; }

  // debug 用：返回自上次调用以来动过的轴（没有则返回 -1），配合限频打印找轴号
  int takeMovedAxis();

  const std::string& name() const { return name_; }

 private:
  int fd_ = -1;
  double axes_[kMaxAxes] = {0.0};
  int maxAxisSeen_ = -1;
  int movedAxis_ = -1;
  std::string name_;
};

}  // namespace legged
#endif  // JS_AXIS_READER_H
