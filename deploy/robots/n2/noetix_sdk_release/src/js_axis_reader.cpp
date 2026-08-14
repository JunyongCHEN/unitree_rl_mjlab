#include "js_axis_reader.h"

#include <fcntl.h>
#include <linux/joystick.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace legged {

JsAxisReader::~JsAxisReader() {
  if (fd_ >= 0) ::close(fd_);
}

bool JsAxisReader::open(const std::string& device) {
  if (fd_ >= 0) ::close(fd_);
  fd_ = ::open(device.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd_ < 0) {
    std::printf("[js] cannot open %s (横移轴将按 0 处理)\n", device.c_str());
    return false;
  }
  char buf[128] = {0};
  if (::ioctl(fd_, JSIOCGNAME(sizeof(buf) - 1), buf) >= 0) name_ = buf;
  uint8_t nAxes = 0;
  ::ioctl(fd_, JSIOCGAXES, &nAxes);
  std::printf("[js] opened %s  name='%s'  axes=%u\n", device.c_str(), name_.c_str(),
              static_cast<unsigned>(nAxes));
  // 首批 JS_EVENT_INIT 事件会带出所有轴的当前值，poll() 一次即可拿到初值
  poll();
  return true;
}

void JsAxisReader::poll() {
  if (fd_ < 0) return;
  struct js_event ev;
  // 非阻塞：一次把积压事件全部排空，避免队列涨满后驱动丢事件
  while (::read(fd_, &ev, sizeof(ev)) == static_cast<ssize_t>(sizeof(ev))) {
    if ((ev.type & JS_EVENT_AXIS) == 0) continue;   // 按键交给 SDK 那一路，这里只要轴
    if (ev.number >= kMaxAxes) continue;
    const double v = static_cast<double>(ev.value) / 32767.0;
    if (ev.number > maxAxisSeen_) maxAxisSeen_ = ev.number;
    // JS_EVENT_INIT 只是上报初值，不算“动过”
    if ((ev.type & JS_EVENT_INIT) == 0 && std::fabs(v - axes_[ev.number]) > 0.05) {
      movedAxis_ = ev.number;
    }
    axes_[ev.number] = v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
  }
}

double JsAxisReader::axis(int index) const {
  if (fd_ < 0 || index < 0 || index >= kMaxAxes) return 0.0;
  return axes_[index];
}

int JsAxisReader::takeMovedAxis() {
  const int a = movedAxis_;
  movedAxis_ = -1;
  return a;
}

}  // namespace legged
