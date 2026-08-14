// 手柄轴探测：把每个摇杆推一下，看哪个轴号在动，用来确定
// config/mjlab_policy.yaml 里 command.lateral_axis.index 该填几。
//
// 不动电机，可以随时跑。控制程序跑着的时候也能跑（joydev 允许多个进程各自 open）。
//
//   ./n2_js_probe                 # 默认 /dev/input/js0
//   ./n2_js_probe /dev/input/js1
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>

#include "js_axis_reader.h"

int main(int argc, char** argv) {
  const std::string dev = (argc > 1) ? argv[1] : "/dev/input/js0";
  legged::JsAxisReader js;
  if (!js.open(dev)) return 1;

  std::printf("\n把每个摇杆分别推到底，观察下面哪一路数值在变。\n");
  std::printf("厂家 SDK 用的是 axes[0](转向) 和 axes[1](前后)，\n");
  std::printf("剩下那个左右方向的轴号就是要填进 lateral_axis.index 的值。\n");
  std::printf("Ctrl-C 退出。\n\n");

  double prev[legged::JsAxisReader::kMaxAxes] = {0.0};
  while (true) {
    js.poll();
    const int n = js.numAxesSeen();
    // 只在有轴变化时刷新，避免刷屏
    bool changed = false;
    for (int i = 0; i < n; ++i) {
      if (std::fabs(js.axis(i) - prev[i]) > 0.02) changed = true;
    }
    if (changed) {
      std::printf("\r");
      for (int i = 0; i < n; ++i) {
        prev[i] = js.axis(i);
        std::printf("ax%d=%+.2f  ", i, prev[i]);
      }
      std::fflush(stdout);
    }
    usleep(20000);  // 50 Hz
  }
  return 0;
}
