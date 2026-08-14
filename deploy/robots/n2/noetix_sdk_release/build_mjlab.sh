#!/bin/bash
# 在 N2 板子（RK3588, aarch64）上编译 mjlab 策略控制程序。
# 和厂家的 build_release.sh 用同一套 CMake，只是额外 chmod 新增的两个可执行文件。
set -e

get_arch=$(uname -m)
if [ "$get_arch" != "aarch64" ]; then
    echo "这两个目标只在 aarch64（板子）上编译，当前是 $get_arch"
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p build
cd build
# 不需要 -DRK3588=ON：那个变量 CMakeLists 从来没读过（厂家脚本里传了但没用，
# 所以会看到 "Manually-specified variables were not used" 的警告）。
# 真正的 -DRK3588 编译宏是 CMakeLists 按 CMAKE_SYSTEM_PROCESSOR 自动加的。
cmake ..
make -j"$(nproc)" n2_rknn_check n2_js_probe n2_mjlab_ctrl
cd ..
chmod +x n2_rknn_check n2_js_probe n2_mjlab_ctrl

echo
echo "编译完成。先做上板自检（不会动电机）："
echo "  ./n2_rknn_check ning/policy_mjlab.rknn ning/mjlab_golden.bin"
echo "确认横移摇杆是几号轴（不动电机，可随时跑）："
echo "  ./n2_js_probe"
echo "把轴号填进 config/mjlab_policy.yaml 的 command.lateral_axis.index，然后（机器人先吊装）："
echo "  ./n2_mjlab_ctrl"
