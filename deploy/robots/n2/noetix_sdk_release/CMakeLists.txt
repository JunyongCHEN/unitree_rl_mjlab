cmake_minimum_required( VERSION 3.16.3 )

project( noetix-sdk )
set(USER_NAME $ENV{USER})
message(STATUS "environment user : ${USER_NAME}")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_PREFIX_PATH /home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11)
set(pybind11_DIR /home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/pybind11Config.cmake)
# 找到 Python3 的库文件路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
  set(PYTHON_LIB_PATH /usr/lib/python3.8/config-3.8-aarch64-linux-gnu)
  add_definitions(-DRK3588)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
  set(PYTHON_LIB_PATH /usr/lib/python3.8/config-3.8-x86_64-linux-gnu)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # global PIC

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(onnxruntime REQUIRED)
add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/aarch64)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/x86_64)
endif()
include_directories(/usr/include/python3.8)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(/home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11/include)

MESSAGE("LIB_DIR:${LIB_DIR}")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(noetix_joint_controller src/controller_example.cpp src/usercontroller.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
  target_link_libraries(noetix_joint_controller 
      PRIVATE pybind11::module 
      Python3::Python
      ${LIB_DIR}/librlcontrol.a
      ${LIB_DIR}/libhardware.a
      ${LIB_DIR}/libsoem.a
      ${LIB_DIR}/libyaml-cppd.a
      onnxruntime
      Threads::Threads
      spdlog::spdlog
      rknnrt
  )
else()
  target_link_libraries(noetix_joint_controller
      PRIVATE pybind11::module
      Python3::Python
      ${LIB_DIR}/librlcontrol.a
      ${LIB_DIR}/libhardware.a
      ${LIB_DIR}/libsoem.a
      ${LIB_DIR}/libyaml-cppd.a
      onnxruntime
      Threads::Threads
      spdlog::spdlog
  )
endif()

# =====================================================================
# mjlab (rsl_rl) 速度策略 -> RK3588 NPU
#   n2_mjlab_ctrl   实机控制程序（18 DoF, 65 维观测, 50 Hz）
#   n2_rknn_check   上板自检：精度对 golden.bin + 推理耗时
# librknnrt.so 随 SDK 附带在 lib/aarch64/，头文件在 include/rknpu2/，
# 不依赖板子系统里是否装过 rknpu2。
#
# 注意 include/rknpu2 只加到这两个 target 上，不进全局 include_directories，
# 这样厂家原来的 noetix_joint_controller 仍然用系统的 rknn_api.h，编译行为完全不变。
# =====================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
  add_executable(n2_mjlab_ctrl src/mjlab_controller_main.cpp src/mjlab_policy.cpp
                               src/js_axis_reader.cpp)
  target_include_directories(n2_mjlab_ctrl PRIVATE ${CMAKE_SOURCE_DIR}/include/rknpu2)
  target_link_libraries(n2_mjlab_ctrl
      PRIVATE pybind11::module
      Python3::Python
      ${LIB_DIR}/librlcontrol.a
      ${LIB_DIR}/libhardware.a
      ${LIB_DIR}/libsoem.a
      ${LIB_DIR}/libyaml-cppd.a
      # 策略本身走 NPU，不用 onnxruntime；但 librlcontrol.a 是拿 onnxruntime 编的，
      # 它里面有静态初始化会引用 OrtGetApiBase，所以链它就必须一起链 onnxruntime。
      onnxruntime
      Threads::Threads
      spdlog::spdlog
      ${LIB_DIR}/librknnrt.so
  )
  # 运行时从 SDK 自带的 lib/aarch64 找 librknnrt.so（可执行文件输出在 SDK 根目录）
  set_target_properties(n2_mjlab_ctrl PROPERTIES
      BUILD_RPATH   "$ORIGIN/lib/aarch64"
      INSTALL_RPATH "$ORIGIN/lib/aarch64")

  add_executable(n2_rknn_check tools/n2_rknn_check.cpp)
  target_include_directories(n2_rknn_check PRIVATE ${CMAKE_SOURCE_DIR}/include/rknpu2)
  target_link_libraries(n2_rknn_check PRIVATE ${LIB_DIR}/librknnrt.so)
  set_target_properties(n2_rknn_check PROPERTIES
      BUILD_RPATH   "$ORIGIN/lib/aarch64"
      INSTALL_RPATH "$ORIGIN/lib/aarch64")

  # 手柄轴探测，用来确定横移摇杆是几号轴。不依赖 rknn / 厂家静态库。
  add_executable(n2_js_probe tools/n2_js_probe.cpp src/js_axis_reader.cpp)
else()
  message(STATUS "not aarch64: skipping n2_mjlab_ctrl / n2_rknn_check (RKNN targets)")
endif()



