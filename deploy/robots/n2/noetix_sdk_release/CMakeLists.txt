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



