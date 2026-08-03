cmake_minimum_required( VERSION 3.16.3 )
set(CMAKE_PREFIX_PATH /home/oem/.local/lib/python3.8/site-packages/pybind11)
set(pybind11_DIR /home/oem/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/pybind11Config.cmake)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)


pybind11_add_module(noetix_interface 
noetix_pybind.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_compile_options(-pthread)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/aarch64)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/x86_64)
endif()
MESSAGE("LIB_DIR:${LIB_DIR}")
target_link_libraries(noetix_interface PRIVATE 
    ${LIB_DIR}/librlcontrol.a
    ${LIB_DIR}/libhardware.a
    ${LIB_DIR}/libsoem.a
    ${LIB_DIR}/libyaml-cpp.a
    onnxruntime
    Threads::Threads
    spdlog::spdlog
)
target_include_directories(noetix_interface PUBLIC ${EIGEN3_INCLUDE_DIRS})



# # Optional: set the output directory for the built module
set_target_properties(noetix_interface PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(noetix_interface PROPERTIES PREFIX "")
set_target_properties(noetix_interface PROPERTIES OUTPUT_NAME "noetix_interface")
