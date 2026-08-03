#include "controllerbase.h"
#include "usercontroller.h"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdio.h>
// #define usermode
using namespace legged;
int main()
{
  char buf[256];
  bool ret = true;
  getcwd(buf,sizeof(buf)); 
  std::string path=std::string(buf);
  printf("cur path is %s\n",path.c_str());
  #ifdef usermode
  UserController usercontroller;
  ret = usercontroller.init(ControlMode::USERMODE);
  if(ret == false)
  {
    printf("usercontroller.init   failed\n");
    return 0;
  }
  #ifdef RK3588
      usercontroller.loadrknnmodel(path+"/ning");      
  #else
    usercontroller.loadModel(path+"/ning");  
  #endif
  #else
  Controllerbase controller_base;
  ret = controller_base.init(ControlMode::DEFAULT);
  
  if(ret == false)
  {
    return 0;
  }
  #ifdef RK3588
  ret = controller_base.loadModel("walk",path+"/ning","rknn");
  ret = controller_base.loadModel("run",path+"/ning","rknn");
  #else
  ret = controller_base.loadModel("walk",path+"/ning","onnx");
  ret = controller_base.loadModel("run",path+"/ning","onnx");
  #endif
  printf("loadModel ret %d\n",ret);
  controller_base.start();
  #endif
  
  
  while(1)
  {
    usleep(10);
  }
  return 0;
}
