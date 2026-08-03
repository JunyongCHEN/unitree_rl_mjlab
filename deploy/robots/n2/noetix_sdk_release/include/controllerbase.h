#ifndef CONTROLLERBASE_H
#define CONTROLLERBASE_H
#include "common.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <functional>
namespace py = pybind11;
namespace legged
{

typedef void (*ExternFuntionType)(void);
class Controllerbase 
{
   
public:
  
    Controllerbase();

    ~Controllerbase()  = default;

     /* init 
      系统初始化，mode为default时 初始化硬件和自有模型模块，；mode 为usermode时，初始化硬件,所有控制由用户实现；
    */
     bool init(ControlMode mode);//mode:default run noetix model;mode: usermode run custom model
     /*loadModel 
      加载自有模型，可加载多个模型
     */
     bool loadModel(std::string modelname,std::string modelpath);
     bool loadModel(std::string modelname,std::string modelpath,std::string type);
     /*get_joint_state
     获取电机状态，电机顺序固定{  "arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint",
                              "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
                              "arm_r1_joint", "arm_r2_joint", "arm_r3_joint", "arm_r4_joint",
                              "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"};
     */
     const std::array<MotorState,18> get_joint_state();
     /*set_joint
     设置电机命令，目前只支持pos,kp,kd
     */
     void set_joint(MotorCmd motorcmd);
     /*get_jsdata
     获取遥控器数据，只支持自有遥控器
     */
     const joydata get_jsdata();
     /*get_imu_data 
      获取IMU数据 ，目前只支持元生*/
     const NingImuData get_imu_data();
     /*机器人进入standby状态*/
     void start();
     
     /*setcallback
      注册客户自有模型处理函数，可以在进行数据的前后处理，调用set_joint下发控制命令*/
      void setcallback(ExternFuntionType cb);
      // 注册python回调函数
      void setpycallback(py::function callback) ;

};
} 
#endif
