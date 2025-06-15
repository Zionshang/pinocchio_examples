#include <iostream>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joints.hpp>
#include <pinocchio/parsers/urdf.hpp>

int main()
{
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);

    ///// 打印关节信息 /////
    std::cout << "============== Joint Information ==============" << std::endl;
    for (int i = 0; i < model.njoints; ++i)
    {
        std::cout << std::setw(4) << i << ": "
                  << std::setw(30) << std::left << model.names[i] << ": "
                  << std::fixed << std::setw(25) << model.joints[i].shortname() << std::endl;
    }

    ///// 打印坐标系信息 /////
    std::cout << "============== Frame Information ==============" << std::endl;
    for (int i = 0; i < model.nframes; ++i)
    {
        std::cout << std::setw(4) << i << ": "
                  << std::setw(30) << std::left << model.frames[i].name << std::endl;
    }

    return 0;
}