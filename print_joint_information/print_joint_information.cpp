#include <iostream>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joints.hpp>
#include "pinocchio/parsers/urdf.hpp"

int main()
{
    std::string urdf_filename = "/home/zishang/Cpp_workspace/pinocchio_examples/print_joint_information/robot/mini_cheetah_mesh_v2.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);

    for (int i = 0; i < model.njoints; ++i)
    {
        const auto joint_model = model.joints[i];
        auto joint_data = joint_model.createData();
        std::cout
            << std::setw(4) << i << ": "
            << std::setw(30) << std::left << model.names[i] << ": "
            << std::fixed << std::setw(25) << model.joints[i].shortname() << std::endl;
    }
    return 0;
}