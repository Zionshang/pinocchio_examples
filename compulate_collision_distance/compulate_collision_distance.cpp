#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/collision/collision.hpp"

#include <iostream>

int main()
{
    using namespace pinocchio;

    const std::string urdf_filename = "/home/zishang/Cpp_workspace/pinocchio_examples/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";
    const std::string robots_model_path = "/home/zishang/Cpp_workspace/pinocchio_examples/robot/mini_cheetah/meshes";
    const std::string sdf_filename = "/home/zishang/Cpp_workspace/pinocchio_examples/robot/mini_cheetah/sdf/mini_cheetah.srdf";

    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);

    for (int i = 0; i < model.njoints; ++i)
    {
        const auto joint_model = model.joints[i];
        auto joint_data = joint_model.createData();
        std::cout
            << std::setw(4) << i << ": "
            << std::setw(30) << std::left << model.names[i] << ": "
            << std::fixed << std::setw(25) << model.joints[i].shortname() << std::endl;
    }
    // Load the geometries associated to model which are contained in the URDF file
    GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model, robots_model_path);

    // // Define ground GeometryObject
    // std::shared_ptr<fcl::CollisionGeometry> ground_box(new fcl::Box(1e10, 1e10, 1));
    // Eigen::Vector3d ground_place(0, 0, -0.5);
    // pinocchio::SE3 ground_pose(Eigen::Matrix3d::Identity(), ground_place);
    // GeometryObject ground_object("ground", 0, ground_pose, ground_box);
    // geom_model.addGeometryObject(ground_object);

    // Add possible collision pairs in urdf
    geom_model.addAllCollisionPairs();
    pinocchio::srdf::removeCollisionPairs(model, geom_model, sdf_filename);

    std::cout << "geom_model\n"
              << geom_model << std::endl;

    // Build the data associated to the geom_model
    GeometryData geom_data(geom_model);

    std::cout << "geom_data\n"
              << geom_data << std::endl;

    // And test all the collision pairs
    pinocchio::computeDistances(model, data, geom_model, geom_data, q);

    return 0;
}