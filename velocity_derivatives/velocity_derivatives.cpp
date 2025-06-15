#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/autodiff/cppad.hpp>
#include <chrono>

using pinocchio::context::Matrix6xs;
int main(int argc, char const *argv[])
{
    ////////////////// Load Model //////////////////
    std::string urdf_filename = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), model);
    pinocchio::Data data(model);
    int nq = model.nq;
    int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(0.);
    model.upperPositionLimit.head<3>().fill(0.5);

    ////////////////// Random configuration and velocity //////////////////
    pinocchio::JointIndex joint_id = 10;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(nv);

    ////////////////// Analytical derivatives //////////////////
    auto t1 = std::chrono::high_resolution_clock::now();
    Matrix6xs v_partial_dq = Matrix6xs::Zero(6, nv);
    Matrix6xs v_partial_dv = Matrix6xs::Zero(6, nv);
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, Eigen::VectorXd::Zero(nv));
    pinocchio::getJointVelocityDerivatives(model, data, joint_id, pinocchio::LOCAL_WORLD_ALIGNED, v_partial_dq, v_partial_dv);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto analytical_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    ////////////////// Kinematics Hessian derivatives //////////////////
    auto t3 = std::chrono::high_resolution_clock::now();
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::computeJointKinematicHessians(model, data);
    pinocchio::Data::Tensor3x H = pinocchio::getJointKinematicHessian(model, data, joint_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::MatrixXd dv_dq(6, model.nv);
    const Eigen::DenseIndex outer_offset = 6 * model.nv;
    for (Eigen::DenseIndex k = 0; k < model.nv; ++k)
    {
        Eigen::Map<pinocchio::Data::Matrix6x> H1(H.data() + k * outer_offset, 6, model.nv);
        dv_dq.col(k).noalias() = H1 * v;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto hessian_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    ////////////////// CPPAD derivatives //////////////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;

    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv + nv); // q, v, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    ADVectorX ad_Y(6);

    ADVectorX ad_q = ad_X.head(nq);
    ADVectorX ad_v = ad_X.segment(nq, nv);
    ADVectorX ad_dq = ad_X.tail(nv);
    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_q, ad_dq);

    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus, ad_v);
    auto joint_motion = pinocchio::getVelocity(ad_model, ad_data, joint_id, pinocchio::LOCAL_WORLD_ALIGNED);
    ad_Y.head(3) = joint_motion.linear();
    ad_Y.tail(3) = joint_motion.angular();
    CppAD::ADFun<double> ad_joint_vel(ad_X, ad_Y);

    Eigen::VectorXd x(nq + nv + nv);
    x << q, v, Eigen::VectorXd::Zero(nv);
    auto t5 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd djoint_vel_dx = ad_joint_vel.Jacobian(x)
                                        .reshaped(ad_X.size(), ad_Y.size())
                                        .transpose();
    Eigen::MatrixXd djoint_vel_dq = djoint_vel_dx.rightCols(nv);
    Eigen::MatrixXd djoint_vel_dv = djoint_vel_dx.middleCols(nq, nv);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto ad_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    std::cout << "==============Joint velocity partial derivative w.r.t. configuration==============" << std::endl;
    std::cout << "pinocchio derivatives:\n"
              << v_partial_dq << std::endl;
    std::cout << "cppad derivatives:\n"
              << djoint_vel_dq << std::endl;
    std::cout << "kinematic hessian derivatives:\n"
              << dv_dq << std::endl;

    std::cout << "==============Joint velocity partial derivative w.r.t. velocity==============" << std::endl;
    std::cout << "pinocchio derivatives:\n"
              << v_partial_dv << std::endl;
    std::cout << "cppad derivatives:\n"
              << djoint_vel_dv << std::endl;

    std::cout << "============== Time taken for each method ==============" << std::endl;
    std::cout << "Analytical derivatives time: " << analytical_us << " us" << std::endl;
    std::cout << "Kinematics Hessian derivatives time: " << hessian_us << " us" << std::endl;
    std::cout << "ad_joint_vel.Jacobian(x) time: " << ad_us << " us" << std::endl;
    return 0;
}
