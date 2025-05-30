#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/autodiff/cppad.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

using CppAD::AD;
using ADVectorX = Eigen::VectorX<AD<double>>;
int main(int argc, char const *argv[])
{
    ////////////// 加载模型 //////////////
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
    pinocchio::Data data(model);
    const int nq = model.nq;
    const int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(-1);
    model.upperPositionLimit.head<3>().fill(1);

    ////////////// 创建CPPAD变量 //////////////
    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv + nv + nv); // q, v, a, dq
    ADVectorX ad_Y(nv);                // tau
    CppAD::ADFun<double> ad_inv_dynamics_fun;
    ad_X.setZero();
    CppAD::Independent(ad_X);

    ////////////// 创建CPPAD逆动力学计算函数 //////////////
    ADVectorX ad_q = ad_X.head(nq);
    ADVectorX ad_v = ad_X.segment(nq, nv);
    ADVectorX ad_a = ad_X.segment(nq + nv, nv);
    ADVectorX ad_dq = ad_X.tail(nv);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_q, ad_dq);
    ad_Y = pinocchio::rnea(ad_model, ad_data, ad_q_plus, ad_v, ad_a);
    ad_inv_dynamics_fun.Dependent(ad_X, ad_Y);

    ////////////// 设置具体测试变量 //////////////
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(nv);
    Eigen::VectorXd a = Eigen::VectorXd::Random(nv);

    ////////////// CPPAD计算逆动力学偏导 //////////////
    Eigen::VectorXd X(nq + nv + nv + nv); // q, v, a, dq
    X << q, v, a, Eigen::VectorXd::Zero(nv);

    Eigen::VectorXd dtau_dx_vec = ad_inv_dynamics_fun.Jacobian(X);
    Eigen::MatrixXd dtau_dx = dtau_dx_vec.reshaped(X.size(), nv).transpose();
    Eigen::MatrixXd dtau_dq = dtau_dx.rightCols(nv);
    Eigen::MatrixXd dtau_dv = dtau_dx.middleCols(nq, nv);
    Eigen::MatrixXd dtau_da = dtau_dx.middleCols(nq + nv, nv);

    ////////////// pinocchio计算解析雅可比偏导数 //////////////
    pinocchio::computeRNEADerivatives(model, data, q, v, a);
    Eigen::MatrixXd dtau_dq_pin = data.dtau_dq;
    Eigen::MatrixXd dtau_dv_pin = data.dtau_dv;
    data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
    Eigen::MatrixXd dtau_da_pin = data.M;

    ////////////// 输出结果对比 //////////////
    if (dtau_dq.isApprox(dtau_dq_pin, 1e-6))
        std::cout << "dtau_dq is correct" << std::endl;
    else
        std::cout << "dtau_dq is wrong" << std::endl;

    if (dtau_dv.isApprox(dtau_dv_pin, 1e-6))
        std::cout << "dtau_dv is correct" << std::endl;
    else
        std::cout << "dtau_dv is wrong" << std::endl;

    if (dtau_da.isApprox(dtau_da_pin, 1e-6))
        std::cout << "dtau_da is correct" << std::endl;
    else
        std::cout << "dtau_da is wrong" << std::endl;

    return 0;
}
