#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/autodiff/cppad.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
using CppAD::AD;
using ADVectorX = Eigen::VectorX<AD<double>>;

int main()
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

    ////////////// 设置具体测试变量 //////////////
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    double dt = 0.01;
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(nv);
    Eigen::VectorXd a = Eigen::VectorXd::Random(nv);

    ////////////// 创建CPPAD逆动力学计算函数 //////////////
    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv + nv + nv); // q, v, a, dq
    ADVectorX ad_Y(nv);                // q_next
    CppAD::ADFun<double> ad_fun_q_next;
    ad_X.setZero();
    CppAD::Independent(ad_X);

    ADVectorX ad_q = ad_X.head(nq);
    ADVectorX ad_v = ad_X.segment(nq, nv);
    ADVectorX ad_a = ad_X.segment(nq + nv, nv);
    ADVectorX ad_dq = ad_X.tail(nv);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_q, ad_dq);
    ADVectorX ad_v_next = ad_v + ad_a * dt; // v_next = v + a * dt
    ADVectorX ad_q_next = pinocchio::integrate(ad_model, ad_q_plus, ad_v_next * dt);
    ad_Y = ad_q_next;
    ad_fun_q_next.Dependent(ad_X, ad_Y);

    ////////////// Cppad 求导 //////////////
    Eigen::VectorXd x(nq + nv + nv + nv); // q, v, a, dq
    x << q, v, a, Eigen::VectorXd::Zero(nv);

    Eigen::MatrixXd dqnext_dx_cppad = ad_fun_q_next.Jacobian(x)
                                          .reshaped(ad_X.size(), ad_Y.size())
                                          .transpose();
    Eigen::MatrixXd dqnext_dq_cppad = dqnext_dx_cppad.rightCols(nv);
    Eigen::MatrixXd dqnext_dv_cppad = dqnext_dx_cppad.middleCols(nq, nv);

    ////////////// 解析求导 //////////////
    Eigen::VectorXd v_next = v + a * dt;
    Eigen::MatrixXd dqnext_dq = Eigen::MatrixXd::Zero(nv, nv);
    Eigen::MatrixXd dqnext_dvnext = Eigen::MatrixXd::Zero(nv, nv);
    pinocchio::dIntegrate(model, q, v_next * dt, dqnext_dq, pinocchio::ARG0);
    pinocchio::dIntegrate(model, q, v_next * dt, dqnext_dvnext, pinocchio::ARG1);
    Eigen::MatrixXd dqnext_dv = dqnext_dvnext * dt;

    ////////////// 解析对比 //////////////
    // todo: 不明白为什么结果不对~
    std::cout << "dqnext_dq (CppAD):\n"
              << dqnext_dq_cppad << std::endl;
    std::cout << "dqnext_dq (analytical):\n"
              << dqnext_dq << std::endl;
    std::cout << "dqnext_dv (CppAD):\n"
              << dqnext_dv_cppad << std::endl;
    std::cout << "dqnext_dv (analytical):\n"
              << dqnext_dv << std::endl;
}