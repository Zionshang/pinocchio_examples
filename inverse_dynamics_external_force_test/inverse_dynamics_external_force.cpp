#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include <vector>
#include <string>
#include "Eigen/Core"

using namespace std;

using Vec3 = Eigen::Matrix<double, 3, 1>;
using VecNq = Eigen::Matrix<double, 19, 1>;
using VecNv = Eigen::Matrix<double, 18, 1>;
using VecNj = Eigen::Matrix<double, 12, 1>;

class PinocchioDynamics
{
public:
    PinocchioDynamics(string urdf_filename);

    void PrintModelTree();
    VecNj InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                const VecNv &a, const vector<Vec3> &fext);
    VecNv forwardDynamicsByEom(const VecNq &q, const VecNv &v,
                               const VecNj &tau, const vector<Vec3> &fext);

    const int GetNq() const { return nq_; }
    const int GetNv() const { return nv_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;

    int nq_, nv_;
    int id_feet_body_[4];
    int id_feet_joint_[4];
};

PinocchioDynamics::PinocchioDynamics(string urdf_filename)
{
    pinocchio::urdf::buildModel(urdf_filename, model_);
    data_ = pinocchio::Data(model_);
    nq_ = model_.nq;
    nv_ = model_.nv;

    vector<string> foot_names = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"};
    for (int i = 0; i < 4; i++)
        id_feet_body_[i] = model_.getFrameId(foot_names[i], pinocchio::BODY);
}

void PinocchioDynamics::PrintModelTree()
{
    using pinocchio::FrameIndex;
    using pinocchio::JointIndex;

    for (JointIndex joint_id = 0; joint_id < (JointIndex)model_.njoints; ++joint_id)
        std::cout << std::setw(24) << std::left << joint_id << ": " << model_.names[joint_id] << std::endl;

    for (FrameIndex frame_id = 0; frame_id < (FrameIndex)model_.nframes; ++frame_id)
        std::cout << std::setw(24) << std::left << frame_id << ": " << model_.frames[frame_id].name << std::endl;
}

VecNj PinocchioDynamics::InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                               const VecNv &a, const vector<Vec3> &fext)
{
}

VecNv PinocchioDynamics::forwardDynamicsByEom(const VecNq &q, const VecNv &v,
                                              const VecNj &tau, const vector<Vec3> &fext)
{
    // calculate contact jacobian
    std::vector<Eigen::Matrix<double, 6, 18>> J(4);
    for (int i = 0; i < 4; i++)
    {
        J[i].setZero();
        pinocchio::computeFrameJacobian(model_, data_, q, id_feet_body_[i], pinocchio::LOCAL_WORLD_ALIGNED, J[i]);
    }

    // calculate M and C matrix, such that M * a + C = S * tau + J' * fext
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nv_, nv_);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nv_, 1);
    pinocchio::crba(model_, data_, q);
    pinocchio::nonLinearEffects(model_, data_, q, v);
    M.triangularView<Eigen::Upper>() = data_.M;
    M.triangularView<Eigen::Lower>() = M.triangularView<Eigen::Upper>().transpose();
    C = data_.nle;

    // calculate a by a = M.inv (S * tau - C + J' * fext)
    // Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nv_, 12);
    // S.bottomRows(12) = Eigen::MatrixXd::Identity(12, 12);
    // VecNv a = S * tau - C;
    VecNv S_times_tau;
    S_times_tau << Eigen::VectorXd::Zero(6), tau;
    VecNv a = S_times_tau - C;
    for (int i = 0; i < 4; i++)
        a += J[i].topRows(3).transpose() * fext[i]; // only consider the linear part of jacobian

    return M.inverse() * a;
}

int main()
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/inverse_dynamics_external_force_test/robot/mini_cheetah_mesh_v2.urdf";
    PinocchioDynamics pin_dyn(urdf_filename);
    int nv = pin_dyn.GetNv();
    VecNq q;
    q << 0.0, 0.0, 0.235,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.8, -1.6,
        0.0, -0.8, 1.6,
        0.0, 0.8, -1.6,
        0.0, -0.8, 1.6;
    VecNv v = Eigen::MatrixXd::Random(nv, 1);
    // VecNv a = Eigen::MatrixXd::Random(nv, 1);
    VecNj tau = Eigen::MatrixXd::Random(12, 1);

    Vec3 fext0(10, -20, 100), fext1(-10, 20, 120), fext2(5, -5, 90), fext3(-10, -10, 120);
    vector<Vec3> fext = {fext0, fext1, fext2, fext3};

    VecNv a = pin_dyn.forwardDynamicsByEom(q, v, tau, fext);
    cout << "a from EOM" << a << std::endl;
}
