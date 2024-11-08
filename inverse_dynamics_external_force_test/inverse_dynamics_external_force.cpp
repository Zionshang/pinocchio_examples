#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include <vector>
#include <string>
#include <chrono>
#include "Eigen/Core"

using namespace std;

using Vec3 = Eigen::Matrix<double, 3, 1>;
using VecNq = Eigen::Matrix<double, 19, 1>;
using VecNv = Eigen::Matrix<double, 18, 1>;

class PinocchioDynamics
{
public:
    PinocchioDynamics(string urdf_filename);

    void PrintModelTree();
    VecNv InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                const VecNv &a, const vector<Vec3> &f_ext);
    VecNv forwardDynamicsByEom(const VecNq &q, const VecNv &v,
                               const VecNv &tau, const vector<Vec3> &f_ext);

    const int GetNq() const { return nq_; }
    const int GetNv() const { return nv_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;

    int nq_, nv_;
    int id_feet_[4];
    int id_feet_parentjoint_[4];
};

PinocchioDynamics::PinocchioDynamics(string urdf_filename)
{
    pinocchio::urdf::buildModel(urdf_filename, model_);
    data_ = pinocchio::Data(model_);
    nq_ = model_.nq;
    nv_ = model_.nv;

    vector<string> foot_names = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"};
    vector<string> foot_parent_joint_names = {"thigh_fl_to_knee_fl_j", "thigh_fr_to_knee_fr_j",
                                              "thigh_hl_to_knee_hl_j", "thigh_hr_to_knee_hr_j"};

    for (int i = 0; i < 4; i++)
    {
        id_feet_[i] = model_.getFrameId(foot_names[i], pinocchio::BODY);
        id_feet_parentjoint_[i] = model_.getJointId(foot_parent_joint_names[i]);
    }
}

void PinocchioDynamics::PrintModelTree()
{
    using pinocchio::FrameIndex;
    using pinocchio::JointIndex;

    std::cout << " -------------index and names in joint------------- " << std::endl;
    for (JointIndex joint_id = 0; joint_id < (JointIndex)model_.njoints; ++joint_id)
        std::cout << std::setw(24) << std::left << joint_id << ": " << model_.names[joint_id] << std::endl;

    std::cout << " -------------index and names in frame------------- " << std::endl;
    for (FrameIndex frame_id = 0; frame_id < (FrameIndex)model_.nframes; ++frame_id)
        std::cout << std::setw(24) << std::left << frame_id << ": " << model_.frames[frame_id].name << std::endl;
}

VecNv PinocchioDynamics::InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                               const VecNv &a, const vector<Vec3> &f_ext)
{
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    // transform the f_ext from world frame to foot local frame
    vector<Vec3> f_ext_footlocal(4);
    for (int i = 0; i < 4; i++)
        f_ext_footlocal[i] = data_.oMf[id_feet_[i]].rotation().transpose() * f_ext[i];

    pinocchio::container::aligned_vector<pinocchio::Force> f_6d_ext_jointlocal(model_.njoints, pinocchio::Force::Zero());
    for (int i = 0; i < 4; i++)
    {
        // f_6d_ext_local is expressed in foot local frame, we should transform it to joint local frame
        const pinocchio::Force f_6d_ext_footlocal(f_ext_footlocal[i], Eigen::Vector3d::Zero());
        const pinocchio::SE3 jMf = data_.oMi[id_feet_parentjoint_[i]].inverse() * data_.oMf[id_feet_[i]];
        f_6d_ext_jointlocal[id_feet_parentjoint_[i]] = jMf.act(f_6d_ext_footlocal);
    }

    pinocchio::rnea(model_, data_, q, v, a, f_6d_ext_jointlocal);

    return data_.tau;
}

VecNv PinocchioDynamics::forwardDynamicsByEom(const VecNq &q, const VecNv &v,
                                              const VecNv &tau, const vector<Vec3> &f_ext)
{
    // calculate contact jacobian
    std::vector<Eigen::Matrix<double, 6, 18>> J(4);
    for (int i = 0; i < 4; i++)
    {
        J[i].setZero();
        pinocchio::computeFrameJacobian(model_, data_, q, id_feet_[i], pinocchio::LOCAL_WORLD_ALIGNED, J[i]);
    }

    // calculate M and C matrix, such that M * a + C = S * tau + J' * f_ext
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nv_, nv_);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nv_, 1);
    pinocchio::crba(model_, data_, q);
    pinocchio::nonLinearEffects(model_, data_, q, v);
    M.triangularView<Eigen::Upper>() = data_.M;
    M.triangularView<Eigen::Lower>() = M.triangularView<Eigen::Upper>().transpose();
    C = data_.nle;

    // calculate a by a = M.inv (tau - C + J' * f_ext)
    VecNv a = tau - C;
    for (int i = 0; i < 4; i++)
        a += J[i].topRows(3).transpose() * f_ext[i]; // only consider the linear part of jacobian

    return M.inverse() * a;
}

int main()
{
    std::string urdf_filename = "/home/zishang/Cpp_workspace/pinocchio_examples/inverse_dynamics_external_force_test/robot/mini_cheetah_mesh_v2.urdf";
    PinocchioDynamics pin_dyn(urdf_filename);
    pin_dyn.PrintModelTree();

    int nv = pin_dyn.GetNv();
    VecNq q;
    q << 0.0, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;

    VecNv v = Eigen::MatrixXd::Random(nv, 1);
    VecNv a = Eigen::MatrixXd::Random(nv, 1);
    cout << "a from given:\t" << a.transpose() << std::endl;

    Vec3 fext0(10, -20, 100), fext1(-10, 20, 120), fext2(5, -5, 90), fext3(-10, -10, 120);
    vector<Vec3> f_ext = {fext0, fext1, fext2, fext3};

    auto start = std::chrono::high_resolution_clock::now();
    VecNv tau = pin_dyn.InverseDynamicsByRnea(q, v, a, f_ext);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    cout << "tau from RNEA:\t" << tau.transpose() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    VecNv a2 = pin_dyn.forwardDynamicsByEom(q, v, tau, f_ext);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    cout << "a from EOM:\t" << a2.transpose() << std::endl;

    cout << "errors of a:\t" << (a - a2).squaredNorm() << std::endl;
    return 0;
}
