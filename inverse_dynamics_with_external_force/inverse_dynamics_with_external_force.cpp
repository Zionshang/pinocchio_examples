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

using VecNq = Eigen::Matrix<double, 19, 1>;
using VecNv = Eigen::Matrix<double, 18, 1>;

class PinocchioDynamics
{
public:
    PinocchioDynamics(string urdf_filename, vector<string> foot_names);

    void PrintModelTree();
    VecNv InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                const VecNv &a, const vector<Eigen::Vector3d> &f_ext);
    VecNv InverseDynamicsByEom(const VecNq &q, const VecNv &v,
                               const VecNv &a, const vector<Eigen::Vector3d> &f_ext);

    const int GetNq() const { return nq_; }
    const int GetNv() const { return nv_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;

    int nq_, nv_;
    std::vector<pinocchio::FrameIndex> id_feet_;
    std::vector<pinocchio::JointIndex> id_joint_;
};

PinocchioDynamics::PinocchioDynamics(string urdf_filename, vector<string> foot_names)
{
    pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);
    nq_ = model_.nq;
    nv_ = model_.nv;

    for (int i = 0; i < foot_names.size(); i++)
    {
        id_feet_.push_back(model_.getFrameId(foot_names[i]));
        id_joint_.push_back(model_.frames[id_feet_[i]].parent);
        std::cout << foot_names[i] << " frame id: " << id_feet_[i] << ", parent joint id: " << id_joint_[i] << std::endl;
    }
}

void PinocchioDynamics::PrintModelTree()
{
    using pinocchio::FrameIndex;
    using pinocchio::JointIndex;

    std::cout << " -------------index and names of joint------------- " << std::endl;
    for (JointIndex joint_id = 0; joint_id < (JointIndex)model_.njoints; ++joint_id)
        std::cout << std::setw(24) << std::left << joint_id << ": " << model_.names[joint_id] << std::endl;

    std::cout << " -------------index and names of frame------------- " << std::endl;
    for (FrameIndex frame_id = 0; frame_id < (FrameIndex)model_.nframes; ++frame_id)
        std::cout << std::setw(24) << std::left << frame_id << ": " << model_.frames[frame_id].name << std::endl;
}

VecNv PinocchioDynamics::InverseDynamicsByRnea(const VecNq &q, const VecNv &v,
                                               const VecNv &a, const vector<Eigen::Vector3d> &f_ext)
{
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    // all joints force, expressed in local frame
    pinocchio::container::aligned_vector<pinocchio::Force> f_joints_L(model_.njoints, pinocchio::Force::Zero());

    // transformation of foot relative to world, but only consider rotation part.
    pinocchio::SE3 X_wf_rotation = pinocchio::SE3::Identity();
    // foot force expressed in local_world_aligned frame
    pinocchio::Force f_foot_LWA = pinocchio::Force::Zero();
    // foot force expressed in frame local frame
    pinocchio::Force f_foot_L = pinocchio::Force::Zero();
    for (int i = 0; i < 4; i++)
    {
        X_wf_rotation.rotation(data_.oMf[id_feet_[i]].rotation());
        f_foot_LWA.linear(f_ext[i]);
        f_foot_L = X_wf_rotation.actInv(f_foot_LWA);

        // transformation of foot relative to joint
        const pinocchio::SE3 X_jf_ = data_.oMi[id_joint_[i]].inverse() * data_.oMf[id_feet_[i]];
        f_joints_L[id_joint_[i]] = X_jf_.act(f_foot_L);
    }

    pinocchio::rnea(model_, data_, q, v, a, f_joints_L);

    return data_.tau;
}

VecNv PinocchioDynamics::InverseDynamicsByEom(const VecNq &q, const VecNv &v,
                                              const VecNv &a, const vector<Eigen::Vector3d> &f_ext)
{
    // calculate contact jacobian
    std::vector<Eigen::MatrixXd> J(4);
    for (int i = 0; i < 4; i++)
    {
        J[i].setZero(6, nv_);
        pinocchio::computeFrameJacobian(model_, data_, q, id_feet_[i], pinocchio::LOCAL_WORLD_ALIGNED, J[i]);
    }

    // calculate M and C matrix, such that M * a + C = S * tau + J' * f_ext
    pinocchio::crba(model_, data_, q);
    pinocchio::nonLinearEffects(model_, data_, q, v);
    data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.triangularView<Eigen::StrictlyUpper>().transpose();

    // calculate a by a = M.inv (tau - C + J' * f_ext)
    VecNv tau = data_.M * a + data_.nle;
    for (int i = 0; i < id_feet_.size(); i++)
        tau -= J[i].topRows<3>().transpose() * f_ext[i]; // only consider the linear part of jacobian

    return tau;
}

int main()
{
    std::string urdf_filename = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    vector<string> foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};

    PinocchioDynamics pin_dyn(urdf_filename, foot_names);
    pin_dyn.PrintModelTree();

    int nv = pin_dyn.GetNv();
    VecNq q;
    q << 0.0, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;

    VecNv v = Eigen::VectorXd::Random(nv);
    VecNv a = Eigen::VectorXd::Random(nv);
    cout << "a from given:\t" << a.transpose() << std::endl;

    Eigen::Vector3d fext0(10, -20, 100), fext1(-10, 20, 120), fext2(5, -5, 90), fext3(-10, -10, 120);
    vector<Eigen::Vector3d> f_ext = {fext0, fext1, fext2, fext3};

    auto start = std::chrono::high_resolution_clock::now();
    VecNv tau_rnea = pin_dyn.InverseDynamicsByRnea(q, v, a, f_ext);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "InverseDynamicsByRnea Execution time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    cout << "tau from RNEA:\t" << tau_rnea.transpose() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    VecNv tau_eom = pin_dyn.InverseDynamicsByEom(q, v, a, f_ext);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "InverseDynamicsByEom Execution time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    cout << "tau from EOM:\t" << tau_eom.transpose() << std::endl;

    cout << "errors of a:\t" << (tau_rnea - tau_eom).squaredNorm() << std::endl;
    return 0;
}
