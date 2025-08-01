#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>

using namespace pinocchio;

Data::Tensor3x computeFrameKinematicsHessian(const Model &model, const Data &data,
                                             const FrameIndex frame_id,
                                             const Data::Tensor3x &H_joint_loc,
                                             const Data::Matrix6x &J_frame_lwa)
{
    const Eigen::DenseIndex nv = model.nv;
    Data::Tensor3x H_frame_lwa(6, nv, nv);

    // 常量：关节 → frame 的变换矩阵（LOCAL 下的固定偏置）
    const Motion::ActionMatrixType X_j2f_local = model.frames[frame_id].placement.toActionMatrixInverse();

    // 纯旋转的变换矩阵
    const pinocchio::SE3 R(data.oMf[frame_id].rotation(), Eigen::Vector3d::Zero());
    const Motion::ActionMatrixType X_R = R.toActionMatrix();

    // 纯旋转的变换矩阵关于q_k的导数
    Eigen::Matrix<double, 6, 6> X_w_hat;
    X_w_hat.setZero();

    //    J_f_lwa = X_R * X_j2f_local * J_j_loc
    // => H_f_lwa_k = X_R * X_j2f_local * H_joint_loc_k + d(X_R)/dq_k * X_j2f_local * J_j_loc
    // => H_f_lwa_k = X_R * X_j2f_local * H_joint_loc_k + X(w_hat) * X_R * X_j2f_local * J_j_loc
    // => H_f_lwa_k = X_R * X_j2f_local * H_joint_loc_k + X(w_hat) * J_frame_lwa
    const Eigen::DenseIndex outer_offset = 6 * nv;
    Data::Matrix6x H_frame_lwa_k(6, nv);
    for (Eigen::DenseIndex k = 0; k < nv; ++k)
    {
        Eigen::Map<const Data::Matrix6x> H_joint_loc_k(H_joint_loc.data() + k * outer_offset, 6, nv);

        const Eigen::Vector3d &w_k = J_frame_lwa.bottomRows<3>().col(k);
        const Eigen::Matrix3d w_hat = pinocchio::skew(w_k);
        X_w_hat.template topLeftCorner<3, 3>() = w_hat;
        X_w_hat.template bottomRightCorner<3, 3>() = w_hat;

        H_frame_lwa_k.noalias() = X_R * X_j2f_local * H_joint_loc_k;
        H_frame_lwa_k.noalias() += X_w_hat * J_frame_lwa;

        Eigen::Map<Data::Matrix6x>(H_frame_lwa.data() + k * outer_offset, 6, nv) = H_frame_lwa_k;
    }

    return H_frame_lwa;
}

int main()
{
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    Data data(model);
    std::string frame_name = "FL_foot";
    FrameIndex frame_id = model.getFrameId(frame_name);
    JointIndex joint_id = model.frames[frame_id].parent;
    int nv = model.nv;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeJointJacobians(model, data);
    pinocchio::computeJointKinematicHessians(model, data);

    Data::Tensor3x H_joint_loc(6, model.nv, model.nv);
    H_joint_loc.setZero();
    pinocchio::getJointKinematicHessian(model, data, joint_id,
                                        pinocchio::ReferenceFrame::LOCAL, H_joint_loc);
    Data::Matrix6x J_frame_lwa(6, nv);
    J_frame_lwa.setZero();
    pinocchio::getFrameJacobian(model, data, frame_id,
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                J_frame_lwa);

    Data::Tensor3x H_frame_lwa = computeFrameKinematicsHessian(model, data, frame_id, H_joint_loc, J_frame_lwa);

    // ---------- 有限差分校验 ----------
    Data data_plus(model);
    Data::ConfigVectorType v_plus = Data::ConfigVectorType::Zero(nv);
    Data::Matrix6x J_plus(6, nv);
    J_plus.setZero();
    const Eigen::DenseIndex outer_offset = 6 * nv;
    bool all_ok = true;
    double eps = 1e-8;
    double tol = std::sqrt(eps);

    for (Eigen::DenseIndex k = 0; k < nv; ++k)
    {
        // q_plus = integrate(q, eps * e_k)
        v_plus.setZero();
        v_plus[k] = eps;
        const auto q_plus = pinocchio::integrate(model, q, v_plus);

        // 计算 J_plus（frame@LWA）
        pinocchio::computeJointJacobians(model, data_plus, q_plus);
        pinocchio::updateFramePlacements(model, data_plus);
        J_plus.setZero();
        pinocchio::getFrameJacobian(model, data_plus, frame_id,
                                    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                    J_plus);

        // 数值差分：dJ_dq_fd ≈ (J_plus - J_ref) / eps
        Data::Matrix6x dJ_dq_fd = (J_plus - J_frame_lwa) / eps;

        // 解析切片：H_frame_lwa(:,:,k)
        Eigen::Map<const Data::Matrix6x> dJ_dq_anal(H_frame_lwa.data() + k * outer_offset, 6, nv);

        // 误差度量
        const double err_inf = (dJ_dq_fd - dJ_dq_anal).template lpNorm<Eigen::Infinity>();
        const double err_fro = (dJ_dq_fd - dJ_dq_anal).norm();

        if (!(err_inf <= tol))
        {
            all_ok = false;
            std::cout << "[verify] FAILED at k = " << k
                      << "  |err|_inf = " << err_inf
                      << "  (tol = " << tol << "),  |err|_F = " << err_fro << "\n";
            // 如需查看细节，取消注释：
            std::cout << "dJ_dq_fd:\n"
                      << dJ_dq_fd << "\n";
            std::cout << "dJ_dq_anal:\n"
                      << dJ_dq_anal << "\n";
            // 可以选择 break；这里继续遍历所有 k
        }
    }

    std::cout << "[verify] " << (all_ok ? "PASSED" : "FAILED") << "\n";

    return 0;
}
