#ifndef __DO_REPROJ_CPP__
#define __DO_REPROJ_CPP__

#include "do_reproj.hpp"

#include <opencv2/core/eigen.hpp>

namespace aimer::aim {

// DoReproj：负责将相机内参与 IMU 外参组合，生成两帧间的透视变换
DoReproj::DoReproj() {}



/*
// 输入单位为弧度  输入当前电控返回的 roll, pitch, yaw角度
Eigen::Quaterniond quaternionFromRPY(double roll, double pitch, double yaw) {
    // 注意顺序：先绕 X（roll），再绕 Y（pitch），最后绕 Z（yaw）
    Eigen::AngleAxisd r(roll,  Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd p(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd y(yaw,   Eigen::Vector3d::UnitZ());
    return (y * p * r);  // ZYX
}

// 调用示例 角度转弧度
 double roll  = deg2rad(cur_roll_deg);
double pitch = deg2rad(cur_pitch_deg);
double yaw   = deg2rad(cur_yaw_deg);
Eigen::Quaterniond imu_q = quaternionFromRPY(roll, pitch, yaw); 
//得到q1/q2





Eigen::Matrix3d imu2optical() {
  // 1) 相机在云台上的偏转
  double ry = M_PI/360, rp = M_PI/120;
  Eigen::Matrix3d R_cam_link = 
      Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ());
  // 2) 从相机机体到光心坐标系
  Eigen::Matrix3d R_optical = 
      Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitX());
  // 合成：gimbal_link → camera_optical_frame
  return R_optical * R_cam_link;
}

Eigen::Matrix3d R_imu2cam = imu2optical();
cv::Mat imuMat;
cv::eigen2cv(R_imu2cam, imuMat);
//即可得到imu四元数

*/

// init：用 OpenCV 转换输入的 cv::Mat 为 Eigen 矩阵，并构建齐次相机矩阵
void DoReproj::init(const cv::Mat& cam, const cv::Mat& imu) {
    // cam: 先转换为 3x4，再扩展为 4x4 齐次矩阵
    this->cam = Eigen::Matrix4d();
    Eigen::Matrix<double, 3, 4> mat;
    cv::cv2eigen(cam, mat);
    // for inversibility (théorème de factorisation)
    this->cam.block<3, 4>(0, 0) = mat;
    this->cam(3, 3) = 1;

    // imu: 将 IMU->相机 旋转矩阵读入
    cv::cv2eigen(imu, this->imu);
}

// 构造时可直接调用 init
DoReproj::DoReproj(const cv::Mat& cam, const cv::Mat& imu) {
    this->init(cam, imu);
}

// from_q_get_trans_mat：根据四元数 q 计算 IMU 在机体坐标下的 4x4 变换矩阵
Eigen::Matrix4d DoReproj::from_q_get_trans_mat(const DoReproj::Quat& q) {
    // 上半块为旋转（IMU->机体），下行为 [0,0,0,1]
    Eigen::Matrix4d res = Eigen::Matrix4d::Zero(4, 4);
    res.block<3, 3>(0, 0) = this->imu * q.matrix().inverse();
    res(3, 3) = 1;
    return res;
}

// get_fr_trans_mat：计算两帧（q1→q2）之间的 3x3 透视变换 
// q1上一帧的 IMU 四元数（机体在世界／上一帧坐标系中的旋转）         q2当前帧的 IMU 四元数
Eigen::Matrix3d DoReproj::get_fr_trans_mat(const DoReproj::Quat& q1, const DoReproj::Quat& q2) {
    // 先 cam * T(q2) * (cam * T(q1))^{-1}，再取前 3x3 作为投影变化
    Eigen::Matrix4d mat = this->cam * this->from_q_get_trans_mat(q2)
        * (this->cam * this->from_q_get_trans_mat(q1)).inverse();
    return mat.block<3, 3>(0, 0);
}

// reproj：对输入图像 src 应用透视变换，输出重投影结果
cv::Mat DoReproj::reproj(const cv::Mat& src, const DoReproj::Quat& q1, const DoReproj::Quat& q2) {
    // 1. 计算 3x3 变换矩阵并转为 cv::Mat
    Eigen::Matrix3d mat = this->get_fr_trans_mat(q1, q2);
    cv::Mat cv_mat;
    cv::eigen2cv(mat, cv_mat);

    // 2. warpPerspective 重投影至原图大小
    cv::Mat res;
    cv::warpPerspective(src, res, cv_mat, src.size());
    return res.clone();
}

}  // namespace aimer::aim

#endif
