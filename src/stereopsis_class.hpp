// #ifndef Stereopsis_class_H
// #define Stereopsis_class_H

#include <iostream>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/ximgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

class Stereopsis
{
    private:
        cv::Mat img_input_L;
        cv::Mat img_input_R;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;

        Eigen::Matrix3d K1;
        cv::Matx33d K1_;
        cv::Mat dist_coeff1 = cv::Mat::zeros(1, 5, CV_64FC1);

        Eigen::Matrix3d K2;
        cv::Matx33d K2_;
        cv::Mat dist_coeff2 = cv::Mat::zeros(1, 5, CV_64FC1);


    public:
        explicit Stereopsis(cv::Mat& img_left, cv::Mat& img_right, Eigen::Matrix3d R_L2R, Eigen::Vector3d t_L2R);
        ~Stereopsis();
        void ApplyHomography(cv::Mat &img, cv::Mat &img_out, Eigen::MatrixXd K, cv::Mat dist_coeff, Eigen::MatrixXd R, Eigen::MatrixXd P);
        void drawLines(cv::Mat &imgL, cv::Mat &imgR, cv::Mat &img_out, std::vector<cv::Point2d> points_L, std::vector<cv::Point2d> points_R);
        Eigen::Vector4d find_position(Eigen::Vector3d left_point, Eigen::Vector3d right_point, Eigen::Matrix3d K1, Eigen::Matrix3d K2, Eigen::Vector3d t);
        cv::Mat disparityMap(cv::Mat &imgL, cv::Mat &imgR);
        cv::Point3d run(); 
};


// #endif // Stereopsis_class_H