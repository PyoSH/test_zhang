// #ifndef Stereopsis_class
// #define Stereopsis_class

// #include "include/stereopsis_class.hpp"
#include "stereopsis_class.hpp"


Stereopsis::Stereopsis(cv::Mat& img_left, cv::Mat& img_right, Eigen::Matrix3d R_L2R, Eigen::Vector3d t_L2R)
{
    img_input_L = img_left.clone();
    img_input_R = img_right.clone(); 

    R = R_L2R;
    t = t_L2R;

    // BFS-U3--23287701
    K1 << 1153.262356, 0.000000, 495.886158,
        0.0, 1153.645224, 297.251145,
        0, 0, 1;
    
    cv::eigen2cv(K1, K1_); 
    dist_coeff1 = (cv::Mat1d(1, 5) <<-0.397690, 0.229652, 0.000536, -0.000847, 0.000000);
    
    // BFS-U3--23287707
    K2 << 1152.0, 0.000000, 471.3470,
        0.0, 1152.30, 289.5418,
        0, 0, 1;
    cv::eigen2cv(K2, K2_);
    dist_coeff2 = (cv::Mat1d(1, 5) << -0.4004, 0.2286, 0.00, 0.00, 0.00);

    std::cout << "--Class Stereopsis init--" << std::endl;
}

Stereopsis::~Stereopsis()
{
    std::cout << "--Class Stereopsis end--" << std::endl;
}

void Stereopsis::ApplyHomography(cv::Mat &img, cv::Mat &img_out, Eigen::MatrixXd K, cv::Mat dist_coeff, Eigen::MatrixXd R, Eigen::MatrixXd P)
{
    // cv::Mat _H;
    // cv::eigen2cv(H,_H);
    // cv::warpPerspective(img, img_out, _H, cv::Size(img.size().width * 1, img.size().height * 1));

    cv::Mat K_, R_, P_;
    cv::eigen2cv(K, K_);
    cv::eigen2cv(R, R_);
    cv::eigen2cv(P, P_);

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(K_, dist_coeff, R_, P_, img.size(), CV_16SC2, map11, map12);

    cv::remap(img, img_out, map11, map12, cv::INTER_LINEAR);
    
}

void Stereopsis::drawLines(cv::Mat &imgL, cv::Mat &imgR, cv::Mat &img_out, std::vector<cv::Point2d> points_L, std::vector<cv::Point2d> points_R)
{
    cv::hconcat(imgL.clone(), imgR.clone(), img_out);

    for(size_t i=0; i< points_L.size(); i++){
        cv::Point2d tmp_left, tmp_right;
        tmp_left = points_L[i];
        tmp_right.y = points_R[i].y;
        tmp_right.x = points_R[i].x + imgL.size().width;

        cv::Scalar color_ = cv::Scalar(rand()%255, rand()%255, rand()%255);

        cv::circle(img_out, cv::Point(tmp_left.x, tmp_left.y), 5, color_,
                    cv::FILLED, 4, 0);
        cv::circle(img_out, cv::Point(tmp_right.x, tmp_right.y), 5, color_ ,
                    cv::FILLED, 4, 0);

        cv::line(img_out, tmp_left, tmp_right, color_ , 2);
    }
}

Eigen::Vector4d Stereopsis::find_position(Eigen::Vector3d left_point, Eigen::Vector3d right_point, Eigen::Matrix3d Kl, Eigen::Matrix3d Kr, Eigen::Vector3d t)
{	
	double o_xl, o_yl;
    double o_xr, o_yr;
    o_xl = Kl(0,2);
    o_yl = Kl(1,2);
    o_xr = Kr(0,2);
    o_yr = Kr(1,2);

    double f_xl = Kl(0,0);
    double f_yl = Kl(1,1);
	double f_xr = Kr(0,0);
    double f_yr = Kr(1,1);

	double ul = left_point(0);
	double ur = right_point(0);
	double vl = left_point(1);
	double vr = right_point(1);

	double X_c = 0;
	double Y_c = 0;
	double Z_c_x = 0;
    double Z_c_y = 0;

    X_c = (ul-o_xl)*(t(2)*(ur - o_xl)-t(0)*f_xl) / (f_xl * (ur - ul));
    Y_c = (vl - o_yl) * (t(2) * (vr - o_yl) - t(1) * f_yl) / (f_yl * (vr - vl));
    Z_c_x = ( t(2) * (ur - o_xl) - t(0) * f_xl ) / (ur - ul);
    Z_c_y = ( t(2) * (vr - o_yl) - t(1) * f_yl ) / (vr - vl);

    

    if( X_c == 0){
        X_c = (ul - o_xl) * f_yl * Y_c / ((vl - o_yl) * f_yl);
    }
    else if ( Y_c == 0 )
    {
        Y_c = (vl - o_yl) * f_xl * X_c / ((ul - o_xl) * f_xl);
    }
    
    
    // std::cout << "X: " << X_c << std::endl;
    // std::cout << "Y: " << Y_c << std::endl;
    std::cout << "Z by tx: " << Z_c_x << "  Z by ty: " << Z_c_y << "\n" << std::endl;
    

	return {X_c, Y_c, Z_c_x, 1};
}

cv::Mat Stereopsis::disparityMap(cv::Mat &imgL, cv::Mat &imgR)
{
    cv::Mat imgL_mono, imgR_mono;
    cv::cvtColor(imgL, imgL_mono, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, imgR_mono, cv::COLOR_BGR2GRAY);

    int ndisparities = 112;
    int blocksize = 11;
    
    cv::Mat img_disparity_16s, img_disparity_8u;
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(ndisparities, blocksize);
    
    cv::Ptr<cv::ximgproc::DisparityFilter> wls_filter;
    wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);

    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(stereo);


    stereo->compute(imgL_mono, imgR_mono, img_disparity_16s);

    double minVal; double maxVal;
    minMaxLoc( img_disparity_16s, &minVal, &maxVal );
    std::cout << "min : " << minVal << ", max : " << maxVal << std::endl;
    img_disparity_16s.convertTo(img_disparity_8u, CV_8UC1, 255/(maxVal - minVal));

    return img_disparity_8u;
}

cv::Point3d Stereopsis::run()
{
    /*
    ( 2. 좌, 우 이미지 undistort )
    일단 K는 내장으로. */
    cv::Mat L_undist = img_input_L.clone();
    cv::Mat R_undist = img_input_R.clone();

    /*
    3. R, t로 호모그래피 작성*/
    Eigen::Matrix3d H1;
    Eigen::Matrix3d H2;    
    H1 = Eigen::Matrix3d::Identity();
    // H2 = K * R.transpose() * K.inverse(); 

    Eigen::Matrix<double, 3,4> P_rect_l, P_rect_r;
    // P_rect_l.Zero();
    P_rect_l.col(0) = K1.col(0);
    P_rect_l.col(1) = K1.col(1);
    P_rect_l.col(2) = K1.col(2);
    P_rect_l.col(3) << 0, 0, 0; 

    P_rect_r.col(0) = K2.col(0);
    P_rect_r.col(1) = K2.col(1);
    P_rect_r.col(2) = K2.col(2);
    P_rect_r.col(3) << 0, 0, 0; 

    Eigen::Matrix<double, 4, 4> MatExtrinsic;
    MatExtrinsic << R(0,0), R(0,1), R(0,2), t(0),
                    R(1,0), R(1,1), R(1,2), t(1),
                    R(2,0), R(2,1), R(2,2), t(2),
                    0, 0, 0, 1;
    P_rect_r = P_rect_r * MatExtrinsic;

    // std::cout << "P_rect_l: \n" << P_rect_l << "\n P_rect_r: \n" << P_rect_r << std::endl;

    /*
    4. 호모그래피 좌, 우 적용*/
    cv::Mat img_left_warped, img_right_warped;
    ApplyHomography(L_undist, img_left_warped, K1, dist_coeff1, R, P_rect_l);
    ApplyHomography(R_undist, img_right_warped, K2, dist_coeff2, R, P_rect_r);


    /*
    5. 변환된 이미지에서 특징점 검출*/
    
    cv::Size board_pattern;
    board_pattern.height = 3;
    board_pattern.width = 5;

    std::vector<cv::Point2d> L_img_points;
    bool isBoardL = cv::findChessboardCorners(img_left_warped, board_pattern , L_img_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE  + cv::CALIB_CB_FAST_CHECK);
    
    std::vector<cv::Point2d> R_img_points;
    bool isBoardR = cv::findChessboardCorners(img_right_warped, board_pattern , R_img_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE  + cv::CALIB_CB_FAST_CHECK);

    std::cout << "Num of points -L: " << L_img_points.size() << " -R: " << R_img_points.size() << std::endl;

    cv::Point3d tgt_position = {0,0,0};

    if(isBoardL && isBoardR){
        // 임시로 만든 그림 그리기
        cv::Mat img_matches;
        drawLines(img_left_warped, img_right_warped, img_matches, L_img_points, R_img_points);

        /* -이번 인턴기간에는 PASS-
        6. 검출된 특징점 매칭*/
        /*
        // 1. Feature point detecting
        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        detector -> detectAndCompute(L_undist, cv::noArray(), keypoints1, descriptors1);
        detector -> detectAndCompute(R_undist, cv::noArray(), keypoints2, descriptors2);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        //-- Draw matches
        cv::Mat img_matches;
        drawMatches( L_undist, keypoints1, R_undist, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Show detected matches
        cv::imshow("Good Matches", img_matches );
        */

        /* -이번 인턴기간에는 PASS-
        7. 매칭 리스트에서 epipolar constraint로 outlier 제거
        걸러내고 난 다음에 L_img_points 변수에 넣자. */


        /* 
        8. 점들에 대해 disparity 계산 or disparity map 작성*/
        // cv::Mat img_disparityMap;
        // img_disparityMap = disparityMap(img_left_warped, img_right_warped);    

        /*
        9. 점들에 대해 XYZ1 상대거리 계산*/
        Eigen::Vector3d L_px, R_px;


        cv::Point2d min_left= {INFINITY,INFINITY};
        cv::Point2d min_right = {INFINITY,INFINITY};

        std::vector<cv::Point2d>::iterator iter, iter2;
        
        for(iter = L_img_points.begin(); iter != L_img_points.end(); iter++){
            cv::Point2d tmp = *iter;
            
            if( tmp.x <= min_left.x && tmp.y <= min_left.y){
                min_left = tmp;
            }
        }

        for(iter2 = R_img_points.begin(); iter2 != R_img_points.end(); iter2++){
            cv::Point2d tmp = *iter2;
            
            if( tmp.x <= min_left.x && tmp.y <= min_right.y){
                min_right = tmp;
            }
        }

        L_px <<  min_left.x, min_left.y, 1;
        R_px <<  min_right.x, min_right.y, 1;

        Eigen::Vector4d tmp = find_position( L_px, R_px, K1, K2, t);

        
        tgt_position.x = tmp(0);
        tgt_position.y = tmp(1);
        tgt_position.z = tmp(2);

        cv::String info = cv::format("XYZ: [%.3f, %.3f, %.3f]", tgt_position.x, tgt_position.y, tgt_position.z);
        cv::putText(img_matches, info, cv::Point(5,45), cv::FONT_HERSHEY_PLAIN, 3, cv::Vec3b(0, 255, 0));
        
        // cv::imshow("warped left", img_left_warped);
        // cv::imshow("warped right", img_right_warped);
        cv::imshow("All Matches", img_matches);
        
        // cv::imshow("StereoBM", img_disparityMap);
        // cv::moveWindow("warped left", 0, 600);
        // cv::moveWindow("warped right", 1000, 600);
        cv::waitKey(1);


    }
    else{
        std::cout << "There is no board" << std::endl;
        
    }

    
    return tgt_position;
    

}

// #endif // Stereopsis_class