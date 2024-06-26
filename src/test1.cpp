#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


void GetChessboardCorners(std::vector <Eigen::Matrix<double, 48, 2>> &pixel_coners){
    Eigen::Matrix<double, 48, 2> c1,c2,c3,c4,c5;

    c1 <<  254.1017,   83.3595, 251.8365,  151.6737,
            250.2021,  220.1422, 249.2022,  288.1765,
            248.8231,  355.2133, 249.0360,  420.7380,
            321.8561,   84.1562, 319.5507,  152.9469,
            317.6887,  221.8905, 316.2783,  290.3846,
            315.3153,  357.8543, 314.7848,  423.7753,
            390.7910,   85.5584, 388.4527,  154.6322,
            386.3588,  223.8546, 384.5229,  292.6153,
            382.9515,  360.3328, 381.6444,  426.4778,
            460.3207,   87.5705, 457.9561,  156.7253,
            455.6311,  226.0225, 453.3651,  294.8496,
            451.1759,  362.6237, 449.0783,  428.8142,
            529.8325,   90.1794, 527.4485,  159.2105,
            524.8991,  228.3764, 522.2098,  297.0681,
            519.4092,  364.7060, 516.5271,  430.7626,
            598.7150,   93.3558, 596.3188,  162.0619,
            593.5578,  230.8938, 590.4629,  299.2518,
            587.0728,  366.5642, 583.4316,  432.3107,
            666.3859,   97.0553, 663.9839,  165.2442,
            661.0291,  233.5486, 657.5562,  301.3829,
            653.6129,  368.1887, 649.2566,  433.4565,
            732.3190,  101.2196, 729.9150,  168.7142,
            726.7870,  236.3114, 722.9717,  303.4458,
            718.5241,  369.5762, 713.5129,  434.2086;

    c2 << 227.1103,  201.0992, 239.5458,  250.8074,
            252.2802,  300.3079, 265.2465,  349.3672,
            278.3758,  397.7637, 291.5990,  445.2926,
            276.4663,  188.2769, 288.8216,  238.0660,
            301.3981,  287.6668, 314.1306,  336.8441,
            326.9534,  385.3746, 339.8016,  433.0518,
            326.1649,  175.6110, 338.4187,  225.4004,
            350.8144,  275.0215, 363.2892,  324.2388,
            375.7810,  372.8285, 388.2296,  420.5833,
            375.9644,  163.1629, 388.0965,  212.8717,
            400.2909,  262.4327, 412.4878,  311.6115,
            424.6289,  360.1847, 436.6588,  407.9450,
            425.6230,  150.9902, 437.6146,  200.5384,
            449.5901,  249.9600, 461.4929,  299.0222,
            473.2685,  347.5033, 484.8666,  395.1971,
            474.9048,  139.1452, 486.7393,  188.4554,
            498.4815,  237.6600, 510.0778,  286.5294,
            521.4782,  334.8443, 532.6371,  382.4003,
            523.5856,  127.6745, 535.2485,  176.6725,
            546.7459,  225.5859, 558.0277,  274.1888,
            569.0479,  322.2654, 579.7657,  369.6143,
            571.4574,  116.6179, 582.9365,  165.2338,
            594.1808,  213.7854, 605.1437,  262.0519,
            615.7832,  309.8216, 626.0632,  356.8966;

    c3 << 187.7422,  129.1586, 185.1786,  177.7727,
            183.0375,  226.2673, 181.3202,  274.4352,
            180.0218,  322.0780, 179.1320,  369.0105,
            235.7106,  131.0186, 233.0506,  179.4648,
            230.7435,  227.7908, 228.7923,  275.7910,
            227.1949,  323.2684, 225.9449,  370.0393,
            283.6222,  133.0912, 280.8713,  181.3013,
            278.4030,  229.3905, 276.2221,  277.1557,
            274.3296,  324.4027, 272.7226,  370.9501,
            331.2718,  135.3631, 328.4356,  183.2711,
            325.8124,  231.0577, 323.4088,  278.5234,
            321.2287,  325.4775, 319.2729,  371.7417,
            378.4614,  137.8178, 375.5460,  185.3612,
            372.7758,  232.7830, 370.1591,  279.8882,
            367.7024,  326.4902, 365.4102,  372.4147,
            425.0059,  140.4368, 422.0182,  187.5572,
            419.1102,  234.5563, 416.2921,  281.2442,
            413.5730,  327.4392, 410.9603,  372.9713,
            470.7364,  143.2004, 467.6839,  189.8442,
            464.6489,  236.3676, 461.6432,  282.5863,
            458.6782,  328.3238, 455.7642,  373.4151,
            515.5034,  146.0882, 512.3940,  192.2071,
            509.2437,  238.2070, 506.0657,  283.9095,
            502.8733,  329.1440, 499.6793,  373.7504;

    c4 << 515.4911, 199.4973, 515.1737, 249.5788,
            514.7865, 299.4982, 514.3323, 349.0303,
            513.8150, 397.9603, 513.2392, 446.0879,
            564.9253, 199.9341, 564.5379, 249.7250,
            564.0073, 299.3565, 563.3384, 348.6072,
            562.5378, 397.2659, 561.6137, 445.1353,
            613.5765, 200.4973, 613.1226, 249.9281,
            612.4557, 299.2028, 611.5822, 348.1048,
            610.5116, 396.4270, 609.2552, 443.9761,
            661.2551, 201.1790, 660.7389, 250.1848,
            659.9439, 299.0383, 658.8784, 347.5285,
            657.5540, 395.4532, 655.9852, 442.6240,
            707.7936, 201.9697, 707.2195, 250.4910,
            706.3060, 298.8643, 705.0627, 346.8847,
            703.5032, 394.3560, 701.6450, 441.0948,
            753.0495, 202.8589, 752.4220, 250.8424,
            751.4002, 298.6822, 749.9947, 346.1805,
            748.2210, 393.1476, 746.0990, 439.4063,
            796.9083, 203.8351, 796.2315, 251.2338,
            795.1119, 298.4934, 793.5607, 345.4236,
            791.5951, 391.8419, 789.2371, 437.5777,
            839.2855, 204.8855, 838.5628, 251.6599,
            837.3557, 298.2994, 835.6760, 344.6221,
            833.5417, 390.4535, 830.9769, 435.6298;

    c5 << 287.6053, 134.0422, 288.1820, 182.5803,
            289.0145, 230.9630, 290.0934, 278.9841,
            291.4063, 326.4473, 292.9384, 373.1705,
            332.5066, 133.4396, 332.9827, 182.7598,
            333.6569, 231.9157, 334.5217, 280.6911,
            335.5669, 328.8809, 336.7807, 376.2947,
            378.9914, 133.0229, 379.3566, 183.0845,
            379.8556, 232.9729, 380.4830, 282.4623,
            381.2315, 331.3385, 382.0928, 379.4038,
            426.9495, 132.8118, 427.1938, 183.5661,
            427.5013, 234.1385, 427.8688, 284.2937,
            428.2927, 333.8094, 428.7684, 382.4803,
            476.2401, 132.8251, 476.3543, 184.2154,
            476.4549, 235.4152, 476.5417, 286.1806,
            476.6147, 336.2814, 476.6743, 385.5055,
            526.6915, 133.0803, 526.6675, 185.0417,
            526.5477, 236.8045, 526.3351, 288.1170,
            526.0341, 338.7415, 525.6501, 388.4598,
            578.1019, 133.5923, 577.9334, 186.0523,
            577.5820, 238.3064, 577.0545, 290.0956,
            576.3598, 341.1758, 575.5090, 391.3232,
            630.2425, 134.3734, 629.9244, 187.2528,
            629.3332, 239.9193, 628.4790, 292.1082,
            627.3757, 343.5698, 626.0404, 394.0756;

    pixel_coners[0] = c1;
    pixel_coners[1] = c2;
    pixel_coners[2] = c3;
    pixel_coners[3] = c4;
    pixel_coners[4] = c5;
}

std::vector<Eigen::Matrix3d> ComputeHomography8Point(std::vector<Eigen::Matrix<double,48,2>> pixel_coners, Eigen::Matrix<double, 48,4> world_points){
    std::vector<Eigen::Matrix3d> Hs(5);

    Eigen::MatrixXd A(2*48, 9); // H가 3x3 -> 9
    for(int i=0; i<5; i++){ // 이미지에 대해서
        for(int j=0; j< 48; j++){ // 검출된 각 chessboard 포인트에 대해서
            double X = world_points.row(j)(0);
            double Y = world_points.row(j)(1);
            double W = world_points.row(j)(3);
            double u = pixel_coners[i].row(j)(0);
            double v = pixel_coners[i].row(j)(1);

            Eigen::Matrix<double, 9,1> a1, a2; // Direct Linear Transformation 식 세우기
            a1 << X, Y, W, 0, 0, 0, -u*X, -u*Y, -u*W;
            a2 << 0, 0, 0, X, Y, W, -v*X, -v*Y, -v*W;

            int k = 2* j;
            A.row(k) = a1.transpose();
            A.row(k+1) = a2.transpose();
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::Matrix<double, 9, 1> h = V.col(V.cols()-1);
        Eigen::Matrix3d H = Eigen::Map<Eigen::Matrix3d>(h.data());
        H = H/H(2,2);
        Hs[i] = H.transpose();
    }

    return Hs;
}

std::vector<Eigen::Vector3d> Project3Dto2D(Eigen::Matrix3d K, Eigen::Matrix3d R, Eigen::Vector3d t, Eigen::Matrix<double, 48, 4> world_points){
	std::vector<Eigen::Vector3d> projected_points(48);

	Eigen::MatrixXd H(3,4);
	H.col(0) = R.col(0);
	H.col(1) = R.col(1);
	H.col(2) = R.col(2);
	H.col(3) = t.transpose();

	Eigen::Vector4d curr_world_point;
	Eigen::Vector3d curr_px_point;
	for(int i=0; i< world_points.rows(); i++){
		curr_world_point = world_points.row(i);
		curr_px_point = K * H * curr_world_point;
		curr_px_point = curr_px_point / curr_px_point(2);
		projected_points.push_back(curr_px_point);
	}

	return projected_points;
}

void DrawPoints(std::vector<Eigen::Vector3d> points_per_image, cv::Mat& curr_img){
	cv::Mat img = curr_img;

	for(int i=0; i<points_per_image.size(); i++){
		cv::Point location_pt(points_per_image[i](0), points_per_image[i](1));
		cv::circle(img, location_pt, 5, 255, -1);
	}

	cv::imshow("12901", img);
	cv::waitKey(0);
}

int main(int argc, char **argv)
{
    std::vector<cv::Mat> images;

    // 이미지 불러오기
    for(int i=1; i<=5; i++){
        std::string curr_img = "/home/masterpyo/eclipse-workspace/test_zhang-master/imgs/"+std::to_string(i)+".jpg";
        cv::Mat img = cv::imread(curr_img, cv::IMREAD_GRAYSCALE);
        if(img.empty())	{
        	std::cout<< "이미지가 없어부러요" << std::endl;
        	return -1;
        }
        else{
        	std::cout<< "이미지가 있으니 저장해유" << std::endl;
        	images.push_back(img);
        }

    }

    // chessboard의 교차점 검출
    std::vector<Eigen::Matrix<double, 48, 2>> pixel_coners(5);
    GetChessboardCorners(pixel_coners);

    // chessboard의 world coordinates 만들기
    Eigen::Matrix<double, 48, 4> world_points;
    int gap = 110; // mm dimension
    for(int r=0; r<8; r++){
        for(int c=0; c<6; c++){
            world_points.row(6*r+c) =
                    Eigen::Vector4d(r*gap, c*gap, 0, 1).transpose();
        }
    }

    // 이미지 각각으로부터 3x3 Homography 구하기
    std::vector<Eigen::Matrix3d> Hs;
    Hs = ComputeHomography8Point(pixel_coners, world_points);

    // Homography로부터 K 구하기
    Eigen::Matrix<double, 10, 6> A;

    for(int i=0; i<5; i++){
        Eigen::Matrix<double, 3, 3> H = Hs[i];
        Eigen::Matrix<double, 6, 1> a1, a2;

        double h1 = H(0,0);
        double h2 = H(0,1);
        double h3 = H(0,2);
        double h4 = H(1,0);
        double h5 = H(1,1);
        double h6 = H(1,2);
        double h7 = H(2,0);
        double h8 = H(2,1);
        double h9 = H(2,2);

        a1 << h1*h2, h1*h5+h2*h4, h1*h8+h2*h7, h4*h5, h4*h8+h5*h7, h7*h8;
        a2 << h1*h1-h2*h2, 2*h1*h4-2*h2*h5, 2*h1*h7-2*h2*h8, h4*h4-h5*h5, 2*h4*h7-2*h5*h8, h7*h7-h8*h8;

        int j = 2*i;
        A.row(j) = a1;
        A.row(j+1) = a2;
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 10, 6>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 6,6> V = svd.matrixV();
    Eigen::Matrix<double, 6,1> v = V.col(V.cols()-1);
    Eigen::Matrix3d KtinvKinv;
    KtinvKinv << v(0), v(1), v(2), v(1), v(3), v(4), v(2), v(4), v(5);

    Eigen::Matrix3d pseudoKinv = KtinvKinv.llt().matrixL();
    Eigen::Matrix3d K = pseudoKinv.inverse().transpose();
    K = K / K(2,2);
    Eigen::Matrix3d Kinv = K.inverse();

    // R, t 구하기
    std::vector<Eigen::Matrix3d> Rs(5);
    std::vector<Eigen::Vector3d> ts(5);

    std::cout << "K\n" << K << std::endl;
    std::cout << "---------------------------------" << std::endl;

    for(int i=0; i<5; i++){
        Eigen::Vector3d r1 = Kinv * Hs[i].col(0);
        Eigen::Vector3d r2 = Kinv * Hs[i].col(1);
        Eigen::Vector3d r3 = r1.cross(r2);
        Eigen::Matrix3d R;
        R << r1, r2, r3;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        Rs[i] = R;

        double labmda = 1/r1.norm();
        Eigen::Vector3d t = labmda * Kinv * Hs[i].col(2);
        ts[i] = t;

        std::cout << "img : " << i << std::endl;
        std::cout << "R \n" << R << std::endl;
        std::cout << "t\n" << t<< std::endl;
        std::cout << "---------------------------------" << std::endl;


        // 점 재투영해서 띄우기
		std::vector<Eigen::Vector3d> projected_pt = Project3Dto2D(K, R, t, world_points);
		DrawPoints(projected_pt, images[i]);
    }

    return 0;
}
