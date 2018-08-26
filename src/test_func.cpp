//
// Created by eli on 18-8-25.
//

#include "ImageDetect.h"
#include "darknet.h"
#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"
using namespace std;

void printTestRc(string test_name,int rc)
{
    cout<<test_name;
    if (rc==0)
        cout<<" [passed]\n";
    else
        cout<<" [failed]\n";
}

int test_image_color_mat_to_image()
{
    cv::Mat src=cv::imread("test/test.jpg");
//    cv::Mat src=cv::imread("test/test_image_color_to_image.png");
    image dst=color_mat_to_image(src);
    int rc=0;
    if (src.rows!=dst.h || src.cols!=dst.w || src.channels()!=dst.c)
        rc=1;
    std::cout<<src.channels()<<std::endl;
    show_image(dst,"test/test_image_color_to_image");

    cv::Mat img2=image_to_color_cv_mat(dst);
    return rc;
}

int test_detector()
{
    int rc=0;

    cv::Mat src=cv::imread("test/test.jpg");
    image dst=color_mat_to_image(src);
    cv::Mat predict_img;
    ObjDetector detector("cfg/voc.data","cfg/test.cfg","cfg/model.weights","cfg/voc.name",0.99, 0.5);
    detector.predict(src,predict_img);
    cv::imshow("predict img",predict_img);
    cv::waitKey(1000);
    cv::imwrite("test/predict_rc.jpg",predict_img);
    return rc;
}
void test()
{

    int rc = test_image_color_mat_to_image();
    printTestRc("test image color mat to image",rc);

    rc=test_detector();
    printTestRc("test detector class prediction",rc);

}