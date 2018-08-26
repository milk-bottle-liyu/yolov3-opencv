#ifndef EP_OBJECT_DETECT_H
#define EP_OBJECT_DETECT_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "darknet.h"

image color_mat_to_image(cv::Mat &src);
cv::Mat image_to_color_cv_mat(image &src);

class ObjDetector
{
public:
    ObjDetector(char * datacfg,char *cfgfile,char * weightfile,char *namefile,float thresh,float hier_thresh);
    ~ObjDetector();

private:
    list *options_;
    char *name_list_;
    char **names_;

    image **alphabet_;
    network *net_;

    float thresh_,hier_thresh_;

public:
    int predict(cv::Mat & src, cv::Mat & dst);
};
#endif
