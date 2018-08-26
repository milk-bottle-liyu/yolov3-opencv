#include "ImageDetect.h"

image color_mat_to_image(cv::Mat &src)
{
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();

    image out = make_image(w, h, c);
    IplImage src_tmp(src);

    unsigned char *data = (unsigned char *)src_tmp.imageData;
    h = src_tmp.height;
    w = src_tmp.width;
    c = src_tmp.nChannels;
    int step = src_tmp.widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                out.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return out;
}

cv::Mat image_to_color_cv_mat(image &src)
{
    cv::Mat out(src.h,src.w,CV_8UC3);
    for (unsigned i=0;i<out.rows;i++)
    {
        uchar * ptr=out.ptr<uchar>(i);
        for (unsigned j=0;j<out.cols;j++)
        {
            //b
            ptr[j*3+0]=src.data[0*src.h*src.w+i*src.w+j]*255;
            //g
            ptr[j*3+1]=src.data[1*src.h*src.w+i*src.w+j]*255;
            //r
            ptr[j*3+2]=src.data[2*src.h*src.w+i*src.w+j]*255;
        }
    }
    return out;
}

ObjDetector::ObjDetector(char * datacfg,char *cfgfile,char * weightfile,char * namefile, float thresh,float hier_thresh)
{
    options_ = read_data_cfg(datacfg);
    name_list_ = option_find_str(options_, "names", namefile);
    names_ = get_labels(name_list_);

    alphabet_ = load_alphabet();
    net_ = load_network(cfgfile, weightfile, 0);
    set_batch_network(net_, 1);
    srand(2222222);
    thresh_=thresh;
    hier_thresh_=hier_thresh;
}

ObjDetector::~ObjDetector(){
    free_network(net_);
}

int ObjDetector::predict(cv::Mat & src, cv::Mat & dst)
{
    double time;
    float nms=.45;

    image im = color_mat_to_image(src);
    image sized = letterbox_image(im, net_->w, net_->h);
    layer l = net_->layers[net_->n-1];


    float *X = sized.data;
    time=what_time_is_it_now();
    network_predict(net_, X);
    int nboxes = 0;
    detection *dets = get_network_boxes(net_, im.w, im.h, thresh_, hier_thresh_, 0, 1, &nboxes);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    draw_detections(im, dets, nboxes, thresh_, names_, alphabet_, l.classes);

    dst=image_to_color_cv_mat(im);
    free_image(im);
    free_image(sized);
    free_detections(dets,nboxes);
    return 0;
}
