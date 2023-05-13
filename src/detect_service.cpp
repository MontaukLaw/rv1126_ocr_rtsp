#include <iostream>
#include "DbNet.h"
#include "Crnn.h"

#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "im2d.h"
#include "rga.h"
#include "rkmedia_api.h"
#include "rkmedia_venc.h"
#include "sample_common.h"
#include "opencv2/opencv.hpp"
#include "librtsp/rtsp_demo.h"
#include "output_json.h"

using namespace cv;
using namespace std;

extern DBNet dbNet;
extern CRNN crnn;
extern Mat frameResized;

vector<StringBox> result;

void print_detectFrame_pixel_0(Mat frame)
{
    int x = frame.at<Vec3b>(0, 0)[0];               // getting the pixel values //
    int y = frame.at<Vec3b>(0, 0)[1];               // getting the pixel values //
    int z = frame.at<Vec3b>(0, 0)[2];               // getting the pixel values //
    cout << "Value of blue channel:" << x << endl;  // showing the pixel values //
    cout << "Value of green channel:" << y << endl; // showing the pixel values //
    cout << "Value of red channel:" << z << endl;   // showing the pixel values //
}

static void write_jpg(vector<ImgBox> crop_imgs)
{
    Mat outputFrame = Mat(frameResized.rows, frameResized.cols, CV_8UC3, Scalar(0, 0, 0));
    // 先转成BGR
    cvtColor(frameResized, outputFrame, COLOR_RGB2BGR);

    // 在frame中画出中心点
    for (int i = 0; i < crop_imgs.size(); i++)
    {
        ImgBox box = crop_imgs[i];
        Point center = Point(box.centerX, box.centerY);
        circle(outputFrame, center, 2, Scalar(0, 0, 255), 2);
    }

    // 写入jpeg
    imwrite("detect.jpg", outputFrame);
}
void detect_process()
{
    // 记录时间
    double time0 = static_cast<double>(getTickCount());

    print_detectFrame_pixel_0(frameResized);

    // 获取文本框列表
    vector<ImgBox> crop_imgs;
    crop_imgs = dbNet.getTextImages(frameResized);

    printf("Found %d text boxes\n", crop_imgs.size());
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    printf("Detect spend: %.2fms\n", time0 * 1000);

    // 为了方便观察，将检测结果写入图片
    write_jpg(crop_imgs);

    double time1 = static_cast<double>(getTickCount());
    // 启动crnn模型, 识别文本
    result = crnn.inference(crop_imgs);

    time1 = ((double)getTickCount() - time1) / getTickFrequency();
    printf("Rec used: %.2f\n", time1 * 1000);
    printf("Detect + Rec cost: %.2f ms \n", (time0 + time1) * 1000);

    output_json(result);
}