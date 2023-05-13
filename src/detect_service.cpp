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

void detect_process(Mat imgRGB)
{

    // 记录时间
    double time0 = static_cast<double>(getTickCount());

    // 获取文本框列表
    vector<ImgBox> crop_imgs;
    crop_imgs = dbNet.getTextImages(imgRGB);

    printf("Found %d text boxes\n", crop_imgs.size());

    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    printf("Detect time spend: %.2f\n", time0 * 1000);

}