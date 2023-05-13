#include <stdio.h>
#include <semaphore.h>
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>
#include "DbNet.h"
#include "Crnn.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"

#include "im2d.h"
#include "rga.h"
#include "rkmedia_api.h"
#include "rkmedia_venc.h"
#include "sample_common.h"

#include "librtsp/rtsp_demo.h"
#include "output_json.h"

#include "models.h"
#include "detect_service.h"

using namespace std;
using namespace cv;
#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

static Mat originFrame;
static Mat detectFrame;
static Mat frameRGB;

bool detectFinished = true;
bool dataReady = false;

pthread_t detectThread;

char *mjpg_det_model_path = nullptr;
char *mjpg_reg_model_path = nullptr;
char *mjpg_keys_path = nullptr;
Mat frameResized;
int startX = 0;
int startY = 0;
int endX = 0;
int endY = 0;

void *detect_process_thread(void *arg)
{
    pthread_detach(pthread_self());

    while (1)
    {
        while (!dataReady)
        {
            usleep(1000);
        }

        printf("start detection\n");
        // sleep(1);  // for test
        // 把BGR转为RGB
        cv::cvtColor(detectFrame, frameRGB, CV_BGR2RGB);

        frameResized = frameRGB(Range(startY, endY), Range(startX, endX));

        // 开始检测
        // print_detectFrame_pixel_0(frameRGB);
        if (frameResized.data != nullptr)
        {
            // sleep(1);
            detect_process();
        }

        // imwrite("test.jpg", frameRGB);
        // sleep(1);  // for test
        // 检测结束
        printf("end detection\n");
        dataReady = false;
        detectFinished = true;
    }

    return nullptr;
}

int main__(int argc, char **argv)
{
    Mat img = imread("test.jpg");
    cout << "Width : " << img.size().width << endl;
    cout << "Height: " << img.size().height << endl;
    cout << "Channels: :" << img.channels() << endl;

    // Crop image
    int startX = (img.size().width - MODEL_WIDTH) / 2;
    int startY = (img.size().height - MODEL_HEIGHT) / 2;

    int endX = startX + MODEL_WIDTH;
    int endY = startY + MODEL_HEIGHT;

    Mat cropped_image = img(Range(startY, endY), Range(startX, endX));

    // display image
    // imshow("Original Image", img);
    // imshow("Cropped Image", cropped_image);

    // Save the cropped Image
    imwrite("cropped_Image.jpg", cropped_image);

    // 0 means loop infinitely
    // waitKey(0);
    // destroyAllWindows();
    return 0;
}

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        printf("Usage: ./demo [mjpg_det_model_path] [mjpg_reg_model_path] [mjpg_keys_path]\n");
        return -1;
    }
    // 获取参数1：det_model_path
    mjpg_det_model_path = argv[1];
    // 获取参数2：mjpg_reg_model_path
    mjpg_reg_model_path = argv[2];
    // 获取参数3：mjpg_keys_path
    mjpg_keys_path = argv[3];

    // 打印一下
    printf("mjpg_det_model_path: %s\n", mjpg_det_model_path);
    printf("mjpg_reg_model_path: %s\n", mjpg_reg_model_path);
    printf("mjpg_keys_path: %s\n", mjpg_keys_path);

    // 初始化模型
    init_models();
    printf("==> rkmedia_vi_ocr_thread\n");

    // DBNet dbNet;
    // CRNN crnn;

    // int retDbNet = dbNet.initModel(mjpg_det_model_path);
    // int retCrnn = crnn.loadModel_init(mjpg_reg_model_path, mjpg_keys_path);

    // if (retDbNet < 0 || retCrnn < 0)
    // {
    //     printf("load model fail!");
    // }
    // printf("models loaded!\n");

    int res;
    VideoCapture capture("http://192.168.4.1:81/stream");
    // 判断视频是否读取成功，返回true表示成功
    if (!capture.isOpened())
    {
        std::cout << "无法读取视频" << std::endl;
    }

    capture >> detectFrame;

    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    printf("frame_width = %d, frame_height = %d\n", frame_width, frame_height);

    detectFrame = Mat(frame_height, frame_width, CV_8UC3);
    originFrame = Mat(frame_height, frame_width, CV_8UC3);
    frameRGB = Mat(frame_height, frame_width, CV_8UC3);
    frameResized = Mat(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);

    // Crop image
    startX = (frameRGB.size().width - MODEL_WIDTH) / 2;
    startY = (frameRGB.size().height - MODEL_HEIGHT) / 2;

    endX = startX + MODEL_WIDTH;
    endY = startY + MODEL_HEIGHT;

    // 创建线程
    res = pthread_create(&detectThread, NULL, detect_process_thread, NULL);
    if (res != 0)
    {
        printf("producer thread create failed");
        return 0;
    }

    while (1)
    {
        // Capture frame-by-frame
        capture >> originFrame;
        // if (!capture.read(originFrame))
        // {
        //     std::cout << "无法读取帧！" << std::endl;
        //     break;
        // }
        // If the frame is empty, break immediately
        if (originFrame.empty())
        {
            // printf("no frame\n");
            usleep(1000);
            continue;
        }

        if (detectFinished)
        {
            memcpy(&detectFrame.data[0], &originFrame.data[0], frame_width * frame_height * 3);
            detectFinished = false;
            dataReady = true;
        }

        usleep(1000);
    }

    capture.release();

    return 0;
}