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
#include "curl_get_file.h"
#include "rknn_model.h"

#include "gflags/gflags.h"

DEFINE_bool(if_need_rotate, true, "if_need_rotate");
DEFINE_bool(if_save_dect_img, false, "if_save_dect_img");
DEFINE_string(det_model_path, "static_model_0625.rknn", "det_model_path");
DEFINE_string(reg_model_path, "repvgg_s.rknn", "reg_model_path");
DEFINE_string(yolo_model_path, "cell_screen_yolo.rknn", "yolo_model_path");
DEFINE_string(dict_file_path, "dict_text.txt", "dict_file_path");
DEFINE_string(stream_address, "http://192.168.0.119/1685697887.jpg", "stream_address");

using namespace std;
using namespace cv;
using namespace gflags;

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640
#define MIN_INPU_STREAM_HEIGHT 720

static Mat originFrame;
static Mat detectFrame;
static Mat frameRGB;

bool detectFinished = true;
bool dataReady = false;

pthread_t detectThread;

#if 0
char *mjpg_det_model_path = nullptr;
char *mjpg_reg_model_path = nullptr;
char *mjpg_keys_path = nullptr;
char *stream_address = nullptr;
char *cw_ratate = nullptr;
char *create_test_image = nullptr;
bool ifRotate = false;
bool ifSaveImage = false;
#endif

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

        // printf("start detection\n");
        // sleep(1);  // for test
        // 把BGR转为RGB
        cv::cvtColor(detectFrame, frameRGB, CV_BGR2RGB);

        // 然后剪裁取出640*640
        frameResized = frameRGB(Range(startY, endY), Range(startX, endX));

        if (FLAGS_if_need_rotate)
        {
            cv::rotate(frameResized, frameResized, ROTATE_90_CLOCKWISE);
            // frameResized =
        }

        // 开始检测
        // print_detectFrame_pixel_0(frameRGB);
        if (frameResized.data != nullptr)
        {
            // sleep(1);
            imwrite("input_corp.jpg", frameResized);
            // 先做 yolo
            run_model(frameResized);
            yolo_det(frameResized.data);
            // detect_process();
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

// 测试读取视频帧
int main(int argc, char **argv)
{

    ParseCommandLineFlags(&argc, &argv, true);
    // if (FLAGS_languages.find("english") != string::npos)
    //     HandleEnglish();
    // printf("det_model_path: %s \n", FLAGS_det_model_path);
    // printf("reg_model_path: %s \n", FLAGS_reg_model_path);
    // printf("yolo_model_path: %s \n", FLAGS_yolo_model_path);
    // printf("dict_file_path: %s \n", FLAGS_dict_file_path);
    // printf("if_need_rotate: %d \n", FLAGS_if_need_rotate);
    // printf("if_save_dect_img: %d \n", FLAGS_if_save_dect_img);

    std::cout << "det_model_path: " << FLAGS_det_model_path << std::endl;
    std::cout << "reg_model_path: " << FLAGS_reg_model_path << std::endl;
    std::cout << "yolo_model_path: " << FLAGS_yolo_model_path << std::endl;
    std::cout << "dict_file_path: " << FLAGS_dict_file_path << std::endl;
    std::cout << "if_need_rotate: " << FLAGS_if_need_rotate << std::endl;
    std::cout << "if_save_dect_img: " << FLAGS_if_save_dect_img << std::endl;

    int res;

#if 0
    // // 获取参数1：det_model_path
    // mjpg_det_model_path = argv[1];
    // // 获取参数2：mjpg_reg_model_path
    // mjpg_reg_model_path = argv[2];
    // // 获取参数3：mjpg_keys_path
    // mjpg_keys_path = argv[3];
    // // 获取参数4：stream_address
    // stream_address = argv[4];
    // // 看是否旋转
    // cw_ratate = argv[5];
    // // 是否需要保存测试用图片
    // create_test_image = argv[6];

    // if (cw_ratate)
    // {
    //     if (strcmp(cw_ratate, "rotate") == 0)
    //     {
    //         ifRotate = true;
    //         printf("rotate cw 90\n");
    //     }
    // }

    // if (create_test_image)
    // {
    //     if (strcmp(create_test_image, "save_test") == 0)
    //     {
    //         ifSaveImage = true;
    //         printf("save test image\n");
    //     }
    // }
#endif

    init_yolo_model();

    // 初始化模型
    // init_models();
    // stream_address = "http://192.168.0.117/capture";

    printf("==> rkmedia_vi_ocr_thread\n");
    // std::string baseURL = stream_address;
    // std::string baseURL = "http://192.168.0.117/capture?_cb=";
    // VideoCapture capture1(stream_address);
    // printf("stream add:%s \n", baseURL.c_str());
    // printf("stream add:%s \n", "https://upload-images.jianshu.io/upload_images/5809200-a99419bb94924e6d.jpg");
    // if (!capture1.isOpened())
    // {
    //     std::cout << "无法读取视频" << std::endl;
    //     return -1;
    // }
    printf("stream add:%s \n", FLAGS_stream_address);
    detectFrame = get_image_online(FLAGS_stream_address);
    if (detectFrame.empty())
    {
        printf("get image online failed\n");
        return 0;
    }

    // capture1 >> detectFrame;

    int frame_width = detectFrame.cols;  // capture1.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = detectFrame.rows; // .get(CAP_PROP_FRAME_HEIGHT);
    printf("frame_width = %d, frame_height = %d\n", frame_width, frame_height);

    // 检查图像高度是否符合最低高度要求
    if (frame_height < MODEL_HEIGHT)
    {
        printf("frame_height < MIN_INPU_STREAM_HEIGHT\n");
        return 0;
    }

    // imwrite("/tmp/test.jpg", detectFrame);

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
    res = 0;
    res = pthread_create(&detectThread, NULL, detect_process_thread, NULL);
    if (res != 0)
    {
        printf("producer thread create failed");
        return 0;
    }

    char stream_address_buf[512];
    int64_t timestamp = 0;
    sleep(1);
    int counter = 0;
    while (1)
    {
        // printf("getting new pic\n");
        // counter++;
        // std::string imageURL = baseURL + std::to_string(counter)+".jpg";
        // printf("%s\n", imageURL.c_str());
        // VideoCapture cap("https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/d4grhzment0lwrtpoc8o");
        // VideoCapture cap("http://192.168.0.122/capture");
        // memset(stream_address_buf, 0, sizeof(stream_address_buf));
        // timestamp++;
        // http://192.168.0.117/capture?_cb=1684328152623
        // sprintf(stream_address_buf, "%s?_cb=%ld", stream_address,timestamp);
        // sprintf(stream_address_buf, "%ld", stream_address_buf, timestamp);
        // printf("stream_address_buf: %s\n", stream_address_buf);
        // capture = new VideoCapture(stream_address_buf);

        // Capture frame-by-frame
        originFrame = get_image_online(FLAGS_stream_address);
        // originFrame = imread("1687224690.jpg");
        // cap >> originFrame;
        if (originFrame.empty())
        {
            // printf("no frame\n");
            usleep(1000);
            continue;
        }
        // printf("get 1 frame\n");
        // imwrite("/tmp/test.jpg", originFrame);

        if (detectFinished)
        {
            memcpy(&detectFrame.data[0], &originFrame.data[0], frame_width * frame_height * 3);
            detectFinished = false;
            dataReady = true;
        }
        // sleep(1);
        usleep(10000);
        // cap.release();
    }

    // capture1.release();

    return 0;
}
