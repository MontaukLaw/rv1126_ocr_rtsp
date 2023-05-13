#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
using namespace std;
using namespace cv;

VideoCapture capture("http://192.168.4.1:81/stream");
Mat originFrame;
Mat detectFrame;
bool detecting = false;
bool startDetecting = false;
pthread_t detectThread;

void print_detectFrame_pixel_0()
{
    int x = detectFrame.at<Vec3b>(0, 0)[0];         // getting the pixel values //
    int y = detectFrame.at<Vec3b>(0, 0)[1];         // getting the pixel values //
    int z = detectFrame.at<Vec3b>(0, 0)[2];         // getting the pixel values //
    cout << "Value of blue channel:" << x << endl;  // showing the pixel values //
    cout << "Value of green channel:" << y << endl; // showing the pixel values //
    cout << "Value of red channel:" << z << endl;   // showing the pixel values //
}

void *detect_process(void *arg)
{
    while (1)
    {
        while (!startDetecting)
        {
            usleep(1000);
        }
        detecting = true;
        // 开始检测


        // 检测结束
        detecting = false;
        startDetecting = false;
    }
}

int main()
{
    int res;

    // 判断视频是否读取成功，返回true表示成功
    if (!capture.isOpened())
    {
        std::cout << "无法读取视频" << std::endl;
    }

    capture >> detectFrame;

    // 创建线程
    res = pthread_create(&detectThread, NULL, detect_process, NULL);
    if (res != 0)
    {
        printf("producer thread create failed");
        return 0;
    }

    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    printf("frame_width = %d, frame_height = %d\n", frame_width, frame_height);

    while (1)
    {
        // Capture frame-by-frame
        capture >> originFrame;

        // If the frame is empty, break immediately
        if (originFrame.empty())
            break;

        if (!detecting)
        {
            startDetecting = true;
            memcpy(&detectFrame.data[0], &originFrame.data[0], frame_width * frame_height * 3);
            // memcpy(&detectFrame.data, &originFrame.data, frame_width * frame_height * 3);
        }
        static int i = 0;
        // for debug
        if (i == 0)
        {
            // imwrite("test.jpg", originFrame);
        }
        // imwrite("test.jpg", originFrame);
        i++;
        usleep(1000);
    }

    capture.release();

    return 0;
}