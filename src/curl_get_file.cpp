#include "curl_get_file.h"

using namespace std;
using namespace cv;

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, std::vector<uchar> *buffer)
{
    size_t total_size = size * nmemb;
    buffer->insert(buffer->end(), (uchar *)contents, (uchar *)contents + total_size);
    return total_size;
}

Mat get_image_online(string url)
{
    Mat img;
    // 初始化cURL库
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // 创建cURL会话句柄
    CURL *curl_handle = curl_easy_init();

    // 设置目标URL
    // const char *url_const = "http://192.168.0.122/capture";

    // 坑!
    char *url_char = (char *)malloc(url.length() + 1);
    strcpy(url_char, url.c_str());
    const char *url_const = url_char;

    // 设置cURL选项
    // curl_easy_setopt(curl_handle, CURLOPT_URL, url_const);
    curl_easy_setopt(curl_handle, CURLOPT_URL, url_const);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 5L);

    // 创建内存缓冲区
    std::vector<uchar> jpeg_data;

    // 设置cURL选项，将接收到的数据保存到内存缓冲区中
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &jpeg_data);

    // 执行HTTP请求
    CURLcode res = curl_easy_perform(curl_handle);

    // 检查请求是否成功
    if (res != CURLE_OK)
    {
        // 处理错误情况
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        // return img;
    }
    else
    {
        // 解码JPEG数据为Mat对象
        img = imdecode(jpeg_data, IMREAD_COLOR);

        // 对解码后的图像进行处理，例如显示图像
        // imshow("Decoded Image", img);
        // imwrite("test0520.jpg", img);
        // waitKey(0);
    }

    // 释放内存缓冲区
    jpeg_data.clear();

    // 释放url_char
    free(url_char);

    // 清理cURL会话句柄
    curl_easy_cleanup(curl_handle);

    // 清理cURL库
    curl_global_cleanup();

    return img;
}

static int uint_test(void)
{
    string url = "http://192.168.0.122/capture"; // 下载文件的URL

    Mat img = get_image_online(url);
    if (img.empty())
    {
        printf("img is empty\n");
        return 0;
    }

    // 保存img
    imwrite("test0520.jpg", img);
}
