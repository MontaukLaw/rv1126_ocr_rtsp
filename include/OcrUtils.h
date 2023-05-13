//
// Created by cai on 2021/10/14.
//

#ifndef CRNN_NCNN_OCRSTRUCT_H
#define CRNN_NCNN_OCRSTRUCT_H

#include <opencv2/core.hpp>

#include <sys/stat.h>
/*#define __ENABLE_CONSOLE__ true
#define Logger(format, ...) {\
  if(__ENABLE_CONSOLE__) printf(format,##__VA_ARGS__); \
}*/

double getCurrentTime();

inline bool isFileExists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

struct TextBox
{
    std::vector<cv::Point> boxPoint;
    float score;
};

struct ImgBox
{
    cv::Mat img;
    std::vector<cv::Point> imgPoint;
    int centerX;
    int centerY;
};

struct StringBox
{
    std::string txt;
    std::vector<cv::Point> txtPoint;
    int centerX;
    int centerY;
};

struct SizeImg
{
    cv::Mat img;
    float scale;
};

std::vector<TextBox> findRsBoxes(const cv::Mat &src, const cv::Mat &fMapMat, const cv::Mat &norfMapMat,
                                 const float boxScoreThresh, const float unClipRatio, float scale);

std::vector<ImgBox> getImage(std::vector<TextBox> box, cv::Mat &src);

std::vector<SizeImg> db_narrow_pad(cv::Mat &img);

cv::Mat crnn_narrow_32_pad(cv::Mat &img, int W);

std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize);

float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox);

std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio);

void drawTextBox(cv::Mat &boxImg, const std::vector<cv::Point> &box, int thickness, int x, int y);

#endif // CRNN_NCNN_OCRSTRUCT_H
