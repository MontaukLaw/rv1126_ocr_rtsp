//
// Created by cai on 2021/10/14.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "clipper.h"
#include "OcrUtils.h"
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat crnn_narrow_32_pad(cv::Mat &img, int W) {                                          // resize到指定尺度
    float scale = 32.0 / img.rows;
    int nw = int(img.cols * scale);
    int nh = int(img.rows * scale);
    resize(img, img, cv::Size(nw, nh),0,0, INTER_CUBIC);

    int top = 0;
    int bottom = 32 - nh;
    int left = 0;
    int right = W - nw;

    Mat resize_img;

    if (bottom < 0 || right < 0){
        resize(img, resize_img, cv::Size(W, 32));
    } else{
        copyMakeBorder(img, resize_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    }

    return resize_img;
}

std::vector<SizeImg> db_narrow_pad(cv::Mat& img) {                  // 图片预处理
    std::vector<SizeImg> img_scale;
    cv::Mat resize_img;

    float scale = 640.0 / std::max(img.cols,img.rows);
    int nw = int(img.cols * scale);
    int nh = int(img.rows * scale);

    resize(img,resize_img,cv::Size(nw,nh),cv::INTER_CUBIC);

    int top = 0;
    int bottom = 640 - resize_img.rows;
    int left = 0;
    int right = 640 - resize_img.cols ;

    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    img_scale.push_back(SizeImg{resize_img,scale});

    return img_scale;
}

std::vector<ImgBox> getImage(std::vector<TextBox> box ,cv::Mat &src) {

    std::vector<ImgBox> crop_img;
    crop_img.clear();

    Mat src_copy = src.clone();

    for (int i = 0; i < box.size(); i++) {
//        int x[4]={box[i].boxPoint[0].x, box[i].boxPoint[1].x,box[i].boxPoint[2].x, box[i].boxPoint[3].x};
//        int y[4]={box[i].boxPoint[0].y, box[i].boxPoint[1].y,box[i].boxPoint[2].y, box[i].boxPoint[3].y};

//        int min_x = (*std::min_element(x,x+4));
//        int min_y = (*std::min_element(y,y+4));
//        int max_x = (*std::max_element(x,x+4));
//        int max_y = (*std::max_element(y,y+4));

        int img_w = int(sqrt(pow(box[i].boxPoint[0].x - box[i].boxPoint[1].x, 2) +
                             pow(box[i].boxPoint[0].y - box[i].boxPoint[1].y, 2)));
        int img_h = int(sqrt(pow(box[i].boxPoint[0].x - box[i].boxPoint[3].x, 2) +
                             pow(box[i].boxPoint[0].y - box[i].boxPoint[3].y, 2)));

        cv::Point2f src_points[] ={
                cv::Point2f(box[i].boxPoint[0].x,box[i].boxPoint[0].y),
                cv::Point2f(box[i].boxPoint[1].x,box[i].boxPoint[1].y),
                cv::Point2f(box[i].boxPoint[2].x,box[i].boxPoint[2].y),
                cv::Point2f(box[i].boxPoint[3].x,box[i].boxPoint[3].y),
        };

        cv::Point2f dst_points[] ={
                cv::Point2f(0,0),
                cv::Point2f(img_w,0),
                cv::Point2f(img_w,img_h),
                cv::Point2f(0,img_h),
        };

        cv::Mat rot_img,img_warp;
        rot_img = cv::getPerspectiveTransform(src_points,dst_points);
        cv::warpPerspective(src_copy,img_warp,rot_img,cv::Size(img_w,img_h),cv::INTER_LINEAR);

//        cv::Mat img = src(cv::Rect(min_x, min_y, img_w, img_h));
        crop_img.emplace_back(ImgBox{img_warp,box[i].boxPoint});

//        line(src,Point(box[i].boxPoint[0].x,box[i].boxPoint[0].y),Point(box[i].boxPoint[1].x,box[i].boxPoint[1].y),Scalar(0,0,255),1,CV_AA);
//        line(src,Point(box[i].boxPoint[1].x,box[i].boxPoint[1].y),Point(box[i].boxPoint[2].x,box[i].boxPoint[2].y),Scalar(0,0,255),1,CV_AA);
//        line(src,Point(box[i].boxPoint[2].x,box[i].boxPoint[2].y),Point(box[i].boxPoint[3].x,box[i].boxPoint[3].y),Scalar(0,0,255),1,CV_AA);
//        line(src,Point(box[i].boxPoint[3].x,box[i].boxPoint[3].y),Point(box[i].boxPoint[0].x,box[i].boxPoint[0].y),Scalar(0,0,255),1,CV_AA);

    }
    return crop_img;
}


std::vector<TextBox> findRsBoxes(const cv::Mat &src ,const cv::Mat &fMapMat, const cv::Mat &norfMapMat,
                                 const float boxScoreThresh, const float unClipRatio,float s) {
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);

//        ---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / s);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), src.cols);

            clipMinBox[j].y = (clipMinBox[j].y / s);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), src.rows);
        }

        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

bool cvPointCompare(const cv::Point &a, const cv::Point &b) {
    return a.x < b.x;
}

std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize) {
    std::vector<cv::Point> minBoxVec;
    cv::RotatedRect textRect = cv::minAreaRect(inVec);
    cv::Mat boxPoints2f;
    cv::boxPoints(textRect, boxPoints2f);

    float *p1 = (float *) boxPoints2f.data;
    std::vector<cv::Point> tmpVec;
    for (int i = 0; i < 4; ++i, p1 += 2) {
        tmpVec.emplace_back(int(p1[0]), int(p1[1]));
    }

    std::sort(tmpVec.begin(), tmpVec.end(), cvPointCompare);

    minBoxVec.clear();

    int index1, index2, index3, index4;
    if (tmpVec[1].y > tmpVec[0].y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }

    if (tmpVec[3].y > tmpVec[2].y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    minBoxVec.clear();

    minBoxVec.push_back(tmpVec[index1]);
    minBoxVec.push_back(tmpVec[index2]);
    minBoxVec.push_back(tmpVec[index3]);
    minBoxVec.push_back(tmpVec[index4]);

    minSideLen = (std::min)(textRect.size.width, textRect.size.height);
    allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

    return minBoxVec;
}

float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox) {
    std::vector<cv::Point> box = inBox;
    int width = inMat.cols;
    int height = inMat.rows;
    int maxX = -1, minX = 1000000, maxY = -1, minY = 1000000;
    for (int i = 0; i < box.size(); ++i) {
        if (maxX < box[i].x)
            maxX = box[i].x;
        if (minX > box[i].x)
            minX = box[i].x;
        if (maxY < box[i].y)
            maxY = box[i].y;
        if (minY > box[i].y)
            minY = box[i].y;
    }
    maxX = (std::min)((std::max)(maxX, 0), width - 1);
    minX = (std::max)((std::min)(minX, width - 1), 0);
    maxY = (std::min)((std::max)(maxY, 0), height - 1);
    minY = (std::max)((std::min)(minY, height - 1), 0);

    for (int i = 0; i < box.size(); ++i) {
        box[i].x = box[i].x - minX;
        box[i].y = box[i].y - minY;
    }

    std::vector<std::vector<cv::Point>> maskBox;
    maskBox.push_back(box);
    cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);

    return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                    maskMat).val[0];
}

// use clipper
std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;
    ClipperLib::Path poly;

    for (int i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    double distance = unClipRatio * ClipperLib::Area(poly) / (double) perimeter;

    ClipperLib::ClipperOffset clipperOffset;
    clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipperOffset.Execute(polys, distance);

    outBox.clear();
    std::vector<cv::Point> rsVec;
    for (int i = 0; i < polys.size(); ++i) {
        ClipperLib::Path tmpPoly = polys[i];
        for (int j = 0; j < tmpPoly.size(); ++j) {
            outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
        }
    }
    return outBox;
}

void drawTextBox(cv::Mat &boxImg, const std::vector<cv::Point> &box, int thickness) {
    auto color = cv::Scalar(255, 255, 0);// R(255) G(0) B(0)
    cv::line(boxImg, box[0], box[1], color, thickness);
    cv::line(boxImg, box[1], box[2], color, thickness);
    cv::line(boxImg, box[2], box[3], color, thickness);
    cv::line(boxImg, box[3], box[0], color, thickness);
}

void printGpuInfo() {
#ifdef __VULKAN__
    auto gpuCount = ncnn::get_gpu_count();
    if (gpuCount != 0) {
        printf("This device has %d GPUs\n", gpuCount);
    } else {
        printf("This device does not have a GPU\n");
    }
#endif
}


