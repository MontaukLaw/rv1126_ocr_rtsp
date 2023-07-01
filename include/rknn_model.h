#ifndef __MY_RKMEDIA_RKNN_MODEL_H
#define __MY_RKMEDIA_RKNN_MODEL_H

#include "comm.h"

int init_yolo_model(void);

// int predict(void *bufData);
int predict(void *bufData, detect_result_group_t *detect_result_group);

void yolo_det(void *bufData);

void run_model(cv::Mat &src);

#endif // __MY_RKMEDIA_RKNN_MODEL_H