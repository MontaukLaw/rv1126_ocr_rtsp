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

DBNet dbNet;
CRNN crnn;

extern char *det_model_path = nullptr;
extern char *reg_model_path = nullptr;
extern char *keys_path = nullptr;

void init_models()
{

    printf("==> rkmedia_vi_ocr_thread\n");

    int retDbNet = dbNet.initModel(det_model_path);
    int retCrnn = crnn.loadModel_init(reg_model_path, keys_path);

    if (retDbNet < 0 || retCrnn < 0)
    {
        printf("load model fail!");
    }
    printf("models loaded!\n");
}