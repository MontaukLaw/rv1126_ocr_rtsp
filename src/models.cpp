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

extern char *mjpg_det_model_path;
extern char *mjpg_reg_model_path;
extern char *mjpg_keys_path;

void init_models()
{

    printf("==> rkmedia_vi_ocr_thread\n");

    int retDbNet = dbNet.initModel(mjpg_det_model_path);
    int retCrnn = crnn.loadModel_init(mjpg_reg_model_path, mjpg_keys_path);

    if (retDbNet < 0 || retCrnn < 0)
    {
        printf("load model fail!");
    }
    printf("models loaded!\n");
}