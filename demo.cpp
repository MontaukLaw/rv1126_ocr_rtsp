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

// #define FRAME_WIDTH 1920
// #define FRAME_HEIGHT 1080

#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720

// RTSP
static bool quit = false;
rtsp_demo_handle g_rtsplive = NULL;
static rtsp_session_handle g_rtsp_session;

static void *venc_rtsp_tidp(void *args);
static void *rkmedia_vi_ocr_thread(void *args);
static void *main_process(void *arg);

char mediaBuffer[FRAME_WIDTH * FRAME_HEIGHT * 3];
RK_U32 Media_Buffer_size = 0;

char isMBCopy = false;
bool ifInferFinished = false;
Mat imageRGB = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);

vector<StringBox> result;

char *det_model_path = nullptr;
char *reg_model_path = nullptr;
char *keys_path = nullptr;

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        printf("Usage: ./demo [det_model_path] [reg_model_path] [keys_path]\n");
        return -1;
    }

    // 获取参数1：det_model_path
    det_model_path = argv[1];
    // 获取参数2：reg_model_path
    reg_model_path = argv[2];
    // 获取参数3：keys_path
    keys_path = argv[3];

    // 打印一下
    printf("det_model_path: %s\n", det_model_path);
    printf("reg_model_path: %s\n", reg_model_path);
    printf("keys_path: %s\n", keys_path);

    RK_U32 video_width = FRAME_WIDTH;
    RK_U32 video_height = FRAME_HEIGHT;

    RK_CHAR *pDeviceName = "rkispp_scale0";
    RK_CHAR *pcDevNode = "/dev/dri/card0";
    // char *iq_file_dir = "/etc/iqfiles";
    char *iq_file_dir = "/oem/etc/iqfiles";
    RK_S32 s32CamId = 0;
    RK_U32 u32BufCnt = 3;
    RK_U32 fps = 20;
    int ret;
    pthread_t rkmedia_vi_ocr_tidp;
    pthread_t get_media_tidp;
    pthread_t rtsp_tidp;
    RK_BOOL bMultictx = RK_FALSE;
    CODEC_TYPE_E enCodecType = RK_CODEC_TYPE_H264;

    printf("\n###############################################\n");
    printf("VI CameraIdx: %d\npDeviceName: %s\nResolution: %dx%d\n\n",
           s32CamId, pDeviceName, video_width, video_height);
    printf("###############################################\n\n");

    if (iq_file_dir)
    {
#ifdef RKAIQ
        printf("#Rkaiq XML DirPath: %s\n", iq_file_dir);
        printf("#bMultictx: %d\n\n", bMultictx);
        rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
        SAMPLE_COMM_ISP_Init(s32CamId, hdr_mode, bMultictx, iq_file_dir);
        SAMPLE_COMM_ISP_Run(s32CamId);
        SAMPLE_COMM_ISP_SetFrameRate(s32CamId, fps);
#endif
    }

    // init rtsp
    g_rtsplive = create_rtsp_demo(554);
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/main_stream");
    if (enCodecType == RK_CODEC_TYPE_H264)
    {
        rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    }
    else if (enCodecType == RK_CODEC_TYPE_H265)
    {
        rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H265, NULL, 0);
    }
    else
    {
        printf("not support other type\n");
        return -1;
    }
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

    RK_MPI_SYS_Init();
    VI_CHN_ATTR_S vi_chn_attr;
    vi_chn_attr.pcVideoNode = pDeviceName;
    vi_chn_attr.u32BufCnt = u32BufCnt;
    vi_chn_attr.u32Width = video_width;
    vi_chn_attr.u32Height = video_height;
    vi_chn_attr.enPixFmt = IMAGE_TYPE_NV12;
    vi_chn_attr.enBufType = VI_CHN_BUF_TYPE_MMAP;
    vi_chn_attr.enWorkMode = VI_WORK_MODE_NORMAL;
    ret = RK_MPI_VI_SetChnAttr(s32CamId, 0, &vi_chn_attr);
    ret |= RK_MPI_VI_EnableChn(s32CamId, 0);
    if (ret)
    {
        printf("ERROR: create VI[0:0] error! ret=%d\n", ret);
        return -1;
    }

    // RGA负责把数据从NV12转换成BGR
    RGA_ATTR_S stRgaAttr;
    memset(&stRgaAttr, 0, sizeof(stRgaAttr));
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = u32BufCnt;
    stRgaAttr.u16Rotaion = 0;
    stRgaAttr.stImgIn.u32X = 0;
    stRgaAttr.stImgIn.u32Y = 0;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_NV12;
    stRgaAttr.stImgIn.u32Width = video_width;
    stRgaAttr.stImgIn.u32Height = video_height;
    stRgaAttr.stImgIn.u32HorStride = video_width;
    stRgaAttr.stImgIn.u32VirStride = video_height;
    stRgaAttr.stImgOut.u32X = 0;
    stRgaAttr.stImgOut.u32Y = 0;
    stRgaAttr.stImgOut.imgType = IMAGE_TYPE_BGR888;
    stRgaAttr.stImgOut.u32Width = video_width;
    stRgaAttr.stImgOut.u32Height = video_height;
    stRgaAttr.stImgOut.u32HorStride = video_width;
    stRgaAttr.stImgOut.u32VirStride = video_height;
    ret = RK_MPI_RGA_CreateChn(0, &stRgaAttr);
    if (ret)
    {
        printf("ERROR: create RGA[0:0] falied! ret=%d\n", ret);
        return -1;
    }

    VENC_CHN_ATTR_S venc_chn_attr;
    memset(&venc_chn_attr, 0, sizeof(venc_chn_attr));
    switch (enCodecType)
    {
    case RK_CODEC_TYPE_H265:
        venc_chn_attr.stVencAttr.enType = RK_CODEC_TYPE_H265;
        venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
        venc_chn_attr.stRcAttr.stH265Cbr.u32Gop = 30;
        venc_chn_attr.stRcAttr.stH265Cbr.u32BitRate = video_width * video_height;
        // frame rate: in 30/1, out 30/1.
        venc_chn_attr.stRcAttr.stH265Cbr.fr32DstFrameRateDen = 1;
        venc_chn_attr.stRcAttr.stH265Cbr.fr32DstFrameRateNum = 30;
        venc_chn_attr.stRcAttr.stH265Cbr.u32SrcFrameRateDen = 1;
        venc_chn_attr.stRcAttr.stH265Cbr.u32SrcFrameRateNum = 30;
        break;
    case RK_CODEC_TYPE_H264:
    default:
        venc_chn_attr.stVencAttr.enType = RK_CODEC_TYPE_H264;
        venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
        venc_chn_attr.stRcAttr.stH264Cbr.u32Gop = 30;
        venc_chn_attr.stRcAttr.stH264Cbr.u32BitRate = video_width * video_height * 3;
        // frame rate: in 30/1, out 30/1.
        venc_chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRateDen = 1;
        venc_chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRateNum = 30;
        venc_chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRateDen = 1;
        venc_chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRateNum = 30;
        break;
    }

    venc_chn_attr.stVencAttr.imageType = IMAGE_TYPE_RGB888;
    venc_chn_attr.stVencAttr.u32PicWidth = video_width;
    venc_chn_attr.stVencAttr.u32PicHeight = video_height;
    venc_chn_attr.stVencAttr.u32VirWidth = video_width;
    venc_chn_attr.stVencAttr.u32VirHeight = video_height;
    venc_chn_attr.stVencAttr.u32Profile = 66;
    ret = RK_MPI_VENC_CreateChn(0, &venc_chn_attr);
    if (ret)
    {
        printf("ERROR: create VENC[0:0] error! ret=%d\n", ret);
        return -1;
    }

    // 绑定VI和RGA
    MPP_CHN_S stSrcChn;
    MPP_CHN_S stDestChn;
    printf("Bind VI[0:0] to RGA[0:0]....\n");
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = s32CamId;
    stSrcChn.s32ChnId = 0;
    stDestChn.enModId = RK_ID_RGA;
    stDestChn.s32DevId = s32CamId;
    stDestChn.s32ChnId = 0;
    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret)
    {
        printf("ERROR: Bind VI[0:0] to RGA[0:0] failed! ret=%d\n", ret);
        return -1;
    }

    pthread_create(&rkmedia_vi_ocr_tidp, NULL, rkmedia_vi_ocr_thread, NULL);
    pthread_create(&get_media_tidp, NULL, main_process, NULL);
    pthread_create(&rtsp_tidp, NULL, venc_rtsp_tidp, NULL);

    printf("%s initial finish\n", __func__);

    while (!quit)
    {
        usleep(500000);
    }

    // 清理工作
    printf("%s exit!\n", __func__);
    printf("Unbind VI[0:0] to RGA[0:0]....\n");
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = s32CamId;
    stSrcChn.s32ChnId = 0;
    stDestChn.enModId = RK_ID_RGA;
    stSrcChn.s32DevId = s32CamId;
    stDestChn.s32ChnId = 0;
    ret = RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
    if (ret)
    {
        printf("ERROR: unbind VI[0:0] to RGA[0:0] failed! ret=%d\n", ret);
        return -1;
    }

    printf("Destroy VENC[0:0] channel\n");
    ret = RK_MPI_VENC_DestroyChn(0);
    if (ret)
    {
        printf("ERROR: Destroy VENC[0:0] error! ret=%d\n", ret);
        return 0;
    }

    printf("Destroy RGA[0:0] channel\n");
    ret = RK_MPI_RGA_DestroyChn(0);
    if (ret)
    {
        printf("ERROR: Destroy RGA[0:0] error! ret=%d\n", ret);
        return 0;
    }

    printf("Destroy VI[0:0] channel\n");
    ret = RK_MPI_VI_DisableChn(s32CamId, 0);
    if (ret)
    {
        printf("ERROR: destroy VI[0:0] error! ret=%d\n", ret);
        return -1;
    }

    if (iq_file_dir)
    {
#if RKAIQ
        SAMPLE_COMM_ISP_Stop(s32CamId);
#endif
    }
    return 0;
}

/*!
 * \fn     venc_rtsp_tidp
 * \brief  rtsp线程
 *
 * \param  [in] void *args   #
 *
 * \retval void *
 */
static void *venc_rtsp_tidp(void *args)
{

    pthread_detach(pthread_self());
    MEDIA_BUFFER mb = NULL;

    while (!quit)
    {
        mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_VENC, 0, -1);
        if (!mb)
        {
            printf("ERROR: RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
            break;
        }

        rtsp_tx_video(g_rtsp_session, (unsigned char *)RK_MPI_MB_GetPtr(mb), RK_MPI_MB_GetSize(mb), RK_MPI_MB_GetTimestamp(mb));
        RK_MPI_MB_ReleaseBuffer(mb);
        rtsp_do_event(g_rtsplive);
    }

    return NULL;
}

// 主线程 先拿vi数据, 复制内存
// 拿到结果之后发去编码
static void *main_process(void *arg)
{
    pthread_detach(pthread_self());

    printf("==> main_process\n");

    while (!quit)
    {
        MEDIA_BUFFER src_mb = NULL;
        src_mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, -1);
        if (!src_mb)
        {
            printf("ERROR: RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
            break;
        }
        rkMB_IMAGE_INFO ImageInfo = {0};
        RK_MPI_MB_GetImageInfo(src_mb, &ImageInfo);
        // printf("GetImageInfo: width = %d, height = %d, size = %d\n", ImageInfo.u32Width, ImageInfo.u32Height, RK_MPI_MB_GetSize(src_mb));

        if (isMBCopy == false)
        {
            // 将数据复制一份
            memcpy(&mediaBuffer[0], (uint8_t *)RK_MPI_MB_GetPtr(src_mb), RK_MPI_MB_GetSize(src_mb));
            Media_Buffer_size = RK_MPI_MB_GetSize(src_mb);
            printf("Copy Data Size is %d\n", Media_Buffer_size);
            isMBCopy = true;
        }

        Mat image = Mat(ImageInfo.u32Height, ImageInfo.u32Width, CV_8UC3, RK_MPI_MB_GetPtr(src_mb));

        // 打印一次就好
        if (ifInferFinished)
        {
            for (auto &txt : result)
            {
                cout << txt.txt << endl;
            }
            output_json(result);
            ifInferFinished = false;
        }

        for (auto &txt : result)
        {
            drawTextBox(image, txt.txtPoint, 1, txt.centerX, txt.centerY);
        }

        // result.clear();

        RK_MPI_SYS_SendMediaBuffer(RK_ID_VENC, 0, src_mb);
        RK_MPI_MB_ReleaseBuffer(src_mb);
        src_mb = NULL;
    }

    return NULL;
}

/*!
 * \fn     rkmedia_vi_ocr_thread
 * \brief  ocr线程, 用于识别
 *
 * \param  [in] void *args   #
 *
 * \retval void *
 */
static void *rkmedia_vi_ocr_thread(void *args)
{
    pthread_detach(pthread_self());
    // const char *db_model_path = "./det_new.rknn";
    // const char *crnn_model_path = "./repvgg_s.rknn";
    // const char *key_path = "./dict_text.txt";
    // const char *img_path = "./in_img.jpg";

    DBNet dbNet;
    CRNN crnn;

    printf("==> rkmedia_vi_ocr_thread\n");

    int retDbNet = dbNet.initModel(det_model_path);
    int retCrnn = crnn.loadModel_init(reg_model_path, keys_path);

    if (retDbNet < 0 || retCrnn < 0)
    {
        printf("load model fail!");
    }

    printf("==> loadModel finished!\n");
    // MB_IMAGE_INFO_S stImageInfo = {1920, 1080, 1920, 1080, IMAGE_TYPE_BGR888};
    MB_IMAGE_INFO_S stImageInfo = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT, IMAGE_TYPE_BGR888};

    while (!quit)
    {
        if (isMBCopy == true)
        {

            // ifInferFinished = false;
            MEDIA_BUFFER mb = RK_MPI_MB_CreateImageBuffer(&stImageInfo, RK_TRUE, MB_FLAG_NOCACHED);
            if (!mb)
            {
                printf("ERROR: no space left!\n");
                break;
            }
            printf("==> RK_MPI_MB_CreateImageBuffer\n");

            memcpy(RK_MPI_MB_GetPtr(mb), mediaBuffer, Media_Buffer_size);
            RK_MPI_MB_SetSize(mb, Media_Buffer_size);

#if 1
            // 直接将获取的码流, 转换为Mat, mb就是直接RGA获取的数据备份
            Mat imageOrigin = Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, RK_MPI_MB_GetPtr(mb));

            // 把BGR转为RGB
            cv::cvtColor(imageOrigin, imageRGB, CV_BGR2RGB);

            // cv::imwrite("in_img.jpg",image);
#else
            Mat image = imread(img_path);
            if (image.empty())
            {
                printf("Error: Could not load image\n");
                return NULL;
            }
#endif
            // 记录时间
            double time0 = static_cast<double>(getTickCount());

            // 获取文本框列表
            vector<ImgBox> crop_imgs;
            crop_imgs = dbNet.getTextImages(imageRGB);

            printf("Found %d text boxes\n", crop_imgs.size());

            time0 = ((double)getTickCount() - time0) / getTickFrequency();
            printf("Detect time spend: %.2f\n", time0 * 1000);
            double time1 = static_cast<double>(getTickCount());

            // 启动crnn模型, 识别文本
            result = crnn.inference(crop_imgs);

            time1 = ((double)getTickCount() - time1) / getTickFrequency();
            printf("Rec used: %.2f\n", time1 * 1000);
            printf("Detect + Rec cost: %.2f ms \n", (time0 + time1) * 1000);

            ifInferFinished = true;

            // cv::imwrite("out_img.jpg",image);
            // 释放image
            imageOrigin.release();

            isMBCopy = false;

            RK_MPI_MB_ReleaseBuffer(mb);
        }
        else
        {
            usleep(1000 * 300);
        }
    }

    return NULL;
}
