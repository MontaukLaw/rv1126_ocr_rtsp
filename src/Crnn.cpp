#include "Crnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

CRNN::~CRNN()
{
    clear();
}

int CRNN::loadModel_init(const char *filename, const char *keys_path)
{

    std::ifstream key(keys_path);
    std::string line;
    if (key)
    {
        while (getline(key, line))
        {
            keys.push_back(line);
        }
    }

    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size_t model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        model = nullptr;
        return -2;
    }
    if (fp)
    {
        fclose(fp);
    }
    int ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
        free(model);
        model = nullptr;
        printf("rknn_init fail! ret=%d\n", ret);
        return -3;
    }
    free(model);
    model = nullptr;

    // Get Model Input Output Info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        input_width = input_attrs[0].dims[0];
        input_height = input_attrs[0].dims[1];
    }
    else
    {
        input_width = input_attrs[0].dims[1];
        input_height = input_attrs[0].dims[2];
    }

    printf("model input height=%d ,width=%d\n", input_height, input_width);

    return 0;
}

void CRNN::clear()
{
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
}

// CRNN 文字识别推理过程
std::vector<StringBox> CRNN::inference(std::vector<ImgBox> crop_img)
{
    std::vector<StringBox> result;
    int dstWidth = input_width;

    int counter = 0;

    for (auto &img : crop_img)
    {
        double time0 = static_cast<double>(getTickCount());
        cv::Mat orig_img = img.img.clone();
        // Set Input Data

        cv::Mat bgr = crnn_narrow_32_pad(img.img, dstWidth);
        ;
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = bgr.cols * bgr.rows * bgr.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = bgr.data;

        int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
        }

        ret = rknn_run(ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
        }

        // Get Output
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;

        ret = rknn_outputs_get(ctx, 1, outputs, NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
        }

        float *buffer = (float *)outputs[0].buf;

        int keySize = keys.size();
        vector<float> outputData(buffer, buffer + (input_width / 4) * (keySize + 1));

        int lastIndex = 0;
        int maxIndex;
        std::string strRes;

        int num = ceil(int(orig_img.cols * (32.0 / orig_img.rows)) / 4);
        num = std::min(num, (bgr.cols / 4));

        for (int n = 0; n < num; n++)
        {
            maxIndex = int(argmax(&outputData[n * (keySize + 1)], &outputData[(n + 1) * (keySize + 1)]));
            if (maxIndex > 0 && maxIndex < keySize && (!(n > 0 && maxIndex == lastIndex)))
            {
                strRes.append(keys[maxIndex - 1]);
            }
            lastIndex = maxIndex;
        }

        result.emplace_back(StringBox{strRes, img.imgPoint, img.centerX, img.centerY});
        time0 = ((double)getTickCount() - time0) / getTickFrequency();
        // 如果需要统计每次文字识别的时间，可以打开下面的注释
        // printf("The single identification time is: %lf\n", time0);
        counter++;

        rknn_outputs_release(ctx, 1, outputs);
    }

    return result;
}