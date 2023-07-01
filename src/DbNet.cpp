#include "DbNet.h"
#include "iostream"

using namespace cv;
using namespace std;

DBNet::~DBNet()
{
    clear();
}

void DBNet::clear()
{
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
}

int DBNet::initModel(const char *filename)
{
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

std::vector<ImgBox> DBNet::getTextImages(cv::Mat &src)
{
    std::vector<SizeImg> img_scale = db_narrow_pad(src);

    cv::Mat img = img_scale[0].img;
    float scale = img_scale[0].scale;

    // Set Input Data  
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    // inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = img.cols * img.rows * img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs); 
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        // return -1;
    }

    ret = rknn_run(ctx, nullptr); 
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        //        return -1;
    }

    // Get Output    
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(ctx, 1, outputs, NULL); 
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        // return -1;
    }

    cv::Mat fMapMat(img.rows, img.cols, CV_32FC1);
    memcpy(fMapMat.data, (float *)outputs[0].buf, img.rows * img.cols * sizeof(float));

    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::dilate(fMapMat, fMapMat, element);

    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    std::vector<TextBox> box;
    box = findRsBoxes(src, fMapMat, norfMapMat, boxScoreThresh, unClipRatio, scale);

    std::vector<ImgBox> crop_img;
    crop_img = getImage(box, src);
    return crop_img;
}
