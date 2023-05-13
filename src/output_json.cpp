#include <json-c/json.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "OcrUtils.h"

void output_json(std::vector<StringBox> boxes)
{
    json_object *root = json_object_new_object();
    int nbr = boxes.size();
    int i = 0;
    json_object_object_add(root, "nbr", json_object_new_int(nbr));

    // 添加结果数组
    json_object *boxArray = json_object_new_array();

    for (i = 0; i < nbr; i++)
    {
        json_object *box = json_object_new_object();
        json_object_object_add(box, "txt", json_object_new_string(boxes[i].txt.c_str()));
        json_object_object_add(box, "centerX", json_object_new_int(boxes[i].centerX));
        json_object_object_add(box, "centerY", json_object_new_int(boxes[i].centerY));

        json_object_array_add(boxArray, box);
    }

    json_object_object_add(root, "boxes", boxArray);

    printf("RESULT:\n%s\n", json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY));

    // 释放资源
    json_object_put(root);
}
