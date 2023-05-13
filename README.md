## build
cmake -S . -B build
cmake --build build

## install
## run

adb push 到板子上
在板子上运行
./rknn_ocr model/det_new.rknn model/repvgg_s.rknn model/dict_text.txt

模型文件, 字典文件路径都根据实际调整一下.

```
