## build

modify `GCC_COMPILER` on `build.sh` for target platform, then execute

```
./build.sh
```

## install

connect device and push build output into `/`

```
adb push install/rknn_ocr /
```

## run

```
adb shell
cd /rknn_ocr/
./rknn_ocr model/det_new.rknn model/repvgg_s.rknn model/dict_text.txt model/1.png
```
