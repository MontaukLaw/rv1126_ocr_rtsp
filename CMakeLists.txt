cmake_minimum_required(VERSION 3.8)
message(STATUS "cmake version ${CMAKE_VERSION}")
project(rknn_ocr_rtsp)

set(TOOLCHAIN_DIR /opt/atk-dlrv1126-toolchain)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)

set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_features2d opencv_flann opencv_highgui opencv_freetype)
set(RKNN_LIBS easymedia rga rkaiq rknn_api sample_common_isp)
link_directories(include lib)
link_directories(include ./librtsp)
message(STATUS "OPENCV_LIBS ${OPENCV_LIBS}")
	
include_directories(${SYSROOT})
include_directories(${SYSROOT}/rga)
include_directories(${SYSROOT}/easymedia)
include_directories(${SYSROOT}/rkaiq/uAPI)
include_directories(${SYSROOT}/rkaiq/xcore)
include_directories(${SYSROOT}/rkaiq/algos)
include_directories(${SYSROOT}/rkaiq/common)
include_directories(${SYSROOT}/rkaiq/iq_parser)
include_directories(${SYSROOT}/rknn)
include_directories(.)
include_directories(./include)
include_directories(./librknn_api/3rdparty)
include_directories(./librknn_api/include)
include_directories(./librtsp)

add_definitions(-DRKAIQ)

# opencv
set(AMRHF_SYS_LIB ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/lib)
set(OPENCV_PATH ${AMRHF_SYS_LIB})
set(OPENCV_INCLUDE ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/include)
include_directories(${OPENCV_INCLUDE})
# set(OPENCV_LIB_PATH ${OPENCV_PATH}/lib)
link_directories(${AMRHF_SYS_LIB})
set(OPENCV_LIBS ${OPENCV_PATH}/libopencv_calib3d.so
                ${OPENCV_PATH}/libopencv_core.so
                ${OPENCV_PATH}/libopencv_dnn.so
                ${OPENCV_PATH}/libopencv_features2d.so 
                ${OPENCV_PATH}/libopencv_flann.so
                ${OPENCV_PATH}/libopencv_freetype.so
                ${OPENCV_PATH}/libopencv_highgui.so
                ${OPENCV_PATH}/libopencv_imgcodecs.so
                ${OPENCV_PATH}/libopencv_imgproc.so
                ${OPENCV_PATH}/libopencv_ml.so
                ${OPENCV_PATH}/libopencv_objdetect.so
                ${OPENCV_PATH}/libopencv_photo.so
                ${OPENCV_PATH}/libopencv_shape.so
                ${OPENCV_PATH}/libopencv_stitching.so
                ${OPENCV_PATH}/libopencv_superres.so
                ${OPENCV_PATH}/libopencv_videoio.so
                ${OPENCV_PATH}/libopencv_video.so
                ${OPENCV_PATH}/libopencv_videostab.so )
message(STATUS "OPENCV_LIBS = ${OPENCV_LIBS}")


# ffmpeg arm的库
set(FFMpeg_INCLUDE_DIRS /home/marc/out_ffmpeg_arm/include)
link_directories(/home/marc/out_ffmpeg_arm/lib) 

set(FFMpeg_LIBS avcodec avformat avutil swresample swscale)

add_executable(rknn_ocr_rtsp demo.cpp src/output_json.cpp
              src/clipper.cpp src/Crnn.cpp 
              src/DbNet.cpp src/OcrUtils.cpp)

target_link_libraries(rknn_ocr_rtsp ${OPENCV_LIBS} ${RKNN_LIBS} pthread dl librtsp.a json-c)

# ******************** mjpeg **************************
add_executable(mjpg_ocr mjpg_ocr.cpp src/output_json.cpp
              src/clipper.cpp src/Crnn.cpp 
              src/DbNet.cpp src/OcrUtils.cpp
              src/detect_service.cpp src/models.cpp
              src/curl_get_file.cpp
              )

include_directories(mjpg_ocr ${FFMpeg_INCLUDE_DIRS} ${OPENCV_INCLUDE})

target_link_libraries(mjpg_ocr ${OPENCV_LIBS} ${RKNN_LIBS} 
                       pthread dl  librtsp.a json-c
                       ${FFMpeg_LIBS} z curl)
