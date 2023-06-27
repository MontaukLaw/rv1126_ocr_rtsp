#ifndef __CURL_GET_FILE_H__
#define __CURL_GET_FILE_H__
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <curl/curl.h>

cv:: Mat get_image_online(std::string url);

#endif

