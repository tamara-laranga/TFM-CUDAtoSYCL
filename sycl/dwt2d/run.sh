ONEAPI_DEVICE_SELECTOR=cuda:* SYCL_PI_TRACE=1  ./dwt2d 192.bmp -d 192x192 -f -5 -l 3
ls
ONEAPI_DEVICE_SELECTOR=cuda:* SYCL_PI_TRACE=1  ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3
