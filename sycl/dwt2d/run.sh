SYCL_DEVICE_FILTER=cuda SYCL_PI_TRACE=1  ./dwt2d 192.bmp -d 192x192 -f -5 -l 3
ls
SYCL_DEVICE_FILTER=cuda SYCL_PI_TRACE=1  ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3
