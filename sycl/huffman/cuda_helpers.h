#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void) try {
        int count = 0;
	int i = 0;

        count = dpct::dev_mgr::instance().device_count();
        if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
                dpct::device_info prop;
                if (DPCT_CHECK_ERROR(dpct::get_device_info(
                        prop, dpct::dev_mgr::instance().get_device(i))) == 0) {
                        if (prop.get_major_version() >= 1) {
                                break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
        /*
	DPCT1093:22: The "i" device may be not the one intended for use.
         * Adjust the selected device if needed.
	*/
        dpct::select_device(i);

        printf("CUDA initialized.\n");
	return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif
#endif