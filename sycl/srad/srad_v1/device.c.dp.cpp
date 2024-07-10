#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
//======================================================================================================================================================150
//	FUNCTIONS
//======================================================================================================================================================150

//====================================================================================================100
//	SET DEVICE
//====================================================================================================100

void setdevice(void){

	// variables
	int num_devices;
	int device;

	// work
        num_devices = dpct::dev_mgr::instance().device_count();
        if (num_devices > 1) {
		
		// variables
		int max_multiprocessors; 
		int max_device;
                dpct::device_info properties;

                // initialize variables
		max_multiprocessors = 0;
		max_device = 0;
		
		for (device = 0; device < num_devices; device++) {
                        dpct::get_device_info(
                            properties,
                            dpct::dev_mgr::instance().get_device(device));
                        if (max_multiprocessors < properties.get_max_compute_units()) {
                                max_multiprocessors = properties.get_max_compute_units();
                                max_device = device;
			}
		}
                /*
		DPCT1093:13: The "max_device" device may be not
                 * the one intended for use. Adjust the selected device if
                 * needed.
		*/
                dpct::select_device(max_device);
        }

}

//====================================================================================================100
//	GET LAST ERROR
//====================================================================================================100

void checkCUDAError(const char *msg)
{
        /*
	DPCT1010:16: SYCL uses exceptions to report errors and does not
         * use the error codes. The call was replaced with 0. You need to
         * rewrite this code.
	*/
        dpct::err0 err = 0;
        /*
	DPCT1000:15: Error handling if-stmt was detected but could not
         * be rewritten.
	*/
        if (0 != err) {
                // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
                /*
		DPCT1009:17: SYCL uses exceptions to report
                 * errors and does not use the error codes. The call was
                 * replaced by a placeholder string. You need to rewrite this
                 * code.
		*/
                printf("Cuda error: %s: %s.\n", msg, "<Placeholder string>");
                /*
		DPCT1001:14: The statement could not be
                 * removed.
		*/
                fflush(NULL);
                exit(EXIT_FAILURE);
	}
}

//===============================================================================================================================================================================================================200
//	END SET_DEVICE CODE
//===============================================================================================================================================================================================================200
