#ifdef __cplusplus
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"									// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/cuda/cuda.h"							// (in library path specified to compiler)	needed by for device functions
#include "../util/timer/timer.h"						// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_2.dp.cpp" // (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper_2.h"				// (in the current directory)

//========================================================================================================================================================================================================200
//	FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_cuda_wrapper_2(	knode *knodes,
							long knodes_elem,
							long knodes_mem,

							int order,
							long maxheight,
							int count,

							long *currKnode,
							long *offset,
							long *lastKnode,
							long *offset_2,
							int *start,
							int *end,
							int *recstart,
							int *reclength)
{

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

        dpct::get_current_device().queues_wait_and_throw();

        //====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	int numBlocks;
	numBlocks = count;
	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

	time1 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY				MALLOC
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50

	//knode *knodesD;
    //    knodesD = (knode *)sycl::malloc_device(knodes_mem, dpct::get_in_order_queue());
    //    checkCUDAError("cudaMalloc  recordsD");

	knode *knodesD;
	try {
		knodesD = static_cast<knode *>(sycl::malloc_device(knodes_mem, dpct::get_in_order_queue()));
		// No need for checkCUDAError as SYCL throws exceptions on errors
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for knodesD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	currKnodeD
	//==================================================50

	long *currKnodeD;
	try {
        currKnodeD = sycl::malloc_device<long>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for currKnodeD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	offsetD
	//==================================================50

	long *offsetD;
	try{
		offsetD = sycl::malloc_device<long>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for offsetD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	lastKnodeD
	//==================================================50

	long *lastKnodeD;
	try{
        lastKnodeD = sycl::malloc_device<long>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for lastKnodeD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	offset_2D
	//==================================================50

	long *offset_2D;
	try {
        offset_2D = sycl::malloc_device<long>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for offset_2D", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	startD
	//==================================================50

	int *startD;
	try {
		startD = sycl::malloc_device<int>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for startD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	endD
	//==================================================50

	int *endD;
	try{
		endD = sycl::malloc_device<int>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for endD", e.what());
		exit(EXIT_FAILURE);
	}

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	int *ansDStart;
	try {
        ansDStart = sycl::malloc_device<int>(count, dpct::get_in_order_queue());
	} catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for ansDStart", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	ansDLength
	//==================================================50

	int *ansDLength;
	try {
        ansDLength = sycl::malloc_device<int>(count, dpct::get_in_order_queue());
    } catch (sycl::exception const &e){
		fprintf(stderr, "SYCL exception: %s: %s.\n", "malloc_device for ansDLength", e.what());
		exit(EXIT_FAILURE);
	}

	time2 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN
	//====================================================================================================100

	//==================================================50
	//	knodesD
	//==================================================50
	try {
		dpct::get_in_order_queue().memcpy(knodesD, knodes, knodes_mem).wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy knodesD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	currKnodeD
	//==================================================50
	
	try {
		dpct::get_in_order_queue()
			.memcpy(currKnodeD, currKnode, count * sizeof(long))
			.wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for currKnodeD", e.what());
		exit(EXIT_FAILURE);
	}


	//==================================================50
	//	offsetD
	//==================================================50
	try{
		dpct::get_in_order_queue()
			.memcpy(offsetD, offset, count * sizeof(long))
			.wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for offsetD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	lastKnodeD
	//==================================================50
	try {
        dpct::get_in_order_queue()
            .memcpy(lastKnodeD, lastKnode, count * sizeof(long))
            .wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for lastKnodeD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	offset_2D
	//==================================================50
	try{
		dpct::get_in_order_queue()
            .memcpy(offset_2D, offset_2, count * sizeof(long))
            .wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for offset_2D", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	startD
	//==================================================50
	try {
		dpct::get_in_order_queue().memcpy(startD, start, count * sizeof(int)).wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for startD", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	endD
	//==================================================50
	try{
		dpct::get_in_order_queue().memcpy(endD, end, count * sizeof(int)).wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for endD", e.what());
		exit(EXIT_FAILURE);
	}

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50

	try {
		dpct::get_in_order_queue()
            .memcpy(ansDStart, recstart, count * sizeof(int))
            .wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for ansDStart", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	ansDLength
	//==================================================50

	try {
		dpct::get_in_order_queue()
			.memcpy(ansDLength, reclength, count * sizeof(int))
			.wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for ansDLength", e.what());
		exit(EXIT_FAILURE);
	}

	time3 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// [GPU] findRangeK kernel
        /*
	DPCT1049:2: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
	try {
		        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                  sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                    findRangeK(maxheight, knodesD, knodes_elem, currKnodeD,
                               offsetD, lastKnodeD, offset_2D, startD, endD,
                               ansDStart, ansDLength, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "parallel_for findRangeK", e.what());
    	exit(EXIT_FAILURE);
	}

	time4 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	DEVICE IN/OUT
	//====================================================================================================100

	//==================================================50
	//	ansDStart
	//==================================================50
	try {
		dpct::get_in_order_queue()
			.memcpy(recstart, ansDStart, count * sizeof(int))
			.wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for ansDStart", e.what());
		exit(EXIT_FAILURE);
	}

	//==================================================50
	//	ansDLength
	//==================================================50
	try{
		dpct::get_in_order_queue()
			.memcpy(reclength, ansDLength, count * sizeof(int))
			.wait();
	} catch (sycl::exception const &e) {
		fprintf(stderr, "SYCL exception: %s: %s.\n", "memcpy for ansDLength", e.what());
		exit(EXIT_FAILURE);
	}

	time5 = get_time();

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

        dpct::dpct_free(knodesD, dpct::get_in_order_queue());

        dpct::dpct_free(currKnodeD, dpct::get_in_order_queue());
        dpct::dpct_free(offsetD, dpct::get_in_order_queue());
        dpct::dpct_free(lastKnodeD, dpct::get_in_order_queue());
        dpct::dpct_free(offset_2D, dpct::get_in_order_queue());
        dpct::dpct_free(startD, dpct::get_in_order_queue());
        dpct::dpct_free(endD, dpct::get_in_order_queue());
        dpct::dpct_free(ansDStart, dpct::get_in_order_queue());
        dpct::dpct_free(ansDLength, dpct::get_in_order_queue());

        time6 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time6-time0) / 1000000);

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
