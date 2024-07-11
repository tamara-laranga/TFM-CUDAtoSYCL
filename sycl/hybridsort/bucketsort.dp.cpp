#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper_cuda.h"
#include "bucketsort.dp.hpp"
// includes, kernels
#include "bucketsort_kernel.dp.cpp"
#include "histogram1024_kernel.dp.cpp"

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize, 
					 int divisions, float min, float max, float *pivotPoints, 
					 float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
const int histosize = 1024; 
unsigned int* h_offsets = NULL; 
unsigned int* d_offsets = NULL; 
int *d_indice = NULL; 
float *pivotPoints = NULL; 
float *historesult = NULL; 
float *l_pivotpoints = NULL; 
unsigned int *d_prefixoffsets = NULL; 
unsigned int *l_offsets = NULL; 

////////////////////////////////////////////////////////////////////////////////
// Initialize the bucketsort algorithm 
////////////////////////////////////////////////////////////////////////////////
void init_bucketsort(int listsize)
{
	h_offsets = (unsigned int *) malloc(histosize * sizeof(int));
    checkCudaErrors(
        DPCT_CHECK_ERROR(d_offsets = sycl::malloc_device<unsigned int>(
                             histosize, dpct::get_in_order_queue())));
        pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));

    checkCudaErrors(DPCT_CHECK_ERROR(
        d_indice =
            sycl::malloc_device<int>(listsize, dpct::get_in_order_queue())));
        historesult = (float *)malloc(histosize * sizeof(float));

        checkCudaErrors(
            DPCT_CHECK_ERROR(l_pivotpoints = sycl::malloc_device<float>(
                                 DIVISIONS, dpct::get_in_order_queue())));
        checkCudaErrors(DPCT_CHECK_ERROR(
            l_offsets = (unsigned int *)sycl::malloc_device(
                DIVISIONS * sizeof(int), dpct::get_in_order_queue())));

        int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
        checkCudaErrors(DPCT_CHECK_ERROR(
            d_prefixoffsets = (unsigned int *)sycl::malloc_device(
                blocks * BUCKET_BLOCK_MEMORY * sizeof(int),
                dpct::get_in_order_queue())));

        initHistogram1024();
}

////////////////////////////////////////////////////////////////////////////////
// Uninitialize the bucketsort algorithm 
////////////////////////////////////////////////////////////////////////////////
void finish_bucketsort()
{
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_indice, dpct::get_in_order_queue())));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dpct_free(d_offsets, dpct::get_in_order_queue())));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dpct_free(l_pivotpoints, dpct::get_in_order_queue())));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dpct_free(l_offsets, dpct::get_in_order_queue())));
        free(pivotPoints); 
	free(h_offsets);
	free(historesult);
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::dpct_free(d_prefixoffsets, dpct::get_in_order_queue())));
        closeHistogram1024();
}

////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
				int *sizes, int *nullElements, float minimum, float maximum, 
				unsigned int *origOffsets)
{
	////////////////////////////////////////////////////////////////////////////
	// First pass - Create 1024 bin histogram 
	////////////////////////////////////////////////////////////////////////////
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset((void *)d_offsets, 0, histosize * sizeof(int))
                .wait()));
        histogram1024GPU(h_offsets, d_input, minimum, maximum, listsize); 
	for(int i=0; i<histosize; i++) historesult[i] = (float)h_offsets[i];

	///////////////////////////////////////////////////////////////////////////
	// Calculate pivot points (CPU algorithm)
	///////////////////////////////////////////////////////////////////////////
	calcPivotPoints(historesult, histosize, listsize, DIVISIONS, 
			minimum, maximum, pivotPoints,
			(maximum - minimum)/(float)histosize); 
	///////////////////////////////////////////////////////////////////////////
	// Count the bucket sizes in new divisions
	///////////////////////////////////////////////////////////////////////////
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(l_pivotpoints, pivotPoints, (DIVISIONS) * sizeof(int))
                .wait()));
        checkCudaErrors(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset((void *)d_offsets, 0, DIVISIONS * sizeof(int))
                .wait()));
        //checkCudaErrors(cudaBindTexture(0, texPivot, l_pivotpoints, DIVISIONS * sizeof(int))); 
	// Setup block and grid
    sycl::range<3> threads(1, 1, BUCKET_THREAD_N);
        int blocks = ((listsize - 1) / (threads[2] * BUCKET_BAND)) + 1;
    sycl::range<3> grid(1, 1, blocks);

        dpct::image_wrapper_base_p texPivotObj = 0;
        dpct::image_data resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.set_data_type(dpct::image_data_type::linear);
        resDesc.set_data_ptr(l_pivotpoints);
        /*
	DPCT1059:5: SYCL only supports 4-channel image format. Adjust
         * the code.
	*/
        resDesc.set_channel(dpct::image_channel::create<int>());
        resDesc.set_x(DIVISIONS * sizeof(int));

        dpct::sampling_info texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        /*
	DPCT1062:6: SYCL Image doesn't support normalized read mode.

         */

        checkCudaErrors(DPCT_CHECK_ERROR(
            texPivotObj = dpct::create_image_wrapper(resDesc, texDesc)));

        // Find the new indice for all elements
        /*
	DPCT1049:2: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                /*
	    DPCT1101:39: 'BUCKET_BLOCK_MEMORY' expression was
                 * replaced with a value. Modify the code to use the original
                 * expression, provided in comments, if it is correct.
	    */
                sycl::local_accessor<volatile unsigned int, 1> s_offset_acc_ct1(
                    sycl::range<1>(BUCKET_BLOCK_MEMORY), cgh);

                auto texPivotObj_acc =
                    static_cast<dpct::image_wrapper<float, 1> *>(texPivotObj)
                        ->get_access(cgh);

                auto texPivotObj_smpl = texPivotObj->get_sampler();

                int *d_indice_ct2 = d_indice;
                unsigned int *d_prefixoffsets_ct3 = d_prefixoffsets;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                            bucketcount(dpct::image_accessor_ext<float, 1>(
                                            texPivotObj_smpl, texPivotObj_acc),
                                        d_input, d_indice_ct2,
                                        d_prefixoffsets_ct3, listsize, item_ct1,
                                        s_offset_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                    });
        });
        ///////////////////////////////////////////////////////////////////////////
	// Prefix scan offsets and align each division to float4 (required by 
	// mergesort)
	///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
threads.x = BUCKET_WG_SIZE_0;
#else
        threads[2] = 128;
#endif
        grid[2] = DIVISIONS / threads[2];
        /*
	DPCT1049:3: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                unsigned int *d_prefixoffsets_ct0 = d_prefixoffsets;
                unsigned int *d_offsets_ct1 = d_offsets;

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         bucketprefixoffset(d_prefixoffsets_ct0,
                                                            d_offsets_ct1,
                                                            blocks, item_ct1);
                                 });
        });

        // copy the sizes from device to host
        dpct::get_in_order_queue()
            .memcpy(h_offsets, d_offsets, DIVISIONS * sizeof(int))
            .wait();

        origOffsets[0] = 0;
	for(int i=0; i<DIVISIONS; i++){
		origOffsets[i+1] = h_offsets[i] + origOffsets[i]; 
		if((h_offsets[i] % 4) != 0){
			nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i]; 
		}
		else nullElements[i] = 0; 
	}
	for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4; 
	for(int i=0; i<DIVISIONS; i++) {
		if((h_offsets[i] % 4) != 0)	h_offsets[i] = (h_offsets[i] & ~3) + 4; 
	}
	for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i]; 
	for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1]; 
	h_offsets[0] = 0; 
	///////////////////////////////////////////////////////////////////////////
	// Finally, sort the lot
	///////////////////////////////////////////////////////////////////////////
        dpct::get_in_order_queue()
            .memcpy(l_offsets, h_offsets, (DIVISIONS) * sizeof(int))
            .wait();
        dpct::get_in_order_queue()
            .memset(d_output, 0x0, (listsize + (DIVISIONS * 4)) * sizeof(float))
            .wait();
    threads[2] = BUCKET_THREAD_N;
        blocks = ((listsize - 1) / (threads[2] * BUCKET_BAND)) + 1;
    grid[2] = blocks;
        /*
	DPCT1049:4: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                /*
	    DPCT1101:40: 'BUCKET_BLOCK_MEMORY' expression was
                 * replaced with a value. Modify the code to use the original
                 * expression, provided in comments, if it is correct.
	    */
                sycl::local_accessor<volatile unsigned int, 1> s_offset_acc_ct1(
                    sycl::range<1>(BUCKET_BLOCK_MEMORY), cgh);

                int *d_indice_ct1 = d_indice;
                unsigned int *d_prefixoffsets_ct4 = d_prefixoffsets;
                unsigned int *l_offsets_ct5 = l_offsets;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                            bucketsort(d_input, d_indice_ct1, d_output,
                                       listsize, d_prefixoffsets_ct4,
                                       l_offsets_ct5, item_ct1,
                                       s_offset_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                    });
        });
}


////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize, 
					 int divisions, float min, float max, float *pivotPoints, float histo_width)
{
	float elemsPerSlice = listsize/(float)divisions; 
	float startsAt = min; 
	float endsAt = min + histo_width; 
	float we_need = elemsPerSlice; 
	int p_idx = 0; 
	for(int i=0; i<histosize; i++)
	{
		if(i == histosize - 1){
			if(!(p_idx < divisions)){
				pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			}
			break; 
		}
		while(histogram[i] > we_need){
			if(!(p_idx < divisions)){
				printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions); 
				exit(0);
			}
			pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			startsAt += (we_need/histogram[i]) * histo_width; 
			histogram[i] -= we_need; 
			we_need = elemsPerSlice; 
		}
		// grab what we can from what remains of it
		we_need -= histogram[i]; 

		startsAt = endsAt; 
		endsAt += histo_width; 
	}
	while(p_idx < divisions){
		pivotPoints[p_idx] = pivotPoints[p_idx-1]; 
		p_idx++; 
	}
}

