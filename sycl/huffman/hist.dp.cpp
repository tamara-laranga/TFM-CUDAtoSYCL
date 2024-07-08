/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>

#define CHECK(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(dpct::err0 code, const char *file, int line,
                      bool abort = true)
{
}

using namespace std;

#define SIZE    (100*1024*1024)


void histo_kernel( unsigned char *buffer,
        long size,
        unsigned int *histo ,
        const sycl::nd_item<3> &item_ct1,
        unsigned int *temp) {

    temp[item_ct1.get_local_id(2)] = 0;
    /*
    DPCT1065:25: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int offset = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    while (i < size) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &temp[buffer[i]], 1);
        i += offset;
    }

    /*
    DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance if there is no access to global memory.
    */
    item_ct1.barrier();
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(histo[item_ct1.get_local_id(2)]), temp[item_ct1.get_local_id(2)]);
}

int runHisto(char* file, unsigned int* freq, unsigned int memSize, unsigned int *source) {

    FILE *f = fopen(file,"rb");
    if (!f) {perror(file); exit(1);}
    fseek(f,0,SEEK_SET);
    size_t result = fread(source,1,memSize,f);
    if(result != memSize) fputs("Cannot read input file", stderr);

    fclose(f);

    unsigned char *buffer = (unsigned char*)source;

    dpct::device_info prop;
    (DPCT_CHECK_ERROR(
        dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
    int blocks = prop.get_max_compute_units();

    if (!true)
    {
        cout << "No overlaps, so no speedup from streams" << endl;
        return 0;
    }

    // allocate memory on the GPU for the file's data
    int partSize = memSize/32;
    int totalNum = memSize/sizeof(unsigned int);
    int partialNum = partSize/sizeof(unsigned int);

    unsigned char *dev_buffer0; 
    unsigned char *dev_buffer1;
    unsigned int *dev_histo;
    dev_buffer0 = (unsigned char *)sycl::malloc_device(
        partSize, dpct::get_in_order_queue());
    dev_buffer1 = (unsigned char *)sycl::malloc_device(
        partSize, dpct::get_in_order_queue());
    dev_histo = (unsigned int *)sycl::malloc_device(256 * sizeof(int),
                                                    dpct::get_in_order_queue());
    dpct::get_in_order_queue().memset(dev_histo, 0, 256 * sizeof(int)).wait();
    dpct::queue_ptr stream0, stream1;
    CHECK(DPCT_CHECK_ERROR(stream0 = dpct::get_current_device().create_queue()));
    CHECK(DPCT_CHECK_ERROR(stream1 = dpct::get_current_device().create_queue()));
    dpct::event_ptr start, stop;
    (DPCT_CHECK_ERROR(start = new sycl::event()));
    (DPCT_CHECK_ERROR(stop = new sycl::event()));
    /*
    DPCT1024:28: The original code returned the error code that was
     * further consumed by the program logic. This original code was replaced
     * with 0. You may need to rewrite the program logic consuming the error
     * code.
    */
    (DPCT_CHECK_ERROR(dpct::sync_barrier(start, &dpct::get_in_order_queue())));
    
    *start = stream0->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1>) {}); 
    });

    for(int i = 0; i < totalNum; i+=partialNum*2)
    {

        /*
        DPCT1124:29: cudaMemcpyAsync is migrated to asynchronous
         * memcpy API. While the origin API might be synchronous, it depends on
         * the type of operand memory, so you may need to call wait() on event
         * return by memcpy API to ensure synchronization behavior.
        */
        CHECK(DPCT_CHECK_ERROR(stream0->memcpy(dev_buffer0, buffer + i, partSize)));
        /*
        DPCT1124:30: cudaMemcpyAsync is migrated to asynchronous
         * memcpy API. While the origin API might be synchronous, it depends on
         * the type of operand memory, so you may need to call wait() on event
         * return by memcpy API to ensure synchronization behavior.
        */
        dpct::get_current_device().default_queue().memcpy(dev_buffer0, buffer + i, partSize).wait();
        dpct::get_current_device().default_queue().memcpy(dev_buffer1, buffer + i + partialNum, partSize).wait();

        CHECK(DPCT_CHECK_ERROR(
            stream1->memcpy(dev_buffer1, buffer + i + partialNum, partSize)));

        // kernel launch - 2x the number of mps gave best timing
        stream0->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned int, 1> temp_acc_ct1(
                sycl::range<1>(256), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * 2) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    histo_kernel(
                        dev_buffer0, partSize, dev_histo, item_ct1,
                        temp_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        stream1->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned int, 1> temp_acc_ct1(
                sycl::range<1>(256), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * 2) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    histo_kernel(
                        dev_buffer1, partSize, dev_histo, item_ct1,
                        temp_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }

    stream0->wait();
    stream1->wait();

    CHECK(DPCT_CHECK_ERROR(stream0->wait()));
    CHECK(DPCT_CHECK_ERROR(stream1->wait()));
    dpct::get_in_order_queue().memcpy(freq, dev_histo, 256 * sizeof(int)).wait();
    /*
    DPCT1024:31: The original code returned the error code that was
     * further consumed by the program logic. This original code was replaced
     * with 0. You may need to rewrite the program logic consuming the error
     * code.
    */

    dpct::get_current_device().default_queue().memcpy(freq, dev_histo, 256 * sizeof(int)).wait();
    
    *stop = stream0->submit([&](sycl::handler &cgh){
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1>){});
    });

    stream0->wait();
    (DPCT_CHECK_ERROR(dpct::sync_barrier(stop, &dpct::get_in_order_queue())));
    (DPCT_CHECK_ERROR(stop->wait_and_throw()));
    float   elapsedTime;
    (DPCT_CHECK_ERROR(elapsedTime =
                          (stop->get_profiling_info<
                               sycl::info::event_profiling::command_end>() -
                           start->get_profiling_info<
                               sycl::info::event_profiling::command_start>()) /
                          1000000.0f));
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    dpct::dpct_free(dev_histo, dpct::get_in_order_queue());
    dpct::dpct_free(dev_buffer0, dpct::get_in_order_queue());
    dpct::dpct_free(dev_buffer1, dpct::get_in_order_queue());
    return 0;
}
