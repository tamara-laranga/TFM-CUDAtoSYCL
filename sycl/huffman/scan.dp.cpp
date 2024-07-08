/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "scanLargeArray_kernel.dp.cpp"
#include <assert.h>
#include <stdio.h>
#include "cutil.h"
#include <cmath>

inline bool
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

static unsigned int** g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static unsigned int g_numLevelsAllocated = 0;

static void preallocBlockSums(unsigned int maxNumElements) try {
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {
        unsigned int numBlocks =
            std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (unsigned int**) malloc(level * sizeof(unsigned int*));
    g_numLevelsAllocated = level;
    numElts = maxNumElements;
    level = 0;
    
    do {
        unsigned int numBlocks =
            std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)

            CUDA_SAFE_CALL(DPCT_CHECK_ERROR(
                g_scanBlockSums[level++] = sycl::malloc_device<unsigned int>(
                    numBlocks, dpct::get_in_order_queue())));
        numElts = numBlocks;
    } while (numElts > 1);

    CUT_CHECK_ERROR("preallocBlockSums");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        dpct::dpct_free(g_scanBlockSums[i], dpct::get_in_order_queue());
    }

    CUT_CHECK_ERROR("deallocBlockSums");
    
    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(unsigned int *outArray, 
                           const unsigned int *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks =
        std::max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = dpct::max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock =

            sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize =

        sizeof(unsigned int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    sycl::range<3> grid(1, 1, dpct::max(1, numBlocks - np2LastBlock));
    sycl::range<3> threads(1, 1, numThreads);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

    // execute the scan
    if (numBlocks > 1)
    {
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(sharedMemSize), cgh);

            unsigned int *g_scanBlockSums_level_ct2 = g_scanBlockSums[level];
            int numThreads_ct3 = numThreads * 2;

            cgh.parallel_for(
                sycl::nd_range<3>(grid * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                    prescan<true, false>(
                        outArray, inArray, g_scanBlockSums_level_ct2,
                        numThreads_ct3, 0, 0, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        CUT_CHECK_ERROR("prescanWithBlockSums");
        if (np2LastBlock)
        {
            /*
            DPCT1049:7: The work-group size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the work-group size if
             * needed.
            */
            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(sharedMemLastBlock), cgh);

                unsigned int *g_scanBlockSums_level_ct2 =
                    g_scanBlockSums[level];
                int numBlocks_ct4 = numBlocks - 1;
                int numElements_numEltsLastBlock_ct5 =
                    numElements - numEltsLastBlock;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, numThreadsLastBlock),
                        sycl::range<3>(1, 1, numThreadsLastBlock)),
                    [=](sycl::nd_item<3> item_ct1) {
                        prescan<true, true>(
                            outArray, inArray, g_scanBlockSums_level_ct2,
                            numEltsLastBlock, numBlocks_ct4,
                            numElements_numEltsLastBlock_ct5, item_ct1,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
            CUT_CHECK_ERROR("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned int, 0> uni_acc_ct1(cgh);

            unsigned int *g_scanBlockSums_level_ct1 = g_scanBlockSums[level];
            int numElements_numEltsLastBlock_ct2 =
                numElements - numEltsLastBlock;

            cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 uniformAdd(outArray, g_scanBlockSums_level_ct1,
                                            numElements_numEltsLastBlock_ct2, 0,
                                            0, item_ct1, uni_acc_ct1);
                             });
        });
        CUT_CHECK_ERROR("uniformAdd");
        if (np2LastBlock)
        {
            /*
            DPCT1049:9: The work-group size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the work-group size if
             * needed.
            */
            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<unsigned int, 0> uni_acc_ct1(cgh);

                unsigned int *g_scanBlockSums_level_ct1 =
                    g_scanBlockSums[level];
                int numBlocks_ct3 = numBlocks - 1;
                int numElements_numEltsLastBlock_ct4 =
                    numElements - numEltsLastBlock;

                cgh.parallel_for(sycl::nd_range<3>(
                                     sycl::range<3>(1, 1, numThreadsLastBlock),
                                     sycl::range<3>(1, 1, numThreadsLastBlock)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     uniformAdd(
                                         outArray, g_scanBlockSums_level_ct1,
                                         numEltsLastBlock, numBlocks_ct3,
                                         numElements_numEltsLastBlock_ct4,
                                         item_ct1, uni_acc_ct1);
                                 });
            });
            CUT_CHECK_ERROR("uniformAdd");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel
         * may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(sharedMemSize), cgh);

            int numThreads_ct3 = numThreads * 2;

            cgh.parallel_for(
                sycl::nd_range<3>(grid * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                    prescan<false, false>(
                        outArray, inArray, 0, numThreads_ct3, 0, 0, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        CUT_CHECK_ERROR("prescan");
    }
    else
    {
         /*
         DPCT1049:11: The work-group size passed to the SYCL kernel
          * may exceed the limit. To get the device limit, query
          * info::device::max_work_group_size. Adjust the work-group size if
          * needed.
         */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(sharedMemSize), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * threads, threads),
                [=](sycl::nd_item<3> item_ct1) {
                    prescan<false, true>(
                        outArray, inArray, 0, numElements, 0, 0, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
         CUT_CHECK_ERROR("prescanNP2");
    }
}

static void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}

#endif // _PRESCAN_CU_
