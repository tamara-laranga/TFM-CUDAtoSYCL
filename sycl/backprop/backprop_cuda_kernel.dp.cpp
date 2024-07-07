

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "backprop.h"
#include "math.h"

void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid,
					   const sycl::nd_item<3> &item_ct1,
					   float *input_node,
					   sycl::local_accessor<float, 2> weight_matrix) 
{
   int by = item_ct1.get_group(1);
   int tx = item_ct1.get_local_id(2);
   int ty = item_ct1.get_local_id(1);

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;

   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;

   /*
   DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance if there is no access to global memory.
   */
   item_ct1.barrier();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   /*
   DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance if there is no access to global memory.
   */
   item_ct1.barrier();

   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   /*
   DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance if there is no access to global memory.
   */
   item_ct1.barrier();

   for (int i = 1; i <= sycl::log2((float)HEIGHT); i++) {

           int power_two = dpct::pow(2, i);

           if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

           /*
	   DPCT1118:0: SYCL group functions and algorithms must be
            * encountered in converged control flow. You may need to adjust the
            * code.
	   */
           /*
	   DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
            * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            * better performance if there is no access to global memory.

            */
           item_ct1.barrier();
   }
   
   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   /*
   DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance if there is no access to global memory.
   */
   item_ct1.barrier();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}


void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw,
										 const sycl::nd_item<3> &item_ct1)  									
{

   int by = item_ct1.get_group(1);

   int tx = item_ct1.get_local_id(2);
   int ty = item_ct1.get_local_id(1);

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

   /*
   DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance if there is no access to global memory.
   */
   item_ct1.barrier();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
