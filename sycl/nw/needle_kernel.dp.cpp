
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "needle.h"
#include <stdio.h>


#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

int 
maximum( int a,
		 int b,
		 int c){

int k;
if( a <= b )
k = b;
else 
k = a;

if( k <=c )
return(c);
else
return(k);

}

void
needle_cuda_shared_1(  int* referrence,
			  int* matrix_cuda, 
			  int cols,
			  int penalty,
			  int i,
			  int block_width,
			  const sycl::nd_item<3> &item_ct1,
			  sycl::local_accessor<int, 2> temp,
			  sycl::local_accessor<int, 2> ref) 
{
  int bx = item_ct1.get_group(2);
  int tx = item_ct1.get_local_id(2);

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[0][tx + 1] = matrix_cuda[index_n];

  /*
  DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);

		  
	  
	  }

          /*
	  DPCT1118:0: SYCL group functions and algorithms must be encountered in
 * converged control flow. You may need to adjust the code.
	  */
          /*
	  DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
	  */
          item_ct1.barrier();
    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);
	   
	  }

          /*
	  DPCT1118:1: SYCL group functions and algorithms must be encountered in
 * converged control flow. You may need to adjust the code.
	  */
          /*
	  DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
	  */
          item_ct1.barrier();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}


void
needle_cuda_shared_2(  int* referrence,
			  int* matrix_cuda, 
			 
			  int cols,
			  int penalty,
			  int i,
			  int block_width,
			  const sycl::nd_item<3> &item_ct1,
			  sycl::local_accessor<int, 2> temp,
			  sycl::local_accessor<int, 2> ref) 
{

  int bx = item_ct1.get_group(2);
  int tx = item_ct1.get_local_id(2);

  int b_index_x = bx + block_width - i  ;
  int b_index_y = block_width - bx -1;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
    int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  /*
  DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];
 
 
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  /*
  DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[0][tx + 1] = matrix_cuda[index_n];

  /*
  DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);	  
	  
	  }

          /*
	  DPCT1118:2: SYCL group functions and algorithms must be encountered in
 * converged control flow. You may need to adjust the code.
	  */
          /*
	  DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
	  */
          item_ct1.barrier();
    }


 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);


	  }

          /*
	  DPCT1118:3: SYCL group functions and algorithms must be encountered in
 * converged control flow. You may need to adjust the code.
	  */
          /*
	  DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
 * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
 * performance if there is no access to global memory.
	  */
          item_ct1.barrier();
  }


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}

