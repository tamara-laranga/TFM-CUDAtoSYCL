#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

void
init(int argc, char** argv)
{
	if(argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border,
                const sycl::nd_item<3> &item_ct1,
                int *prev,
                int *result)
{

        int bx = item_ct1.get_group(2);
        int tx = item_ct1.get_local_id(2);

        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
        item_ct1.barrier(
            sycl::access::fence_space::local_space); // [Ronny] Added sync to
                                                     // avoid race on prev Aug.
                                                     // 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            /*
            DPCT1118:0: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
            /*
            DPCT1118:1: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(
                sycl::access::fence_space::local_space); // [Ronny] Added sync
                                                         // to avoid race on
                                                         // prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
        sycl::range<3> dimGrid(1, 1, blockCols);

        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                  /*
                  DPCT1101:3: 'BLOCK_SIZE' expression was replaced with a value.
                  Modify the code to use the original expression, provided in
                  comments, if it is correct.
                  */
                  sycl::local_accessor<int, 1> prev_acc_ct1(
                      sycl::range<1>(BLOCK_SIZE), cgh);
                  /*
                  DPCT1101:4: 'BLOCK_SIZE' expression was replaced with a value.
                  Modify the code to use the original expression, provided in
                  comments, if it is correct.
                  */
                  sycl::local_accessor<int, 1> result_acc_ct1(
                      sycl::range<1>(BLOCK_SIZE), cgh);

                  int MIN_pyramid_height_rows_t_ct0 =
                      MIN(pyramid_height, rows - t - 1);
                  int *gpuResult_src_ct2 = gpuResult[src];
                  int *gpuResult_dst_ct3 = gpuResult[dst];

                  cgh.parallel_for(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                      [=](sycl::nd_item<3> item_ct1) {
                            dynproc_kernel(
                                MIN_pyramid_height_rows_t_ct0, gpuWall,
                                gpuResult_src_ct2, gpuResult_dst_ct3, cols,
                                rows, t, borderCols, item_ct1,
                                prev_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                result_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                      });
            });

            // for the measurement fairness
            dpct::get_current_device().queues_wait_and_throw();
        }
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    num_devices = dpct::dev_mgr::instance().device_count();
    /*
    DPCT1093:2: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    if (num_devices > 1) dpct::select_device(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    gpuResult[0] = sycl::malloc_device<int>(cols, dpct::get_in_order_queue());
    gpuResult[1] = sycl::malloc_device<int>(cols, dpct::get_in_order_queue());
    dpct::get_in_order_queue()
        .memcpy(gpuResult[0], data, sizeof(int) * cols)
        .wait();
    gpuWall = sycl::malloc_device<int>((size - cols), dpct::get_in_order_queue());
    dpct::get_in_order_queue()
        .memcpy(gpuWall, data + cols, sizeof(int) * (size - cols))
        .wait();

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    dpct::get_in_order_queue()
        .memcpy(result, gpuResult[final_ret], sizeof(int) * cols)
        .wait();

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    dpct::dpct_free(gpuWall, dpct::get_in_order_queue());
    dpct::dpct_free(gpuResult[0], dpct::get_in_order_queue());
    dpct::dpct_free(gpuResult[1], dpct::get_in_order_queue());

    delete [] data;
    delete [] wall;
    delete [] result;

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
}

