// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project

// includes, kernels
#include "srad_kernel.dp.cpp"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	//Allocate device memory
    J_cuda = sycl::malloc_device<float>(size_I, q_ct1);
    C_cuda = sycl::malloc_device<float>(size_I, q_ct1);
        E_C = sycl::malloc_device<float>(size_I, q_ct1);
        W_C = sycl::malloc_device<float>(size_I, q_ct1);
        S_C = sycl::malloc_device<float>(size_I, q_ct1);
        N_C = sycl::malloc_device<float>(size_I, q_ct1);

#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
        
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU


#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
        sycl::range<3> dimGrid(1, block_y, block_x);

        //Copy data from main memory to device memory
        q_ct1.memcpy(J_cuda, J, sizeof(float) * size_I).wait();

        //Run kernels
        /*
	DPCT1049:0: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        q_ct1.submit([&](sycl::handler &cgh) {
            /*
	    DPCT1101:7: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:8: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> temp_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:9: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:10: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> temp_result_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:11: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:12: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> north_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:13: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:14: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> south_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:15: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:16: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> east_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:17: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:18: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> west_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 srad_cuda_1(E_C, W_C, N_C, S_C, J_cuda, C_cuda,
                                             cols, rows, q0sqr, item_ct1,
                                             temp_acc_ct1, temp_result_acc_ct1,
                                             north_acc_ct1, south_acc_ct1,
                                             east_acc_ct1, west_acc_ct1);
                             });
        });
        /*
	DPCT1049:1: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        q_ct1.submit([&](sycl::handler &cgh) {
            /*
	    DPCT1101:19: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:20: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> south_c_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:21: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:22: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> east_c_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:23: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:24: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> c_cuda_temp_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:25: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:26: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> c_cuda_result_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            /*
	    DPCT1101:27: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            /*
	    DPCT1101:28: 'BLOCK_SIZE' expression was replaced with a
             * value. Modify the code to use the original expression, provided
             * in comments, if it is correct.
	    */
            sycl::local_accessor<float, 2> temp_acc_ct1(
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    srad_cuda_2(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows,
                                lambda, q0sqr, item_ct1, south_c_acc_ct1,
                                east_c_acc_ct1, c_cuda_temp_acc_ct1,
                                c_cuda_result_acc_ct1, temp_acc_ct1);
                });
        });

        //Copy data from device memory to main memory
    q_ct1.memcpy(J, J_cuda, sizeof(float) * size_I).wait();

#endif   
}

    dev_ct1.queues_wait_and_throw();

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");

	free(I);
	free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
    dpct::dpct_free(C_cuda, q_ct1);
        dpct::dpct_free(J_cuda, q_ct1);
        dpct::dpct_free(E_C, q_ct1);
        dpct::dpct_free(W_C, q_ct1);
        dpct::dpct_free(N_C, q_ct1);
        dpct::dpct_free(S_C, q_ct1);
#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

