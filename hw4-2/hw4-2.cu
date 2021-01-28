// This version: c04: 6.45 seconds.
/*
This version is "NO Streaming" version.

0102 TODOs;

(V) 1. Correctness
// c01~07, p31~36
// c06 after: TOO SLOW to get answerQQ

(V) 2. Larger Blocking_Factor B

( ) 3. Initial padding (Remove if(i<n && j<n))

( ) 4. Asynchronous Peer Copy

( ) 5. Use different streams in cudaMemcpyPeerAsync! (stream 0~3)

( ) 6. Less cudaDeviceSynchronize();

( ) 7. #pragma omp parallel 

(V) N. Hide printf into #ifdef DEBUG_PHASE1, PHASE2, PHASE3 

*/


// System includes

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

// CUDA runtime
#include <cuda_runtime.h>

#include <omp.h>


#include <time.h>
#define TIME

// #define CUDA_NVPROF

// #define DEBUG_DIST
// #define DEBUG_DEVICE_DIST
// #define DEBUG_DEVICE_DIST1
// #define DEBUG_PHASE1
// #define DEBUG_PHASE2
// #define DEBUG_PHASE3
// #define CHECK_CORRECTNESS

const int BLOCKING_FACTOR = 32; // 32, 16, 8, 4, 2

const int INF = ((1 << 30) - 1);
// Global var stored in Data Section.
// const int V = 40010;
void input(char* inFileName);
void output(char* outFileName);

void print_ans(int num_V, char* ans_file);
void print_Dist(int num_V);

void block_FW(int B);
// void block_FW_small_n(int B);
void block_FW_MultiGPU_Old(int B);
void block_FW_MultiGPU(int B);
int ceil(int a, int b); // min num that >= a/b
                // floor: max num <= a/b
int floor(int a, int b);

__device__ inline int Addr(int matrixIdx, int i, int j, int N){
    return( N*N*matrixIdx + i*N + j);
}

// W: width, H: height
// __device__ inline int Addr2(int matrixIdx, int i, int j, int W, int H){
//     return( W*H*matrixIdx + i*W + j);
// }


// Device_Boundary: in i direction, where you Split data Initially.


// PHASE 1 : ONE Block do k iterations with B*B threads.
// __global__ void cal(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
__global__ void cal(int device_Boundary, int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){
       
    __shared__ int S[32*32*3];
    int i = block_start_x*B + threadIdx.y;
    int j = block_start_y*B + threadIdx.x;

    if(i<device_Boundary && j<n){
        // S[ (i%B)*B + (j%B)  ] = device_Dist[i*n + j];
        S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
        // S[Addr(0, (i%B), (j%B), B)] = device_Dist[Addr(0,i,j,n)];
        // S[ (i%B)*(B+1) + (j%(B+1))  ] = device_Dist[i*n + j];
        
        // __syncthreads();

            // This for-loop CANNOT be serialize!
            // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
            for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ 
                __syncthreads();
                // if (S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)];
                // } 
                S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)]);                                  
                                                      
            }
            device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
    }// end if(i<n && j<n )
}


// phase 1 for device 1
__global__ void cal_1(int device_Boundary, int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){
       
        __shared__ int S[32*32*3];
        int i = block_start_x*B + threadIdx.y;
        int j = block_start_y*B + threadIdx.x;
    
        if(i >= device_Boundary && i<n && j<n){
            // S[ (i%B)*B + (j%B)  ] = device_Dist[i*n + j];
            S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
            // S[Addr(0, (i%B), (j%B), B)] = device_Dist[Addr(0,i,j,n)];
            // S[ (i%B)*(B+1) + (j%(B+1))  ] = device_Dist[i*n + j];
            
            // __syncthreads();
    
                // This for-loop CANNOT be serialize!
                // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ 
                    __syncthreads();
                    // if (S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                    //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)];
                    // } 
                    S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)]);                                  
                                                          
                }
                device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
        }// end if(i<n && j<n )
}
    
// __global__ void cal3(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
__global__ void cal3(int device_Boundary, int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){

    __shared__ int S[32*32*3];
    // int i = block_start_y* B + blockIdx.y * B + threadIdx.y;
    // int j = block_start_x* B + blockIdx.x * B + threadIdx.x;
    int i = block_start_x* B + blockIdx.x * B + threadIdx.y;
    int j = block_start_y* B + blockIdx.y * B + threadIdx.x;


    // S[Addr(1, threadIdx.y, ((Round*B + threadIdx.x)%B), B)] = device_Dist[Addr(0,i,(Round*B + threadIdx.x),n)];
    // S[Addr(2, ((Round*B + threadIdx.y)%B), threadIdx.x, B)] = device_Dist[Addr(0,(Round*B + threadIdx.y),j,n)];

    if(i<device_Boundary && (Round*B + threadIdx.x) <n) S[Addr(1, threadIdx.y, ((Round*B + threadIdx.x)%B), B)] = device_Dist[Addr(0,i,(Round*B + threadIdx.x),n)];
    if(j<n && (Round*B + threadIdx.y)<n) S[Addr(2, ((Round*B + threadIdx.y)%B), threadIdx.x, B)] = device_Dist[Addr(0,(Round*B + threadIdx.y),j,n)];
    

    if(i<device_Boundary && j<n){
    // For each thread, calculate one edge.
        S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
        __syncthreads();

            // This for-loop CANNOT be parallelize!
            // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
            /// KEY!! Don't USE % on K.
            for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ //k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // __syncthreads();
                        // if (S[Addr(1, (i%B), (k%B), B)]+ S[Addr(2, (k%B), (j%B), B)]  < S[Addr(0, (i%B), (j%B), B)] ) {
                        //     S[Addr(0, (i%B), (j%B), B)] = S[Addr(1, (i%B), (k%B), B)]+ S[Addr(2, (k%B), (j%B), B)];
                        // }
                                // i ,  k                               // k ,  j                            // i ,  j
                        // if (S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                        //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)];
                        // }
                        S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)] );
            }
            device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
    }
}

// Phase 3 for device 1.
__global__ void cal3_1(int device_Boundary, int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){

    __shared__ int S[32*32*3];
    int i = block_start_x* B + blockIdx.x * B + threadIdx.y;
    int j = block_start_y* B + blockIdx.y * B + threadIdx.x;

    if( i<n && (Round*B + threadIdx.x) <n) S[Addr(1, threadIdx.y, ((Round*B + threadIdx.x)%B), B)] = device_Dist[Addr(0,i,(Round*B + threadIdx.x),n)];
    if((Round*B + threadIdx.y)<n &&  j<n ) S[Addr(2, ((Round*B + threadIdx.y)%B), threadIdx.x, B)] = device_Dist[Addr(0,(Round*B + threadIdx.y),j,n)];
    

    if(i>=device_Boundary && i<n && j<n){
    // For each thread, calculate one edge.
        S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
        __syncthreads();

            for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ //k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                        S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)] );
            }
            device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
    }
}


int MAX_GPU_COUNT = 32;

int n, m;
// static int Dist[V][V];
int* Dist;
int * Dist_1;

int main(int argc, char* argv[]) {

    #ifdef TIME
        // struct timespec start, end, temp;
        struct timespec total_starttime;
        struct timespec total_temp;
        struct timespec start;
        struct timespec end;
        struct timespec temp;
        double IO_time=0.0;
        double Total_time = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &total_starttime);
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    input(argv[1]);

    #ifdef DEBUG_DEVICE_DIST
        Dist_1 = (int*)malloc(sizeof(unsigned int)*n*n);
    #endif
    
    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        if ((end.tv_nsec - start.tv_nsec) < 0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        IO_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    #endif
    // printf("%f second on input\n", time_used);

    // we have num_v, num_e, adj_matrix (Dist[V][V]) now
    // int B = 512;
    // Note: Since B*B threads, maximum B : 32 (MAX 1024 threads per block)
    int B;
    B = BLOCKING_FACTOR;

    if(n < B){
        block_FW_MultiGPU_Old(B);
    }
    else{
        block_FW_MultiGPU(B);
    }

    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    output(argv[2]);

    #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC, &end);
        // IO Time
        if ((end.tv_nsec - start.tv_nsec) < 0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        // Total Time
        if ((end.tv_nsec - total_starttime.tv_nsec) < 0) {
            total_temp.tv_sec = end.tv_sec-total_starttime.tv_sec-1;
            total_temp.tv_nsec = 1000000000 + end.tv_nsec - total_starttime.tv_nsec;
        } else {
            total_temp.tv_sec = end.tv_sec - total_starttime.tv_sec;
            total_temp.tv_nsec = end.tv_nsec - total_starttime.tv_nsec;
        }

        IO_time += temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
        Total_time = total_temp.tv_sec + (double) total_temp.tv_nsec / 1000000000.0;
    #endif

    #ifdef TIME
        printf("IO Time: %.8f seconds\n", IO_time);
        printf("Total Time: %.8f seconds\n",Total_time);
    #endif

    printf("========== Comparing results... ===========\n");
    #ifdef DEBUG_DIST
        print_Dist(n);   
    #endif
    #ifdef CHECK_CORRECTNESS
        print_ans(n, argv[3]);
    #endif

    printf("Job Finished\n");
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);    // n = num_vertices
    fread(&m, sizeof(int), 1, file);    // m = num_edges

    printf("V: %d, E: %d\n",n,m);

    Dist = (int*) malloc(sizeof(int)*n*n);
    // Initialize adjacency matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i*n+j] = 0;
                // Dist[i][j] = 0;
            } else {
                Dist[i*n+j] = INF;
                // Dist[i][j] = INF;
            }
        }
    }

    // Sequentially read input edges and fill them into adj matrix.
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        // Dist[pair[0]][pair[1]] = pair[2];
        Dist[ pair[0]*n+  pair[1]] = pair[2];
    }
    fclose(file);
}


void print_ans(int num_V, char* ans_file){

    bool wrong = false;
    FILE* file = fopen(ans_file, "rb");
    int* Ans = (int*)malloc(sizeof(int)*n*n);
    fread(Ans, sizeof(int), n*n, file);

    if(num_V > 15) num_V = 15;

    for(int i=0; i<num_V*num_V; i++){
        if(Dist[i] != Ans[i]){
            wrong = true;
            printf("Wrong at offset %d, expected %d but get %d\n", i*4, Ans[i], Dist[i]);
            printf("Fron %d to %d , cost: %d\n", (i/n), (i%n), Ans[i] );
        }
        // printf("offset %d val %d, ans: %d\n", i*4, Dist[i], Ans[i]);
    }

    if(!wrong) printf(" ======= Congratulation! =========\n");


    printf("======== Your Dist ==========\n");
    for(int i=0;i<num_V; i++){
        for(int j=0; j<num_V; j++){
            if(j==num_V-1) printf("%d\n",Dist[i*num_V+j]);
            else printf("%d ", Dist[i*num_V+j]);
        }
        // printf("offset %d val %d, ans: %d\n", i*4, Dist[i], Ans[i]);
    }   

    printf("======== ANSWER ==========\n");
    for(int i=0;i<num_V; i++){
        for(int j=0; j<num_V; j++){
            if(j==num_V-1) printf("%d\n",Ans[i*num_V+j]);
            else printf("%d ", Ans[i*num_V+j]);
        }
        // printf("offset %d val %d, ans: %d\n", i*4, Dist[i], Ans[i]);
    }   

}



void print_Dist(int num_V){
    printf("========= Dist ============\n");
    for(int i=0;i<num_V; i++){
        for(int j=0; j<num_V; j++){
            if(j==num_V-1) printf("%d\n",Dist[i*num_V+j]);
            else printf("%d ", Dist[i*num_V+j]);
        }
        // printf("offset %d val %d, ans: %d\n", i*4, Dist[i], Ans[i]);
    }
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
        }
        fwrite(Dist+i*n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }


int floor(int a, int b){ return a/b >>1<<1; } // remove LSB ( discard the remainder.)

// 1204: Idea1 : one stream with 9 serialize kernel launch?
// memory to pass to GPU: B, r, r, r, 1, 1. ALL constant! No memory copy.


const int device_0 = 0;
const int device_1 = 1;
const int cudaEnablePeerAccess_Flags  = 0;
#define NUM_THREAD 2

// For Large n.: Don't use Synchronize.
// n > 5000
void block_FW_MultiGPU(int B) {

    printf("Large n : \n");
    printf("Blocking factor: %d (num of pixel(adj entries) in a Block)\n",B);
    printf(" %d * %d block\n",B,B);
    int round = ceil(n, B);

    // int cur_device_number;

    int *device_Dist;
    int *device_Dist_1;

    int canGPU0AccessGPU1, canGPU1AccessGPU0;


    int device_0_Boundary = ceil(n, 2); // e.g. 5/2 -> 3,  160/2 -> 80.
    printf("ceil(%d, 2) :%d\n",n,device_0_Boundary);

    printf("ceil % B remainder : %d\n",device_0_Boundary%B);
    // Avoid cross pivot.
    // 80 % 32 = 16. => (80 - 16 + 32) = 96.
    // ceil(999,2) = 500,  500 % 32 = 20.  500 - 20 + 32 = 512.
    if( device_0_Boundary%B !=0) device_0_Boundary = (device_0_Boundary- (device_0_Boundary%B) + B);
      
    printf("device_0_Boundary: %d\n",device_0_Boundary);


    // Record Computation time
    #ifdef TIME 
        cudaSetDevice(0);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    #ifdef TIME 
        float Total_comm_time = 0;

        cudaSetDevice(1);
        cudaEvent_t Commstart_device_1, Commstop_device_1;
        cudaEventCreate(&Commstart_device_1);
        cudaEventCreate(&Commstop_device_1);        

        cudaSetDevice(0);
        cudaEvent_t Commstart, Commstop;
        cudaEventCreate(&Commstart);
        cudaEventCreate(&Commstop);        
        cudaEventRecord(Commstart);
    #endif

    // Data Partition 1 : Split Top to device 0

    //  Bottom to device 1.


	#pragma omp parallel num_threads(NUM_THREAD) //reduction(+:pixels)
	{
        int omp_id, omp_thread_num;
        omp_id = omp_get_thread_num();
        omp_thread_num = omp_get_num_threads();

        if(omp_id==0){
            cudaSetDevice(0);
            cudaDeviceCanAccessPeer ( &canGPU0AccessGPU1, device_0, device_1 );
            if(canGPU0AccessGPU1==1){
                 printf("Can 0 access 1? %d\n",canGPU0AccessGPU1);

                cudaDeviceEnablePeerAccess ( device_1, cudaEnablePeerAccess_Flags );
                cudaMalloc(&device_Dist,   n * n* sizeof(unsigned int));
                
                #ifdef TIME
                    cudaEventRecord(Commstart);
                #endif
                cudaMemcpyAsync(device_Dist,   Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);
                
                // cudaMemcpyAsync(device_Dist,   Dist, n*device_0_Boundary*sizeof(unsigned int), cudaMemcpyHostToDevice);
                printf("omp t%d allocate & copy gpu 0\n",omp_id);
            }
            else{
                printf("Error, gpu 0 cannot directly access gpu 1\n");
                // return 2;
            }
        }
        else{
            cudaSetDevice(1);
            cudaDeviceCanAccessPeer ( &canGPU1AccessGPU0, device_1, device_0 );
            if(canGPU1AccessGPU0==1){
                printf("Can 1 access 0? %d\n",canGPU1AccessGPU0);

                cudaDeviceEnablePeerAccess ( device_0, cudaEnablePeerAccess_Flags );
                // cudaGetDevice(&cur_device_number);
                cudaMalloc(&device_Dist_1, n * n* sizeof(unsigned int));
                cudaMemcpyAsync(device_Dist_1,   Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);
                // cudaMemcpyAsync(device_Dist_1+device_0_Boundary*n, Dist+device_0_Boundary*n, ( n*n -n*device_0_Boundary)*sizeof(unsigned int), cudaMemcpyHostToDevice);
                printf("omp t%d allocate & copy gpu 1\n",omp_id);
            }
            else{
                printf("Error, gpu 1 cannot directly access gpu 0\n");
                // return 2;
            }
        }
    }


    #ifdef TIME 
        float Commtime;
        cudaSetDevice(0);
        cudaEventRecord(Commstop);
        cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
        cudaEventElapsedTime(&Commtime, Commstart, Commstop);
        
        printf("H2D copy took %.8f seconds\n",Commtime/1000);
        Total_comm_time += Commtime;
    #endif



        #ifdef DEBUG_DEVICE_DIST
            printf("========== Initial Condition =========\n");
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("Initial, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        #ifdef DEBUG_DEVICE_DIST1
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist_1, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("Initial, device_Dist_1: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif






    dim3 num_threads(B,B);


    for (int r = 0; r < round; ++r) {

        #ifdef DEBUG_DIST
            printf("========== Round %d ================\n",r);
            // print_Dist(n);
        #endif
        // printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/

        if(r*B < device_0_Boundary) { // Device 0 do pivot
            // printf("Pivot at GPU 0!\n");
            cudaSetDevice(0);
            cal<<< 1, num_threads , sizeof(int)*B*B>>> (device_0_Boundary, device_Dist, n, B, r,   r, r);
            
            #ifdef TIME 
                cudaEventRecord(Commstart);
            #endif
            // Copy WHOLE pivot ROW to the other device.
            for(int i= r*B; i<(r+1)*B && i<n; i++)
                cudaMemcpyPeer(device_Dist_1+i*n,1, device_Dist+i*n,0 , n*sizeof(unsigned int));
        
            #ifdef TIME 
                float Commtime_Phase1;
                cudaEventRecord(Commstop);
                cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
                cudaEventElapsedTime(&Commtime_Phase1, Commstart, Commstop);
                // printf("Phase1 mem copy took %.8f seconds\n",Commtime_Phase1/1000);
                Total_comm_time += Commtime_Phase1;
            #endif


        }
        else{       // Device 1 do then copy to the other.
            cudaSetDevice(1);
            // printf("Pivot at GPU 1!\n");
            cal_1<<< 1, num_threads , sizeof(int)*B*B>>> (device_0_Boundary, device_Dist_1, n, B, r,   r, r);
            // Copy pivot ROW to the other device.
            #ifdef TIME 
                cudaEventRecord(Commstart_device_1);
            #endif
            for(int i= r*B; i<(r+1)*B && i<n; i++)
                cudaMemcpyPeer(device_Dist+i*n,0, device_Dist_1+i*n,1 , n*sizeof(unsigned int));

            #ifdef TIME 
                float Commtime_Phase1;
                cudaEventRecord(Commstop_device_1);
                cudaEventSynchronize(Commstop_device_1); // WAIT until 'stop' complete.
                cudaEventElapsedTime(&Commtime_Phase1, Commstart_device_1, Commstop_device_1);
                // printf("Phase1 mem copy took %.8f seconds\n",Commtime_Phase1/1000);
                Total_comm_time += Commtime_Phase1;
            #endif

        }

    
        #ifdef DEBUG_DEVICE_DIST
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After phase1, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        #ifdef DEBUG_DEVICE_DIST1
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist_1, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After phase1, device_Dist_1: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif



        

        /* ----------- Phase 2 ------------- */
        cudaSetDevice(0);
        // Compute four sub-phase
        if(r*B < device_0_Boundary){

            // TODO : Modify cal() and cal3() : Need to pass boundary into!!
            // 2-1 
            if(r !=0){
                dim3 nB(1,r); 
                cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_0_Boundary, device_Dist, n,  B, r,         r, 0);
            }
            // 2-2
            if(round -r-1 !=0){
                dim3 nB(1,round - r - 1); 
                cal3<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_0_Boundary, device_Dist, n, B, r,       r, r + 1);
            }
            //2-3
            if(r!=0){
                dim3 nB(r,1); 
                cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_0_Boundary, device_Dist, n, B, r,             0, r);
            }
        
            // 2-4
            if(round-r-1 !=0) {
                dim3 nB(round - r - 1,1); 
                cal3<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_0_Boundary, device_Dist, n, B, r,  r + 1, r);
            }
            
        }
        // Compute ONLY 2-3
        else{
            //2-3
            if(r!=0){
                dim3 nB(r,1); 
                cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_0_Boundary, device_Dist, n, B, r,             0, r);
            }
        }

        cudaSetDevice(1);
        // Compute ONLY 2-4
        if(r*B < device_0_Boundary){   
            // 2-4
            if(round-r-1 !=0) {
                dim3 nB(round - r - 1,1); 
                cal3_1<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_0_Boundary, device_Dist_1, n, B, r,  r + 1, r);
            }           
        }
        // Compute four sub-phase
        else{
            // 2-1 
            if(r !=0){
                dim3 nB(1,r); 
                cal3_1<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_0_Boundary, device_Dist_1, n,  B, r,         r, 0);
            }
            // 2-2
            if(round -r-1 !=0){
                dim3 nB(1,round - r - 1); 
                cal3_1<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_0_Boundary, device_Dist_1, n, B, r,       r, r + 1);
            }
            //2-3
            if(r!=0){
                dim3 nB(r,1); 
                cal3_1<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_0_Boundary, device_Dist_1, n, B, r,             0, r);
            }
        
            // 2-4
            if(round-r-1 !=0) {
                dim3 nB(round - r - 1,1); 
                cal3_1<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_0_Boundary, device_Dist_1, n, B, r,  r + 1, r);
            }
        }        


        #ifdef DEBUG_DEVICE_DIST
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After PHASE 2, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        #ifdef DEBUG_DEVICE_DIST1
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist_1, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After PHASE 2, device_Dist_1: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        
        /* ----------- Phase 3 ------------- */
       
            cudaSetDevice(0);
            if(r != 0){
                dim3 nB(r,r); 
                // cal3<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_Dist, n, B, r, 0, 0, r, r);
                cal3<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_0_Boundary, device_Dist, n, B, r, 0, 0);
            }
    
            if(r !=0 && (round-r-1) !=0){
                dim3 nB(r,(round-r-1)); 
                // cal3<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r,   0, r + 1,   round - r - 1, r);
                cal3<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_0_Boundary,device_Dist, n, B, r,   0, r + 1);
            }
            
            if(r !=0 && round-r-1 !=0){
                dim3 nB((round-r-1),r); 
                // cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r, r + 1, 0, r, round - r - 1);
                cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_0_Boundary,device_Dist, n, B, r, r + 1, 0);
            }
    
            if(round-r-1 !=0){
                dim3 nB_p3(round - r - 1, round - r - 1);
                // cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_Dist, n, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
                cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_0_Boundary,device_Dist, n, B, r, r + 1, r + 1); 
            }

            cudaSetDevice(1);
            if(r != 0){
                dim3 nB(r,r); 
                // cal3<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_Dist, n, B, r, 0, 0, r, r);
                cal3_1<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_0_Boundary,device_Dist_1, n, B, r, 0, 0);
            }
    
            if(r !=0 && (round-r-1) !=0){
                dim3 nB(r,(round-r-1)); 
                // cal3<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r,   0, r + 1,   round - r - 1, r);
                cal3_1<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_0_Boundary,device_Dist_1, n, B, r,   0, r + 1);
            }
            
            if(r !=0 && round-r-1 !=0){
                dim3 nB((round-r-1),r); 
                // cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r, r + 1, 0, r, round - r - 1);
                cal3_1<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_0_Boundary, device_Dist_1, n, B, r, r + 1, 0);
            }
    
            if(round-r-1 !=0){
                dim3 nB_p3(round - r - 1, round - r - 1);
                // cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_Dist, n, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
                cal3_1<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_0_Boundary,device_Dist_1, n, B, r, r + 1, r + 1); 
            }

        #ifdef DEBUG_DEVICE_DIST
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After PHASE3, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }        
        #endif

        #ifdef DEBUG_DEVICE_DIST1
            cudaMemcpy(Dist_1, device_Dist_1, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After PHASE3, device_Dist_1: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }        
        #endif
          
        
    }   // end for(r=0; r<round; r++)

    #ifdef TIME 
        cudaSetDevice(0);
        cudaEventRecord(Commstart);
    #endif

    // Independently copy back to CPU
    cudaMemcpyAsync(Dist, device_Dist, n*device_0_Boundary*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(Dist+device_0_Boundary*n, device_Dist_1+device_0_Boundary*n, ( n*n - n*device_0_Boundary) *sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
   
    #ifdef TIME 
        float Commtime_D2H;
        cudaEventRecord(Commstop);
        cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
        cudaEventElapsedTime(&Commtime_D2H, Commstart, Commstop);
        printf("D2H copy took %.8f seconds\n",Commtime_D2H/1000);
        // printf("Took %.8f milliseconds",time);
        Total_comm_time += Commtime_D2H;
        printf("Communication %.8f seconds\n",Total_comm_time/1000);
    #endif
   

    #ifdef TIME 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // WAIT until 'stop' complete.
        float time;
        cudaEventElapsedTime(&time, start, stop);
        // printf("Took %.8f milliseconds",time);
        printf("Computation(raw): Took %.8f seconds\n",(time)/1000);
        printf("Computation: Took %.8f seconds\n",(time-Total_comm_time)/1000);
    #endif

}

/* ================  For Small n ====================== */
/* Define small n cal & cal3 */
__global__ void cal_small(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){
       
        __shared__ int S[32*32*3];
        int i = block_start_x*B + threadIdx.y;
        int j = block_start_y*B + threadIdx.x;
    
        if(i<n && j<n){
            // S[ (i%B)*B + (j%B)  ] = device_Dist[i*n + j];
            S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
            // S[Addr(0, (i%B), (j%B), B)] = device_Dist[Addr(0,i,j,n)];
            // S[ (i%B)*(B+1) + (j%(B+1))  ] = device_Dist[i*n + j];
            
            // __syncthreads();
    
                // This for-loop CANNOT be serialize!
                // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ 
                    __syncthreads();
                    // if (S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                    //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)];
                    // } 
                    S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)]);                                  
                                                          
                }
                device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
        }// end if(i<n && j<n )
}
    
__global__ void cal3_small(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){
    
        __shared__ int S[32*32*3];
        // int i = block_start_y* B + blockIdx.y * B + threadIdx.y;
        // int j = block_start_x* B + blockIdx.x * B + threadIdx.x;
        int i = block_start_x* B + blockIdx.x * B + threadIdx.y;
        int j = block_start_y* B + blockIdx.y * B + threadIdx.x;
    
    
        // S[Addr(1, threadIdx.y, ((Round*B + threadIdx.x)%B), B)] = device_Dist[Addr(0,i,(Round*B + threadIdx.x),n)];
        // S[Addr(2, ((Round*B + threadIdx.y)%B), threadIdx.x, B)] = device_Dist[Addr(0,(Round*B + threadIdx.y),j,n)];
    
        if(i<n && (Round*B + threadIdx.x) <n) S[Addr(1, threadIdx.y, ((Round*B + threadIdx.x)%B), B)] = device_Dist[Addr(0,i,(Round*B + threadIdx.x),n)];
        if(j<n && (Round*B + threadIdx.y)<n) S[Addr(2, ((Round*B + threadIdx.y)%B), threadIdx.x, B)] = device_Dist[Addr(0,(Round*B + threadIdx.y),j,n)];
        
    
        if(i<n && j<n){
        // For each thread, calculate one edge.
            S[ Addr(0,threadIdx.y, threadIdx.x, B)  ] = device_Dist[Addr(0,i,j,n)];
            __syncthreads();
    
                // This for-loop CANNOT be parallelize!
                // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                /// KEY!! Don't USE % on K.
                for (int iter = 0; iter<B &&  Round*B+iter <n; iter++){ //k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                    // __syncthreads();
                            // if (S[Addr(1, (i%B), (k%B), B)]+ S[Addr(2, (k%B), (j%B), B)]  < S[Addr(0, (i%B), (j%B), B)] ) {
                            //     S[Addr(0, (i%B), (j%B), B)] = S[Addr(1, (i%B), (k%B), B)]+ S[Addr(2, (k%B), (j%B), B)];
                            // }
                                    // i ,  k                               // k ,  j                            // i ,  j
                            // if (S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                            //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)];
                            // }
                            S[Addr(0,threadIdx.y, threadIdx.x, B)] = min(S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)], S[Addr(0,threadIdx.y, threadIdx.x, B)] );
                }
                device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
        }
}


void block_FW_MultiGPU_Old(int B){

    printf("Small n: \n");
    printf("Blocking factor: %d (num of pixel(adj entries) in a Block)\n",B);
    printf(" %d * %d block\n",B,B);
    int round = ceil(n, B);

    // int cur_device_number;

    // cudaMemcpy();
    // int *device_Dist;
    int *device_Dist;
    int *device_Dist_1;

    int canGPU0AccessGPU1, canGPU1AccessGPU0;


    #ifdef TIME 
        float Total_comm_time = 0;
        float Commtime;
        cudaEvent_t Commstart, Commstop;
        cudaEventCreate(&Commstart);
        cudaEventCreate(&Commstop);
    #endif


	#pragma omp parallel num_threads(NUM_THREAD)  //reduction(+:pixels)
	{
        int omp_id, omp_thread_num;
        omp_id = omp_get_thread_num();
        omp_thread_num = omp_get_num_threads();

        if(omp_id==0){
            cudaSetDevice(0);
            cudaDeviceCanAccessPeer ( &canGPU0AccessGPU1, device_0, device_1 );
            if(canGPU0AccessGPU1==1){
                 printf("Can 0 access 1? %d\n",canGPU0AccessGPU1);

                cudaDeviceEnablePeerAccess ( device_1, cudaEnablePeerAccess_Flags );
                cudaMalloc(&device_Dist,   n * n* sizeof(unsigned int));
                #ifdef TIME
                    cudaEventRecord(Commstart);
                #endif
                cudaMemcpyAsync(device_Dist,   Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);
                printf("omp t%d allocate & copy gpu 0\n",omp_id);
            }
            else{
                printf("Error, gpu 0 cannot directly access gpu 1\n");
                // return 2;
            }
        }
        else{
            cudaSetDevice(1);
            cudaDeviceCanAccessPeer ( &canGPU1AccessGPU0, device_1, device_0 );
            if(canGPU1AccessGPU0==1){
                printf("Can 1 access 0? %d\n",canGPU1AccessGPU0);

                cudaDeviceEnablePeerAccess ( device_0, cudaEnablePeerAccess_Flags );
                // cudaGetDevice(&cur_device_number);
                cudaMalloc(&device_Dist_1, n * n* sizeof(unsigned int));
                // cudaMemcpyAsync(device_Dist_1, Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);
                printf("omp t%d allocate & copy gpu 1\n",omp_id);
            }
            else{
                printf("Error, gpu 1 cannot directly access gpu 0\n");
                // return 2;
            }
        }
    }

    #ifdef TIME 
        cudaEventRecord(Commstop);
        cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
        cudaEventElapsedTime(&Commtime, Commstart, Commstop);
        // printf("Took %.8f milliseconds",time);
        Total_comm_time += Commtime;
    #endif

    #ifdef TIME 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif



    // 2*2 threadIdx.x from 0 to 1, Idx.y from 0 to 1
    dim3 num_threads(B,B);





    for (int r = 0; r < round; ++r) {

        #ifdef DEBUG_DIST
            printf("========== Round %d ================\n",r);
            // print_Dist(n);
        #endif
        // printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        // EX: 3*3 Blocks. At iteration k (round r), send D(r,r)   
        // cal<<< 1, num_threads , sizeof(int)*B*(B+1)>>> (device_Dist, n, B, r,   r, r,    1, 1);
        cudaSetDevice(0);
        cal_small<<< 1, num_threads , sizeof(int)*B*B>>> (device_Dist, n, B, r,   r, r);
        // cudaDeviceSynchronize();

        // // printf("round %d   Phase1: (%d, %d), Each row copy: %d entries. \n",r, r, r, min(B, n-r*B));
        // for(int i= r*B; i<(r+1)*B && i<n ; i++){
        //     // printf("Acutal starting location: (%d, %d). MEM[%d]\n",i, r*B, (i*n+r*B));
        //     cudaMemcpyPeer(device_Dist_1+(i*n+r*B),1, device_Dist+(i*n+r*B),0, min(B, n-r*B)*sizeof(unsigned int));
        // }

        // cudaDeviceSynchronize();
        #ifdef DEBUG_DEVICE_DIST
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After phase1, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        
        /* Phase2 */
        // Width: j direction
        // Height: i direction
        ////////////// WIDTH blocks (height == 1) /////////////////

        // GPU 0 
        // 2-1 
        if(r !=0){
            dim3 nB(1,r); 
            cal3_small<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist, n,  B, r,         r, 0);
        }
        // 2-2
        if(round -r-1 !=0){
            dim3 nB(1,round - r - 1); 
            cal3_small<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,       r, r + 1);
        }
        //2-3
        if(r!=0){
            dim3 nB(r,1); 
            cal3_small<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist, n, B, r,             0, r);
       }
   
       // 2-4
       if(round-r-1 !=0) {
           dim3 nB(round - r - 1,1); 
           cal3_small<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,  r + 1, r);
       }

        #ifdef DEBUG_DEVICE_DIST
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After gpu 0, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
        #endif

        //////////// HEIGHT blocks   (width == 1)   /////////////
        // // GPU 1
        // cudaSetDevice(1);
        // // 2-3
        // if(r!=0){
        //      dim3 nB(r,1); 
        //      cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist_1, n, B, r,             0, r);
        // }
    
        // // 2-4
        // if(round-r-1 !=0) {
        //     dim3 nB(round - r - 1,1); 
        //     cal3<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_Dist_1, n, B, r,  r + 1, r);
        // }

        // // Copy device_Dist_1 to Dist_1 and print out!
        // #ifdef DEBUG_DEVICE_DIST1
        //     cudaDeviceSynchronize();
        //     cudaMemcpy(Dist_1, device_Dist_1, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        //     printf("After gpu 1, device_Dist_1: \n");
        //     for(int i=0; i<n; i++){
        //         for(int j=0; j<n; j++){
        //             if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
        //             else printf("%d ",Dist_1[i*n+j]);
        //         }
        //     }
        // #endif


        // PHASE 2 COPY From gpu0 to gpu1

        #ifdef TIME 
            cudaEventRecord(Commstart);
        #endif

        cudaMemcpyPeerAsync(device_Dist_1,1, device_Dist,0, n*n*sizeof(unsigned int));
        cudaDeviceSynchronize();

        #ifdef TIME 
            cudaEventRecord(Commstop);
            cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
            cudaEventElapsedTime(&Commtime, Commstart, Commstop);
            // printf("Took %.8f milliseconds",time);
            Total_comm_time += Commtime;
        #endif

        
    
        // Copy device_Dist to Dist_1 and print out!

        #ifdef DEBUG_DEVICE_DIST
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After Copy from gpu 1 to gpu 0, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }        
        #endif
        /* Phase 3*/ // => USE 2D block!
        // 計算其他的 block
        // 和pivot block 在 x 軸和 y 軸都沒有交集的 blocks！
       
        // 3-1:
        // From (0,0) do (r, r) Blocks.
        int _3_1_block_idx_j = 0;
        int _3_1_0_start_i = 0;
        int _3_1_0_block_height = ceil(r , 2);
        // int 3_4_0_block_width = r;
        int _3_1_1_start_i =  _3_1_0_start_i + _3_1_0_block_height ;
        int _3_1_1_block_height = r - _3_1_0_block_height;

        cudaSetDevice(0);
        if(r != 0){
            dim3 nB(_3_1_0_block_height,r); 
            cal3_small<<< nB, num_threads            , sizeof(int)*B*B*3  >>>(device_Dist, n, B, r, _3_1_0_start_i, _3_1_block_idx_j);
        }

        cudaSetDevice(1);
        if(r != 0 && _3_1_1_block_height!=0  ){
            dim3 nB(_3_1_1_block_height,r); 
            cal3_small<<< nB, num_threads            , sizeof(int)*B*B*3  >>>(device_Dist_1, n, B, r, _3_1_1_start_i, _3_1_block_idx_j);
        }

        
        
        // 3-2
        // From (0, r+1)  do (r * round-r-1 ) BLOCKS
        int _3_2_block_idx_j = r+1;
        int _3_2_0_start_i = 0; // Note, in Block_Idx, NOT Actual index i in Dist!
        int _3_2_0_block_height = ceil(r , 2);
        // int 3_2_0_block_width = round-r-1 ;
        int _3_2_1_start_i =  _3_2_0_start_i + _3_2_0_block_height ;
        int _3_2_1_block_height = r - _3_2_0_block_height; 
        // int _3_2_block_width = (round-r-1 );

        cudaSetDevice(0);
        if(r !=0 && (round-r-1) !=0 && _3_2_block_idx_j<n  ){
            dim3 nB(_3_2_0_block_height,(round-r-1)); 
            cal3_small<<< nB, num_threads       , sizeof(int)*B*B*3  >>>(device_Dist, n, B, r,   _3_2_0_start_i, _3_2_block_idx_j);
        }

        cudaSetDevice(1);
        if(r !=0 && (round-r-1) !=0 && _3_2_block_idx_j<n && _3_2_1_block_height!=0   ){
            dim3 nB(_3_2_1_block_height,(round-r-1)); 
            cal3_small<<< nB, num_threads       , sizeof(int)*B*B*3  >>>(device_Dist_1, n, B, r,   _3_2_1_start_i, _3_2_block_idx_j);
        }


        // 3-3
        // From (r+1, 0)  DO (round-r-1) * r Blocks!
        int _3_3_block_idx_j = 0;
        int _3_3_0_start_i = r+1; // Note, in Block_Idx, NOT Actual index i in Dist!
        int _3_3_0_block_height = ceil(round-r-1 , 2);
        // int 3_4_0_block_width = r;
        int _3_3_1_start_i =  _3_3_0_start_i + _3_3_0_block_height ;
        int _3_3_1_block_height = (round-r-1) - _3_3_0_block_height; 

        cudaSetDevice(0);
        if(r !=0 && round-r-1 !=0){
            dim3 nB(_3_3_0_block_height,r); 
            cal3_small<<< nB  ,num_threads       , sizeof(int)*B*B*3  >>>(device_Dist, n, B, r, _3_3_0_start_i, _3_3_block_idx_j );
        }

        cudaSetDevice(1);
        if(r !=0 && round-r-1 !=0  && _3_3_1_block_height!=0  ){
            dim3 nB(_3_3_1_block_height,r); 
            cal3_small<<< nB  ,num_threads       , sizeof(int)*B*B*3  >>>(device_Dist_1, n, B, r, _3_3_1_start_i, _3_3_block_idx_j );
        }



        // 3-4:
        // From (r+1, r+1)  do (round-r-1) * (round-r-1) blocks
        int _3_4_block_idx_j = r+1;
        int _3_4_0_start_i = r + 1;
        int _3_4_0_block_height = ceil(round-r-1 , 2);
        // int 3_4_0_block_width = (round-r-1);

        // 3-4-1: from (r+1+ ceil(round-r-1, 2), r+1)  Compute {(round-r-1) - ceil(round-r-1 , 2)}  * (round-r-1) Blocks.
        int _3_4_1_start_i =  _3_4_0_start_i + _3_4_0_block_height;
        int _3_4_1_block_height = (round-r-1) - _3_4_0_block_height;
        // int 3_4_1_block_width = (round-r-1);
        
        // 3-4-0: from (r+1, r+1).   Compute ceil(round-r-1 , 2) * (round-r-1) Blocks.
        cudaSetDevice(0);
        if(round-r-1 !=0  && _3_4_block_idx_j<n ){
            dim3 nB_p3(_3_4_0_block_height, round - r - 1);
            cal3_small<<< nB_p3, num_threads, sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,  _3_4_0_start_i, _3_4_block_idx_j); 
        }
        

        cudaSetDevice(1);
        if(round-r-1 !=0 && _3_4_block_idx_j<n && _3_4_1_block_height!=0   ){
            dim3 nB_p3(_3_4_1_block_height, round - r - 1);
            cal3_small<<< nB_p3, num_threads, sizeof(int)*B*B*3 >>>(device_Dist_1, n, B, r, _3_4_1_start_i, _3_4_block_idx_j); 
        }

        cudaDeviceSynchronize();
        cudaSetDevice(0);
  

        #ifdef TIME 
            cudaEventRecord(Commstart);
        #endif


        /* --------- Copy: gpu 1-> gpu0 ----------- */
 
        // 3-1-1
        if(r !=0  && _3_1_1_block_height!=0 ){
            #ifdef DEBUG_PHASE3
                printf("round %d   Phase3-1: GPU 1 copy from (%d, %d), Each row copy: %d entries. \n",r, _3_1_1_start_i, _3_1_block_idx_j,  (r-0)*B );
            #endif
            for(int i= _3_1_1_start_i*B ; i<( _3_1_1_start_i + _3_1_1_block_height) *B && i<n  ; i++){ // row-wise copy. from  (0, r) Block_width = 1, Block_height=r
                #ifdef DEBUG_PHASE3
                    printf("Actual from (%d, %d), MEM[%d]. \n",i, _3_1_block_idx_j*B, (i*n+ _3_1_block_idx_j*B));
                #endif
                cudaMemcpyPeerAsync(device_Dist+(i*n+ _3_1_block_idx_j*B),0, device_Dist_1+(i*n+ _3_1_block_idx_j*B),1 , (r-0)*B *sizeof(unsigned int));
            }
        }        
        // 3-2-1
        if(r !=0 && (round-r-1) !=0  && _3_2_block_idx_j<n && _3_2_1_block_height!=0 ){
            #ifdef DEBUG_PHASE3
                printf("round %d   Phase3-2: GPU 1 copy from (%d, %d), Each row copy: %d entries. \n",r, _3_2_1_start_i, _3_2_block_idx_j,  (n-_3_2_block_idx_j*B)  );
            #endif
            for(int i= _3_2_1_start_i*B ; i<( _3_2_1_start_i + _3_2_1_block_height) *B && i<n  ; i++){ // row-wise copy. from  (0, r) Block_width = 1, Block_height=r
                #ifdef DEBUG_PHASE3
                    printf("Actual from (%d, %d), MEM[%d]. \n",i, _3_2_block_idx_j*B, i*n+ _3_2_block_idx_j*B );
                #endif
                cudaMemcpyPeerAsync(device_Dist+(i*n+ _3_2_block_idx_j*B),0, device_Dist_1+(i*n+ _3_2_block_idx_j*B),1 , (n-_3_2_block_idx_j*B) *sizeof(unsigned int));
            }
        } 

        // 3-3-1
        if(r !=0 && (round-r-1) !=0 && _3_3_1_block_height!=0){
            #ifdef DEBUG_PHASE3
                printf("round %d   Phase3-3: GPU 1 copy from (%d, %d), Each row copy: %d entries. \n",r, _3_3_1_start_i, _3_3_block_idx_j,  (r-0)*B );
            #endif
            for(int i= _3_3_1_start_i*B ; i<( _3_3_1_start_i + _3_3_1_block_height) *B && i<n  ; i++){ // row-wise copy. from  (0, r) Block_width = 1, Block_height=r
                #ifdef DEBUG_PHASE3
                    printf("Actual from (%d, %d), MEM[%d]. \n",i, _3_3_block_idx_j*B,  (i*n+ _3_3_block_idx_j*B));
                #endif
                cudaMemcpyPeerAsync(device_Dist+(i*n+ _3_3_block_idx_j*B),0, device_Dist_1+(i*n+ _3_3_block_idx_j*B),1 , (r-0)*B *sizeof(unsigned int));
            }
        } 

        // 3-4-1
        if(round-r-1 !=0 && _3_4_block_idx_j<n && _3_4_1_block_height!=0    ){
            #ifdef DEBUG_PHASE3
                printf("round %d   Phase3-4: GPU 1 copy from (%d, %d), Each row copy: %d entries. \n",r, _3_4_1_start_i, r+1, (n-(r+1)*B ));
            #endif
            for(int i= _3_4_1_start_i*B; i<  (_3_4_1_start_i+ _3_4_1_block_height)*B  && i<n   ; i++){ // row-wise copy. from  (0, r) Block_width = 1, Block_height=r
                #ifdef DEBUG_PHASE3
                    printf("Actual from (%d, %d), MEM[%d]. \n",i, (r+1)*B, (i*n+ (r+1)*B));
                #endif
                cudaMemcpyPeerAsync(device_Dist+(i*n+ _3_4_block_idx_j*B),0, device_Dist_1+(i*n+ _3_4_block_idx_j*B),1, (n- _3_4_block_idx_j*B )*sizeof(unsigned int));
            }
        }


        #ifdef DEBUG_DEVICE_DIST
            // Copy device_Dist to Dist_1 and print out!
            cudaMemcpy(Dist_1, device_Dist, n*n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("After phase3 copy from gpu1 to gpu0, device_Dist: \n");
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    if(j== n-1) printf("%d\n",Dist_1[i*n+j]);
                    else printf("%d ",Dist_1[i*n+j]);
                }
            }
            printf("\n end round %d \n ===============================\n",r);
        #endif
        
        cudaDeviceSynchronize();

        #ifdef TIME 
            cudaEventRecord(Commstop);
            cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
            cudaEventElapsedTime(&Commtime, Commstart, Commstop);
            // printf("Took %.8f milliseconds",time);
            Total_comm_time += Commtime;
         #endif

        
    }   // end for(r=0; r<round; r++)

    #ifdef TIME 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // WAIT until 'stop' complete.
        float time;
        cudaEventElapsedTime(&time, start, stop);
        // printf("Took %.8f milliseconds",time);
        printf("Computation: Took %.8f seconds\n",(time-Total_comm_time)/1000);
    #endif





    #ifdef TIME 
        cudaEventRecord(Commstart);
    #endif

    cudaMemcpyAsync(Dist, device_Dist, n * n *sizeof(unsigned int), cudaMemcpyDeviceToHost);
   
   
    #ifdef TIME 
        cudaEventRecord(Commstop);
        cudaEventSynchronize(Commstop); // WAIT until 'stop' complete.
        cudaEventElapsedTime(&Commtime, Commstart, Commstop);
        // printf("Took %.8f milliseconds",time);
        Total_comm_time += Commtime;
        printf("Communication %.8f seconds\n",Total_comm_time/1000);
    #endif
   
    // cudaDeviceSynchronize(); // TODO : Can remove this
}


