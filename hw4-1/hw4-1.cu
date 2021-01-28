/*
This version is "NO Streaming" version.

12/16 Try streaming!

*/


#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <time.h>
// #define TIME
// #define CUDA_NVPROF

const int BLOCKING_FACTOR = 32; // 32, 16, 8, 4, 2

const int INF = ((1 << 30) - 1);
// Global var stored in Data Section.
// const int V = 40010;
void input(char* inFileName);
void output(char* outFileName);

void print_ans(int num_V, char* ans_file);

void block_FW(int B);
void block_FW_Large_N(int B);
int ceil(int a, int b);
// void cal(int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

// Shared memory: For each block, each thread brings d[i][j] to s[i][j] !

//

// extern __shared__ int S[];

__device__ inline int Addr(int matrixIdx, int i, int j, int N){
    return( N*N*matrixIdx + i*N + j);
}

// W: width, H: height
// __device__ inline int Addr2(int matrixIdx, int i, int j, int W, int H){
//     return( W*H*matrixIdx + i*W + j);
// }


// TODO: Bank Conflict!

// TRY pahse1: Let thread(Idx.x, Idx.y) access in diagonally! Same WARP NO bank conflict.


// PHASE 1 : ONE Block do k iterations with B*B threads.
// __global__ void cal(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
__global__ void cal(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){
       
    __shared__ int S[32*32];
    int i = block_start_y*B + threadIdx.y;
    int j = block_start_x*B + threadIdx.x;

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
                if (S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                    S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(0, threadIdx.y, iter, B)]+ S[Addr(0, iter, threadIdx.x, B)];
                }                   
                                                      
            }
            device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
    }// end if(i<n && j<n )
}


// Why cal3  don't need sync_threads() and can perform all correct?
// Each thread do k calculation (O(k))
// __global__ void cal3(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
__global__ void cal3(int* device_Dist, int n, int B, int Round, int block_start_x, int block_start_y){

    __shared__ int S[32*32*3];
    int i = block_start_y* B + blockIdx.y * B + threadIdx.y;
    int j = block_start_x* B + blockIdx.x * B + threadIdx.x;


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
                        if (S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                            S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(1, threadIdx.y, iter, B)]+ S[Addr(2, iter, threadIdx.x, B)];
                        }

                        // if (S[Addr(1, threadIdx.y, (k%B), B)]+ S[Addr(2, (k%B), threadIdx.x, B)]  < S[Addr(0,threadIdx.y, threadIdx.x, B)] ) {
                        //     S[Addr(0,threadIdx.y, threadIdx.x, B)] = S[Addr(1, threadIdx.y, (k%B), B)]+ S[Addr(2, (k%B), threadIdx.x, B)];
                        // }
            }
            device_Dist[Addr(0,i,j,n)] = S[Addr(0,threadIdx.y, threadIdx.x, B)];
    }
}



int n, m;
// static int Dist[V][V];
int* Dist;

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
    // B = 32; // 16: faster .(WHY?) communication. MAX: 32
    B = BLOCKING_FACTOR;
    
    // B = 7;
    // int B = 4; // blocking factor.

    // if(n>=5000) block_FW_Large_N(B);
    // else block_FW(B);
    block_FW(B);

    // if(n>=5000) block_FW_Large_N(16);
    // else block_FW(32);


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



    // Communicatoin time: (Memcpy H2D, D2H).
    // printf("Computation Time: %.8f\n",); //GPU Kernel


    // print_ans(n);
    // print_ans(n, argv[3]);
    // output(argv[2]);
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


    FILE* file = fopen(ans_file, "rb");
    int* Ans = (int*)malloc(sizeof(int)*n*n);
    fread(Ans, sizeof(int), n*n, file);

    for(int i=0; i<num_V*num_V; i++){
        if(Dist[i] != Ans[i]){
            printf("Wrong at offset %d, expected %d but get %d\n", i*4, Ans[i], Dist[i]);
            printf("Fron %d to %d , cost: %d\n", (i/n), (i%n), Ans[i] );
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

// 1204: Idea1 : one stream with 9 serialize kernel launch?
// memory to pass to GPU: B, r, r, r, 1, 1. ALL constant! No memory copy.



void block_FW(int B) {

    // printf("Blocking factor: %d (num of pixel(adj entries) in a Block)\n",B);
    // printf(" %d * %d block\n",B,B);
    int round = ceil(n, B);

    // cudaMemcpy();
    int *device_Dist;
    // cudaMalloc(&device_Dist, V * V* sizeof(unsigned int));
    cudaMalloc(&device_Dist, n * n* sizeof(unsigned int));
    


    #ifdef TIME 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif    
    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(device_Dist, Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);

    #ifdef TIME 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // WAIT until 'stop' complete.
        float Comm_time; // H2D
        cudaEventElapsedTime(&Comm_time, start, stop);
        // printf("Took %.8f milliseconds on computation.",time);
    #endif


    // printf("Initial matrix: \n");
    // print_ans(n);

    // 2*2 threadIdx.x from 0 to 1, Idx.y from 0 to 1
    dim3 num_threads(B,B);



    #ifdef TIME 
        cudaEvent_t compt_start, compt_stop;
        cudaEventCreate(&compt_start);
        cudaEventCreate(&compt_stop);
        cudaEventRecord(compt_start);
    #endif

    #ifdef CUDA_NVPROF
    cudaProfilerStart();
    #endif 
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        // EX: 3*3 Blocks. At iteration k (round r), send D(r,r)   
        // cal<<< 1, num_threads , sizeof(int)*B*(B+1)>>> (device_Dist, n, B, r,   r, r,    1, 1);
        // cal<<< 1, num_threads , sizeof(int)*B*B*3>>> (device_Dist, n, B, r,   r, r,    1, 1);
        cal<<< 1, num_threads , sizeof(int)*B*B*3>>> (device_Dist, n, B, r,   r, r);


        /* Phase 2*/
        // cudaProfilerStart(); 
        if(r !=0){
            dim3 nB(1,r); 
            // cal3<<< nB, num_threads      , sizeof(int)*B*B*3>>>(device_Dist, n,  B, r,         r, 0,                             r, 1);
            cal3<<< nB, num_threads      , sizeof(int)*B*B*3>>>(device_Dist, n,  B, r,         r, 0);
        }

        if(round -r-1 !=0){
            dim3 nB(1,round - r - 1); 
            // cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist, n, B, r,       r, r + 1,              round - r - 1, 1);
            cal3<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist, n, B, r,       r, r + 1);
        }
        
        //////////// HEIGHT blocks   (width == 1)   /////////////
        if(r!=0){
             dim3 nB(r,1); 
            //  cal3<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,             0, r,                          1, r);
            cal3<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,             0, r);
        }

        
        if(round-r-1 !=0) {
            dim3 nB(round - r - 1,1); 
            // cal3<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,  r + 1, r,            1, round - r - 1);
            cal3<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,  r + 1, r);
        }
        // cudaProfilerStop(); 


        /* Phase 3*/ // => USE 2D block!
        // 計算其他的 block
        // 和pivot block 在 x 軸和 y 軸都沒有交集的 blocks！

        // cudaProfilerStart(); 
        if(r != 0){
            dim3 nB(r,r); 
            // cal3<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_Dist, n, B, r, 0, 0, r, r);
            cal3<<< nB, num_threads            , sizeof(int)*B*B*3       >>>(device_Dist, n, B, r, 0, 0);
        }

        if(r !=0 && (round-r-1) !=0){
            dim3 nB(r,(round-r-1)); 
            // cal3<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r,   0, r + 1,   round - r - 1, r);
            cal3<<< nB, num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r,   0, r + 1);
        }
        
        if(r !=0 && round-r-1 !=0){
            dim3 nB((round-r-1),r); 
            // cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r, r + 1, 0, r, round - r - 1);
            cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3    >>>(device_Dist, n, B, r, r + 1, 0);
        }

        if(round-r-1 !=0){
            dim3 nB_p3(round - r - 1, round - r - 1);
            // cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_Dist, n, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
            cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3      >>>(device_Dist, n, B, r, r + 1, r + 1); 
        }
        // cudaProfilerStop(); 
    }
    #ifdef CUDA_NVPROF
    cudaProfilerStop(); 
    #endif

    #ifdef TIME 
        cudaEventRecord(compt_stop);
        cudaEventSynchronize(compt_stop); // WAIT until 'stop' complete.
        float compt_time;
        cudaEventElapsedTime(&compt_time, compt_start, compt_stop);
        printf("Computation Time: %.8f seconds\n",compt_time/1000);
    #endif


    #ifdef TIME 
        cudaEventRecord(start);
    #endif   

    cudaMemcpy(Dist, device_Dist, n * n *sizeof(unsigned int), cudaMemcpyDeviceToHost);
    #ifdef TIME 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // WAIT until 'stop' complete.
        float D2H_Comm_time;
        cudaEventElapsedTime(&D2H_Comm_time, start, stop); 
        printf("Memory Copy Time: %.8f seconds\n",  (D2H_Comm_time + Comm_time ) /1000);
    #endif

}



// // For Large n.: Don't use Synchronize.
// // n > 5000
// void block_FW_Large_N(int B) {

//     printf("Blocking factor: %d (num of pixel(adj entries) in a Block)\n",B);
//     printf(" %d * %d block\n",B,B);
//     int round = ceil(n, B);


//     // #ifdef TIME 
//     //     cudaEvent_t start, stop;
//     //     cudaEventCreate(&start);
//     //     cudaEventCreate(&stop);
//     //     cudaEventRecord(start);
//     // #endif


//     // cudaMemcpy();
//     int *device_Dist;
//     // cudaMalloc(&device_Dist, V * V* sizeof(unsigned int));
//     cudaMalloc(&device_Dist, n * n* sizeof(unsigned int));
//     // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
//     cudaMemcpy(device_Dist, Dist, n* n*sizeof(unsigned int), cudaMemcpyHostToDevice);


//     #ifdef TIME 
//         cudaEvent_t start, stop;
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);
//         cudaEventRecord(start);
//     #endif

//     // printf("Initial matrix: \n");
//     // print_ans(n);

//     // 2*2 threadIdx.x from 0 to 1, Idx.y from 0 to 1
//     dim3 num_threads(B,B);


//     /////// CREATE 4 STREAMS ///////////
//     const int num_streams = 4;
//     cudaStream_t streams[num_streams];
//     float *data[num_streams];

//     for (int i = 0; i < num_streams; i++) {
//         cudaStreamCreate(&streams[i]);
//     }

//     // cudaDeviceReset();


//     for (int r = 0; r < round; ++r) {
//         // printf("%d %d\n", r, round);
//         fflush(stdout);
//         /* Phase 1*/
//         // EX: 3*3 Blocks. At iteration k (round r), send D(r,r)   
//         // cal<<< 1, num_threads , sizeof(int)*B*(B+1)>>> (device_Dist, n, B, r,   r, r,    1, 1);
//         cal<<< 1, num_threads , sizeof(int)*B*B>>> (device_Dist, n, B, r,   r, r,    1, 1);

//         // cudaDeviceSynchronize();

//         /* Phase 2*/
//         ////////////// WIDTH blocks (height == 1) /////////////////
//         // if(r !=0){
//         //     dim3 nB(1,r); 
//         //     cal2<<< nB, num_threads      , sizeof(int)*B*B*3>>>(device_Dist, n,  B, r,         r, 0,                             r, 1);
//         // }

//         // if(round -r-1 !=0){
//         //     dim3 nB(1,round - r - 1); 
//         //     cal2<<< nB, num_threads , sizeof(int)*B*B*3>>>(device_Dist, n, B, r,       r, r + 1,              round - r - 1, 1);
//         // }
        
//         // //////////// HEIGHT blocks   (width == 1)   /////////////
//         // if(r!=0){
//         //      dim3 nB(r,1); 
//         //      cal2<<< nB, num_threads , sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,             0, r,                          1, r);
//         // }

        
//         // if(round-r-1 !=0) {
//         //     dim3 nB(round - r - 1,1); 
//         //     cal2<<< nB , num_threads, sizeof(int)*B*B*3 >>>(device_Dist, n, B, r,  r + 1, r,            1, round - r - 1);
//         // }

//         if(r !=0){
//             dim3 nB(1,r); 
//             cal3<<< nB, num_threads      , sizeof(int)*B*B*3,streams[0]>>>(device_Dist, n,  B, r,         r, 0,                             r, 1);
//         }

//         if(round -r-1 !=0){
//             dim3 nB(1,round - r - 1); 
//             cal3<<< nB, num_threads , sizeof(int)*B*B*3,streams[1]>>>(device_Dist, n, B, r,       r, r + 1,              round - r - 1, 1);
//         }
        
//         //////////// HEIGHT blocks   (width == 1)   /////////////
//         if(r!=0){
//              dim3 nB(r,1); 
//              cal3<<< nB, num_threads , sizeof(int)*B*B*3 ,streams[2]>>>(device_Dist, n, B, r,             0, r,                          1, r);
//         }

        
//         if(round-r-1 !=0) {
//             dim3 nB(round - r - 1,1); 
//             cal3<<< nB , num_threads, sizeof(int)*B*B*3,streams[3] >>>(device_Dist, n, B, r,  r + 1, r,            1, round - r - 1);
//         }


//         // cudaDeviceSynchronize();

//         /* Phase 3*/ // => USE 2D block!
//         // 計算其他的 block
//         // 和pivot block 在 x 軸和 y 軸都沒有交集的 blocks！

//         if(r != 0){
//             dim3 nB(r,r); 
//             cal3<<< nB, num_threads            , sizeof(int)*B*B*3     ,streams[0]  >>>(device_Dist, n, B, r, 0, 0, r, r);
//         }

//         if(r !=0 && (round-r-1) !=0){
//             dim3 nB(r,(round-r-1)); 
//             cal3<<< nB, num_threads       , sizeof(int)*B*B*3   ,streams[1] >>>(device_Dist, n, B, r,   0, r + 1,   round - r - 1, r);
//         }
        
//         if(r !=0 && round-r-1 !=0){
//             dim3 nB((round-r-1),r); 
//             cal3<<< nB  ,num_threads       , sizeof(int)*B*B*3   ,streams[2] >>>(device_Dist, n, B, r, r + 1, 0, r, round - r - 1);
//         }

//         if(round-r-1 !=0){
//             dim3 nB_p3(round - r - 1, round - r - 1);
//             cal3<<< nB_p3, num_threads, sizeof(int)*B*B*3   ,streams[3]   >>>(device_Dist, n, B, r, r + 1, r + 1, round - r - 1, round - r - 1); 
//         }

//         // cudaDeviceSynchronize();

//     }

//     // cudaMemcpy(Dist, device_Dist, n * n *sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     #ifdef TIME 
//         cudaEventRecord(stop);
//         cudaEventSynchronize(stop); // WAIT until 'stop' complete.
//         float time;
//         cudaEventElapsedTime(&time, start, stop);
//         // printf("Took %.8f milliseconds",time);
//         printf("Took %.8f seconds",time/1000);
//     #endif
//     cudaMemcpy(Dist, device_Dist, n * n *sizeof(unsigned int), cudaMemcpyDeviceToHost);

// }



