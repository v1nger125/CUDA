
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h> 
#include <random>

using namespace std;

// основной способ сложения данных из статьи nvidia http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// не самый оптимальный приведенный в статье алгоритм, но и не самый худший
__global__ void piCalcKernel(int *d_odata, double* a, double* b, int size)
{
    __shared__ int sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = ((a[i] * a[i] + b[i] * b[i]) < 1 ? 1 : 0);

    __syncthreads();
    for (int j = blockDim.x/2; j > 0; j>>=1)
    {
        if (tid < j)
        {
            sdata[tid] += sdata[tid + j];
        }
        __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t piCalcWithCuda(int* odata, double* a, double* b, int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    int* dev_odata = 0;
    cudaStatus = cudaMalloc((void**)&dev_odata, (size / 1024) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }


    dim3 blockSize = dim3(1024, 1, 1);
    dim3 gridSize = dim3(size/1024, 1, 1);

    // Launch a kernel on the GPU with one thread for each element.
    piCalcKernel <<<gridSize, blockSize >>> (dev_odata, dev_a, dev_b, size);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "piCalcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching piCalcKernel!\n", cudaStatus);
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(odata, dev_odata, (size / 1024) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_odata);

    return cudaStatus;
}

double simplePiCalc(double* a, double* b, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        if (a[i] * a[i] + b[i] * b[i] < 1) 
        {
            sum++;
        }
    }
    return (double) 4*sum / size;
}
int main()
{
    int N;

    cout << "Enter the number of numbers: " << endl;
    cin >> N;

    // выделяем память и генерируем числа
    double* a = new double[N];
    double* b = new double[N];
    int* cudaOut = new int[(N / 1024)];

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (int i = 0; i < N; i++)
    {
        a[i] = distribution(generator);
        b[i] = distribution(generator);
    }

    
    // cuda реализация
    double result = 0;
    clock_t start = clock();
    cudaError_t cudaStatus = piCalcWithCuda(cudaOut, a, b, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "piCalcWithCuda failed!");
    }
    for (int i = 0; i < (N / 1024); i++)
    {
       result += cudaOut[i];
    }
    result = 4 * (double) result / (N - N%1024);
    clock_t end = clock();
    cout << "Cuda time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "Cuda result: " << result << endl;
    // cpu реализация
    result = 0;
    start = clock();
    result = simplePiCalc(a, b, N);
    end = clock();
    cout << "CPU time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "CPU result: " << result << endl;
    delete[] cudaOut;
    delete[] a;
    delete[] b;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

