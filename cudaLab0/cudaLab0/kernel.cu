
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h> 

using namespace std;

// Обычное матричное произведение
void simpleMatMul(int* c, int* a, int* b, int rows1, int cols1, int cols2) {
    for (unsigned int i = 0; i < rows1; i++)
    {
        for (unsigned int j = 0; j < cols2; j++)
        {
            c[i * cols2 + j] = 0;
            for (unsigned int k = 0; k < cols1; k++)
            {
                c[i * cols2 + j] += a[i * cols1 + k] * b[k * cols2 + j];
            }
        }
    }
}

__global__ void matMulKernel(int* c, int* a, int* b, int rows1, int cols1, int cols2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows1 || j >= cols2)
    {
        return;
    }
    c[i * cols2 + j] = 0;
    for (int k = 0; k < cols1; k++)
    {
        c[i * cols2 + j] += a[i * cols1 + k] * b[k * cols2 + j];
    }
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t matMulWithCuda(int* c, int* a, int* b, int rows1, int cols1, int cols2)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, rows1 * cols2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_a, rows1 * cols1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_b, cols1 * cols2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, rows1 * cols1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, cols1 * cols2 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // иногда будем выделять лишнего, но я последую общей концепции Cuda, о том, что 1-2 лишних блока не имеют значения(то есть мне лень писать длинный if)
    dim3 blockSize = dim3(32, 32, 1);
    dim3 gridSize = dim3(rows1 / 32 + 1, cols1 / 32 + 1, 1);
    // Launch a kernel on the GPU with one thread for each element.
    matMulKernel <<< gridSize, blockSize >>> (dev_c, dev_a, dev_b, rows1, cols1, cols2);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matMulKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, rows1 * cols2 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


int main()
{
    int rows1;
    int cols1;
    int cols2;

    cout << "Enter the number of rows and columns" << endl;
    cout << "Number of rows for 1 matrix:" << endl;
    cin >> rows1; 
    cout << "Number of columns for 1 matrix:" << endl;
    cin >> cols1;
    cout << "Number of columns for 2 matrix:" << endl;
    cin >> cols2;

    // выделяем память и заполняем массивы
    int* a = new int[rows1 * cols1];
    int* b = new int[cols1 * cols2];
    int* c = new int[rows1 * cols2];

    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols1; j++)
        {
            a[i * cols1 + j] = i * cols1 + j;
        }
    }
    for (int i = 0; i < cols1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            b[i * cols2 + j] = i * cols2 + j;
        }
    }

    clock_t start = clock();
    // Add vectors in parallel.
    cudaError_t cudaStatus = matMulWithCuda(c, a, b, rows1, cols1, cols2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matMulWithCuda failed!");
    }
    clock_t end = clock();
    cout << "Cuda time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    c = new int[rows1 * cols2];

    start = clock();
    simpleMatMul(c, a, b, rows1, cols1, cols2);
    end = clock();
    cout << "CPU time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    /*for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols1; j++)
        {
            cout << a[i * cols1 + j] << ' ';
        }
        cout << endl;
    }
    for (int i = 0; i < cols1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            cout << b[i * cols2 + j] << ' ';
        }
        cout << endl;
    }
    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            cout << c[i * cols2 + j] << ' ';
        }
        cout << endl;
    }*/
    delete[] a;
    delete[] b;
    delete[] c;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

