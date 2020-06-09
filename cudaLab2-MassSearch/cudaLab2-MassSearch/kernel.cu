
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h> 
#include <random>

using namespace std;

__global__ void massSearchKernel(char* buf, char* rows, bool* result, int bufLength, int rowsCount, int rowsLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= bufLength || j >= rowsCount)
    {
        return;
    }
    result[j * bufLength + i] = true;
    for (int k = 0; k < rowsLength; k++)
    {
        if (buf[i + k] != rows[j * rowsLength + k])
        {
            result[j * bufLength + i] = false;
            break;
        }
    }
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t massSearchWithCuda(char* buf, char* rows, bool* result, int bufLength, int rowsCount, int rowsLength)
{
    char* dev_buf = 0;
    char* dev_rows = 0;
    bool* dev_result = 0;
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_buf, bufLength * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_rows, rowsCount * rowsLength * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_result, rowsCount * bufLength * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_buf, buf, bufLength * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_rows, rows, rowsCount * rowsLength * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }


    dim3 blockSize = dim3(32, 32, 1);
    dim3 gridSize = dim3(bufLength / 32 + 1, rowsCount / 32 + 1, 1);

    // Launch a kernel on the GPU with one thread for each element.
    massSearchKernel <<< gridSize, blockSize >>> (dev_buf, dev_rows, dev_result, bufLength, rowsCount, rowsLength);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "massSearchKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching massSearchKernel!\n", cudaStatus);
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_result, rowsCount * bufLength * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaFree(dev_buf);
    cudaFree(dev_rows);
    cudaFree(dev_result);

    return cudaStatus;
}

void simpleMassSearch(char* buf, char* rows, bool* result, int bufLength, int rowsCount, int rowsLength) 
{
    for (int i = 0; i < bufLength; i++)
    {
        for (int j = 0; j < rowsCount; j++)
        {
            result[j * bufLength + i] = true;
            for (int k = 0; k < rowsLength; k++)
            {
                if (buf[i + k] != rows[j * rowsLength + k])
                {
                    result[j * bufLength + i] = false;
                    break;
                }
            }
        }
    }
}

int main()
{
    int N, H, L;


    cout << "Enter the size of buffer: " << endl;
    cin >> H;
    cout << "Enter the number of rows for search: " << endl;
    cin >> N;
    cout << "Enter rows length: " << endl;
    cin >> L;

    // выделяем память и генерируем числа
    char* buf = new char[H];
    // строки для поиска - двумерный массив, представленный как одномерный
    char* rows = new char[N*L];
    // результат двумерный массив, представленный как одномерный, где true на пересечение строки и столбца означает, что в буфере H на позиции h начинается n-ое слово из списка для поиска N
    bool* result = new bool[H*N];
    bool* cudaResult = new bool[H * N];

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0, 1.0);
    
    // сгенерируем простой пример, где словарь будет иметь 2 символа: a, b. Методы поиска будут работать и на полном 8 битном словаре, но генерировать такие данные довольно сложно
    for (int i = 0; i < H; i++)
    {
        if (distribution(generator) < 0.5) 
        {
            buf[i] = 'a';
        }
        else
        {
            buf[i] = 'b';
        }
    }
    
    // слова для поиска будут случайной комбинацией символов a и b длинной L, повторения возможны, но поиск не будет их учитывать и искать одинаковые слова 2 раза
    for (int i = 0; i < N * L; i++)
    {
        if (distribution(generator) < 0.5)
        {
            rows[i] = 'a';
        }
        else
        {
            rows[i] = 'b';
        }
    }

    // cuda реализация
    clock_t start = clock();
    cudaError_t cudaStatus = massSearchWithCuda(buf, rows, cudaResult, H, N, L);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "massSearchWithCuda failed!");
    }
    clock_t end = clock();
    cout << "Cuda time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    // cpu реализация
    start = clock();
    simpleMassSearch(buf, rows, result, H, N, L);
    end = clock();
    cout << "CPU time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    bool check = true;
    for (int i = 0; i < N*H; i++)
    {
        if (cudaResult[i] != result[i]) 
        {
            check = false;
            break;
        }
    }
    cout << "Result is equal? " << (check ? "YES" : "NO") << endl;
    /*for (int i = 0; i < H; i++)
    {
        cout << buf[i];
    }
    cout << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            cout << rows[i * L + j];
        }
        cout << endl;
    }
    cout << "CPU Result: " << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < H; j++)
        {
            cout << result[i * H + j];
        }
        cout << endl;
    }
    cout << "CUDA Result: " << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < H; j++)
        {
            cout << cudaResult[i * H + j];
        }
        cout << endl;
    }*/
    delete[] buf;
    delete[] rows;
    delete[] result;
    delete[] cudaResult;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

