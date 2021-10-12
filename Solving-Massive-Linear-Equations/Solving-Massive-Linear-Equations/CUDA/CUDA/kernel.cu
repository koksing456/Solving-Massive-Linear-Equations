#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cmath"
#include <omp.h>
using namespace std;
#define N 11
#define THREADS_PER_BLOCK 1024

double* create1DArray();

double* malloc_matrix(const int a, const int b) {
    return (double*)malloc(sizeof(double*) * a * b);
}

void print(double* mat)
{
    printf("Below is the matrix of linear equation: \n");
    int k = 0;
    for (int i = 0; i < N; i++, printf("\n"))
        for (int j = 0; j <= N; j++)
        {
            printf("%lf ", mat[k]);
            k++;
        }
    printf("\n");
}

void printSolution(double* x) {
    printf("\nSolution for the system:\n");
    for (int i = 0; i < N; i++) {
        int k = (i + 1) * (N + 1);


        printf("%lf\n", x[k - 1]);
    }
}

__global__ void replace_zero_gpu(double* AB, int rows, int columns, int column) {
    if (fabs(AB[column * columns + column]) <= 1e-4) {

        int row = column;
        for (; row < rows; row++) {
            if (fabs(AB[row * columns + column]) > 1e-4)
                break;
        }
        int threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if (threadId + column >= columns)
            return;

        int zero = column * columns + column + threadId;
        int chosen = row * columns + column + threadId;
        AB[zero] += AB[chosen];
    }
}


__global__ void column_elimination_gpu(double* AB, int rows, int columns, int column) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId >= (rows - 1 - column) * (columns - column))
        return;

    int el_row = column + threadId / (columns - column) + 1;
    int el_col = column + threadId % (columns - column);
    int el = el_col + el_row * columns;
    int upper_el = el_col + column * columns;

    int main_el = column + column * columns;
    int main2_el = column + el_row * columns;
    double f = AB[main2_el] / AB[main_el];

    AB[el] -= f * AB[upper_el];
}

__global__ void multiple_column(double* AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;
    AB[(threadId * columns) + row] *= AB[columns * (row + 1) - 1];
}

__global__ void reverse_row_elimination(double* AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;
    int cols = columns - 2 - row;

    int start_index = row * columns + row + 1;

    int j = cols % 2;
    for (int i = cols / 2; i > 0; i /= 2) {
        if (threadId >= i)
            return;

        AB[start_index + threadId] += (AB[start_index + threadId + i + j]);
        AB[start_index + threadId + i + j] = 0;
        if (j == 1)
            i++;
        j = i % 2;
        __syncthreads();
    }

    int x_el = (row + 1) * columns - 1;
    int diag_el = row * columns + row;

    if (diag_el + 1 != x_el) {
        AB[x_el] -= AB[diag_el + 1];
        AB[diag_el + 1] = 0.0;
    }

    AB[x_el] /= AB[diag_el];
    AB[diag_el] = 1.0;
}

__global__ void sum_row(double* AB, int rows, int columns, int row) {
    int threadId = threadIdx.x;

    int j = columns % 2;
    for (int i = columns / 2; i > 0; i /= 2) {
        if (threadId >= i)
            return;

        AB[threadId] += AB[threadId + i + j];
        __syncthreads();
        if (j == 1)
            i++;
        j = i % 2;
    }
}


void start_gaussian_elimination_gpu(double* AB, int rows, int cols) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double* AB_gpu;

    cudaMalloc(&AB_gpu, sizeof(double) * rows * cols);
    cudaMemcpy(AB_gpu, (void*)AB, sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
    cudaEventRecord(start);

    for (int column = 0; column < cols - 1; column++) {
        replace_zero_gpu << <1, THREADS_PER_BLOCK >> > (AB_gpu, rows, cols, column);
        cudaThreadSynchronize();

        column_elimination_gpu << < 1, THREADS_PER_BLOCK >> > (AB_gpu, rows, cols, column);
        cudaThreadSynchronize();
    }

    for (int row = rows - 1; row >= 0; row--) {
        reverse_row_elimination << <1, cols >> > (AB_gpu, rows, cols, row);
        multiple_column << <1, row >> > (AB_gpu, rows, cols, row);

        cudaThreadSynchronize();
    }

    cudaMemcpy(AB, (void*)AB_gpu, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost);

    cudaFree(AB_gpu);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Effective Bandwidth (GB/s): %.11fn", milliseconds / 1000);
}


int main(int argc, char** argv) {
    int size = N;
    srand(124);
    double* AB = create1DArray();

    print(AB);

    start_gaussian_elimination_gpu(AB, size, size + 1);

    printf("\n\n");

    printSolution(AB);

    return 0;
}

double* create1DArray()
{
    double* matrix_ab = malloc_matrix(N, N + 1);
    int k = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N + 1; j++) {
            matrix_ab[k] = rand() % 5;

            if (i == j)
            {
                matrix_ab[k] *= -1;
            }

            k++;
        }
    }
    return matrix_ab;
}
