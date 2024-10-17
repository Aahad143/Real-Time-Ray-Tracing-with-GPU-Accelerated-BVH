#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"


#include <stdio.h>
#include <iostream>

#include <chrono>
#include <vector>

#include <SDL.h>
#undef main

#define N 1000000000

__global__ void initVectors(int* a, int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    a[i] = i;
    b[i] = i;
  }
}

__global__ void addVector(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

inline void cudaVectorAdd1() {
  // Host (CPU) pointers
  int* h_a, * h_b, * h_c;

  // Device (GPU) pointers
  int* d_a, * d_b, * d_c;

  // Length of vector
  size_t size = N * sizeof(float);

  // Allocate host memory
  h_a = (int*)malloc(size);
  h_b = (int*)malloc(size);
  h_c = (int*)malloc(size);

  // Initialize host vectors
  for (int i = 0; i < N; i++) {
    h_a[i] = (rand() * 100) / RAND_MAX;
    h_b[i] = (rand() * 100) / RAND_MAX;
  }

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy vectors from host to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Adding vector in parallel using threads on the GPU
  addVector << < blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N);


  // Copy result back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;

  // Print the first few elements of the result
  int elements_to_print = 10;  // Number of elements to print
  printf("First %d elements of the result:\n", elements_to_print);
  for (int i = 0; i < elements_to_print; i++) {
    fprintf(stderr, "h_c[%d] = %d\n", i, h_c[i]);
  }

  // Print the elapsed time
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free host memory
  free(h_a);
  free(h_b);
  free(h_c);
}

inline void cudaVectorAdd2() {

  // Device (GPU) pointers
  int* d_a, * d_b, * d_c;

  // Host pointer
  int* h_c;

  // Length of vector
  size_t size = N * sizeof(float);

  // Allocating memory
  h_c = (int*)malloc(size);

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize vectors on the GPU directly
  initVectors << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, N);

  // Adding vector in parallel using threads on the GPU
  addVector << < blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;

  // Print the first few elements of the result
  int elements_to_print = 10;  // Number of elements to print
  printf("First %d elements of the result:\n", elements_to_print);
  for (int i = 0; i < elements_to_print; i++) {
    fprintf(stderr, "h_c[%d] = %d\n", i, h_c[i]);
  }

  // Print the elapsed time
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free host memory
  free(h_c);
}

inline void cudaThrustVectorAdd() {
  // Initializing host arrays
  std::vector<int> h_a(N);
  std::vector<int> h_b(N);
  std::vector<int> h_c(N);

  // Initializing device arrays
  thrust::device_vector<int> d_a(N);
  thrust::device_vector<int> d_b(N);
  thrust::device_vector<int> d_c(N);

  // Initialize host vectors
  for (int i = 0; i < N; i++) {
    h_a[i] = (rand() * 100) / (int)RAND_MAX;
    h_b[i] = (rand() * 100) / (int)RAND_MAX;
  }

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  // Copy data from host to device
  d_a = h_a;
  d_b = h_b;

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  addVector << <blocksPerGrid, threadsPerBlock >> > (
    thrust::raw_pointer_cast(d_a.data()),
    thrust::raw_pointer_cast(d_b.data()),
    thrust::raw_pointer_cast(d_c.data()),
    N
    );

  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;

  // Print the first few elements of the result
  int elements_to_print = 10;  // Number of elements to print
  printf("First %d elements of the result:\n", elements_to_print);
  for (int i = 0; i < elements_to_print; i++) {
    fprintf(stderr, "h_c[%d] = %d\n", i, h_c[i]);
  }

  // Print the elapsed time
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}

inline void CPUCode() {
  // Host (CPU) pointers
  int* h_a, * h_b, * h_c;

  // Length of vector
  size_t size = N * sizeof(float);

  // Allocate host memory
  h_a = (int*)malloc(size);
  h_b = (int*)malloc(size);
  h_c = (int*)malloc(size);

  // Initialize host vectors
  for (int i = 0; i < N; i++) {
    h_a[i] = (rand() * 100) / (int)RAND_MAX;
    h_b[i] = (rand() * 100) / (int)RAND_MAX;
  }

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  // CPU code
  for (int i = 0; i < N; i++) {
    h_c[i] = h_a[i] + h_b[i];
  }

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = end - start;

  // Print the elapsed time
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Free host memory
  free(h_a);
  free(h_b);
  free(h_c);
}

int main() {
  //cudaVectorAdd1();
  cudaVectorAdd2();
  //CPUCode();
  //cudaThrustVectorAdd();

  return 0;
}