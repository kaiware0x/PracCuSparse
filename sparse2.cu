#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>

int sparse2()
{
    // Initialize the device
    cudaSetDevice(0);

    // Create a cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Define the matrix dimensions and the vector size
    int num_rows = 5;
    int num_cols = 5;
    int nnz = 13; // number of non-zero elements in the matrix
    int vec_size = 5;

    // Define the matrix and vector on the host
    float *h_matrix = (float *)malloc(nnz * sizeof(float));
    int *h_rowPtr = (int *)malloc((num_rows + 1) * sizeof(int));
    int *h_colInd = (int *)malloc(nnz * sizeof(int));
    float *h_vector = (float *)malloc(vec_size * sizeof(float));

    // Fill the matrix and vector with values...

    // Allocate memory on the device
    float *d_matrix, *d_vector, *d_output;
    int *d_rowPtr, *d_colInd;
    cudaMalloc((void **)&d_matrix, nnz * sizeof(float));
    cudaMalloc((void **)&d_rowPtr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_colInd, nnz * sizeof(int));
    cudaMalloc((void **)&d_vector, vec_size * sizeof(float));
    cudaMalloc((void **)&d_output, num_rows * sizeof(float));

    // Copy the matrix and vector to the device
    cudaMemcpy(d_matrix, h_matrix, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vec_size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the matrix-vector multiplication
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                      d_rowPtr, d_colInd, d_matrix,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_cols, d_vector, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_rows, d_output, CUDA_R_32F);

    float alpha = 1.0;
    float beta = 0.0;
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                 CUSPARSE_SPMV_ALG_DEFAULT, NULL);

    // Copy the result back to the host
    float *h_output = (float *)malloc(num_rows * sizeof(float));
    cudaMemcpy(h_output, d_output, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    free(h_matrix);
    free(h_rowPtr);
    free(h_colInd);
    free(h_vector);
    free(h_output);
    cudaFree(d_matrix);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_vector);
    cudaFree(d_output);
    cusparseDestroy(handle);

    return 0;
}
