
#include<iostream>
#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<cusparse_v2.h>
#include<cusparse.h>
#include<thrust/device_vector.h>

const int N = 1024;

int sparse()
{
    /**********************************/
    /********** 入力値の準備 **********/
    /**********************************/

    // CSR形式疎行列のデータ
    //* 要素の値
    //* 列番号
    //* 各行の先頭位置
    double elements[N*3];
    int columnIndeces[N*3];
    int rowOffsets[N+1];

    // 中央差分行列を準備する
    //（対角項が2でその隣が1になる、↓こんなやつ）
    // | 2 1 0 0 0 0 0 0 ・・・ 0 0 0|
    // | 1 2 1 0 0 0 0 0 ・・・ 0 0 0|
    // | 0 1 2 1 0 0 0 0 ・・・ 0 0 0|
    // | 0 0 1 2 1 0 0 0 ・・・ 0 0 0|
    // | 0 0 0 1 2 1 0 0 ・・・ 0 0 0|
    // | 0 0 0 0 1 2 1 0 ・・・ 0 0 0|
    // | 0 0 0 0 0 1 2 1 ・・・ 0 0 0|
    // | 0 0 0 0 0 0 1 2 ・・・ 0 0 0|
    // | 0 0 0 0 0 0 0 0 ・・・ 2 1 0|
    // | 0 0 0 0 0 0 0 0 ・・・ 1 2 1|
    // | 0 0 0 0 0 0 0 0 ・・・ 0 1 2|
    int nonZeroCount = 0;
    rowOffsets[0] = 0;
    for(int i = 0; i < N; i++)
    {
        // 対角項
        elements[nonZeroCount] = 2;
        columnIndeces[nonZeroCount] = i;
        nonZeroCount++;

        // 対角項の左隣
        if(i > 0)
        {
            elements[nonZeroCount] = 1;
            columnIndeces[nonZeroCount] = i - 1;
            nonZeroCount++;
        }

        // 対角項の右隣
        if(i < N-1)
        {
            elements[nonZeroCount] = 1;
            columnIndeces[nonZeroCount] = i + 1;
            nonZeroCount++;
        }

        // 次の行の先頭位置
        rowOffsets[i+1] = nonZeroCount;
    }

    // かけるベクトルを生成
    double vector[N];
    for(int i = 0; i < N; i++)
    {
        vector[i] = i * 0.1;
    }

    // 結果格納ベクトルを生成
    double result[N];

    /**********************************/
    /********** 入力値の転送 **********/
    /**********************************/
    // GPU側の配列を確保
    // （ポインタ管理が面倒なのでthrust使うと便利！）
    thrust::device_vector<double> elementsDevice(N*3);
    thrust::device_vector<int>    columnIndecesDevice(N*3);
    thrust::device_vector<int>    rowOffsetsDevice(N+1);
    thrust::device_vector<double> vectorDevice(N);
    thrust::device_vector<double> resultDevice(N);

    // GPU側配列へ入力値（行列とベクトル）を複製
    thrust::copy_n(elements,      N*3, elementsDevice.begin());
    thrust::copy_n(columnIndeces, N*3, columnIndecesDevice.begin());
    thrust::copy_n(rowOffsets,    N+1, rowOffsetsDevice.begin());
    thrust::copy_n(vector, N, vectorDevice.begin());



    /************************************/
    /********** cuSPARSEの準備 **********/
    /************************************/
    // cuSPARSEハンドルを作成
    ::cusparseHandle_t cusparse;
    ::cusparseCreate(&cusparse);

    // 行列形式を作成
    // * 一般的な形式
    // * 番号は0から開始
    ::cusparseMatDescr_t matDescr;
    ::cusparseCreateMatDescr(&matDescr);
    ::cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    ::cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);



    /******************************************/
    /********** 行列×ベクトルの計算 **********/
    /******************************************/
    // thrust配列からCUDA用ポインタに変換
    double* elementsPtr   = thrust::raw_pointer_cast(&(elementsDevice[0]));
    int* columnIndecesPtr = thrust::raw_pointer_cast(&(columnIndecesDevice[0]));
    int* rowOffsetsPtr    = thrust::raw_pointer_cast(&(rowOffsetsDevice[0]));
    double* vectorPtr     = thrust::raw_pointer_cast(&(vectorDevice[0]));
    double* resultPtr     = thrust::raw_pointer_cast(&(resultDevice[0]));

    // Csrmv（CSR形式行列とベクトルの積）を実行
    // y = α*Ax + β*y;
    const double ALPHA = 1;
    const double BETA = 0;

    ::cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        N, N, nonZeroCount,
        &ALPHA, matDescr, elementsPtr, rowOffsetsPtr, columnIndecesPtr,
        vectorPtr,
        &BETA, resultPtr);

        // Perform SpMV operation with cuSPARSE
float alpha = 1.0f;
float beta = 0.0f;
cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
             &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, 
             CUSPARSE_MV_ALG_DEFAULT, NULL);

    // ::cusparseDcsrmv_v2(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     N, N, nonZeroCount,
    //     &ALPHA, matDescr, elementsPtr, rowOffsetsPtr, columnIndecesPtr,
    //     vectorPtr,
    //     &BETA, resultPtr);



    /************************************/
    /********** 計算結果を取得 **********/
    /************************************/
    // GPU側配列から結果を複製
    thrust::copy_n(resultDevice.begin(), N, resultDevice);

    // 結果の表示
    for(int i = 0; i < N; i++)
    {
        std::cout << result[i] << std::endl;
    }


    return 0;
}
