# matrix_multiplication_ccode_parallelization
c code for matrix multiplication using parallelization using threads and sse 

#define _CRT_SECURE_NO_WARNINGS 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <immintrin.h> // Include for SSE/AVX instructions



//SIMPLE C CODE FUNCTIONS
void multiplyIntMatrices(int m1, int n1, int** matrix1, int m2, int n2, int** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    int** result = (int**)malloc(m1 * sizeof(int*));
    for (int i = 0; i < m1; i++) {
        result[i] = (int*)malloc(n2 * sizeof(int));
    }

    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    fprintf(output, "Matrix data type: %d\n", 1);
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%d ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}

void multiplyFloatMatrices(int m1, int n1, float** matrix1, int m2, int n2, float** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    float** result = (float**)malloc(m1 * sizeof(float*));
    for (int i = 0; i < m1; i++) {
        result[i] = (float*)malloc(n2 * sizeof(float));
    }

    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    fprintf(output, "Matrix data type: %d\n", 2);
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%f ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}

void multiplyLongIntMatrices(int m1, int n1, long int** matrix1, int m2, int n2, long int** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    long int** result = (long int**)malloc(m1 * sizeof(long int*));
    for (int i = 0; i < m1; i++) {
        result[i] = (long int*)malloc(n2 * sizeof(long int));
    }

    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    fprintf(output, "Matrix data type: %d\n", 3);
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%ld ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}

void multiplyDoubleMatrices(int m1, int n1, double** matrix1, int m2, int n2, double** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    double** result = (double**)malloc(m1 * sizeof(double*));
    for (int i = 0; i < m1; i++) {
        result[i] = (double*)malloc(n2 * sizeof(double));
    }

    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    fprintf(output, "Matrix data type: %d\n", 4);
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%lf ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}


//C CODE MULTICORE FUNCTIONS


typedef struct {
    int m1, n1, m2, n2;
    void** matrix1;
    void** matrix2;
    int startRow;
    int endRow;
    void** result;
    int dataType;
} ThreadData;


DWORD WINAPI multiplyRows(LPVOID param) {
    ThreadData* data = (ThreadData*)param;
    int dataType = data->dataType;

    if (dataType == 1) {
        int** matrix1 = (int**)data->matrix1;
        int** matrix2 = (int**)data->matrix2;
        int** result = (int**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                result[i][j] = 0;
                for (int k = 0; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }
    else if (dataType == 2) {
        float** matrix1 = (float**)data->matrix1;
        float** matrix2 = (float**)data->matrix2;
        float** result = (float**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                result[i][j] = 0;
                for (int k = 0; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }
    else if (dataType == 3) {
        long int** matrix1 = (long int**)data->matrix1;
        long int** matrix2 = (long int**)data->matrix2;
        long int** result = (long int**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                result[i][j] = 0;
                for (int k = 0; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }
    else if (dataType == 4) {
        double** matrix1 = (double**)data->matrix1;
        double** matrix2 = (double**)data->matrix2;
        double** result = (double**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                result[i][j] = 0;
                for (int k = 0; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    return 0;
}

void multiplyMatrices(int m1, int n1, void** matrix1, int m2, int n2, void** matrix2, void** result, int numThreads, int dataType) {
    HANDLE* threads = (HANDLE*)malloc(numThreads * sizeof(HANDLE));
    ThreadData* threadData = (ThreadData*)malloc(numThreads * sizeof(ThreadData));

    
    int rowsPerThread = m1 / numThreads;
    int extraRows = m1 % numThreads;
    int currentRow = 0;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].m1 = m1;
        threadData[i].n1 = n1;
        threadData[i].m2 = m2;
        threadData[i].n2 = n2;
        threadData[i].matrix1 = matrix1;
        threadData[i].matrix2 = matrix2;
        threadData[i].result = result;
        threadData[i].dataType = dataType;
        threadData[i].startRow = currentRow;
        threadData[i].endRow = currentRow + rowsPerThread + (i < extraRows ? 1 : 0);

        threads[i] = CreateThread(NULL, 0, multiplyRows, &threadData[i], 0, NULL);
        currentRow = threadData[i].endRow;
    }

  
    WaitForMultipleObjects(numThreads, threads, TRUE, INFINITE);

    
    for (int i = 0; i < numThreads; i++) {
        CloseHandle(threads[i]);
    }

    free(threads);
    free(threadData);
}





//C CODE SSE INSTRUCTIONS FUNCTION

void multiplyIntMatrices_sse(int m1, int n1, int** matrix1, int m2, int n2, int** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

  
    int** result = (int**)malloc(m1 * sizeof(int*));
    for (int i = 0; i < m1; i++) {
        result[i] = (int*)_mm_malloc(n2 * sizeof(int), 16); 
    }

   
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            __m128i sum = _mm_setzero_si128(); 
            int k = 0;
            
            for (; k <= n1 - 4; k += 4) {
                __m128i a = _mm_loadu_si128((__m128i*) & matrix1[i][k]); 
                __m128i b = _mm_set_epi32(matrix2[k + 3][j], matrix2[k + 2][j], matrix2[k + 1][j], matrix2[k][j]); 
                __m128i prod = _mm_mullo_epi32(a, b);
                sum = _mm_add_epi32(sum, prod);
            }
            
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            int buffer[4];
            _mm_storeu_si128((__m128i*)buffer, sum); 
            result[i][j] = buffer[0] + buffer[1] + buffer[2] + buffer[3];

           
            for (; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    
    fprintf(output, "Matrix data type: %d\n", 1); 
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%d ", result[i][j]);
        }
        fprintf(output, "\n");
    }

   
    for (int i = 0; i < m1; i++) {
        _mm_free(result[i]);
    }
    free(result);
}


void multiplyLongIntMatrices_sse(int m1, int n1, long int** matrix1, int m2, int n2, long int** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    
    long int** result = (long int**)malloc(m1 * sizeof(long int*));
    for (int i = 0; i < m1; i++) {
        result[i] = (long int*)malloc(n2 * sizeof(long int));
    }

   
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            __m128i sum = _mm_setzero_si128(); 
            int k = 0;
            
            for (; k <= n1 - 2; k += 2) { 
                __m128i a = _mm_loadu_si128((__m128i*) & matrix1[i][k]); 
                __m128i b = _mm_set_epi64x(matrix2[k + 1][j], matrix2[k][j]);
                sum = _mm_add_epi64(sum, _mm_mul_epu32(a, b)); 
            }
            
            long int buffer[2];
            _mm_storeu_si128((__m128i*)buffer, sum);
            result[i][j] = buffer[0] + buffer[1];

           
            for (; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    
    fprintf(output, "Matrix data type: %d\n", 3); 
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%ld ", result[i][j]);
        }
        fprintf(output, "\n");
    }

   
    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}

void multiplyDoubleMatrices_sse(int m1, int n1, double** matrix1, int m2, int n2, double** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

    
    double** result = (double**)malloc(m1 * sizeof(double*));
    for (int i = 0; i < m1; i++) {
        result[i] = (double*)malloc(n2 * sizeof(double));
    }

   
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            __m128d sum = _mm_setzero_pd(); 
            int k = 0;
            for (; k <= n1 - 2; k += 2) {
                __m128d a = _mm_loadu_pd(&matrix1[i][k]); 
                __m128d b = _mm_set_pd(matrix2[k + 1][j], matrix2[k][j]);
                sum = _mm_add_pd(sum, _mm_mul_pd(a, b));
            }
            
            double buffer[2];
            _mm_storeu_pd(buffer, sum);
            result[i][j] = buffer[0] + buffer[1];

           
            for (; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

  
    fprintf(output, "Matrix data type: %d\n", 4); // Double data type
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%lf ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    
    for (int i = 0; i < m1; i++) {
        free(result[i]);
    }
    free(result);
}


void multiplyFloatMatrices_sse(int m1, int n1, float** matrix1, int m2, int n2, float** matrix2, FILE* output) {
    if (n1 != m2) {
        fprintf(output, "Error: Incompatible matrices for multiplication\n");
        return;
    }

   
    float** result = (float**)malloc(m1 * sizeof(float*));
    for (int i = 0; i < m1; i++) {
        result[i] = (float*)_mm_malloc(n2 * sizeof(float), 16);
    }

    
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            __m128 sum = _mm_setzero_ps(); 
            int k;
            for (k = 0; k <= n1 - 4; k += 4) { 
                __m128 mat1vec = _mm_load_ps(&matrix1[i][k]); 
                
                __m128 mat2vec = _mm_setr_ps(matrix2[k][j], matrix2[k + 1][j], matrix2[k + 2][j], matrix2[k + 3][j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(mat1vec, mat2vec)); 
            }
           
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            
            float partialSum;
            _mm_store_ss(&partialSum, sum);
            result[i][j] = partialSum;

            
            for (; k < n1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    
    fprintf(output, "Matrix data type: %d\n", 2);
    fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            fprintf(output, "%f ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    
    for (int i = 0; i < m1; i++) {
        _mm_free(result[i]);
    }
    free(result);
}



//C CODE FOR COMBINED VECTORIZATION AND MULTICORES


DWORD WINAPI multiplyRows_combined(LPVOID param) {
    ThreadData* data = (ThreadData*)param;
    int dataType = data->dataType;

    if (dataType == 1) {
        int** matrix1 = (int**)data->matrix1;
        int** matrix2 = (int**)data->matrix2;
        int** result = (int**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                __m128i sum = _mm_setzero_si128();
                int k = 0;
                for (; k < data->n1; k++) {
                    
                    __m128i a = _mm_set1_epi32(matrix1[i][k]); 
                    __m128i b = _mm_set1_epi32(matrix2[k][j]);
                    __m128i prod = _mm_mullo_epi32(a, b); 
                    sum = _mm_add_epi32(sum, prod);
                }
                
                sum = _mm_hadd_epi32(sum, sum);
                sum = _mm_hadd_epi32(sum, sum);
                int buffer[4];
                _mm_storeu_si128((__m128i*)buffer, sum); 
                result[i][j] = buffer[0];
            }
        }
    }
    else if (dataType == 2) { // Float
        float** matrix1 = (float**)data->matrix1;
        float** matrix2 = (float**)data->matrix2;
        float** result = (float**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                __m128 sum = _mm_setzero_ps();
                int k = 0;
                for (; k <= data->n1 - 4; k += 4) {
                    __m128 a = _mm_loadu_ps(&matrix1[i][k]); 
                    __m128 b = _mm_set_ps(matrix2[k][j], matrix2[k + 1][j], matrix2[k + 2][j], matrix2[k + 3][j]); 
                    __m128 prod = _mm_mul_ps(a, b); 
                    sum = _mm_add_ps(sum, prod);
                }
                float buffer[4];
                _mm_storeu_ps(buffer, sum); 
                result[i][j] = buffer[0] + buffer[1] + buffer[2] + buffer[3];

                
                for (; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }
    else if (dataType == 3) { // Double
        double** matrix1 = (double**)data->matrix1;
        double** matrix2 = (double**)data->matrix2;
        double** result = (double**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                __m256d sum = _mm256_setzero_pd(); 
                int k = 0;
                for (; k < data->n1 - 3; k += 4) {
                    __m256d a = _mm256_loadu_pd(&matrix1[i][k]); 
                    __m256d b = _mm256_loadu_pd(&matrix2[k][j]); 
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b)); 
                }
                double buffer[4];
                _mm256_storeu_pd(buffer, sum); 
                result[i][j] = buffer[0] + buffer[1] + buffer[2] + buffer[3]; 

                
                for (; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }
    else if (dataType == 4) { // Long int
        long int** matrix1 = (long int**)data->matrix1;
        long int** matrix2 = (long int**)data->matrix2;
        long int** result = (long int**)data->result;

        for (int i = data->startRow; i < data->endRow; i++) {
            for (int j = 0; j < data->n2; j++) {
                __m128i sum = _mm_setzero_si128(); 
                int k = 0;
                for (; k < data->n1 - 1; k += 2) {
                    __m128i a = _mm_loadu_si128((__m128i*) & matrix1[i][k]); 
                    __m128i b = _mm_loadu_si128((__m128i*) & matrix2[k][j]); 
                    __m128i prod = _mm_mullo_epi64(a, b); 
                    sum = _mm_add_epi64(sum, prod);
                }
                long long buffer[2];
                _mm_storeu_si128((__m128i*)buffer, sum); 
                result[i][j] = buffer[0] + buffer[1];

                
                for (; k < data->n1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    return 0;
}


void multiplyMatrices_combined(int m1, int n1, void** matrix1, int m2, int n2, void** matrix2, void** result, int numThreads, int dataType) {
    HANDLE* threads = (HANDLE*)malloc(numThreads * sizeof(HANDLE));
    ThreadData* threadData = (ThreadData*)malloc(numThreads * sizeof(ThreadData));

   
    int rowsPerThread = m1 / numThreads;
    int extraRows = m1 % numThreads;
    int currentRow = 0;

    for (int i = 0; i < numThreads; i++) {
        threadData[i].m1 = m1;
        threadData[i].n1 = n1;
        threadData[i].m2 = m2;
        threadData[i].n2 = n2;
        threadData[i].matrix1 = matrix1;
        threadData[i].matrix2 = matrix2;
        threadData[i].result = result;
        threadData[i].dataType = dataType;
        threadData[i].startRow = currentRow;
        threadData[i].endRow = currentRow + rowsPerThread + (i < extraRows ? 1 : 0);

        threads[i] = CreateThread(NULL, 0, multiplyRows, &threadData[i], 0, NULL);
        currentRow = threadData[i].endRow;
    }

   
    WaitForMultipleObjects(numThreads, threads, TRUE, INFINITE);

    
    for (int i = 0; i < numThreads; i++) {
        CloseHandle(threads[i]);
    }

    free(threads);
    free(threadData);
}



int main()
{
    FILE* input;
    int opcode;


   
    input = fopen("input.txt", "r");
    if (input == NULL) {
        printf("Error opening input file\n");
        return 1;
    }

   
    fscanf(input, "%d", &opcode);

    printf("Opcode: %d\n", opcode);

    //Simple C Code (Serialized)
    if (opcode == 3)
    {
        printf("Running Simple C Code (Serializeable)\n");
        FILE* input, * output;
        int opcode, dataType;
        int m1, n1, m2, n2;

      
        clock_t start = clock();

       
        input = fopen("input.txt", "r");
        if (input == NULL) {
            printf("Error opening input file\n");
            return 1;
        }

        
        fscanf(input, "%d %d", &opcode, &dataType);

       
        fscanf(input, "%d %d", &m1, &n1);

       
        if (dataType == 1) {
            int** matrix1 = (int**)malloc(m1 * sizeof(int*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (int*)malloc(n1 * sizeof(int));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%d", &matrix1[i][j]);
                }
            }

           
            fscanf(input, "%d %d", &m2, &n2);
            int** matrix2 = (int**)malloc(m2 * sizeof(int*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (int*)malloc(n2 * sizeof(int));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%d", &matrix2[i][j]);
                }
            }

            
            fclose(input);

           
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyIntMatrices(m1, n1, matrix1, m2, n2, matrix2, output);

           
            fclose(output);

            
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 2) {
            float** matrix1 = (float**)malloc(m1 * sizeof(float*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (float*)malloc(n1 * sizeof(float));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%f", &matrix1[i][j]);
                }
            }

           
            fscanf(input, "%d %d", &m2, &n2);
            float** matrix2 = (float**)malloc(m2 * sizeof(float*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (float*)malloc(n2 * sizeof(float));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%f", &matrix2[i][j]);
                }
            }

            
            fclose(input);

            
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyFloatMatrices(m1, n1, matrix1, m2, n2, matrix2, output);

            
            fclose(output);

          
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 3) {
            long int** matrix1 = (long int**)malloc(m1 * sizeof(long int*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (long int*)malloc(n1 * sizeof(long int));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%ld", &matrix1[i][j]);
                }
            }

           
            fscanf(input, "%d %d", &m2, &n2);
            long int** matrix2 = (long int**)malloc(m2 * sizeof(long int*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (long int*)malloc(n2 * sizeof(long int));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%ld", &matrix2[i][j]);
                }
            }

           
            fclose(input);

            
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyLongIntMatrices(m1, n1, matrix1, m2, n2, matrix2, output);

            
            fclose(output);

            
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 4) {
            double** matrix1 = (double**)malloc(m1 * sizeof(double*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (double*)malloc(n1 * sizeof(double));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%lf", &matrix1[i][j]);
                }
            }

            
            fscanf(input, "%d %d", &m2, &n2);
            double** matrix2 = (double**)malloc(m2 * sizeof(double*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (double*)malloc(n2 * sizeof(double));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%lf", &matrix2[i][j]);
                }
            }

          
            fclose(input);

           
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyDoubleMatrices(m1, n1, matrix1, m2, n2, matrix2, output);

            
            fclose(output);

            
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else {
            printf("Invalid data type\n");
        }

        
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Total execution time: %f seconds\n", time_spent);


    }

    //Vector Instructions

    else if (opcode == 4)
    {

        printf("C Code Vectorized\n");
        FILE* input, * output;
        int opcode, dataType;
        int m1, n1, m2, n2;

        
        clock_t start = clock();

        input = fopen("input.txt", "r");
        if (input == NULL) {
            printf("Error opening input file\n");
            return 1;
        }

        
        fscanf(input, "%d %d", &opcode, &dataType);

        
        fscanf(input, "%d %d", &m1, &n1);

       
        if (dataType == 1) {
            int** matrix1 = (int**)malloc(m1 * sizeof(int*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (int*)malloc(n1 * sizeof(int));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%d", &matrix1[i][j]);
                }
            }

            
            fscanf(input, "%d %d", &m2, &n2);
            int** matrix2 = (int**)malloc(m2 * sizeof(int*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (int*)malloc(n2 * sizeof(int));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%d", &matrix2[i][j]);
                }
            }

           
            fclose(input);

            
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

            
            multiplyIntMatrices_sse(m1, n1, matrix1, m2, n2, matrix2, output);

           
            fclose(output);

           
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 2) {
            float** matrix1 = (float**)malloc(m1 * sizeof(float*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (float*)malloc(n1 * sizeof(float));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%f", &matrix1[i][j]);
                }
            }

           
            fscanf(input, "%d %d", &m2, &n2);
            float** matrix2 = (float**)malloc(m2 * sizeof(float*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (float*)malloc(n2 * sizeof(float));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%f", &matrix2[i][j]);
                }
            }

            
            fclose(input);

           
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

            
            multiplyFloatMatrices_sse(m1, n1, matrix1, m2, n2, matrix2, output);

           
            fclose(output);

           
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 3) {
            long int** matrix1 = (long int**)malloc(m1 * sizeof(long int*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (long int*)malloc(n1 * sizeof(long int));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%ld", &matrix1[i][j]);
                }
            }

            
            fscanf(input, "%d %d", &m2, &n2);
            long int** matrix2 = (long int**)malloc(m2 * sizeof(long int*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (long int*)malloc(n2 * sizeof(long int));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%ld", &matrix2[i][j]);
                }
            }

            
            fclose(input);

           
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyLongIntMatrices_sse(m1, n1, matrix1, m2, n2, matrix2, output);

            
            fclose(output);

           
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else if (dataType == 4) {
            double** matrix1 = (double**)malloc(m1 * sizeof(double*));
            for (int i = 0; i < m1; i++) {
                matrix1[i] = (double*)malloc(n1 * sizeof(double));
                for (int j = 0; j < n1; j++) {
                    fscanf(input, "%lf", &matrix1[i][j]);
                }
            }

            
            fscanf(input, "%d %d", &m2, &n2);
            double** matrix2 = (double**)malloc(m2 * sizeof(double*));
            for (int i = 0; i < m2; i++) {
                matrix2[i] = (double*)malloc(n2 * sizeof(double));
                for (int j = 0; j < n2; j++) {
                    fscanf(input, "%lf", &matrix2[i][j]);
                }
            }

            
            fclose(input);

            
            output = fopen("matrix_output.txt", "w");
            if (output == NULL) {
                printf("Error opening output file\n");
                return 1;
            }

           
            multiplyDoubleMatrices_sse(m1, n1, matrix1, m2, n2, matrix2, output);

            
            fclose(output);

            
            for (int i = 0; i < m1; i++) {
                free(matrix1[i]);
            }
            free(matrix1);

            for (int i = 0; i < m2; i++) {
                free(matrix2[i]);
            }
            free(matrix2);
        }
        else {
            printf("Invalid data type\n");
        }

        
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Total execution time: %f seconds\n", time_spent);

    }


    //Multiple Cores(Thread)
    else if (opcode == 5)
    {

        printf("Running C based multicore\n");
        FILE* input, * output;
        int opcode, dataType;
        int m1, n1, m2, n2;

        // Open input file
        input = fopen("input.txt", "r");
        if (input == NULL) {
            printf("Error opening input file\n");
            return 1;
        }

        
        fscanf(input, "%d %d", &opcode, &dataType);

       
        fscanf(input, "%d %d", &m1, &n1);

        
        void** matrix1 = (void**)malloc(m1 * sizeof(void*));
        for (int i = 0; i < m1; i++) {
            matrix1[i] = malloc(n1 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
            for (int j = 0; j < n1; j++) {
                if (dataType == 1) {
                    fscanf(input, "%d", &((int*)matrix1[i])[j]);
                }
                else if (dataType == 2) {
                    fscanf(input, "%f", &((float*)matrix1[i])[j]);
                }
                else if (dataType == 3) {
                    fscanf(input, "%ld", &((long int*)matrix1[i])[j]);
                }
                else if (dataType == 4) {
                    fscanf(input, "%lf", &((double*)matrix1[i])[j]);
                }
            }
        }

        
        fscanf(input, "%d %d", &m2, &n2);
        void** matrix2 = (void**)malloc(m2 * sizeof(void*));
        for (int i = 0; i < m2; i++) {
            matrix2[i] = malloc(n2 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
            for (int j = 0; j < n2; j++) {
                if (dataType == 1) {
                    fscanf(input, "%d", &((int*)matrix2[i])[j]);
                }
                else if (dataType == 2) {
                    fscanf(input, "%f", &((float*)matrix2[i])[j]);
                }
                else if (dataType == 3) {
                    fscanf(input, "%ld", &((long int*)matrix2[i])[j]);
                }
                else if (dataType == 4) {
                    fscanf(input, "%lf", &((double*)matrix2[i])[j]);
                }
            }
        }

       
        fclose(input);

       
        void** result = (void**)malloc(m1 * sizeof(void*));
        for (int i = 0; i < m1; i++) {
            result[i] = malloc(n2 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
        }

       
        clock_t startTime = clock();

       
        int numThreads = 8; // Number of threads to use (can be adjusted)
        multiplyMatrices(m1, n1, matrix1, m2, n2, matrix2, result, numThreads, dataType);

        
        clock_t endTime = clock();
        double executionTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        printf("Total execution time: %f seconds\n", executionTime);

       
        output = fopen("matrix_output.txt", "w");
        if (output == NULL) {
            printf("Error opening output file\n");
            return 1;
        }

       
        fprintf(output, "Matrix data type: %d\n", dataType);
        fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
        fprintf(output, "Rows of the resultant matrix after multiplication:\n");
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                if (dataType == 1) {
                    fprintf(output, "%d ", ((int**)result)[i][j]);
                }
                else if (dataType == 2) {
                    fprintf(output, "%f ", ((float**)result)[i][j]);
                }
                else if (dataType == 3) {
                    fprintf(output, "%ld ", ((long int**)result)[i][j]);
                }
                else if (dataType == 4) {
                    fprintf(output, "%lf ", ((double**)result)[i][j]);
                }
            }
            fprintf(output, "\n");
        }

        
        fclose(output);

        for (int i = 0; i < m1; i++) {
            free(matrix1[i]);
        }
        free(matrix1);

        for (int i = 0; i < m2; i++) {
            free(matrix2[i]);
        }
        free(matrix2);

        for (int i = 0; i < m1; i++) {
            free(result[i]);
        }
        free(result);

    }

    else if (opcode == 6)
    {

        printf("Combined C Code of Vectorization and MultipleCores\n");
        FILE* input, * output;
        int opcode, dataType;
        int m1, n1, m2, n2;

       
        input = fopen("input.txt", "r");
        if (input == NULL) {
            printf("Error opening input file\n");
            return 1;
        }

       
        fscanf(input, "%d %d", &opcode, &dataType);

        
        fscanf(input, "%d %d", &m1, &n1);

        
        void** matrix1 = (void**)malloc(m1 * sizeof(void*));
        for (int i = 0; i < m1; i++) {
            matrix1[i] = malloc(n1 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
            for (int j = 0; j < n1; j++) {
                if (dataType == 1) {
                    fscanf(input, "%d", &((int*)matrix1[i])[j]);
                }
                else if (dataType == 2) {
                    fscanf(input, "%f", &((float*)matrix1[i])[j]);
                }
                else if (dataType == 3) {
                    fscanf(input, "%ld", &((long int*)matrix1[i])[j]);
                }
                else if (dataType == 4) {
                    fscanf(input, "%lf", &((double*)matrix1[i])[j]);
                }
            }
        }

       
        fscanf(input, "%d %d", &m2, &n2);
        void** matrix2 = (void**)malloc(m2 * sizeof(void*));
        for (int i = 0; i < m2; i++) {
            matrix2[i] = malloc(n2 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
            for (int j = 0; j < n2; j++) {
                if (dataType == 1) {
                    fscanf(input, "%d", &((int*)matrix2[i])[j]);
                }
                else if (dataType == 2) {
                    fscanf(input, "%f", &((float*)matrix2[i])[j]);
                }
                else if (dataType == 3) {
                    fscanf(input, "%ld", &((long int*)matrix2[i])[j]);
                }
                else if (dataType == 4) {
                    fscanf(input, "%lf", &((double*)matrix2[i])[j]);
                }
            }
        }

       
        fclose(input);

        
        void** result = (void**)malloc(m1 * sizeof(void*));
        for (int i = 0; i < m1; i++) {
            result[i] = malloc(n2 * (dataType == 1 ? sizeof(int) : (dataType == 2 ? sizeof(float) : (dataType == 3 ? sizeof(long int) : sizeof(double)))));
        }

        
        clock_t startTime = clock();

        
        int numThreads = 8; // Number of threads to use (can be adjusted)
        multiplyMatrices(m1, n1, matrix1, m2, n2, matrix2, result, numThreads, dataType);

        
        clock_t endTime = clock();
        double executionTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        printf("Total execution time: %f seconds\n", executionTime);

       
        output = fopen("matrix_output.txt", "w");
        if (output == NULL) {
            printf("Error opening output file\n");
            return 1;
        }

       
        fprintf(output, "Matrix data type: %d\n", dataType);
        fprintf(output, "Dimensions of the resultant matrix: %d %d\n", m1, n2);
        fprintf(output, "Rows of the resultant matrix after multiplication:\n");
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                if (dataType == 1) {
                    fprintf(output, "%d ", ((int**)result)[i][j]);
                }
                else if (dataType == 2) {
                    fprintf(output, "%f ", ((float**)result)[i][j]);
                }
                else if (dataType == 3) {
                    fprintf(output, "%ld ", ((long int**)result)[i][j]);
                }
                else if (dataType == 4) {
                    fprintf(output, "%lf ", ((double**)result)[i][j]);
                }
            }
            fprintf(output, "\n");
        }

        
        fclose(output);

       
        for (int i = 0; i < m1; i++) {
            free(matrix1[i]);
        }
        free(matrix1);

        for (int i = 0; i < m2; i++) {
            free(matrix2[i]);
        }
        free(matrix2);

        for (int i = 0; i < m1; i++) {
            free(result[i]);
        }
        free(result);

    }







    return 0;
}
