#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>

namespace solution {
    #define CUDA_ERROR_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); } 
    inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    __global__ void convolution2D(const float *input_image, float *output_result, const float *conv_kernel, const int N_Rows, const int N_Cols) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
                
        int tx = threadIdx.x;
        int thread_id = bx * blockDim.x + tx;
        int s = N_Rows * N_Cols;
        if(thread_id < s) {
            float sum = 0.0;
            int i = thread_id / N_Cols;
            int j = thread_id % N_Cols;
            for(int d_row = -1; d_row <= 1; d_row++) {
                for(int d_col = -1; d_col <= 1; d_col++) {
                    int new_row = i + d_row, new_col = j + d_col;
                    if(new_row >= 0 && new_row < N_Rows && new_col >= 0 && new_col < N_Cols) {
                        sum += conv_kernel[(d_row + 1) * 3 + (d_col + 1)] * input_image[new_row * N_Cols + new_col];
                    }
                }
            }
            output_result[thread_id] = sum;
        }
    }

    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t N_Rows, const std::int32_t N_Cols) {
        std::string solution_path = std::filesystem::temp_directory_path() / "student_solution.bmp";
        std::ofstream sol_fs(solution_path, std::ios::binary);
        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
        int s = N_Rows * N_Cols;
        const auto image = std::make_unique<float[]>(s);
        bitmap_fs.read(reinterpret_cast<char*>(image.get()), sizeof(float) * s);

        float *device_image, *device_result, *device_kernel;
        CUDA_ERROR_CHECK(cudaMalloc((void **)&device_image, sizeof(float) * s));
        CUDA_ERROR_CHECK(cudaMalloc((void **)&device_result, sizeof(float) * s));
        CUDA_ERROR_CHECK(cudaMalloc((void **)&device_kernel, sizeof(float) * 3 * 3));

        CUDA_ERROR_CHECK(cudaMemcpy(device_image, image.get(), sizeof(float) * s, cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(device_kernel, kernel, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice));

        int size_block = 512;
        int Num_Blocks = (N_Rows * N_Cols + size_block - 1) / size_block;

        convolution2D<<<Num_Blocks, size_block>>>(device_image, device_result, device_kernel, N_Rows, N_Cols);

        float *result = new float[s];
        CUDA_ERROR_CHECK(cudaMemcpy(result, device_result, sizeof(float) * s, cudaMemcpyDeviceToHost));

        sol_fs.write(reinterpret_cast<char*>(result), sizeof(float) * s);

        cudaFree(device_image);
        cudaFree(device_result);
        cudaFree(device_kernel);

        delete[] result;

        return solution_path;
    }
};
