# SPP-CUDA-2D-Conv

## Project Overview
This project is an assignment on optimizing the application of CUDA convolutions to pixel data, developed as part of the SPP (Software Programming for Performance) course. The project demonstrates the use of CUDA to perform 2D convolution operations on images, optimizing the process for high performance on NVIDIA GPUs.

### Explanation of the Code

1. **CUDA Error Checking**: The macro `CUDA_ERROR_CHECK` is used to wrap CUDA function calls and check for errors. If an error occurs, it is reported, and the program exits.
2. **CUDA Kernel (`convolution2D`)**: This kernel performs the 2D convolution operation. Each thread calculates the convolution for one pixel of the output image. The convolution is performed using a 3x3 kernel, and boundary conditions are handled by checking if the neighboring pixel indices are within bounds.
3. **Host Code (`compute` function)**:
    - **File I/O**: Reads the input image and writes the output image.
    - **Memory Allocation**: Allocates memory on the GPU for the input image, output result, and convolution kernel.
    - **Memory Copy**: Copies data from host to device memory and vice versa.
    - **Kernel Launch**: Configures the grid and block dimensions and launches the CUDA kernel.
    - **Result Handling**: Copies the result from device to host memory and writes it to a file.

### Shared Memory and Optimization Techniques

- **Shared Memory**: In the provided code, shared memory is not utilized. Shared memory can be used to optimize the convolution process by reducing the number of global memory accesses. By loading a tile of the input image into shared memory, threads can cooperatively load data, which can significantly improve performance.
- **Future Optimization Techniques**:
    1. **Tiling**: Use tiling to load data into shared memory in a coalesced manner, reducing global memory access latency.
    2. **Loop Unrolling**: Unroll the inner loops to reduce loop overhead and increase instruction-level parallelism.
    3. **Constant Memory**: Store the convolution kernel in constant memory as it remains unchanged during execution.
    4. **Texture Memory**: Use texture memory for read-only data like images, which can exploit spatial locality and provide cached access to global memory.

## How to Build and Run

### Build Instructions

1. Clone the repository:
    
    ```bash
    bashCopy code
    git clone <repository-url>
    cd <repository-directory>
    
    ```
    

### Run the Application

To run the 2D convolution application, use the provided `runner_script.sh`:

```bash
bashCopy code
./runner_script.sh 

```

## Conclusion

This project demonstrates the application of CUDA for optimizing 2D convolution operations on images. By leveraging the parallel processing capabilities of GPUs, significant performance improvements can be achieved.