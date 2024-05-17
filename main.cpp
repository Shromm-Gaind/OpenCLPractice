#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

// Some examples I used to help: https://github.com/yeephycho/OpenCV_based_JuliaSet_Implementation/blob/master/src/opencl_imp/main.cpp
// Psedocode from https://en.wikipedia.org/wiki/Julia_set
// I use opencl c++ bindings and have a time measurement using chrono

const char* kernelSource = R"CLC(
__kernel void julia_set(
    __global unsigned char* output,
    int width,
    int height,
    float realC,
    float imagC,
    float realMin,
    float realMax,
    float imagMin,
    float imagMax,
    int maxIter) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    float real = realMin + x * (realMax - realMin) / width;
    float imag = imagMin + y * (imagMax - imagMin) / height;

    int iter;
    for (iter = 0; iter < maxIter; ++iter) {
        float realTemp = real * real - imag * imag + realC;
        imag = 2 * real * imag + imagC;
        real = realTemp;

        if (real * real + imag * imag > 4.0f) break;
    }

    output[y * width + x] = (unsigned char)(255 * iter / maxIter);
}
)CLC";

void checkError(cl_int error, const char* functionName) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error " << error << " in " << functionName << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 800;
    const int height = 800;
    const int maxIter = 300;
    const float realC = -0.7f;
    const float imagC = 0.27015f;
    const float realMin = -1.5f;
    const float realMax = 1.5f;
    const float imagMin = -1.5f;
    const float imagMax = 1.5f;

    // Get all platforms (drivers)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No platforms found." << std::endl;
        return 1;
    }

    // Select the default platform and create a context using this platform and the GPU
    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        std::cerr << "No devices found." << std::endl;
        return 1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Create the program from source
    cl::Program program(context, kernelSource);

    // Build the program for the devices
    cl_int err = program.build(devices);
    if (err != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // Create the kernel
    cl::Kernel kernel(program, "julia_set");

    // Create the output buffer
    std::vector<unsigned char> output(width * height);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, output.size() * sizeof(unsigned char));

    // Set the kernel arguments
    kernel.setArg(0, outputBuffer);
    kernel.setArg(1, width);
    kernel.setArg(2, height);
    kernel.setArg(3, realC);
    kernel.setArg(4, imagC);
    kernel.setArg(5, realMin);
    kernel.setArg(6, realMax);
    kernel.setArg(7, imagMin);
    kernel.setArg(8, imagMax);
    kernel.setArg(9, maxIter);

    // Define the global and local work sizes
    cl::NDRange globalSize(width, height);
    cl::NDRange localSize(16, 16);

    // Execute the kernel and measure the time
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

    // Read the result
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, output.size() * sizeof(unsigned char), output.data());

    // Save the output image
    cv::Mat image(height, width, CV_8UC1, output.data());
    cv::imwrite("julia_set.png", image);

    std::cout << "Julia set image saved as julia_set.png" << std::endl;

    return 0;
}
