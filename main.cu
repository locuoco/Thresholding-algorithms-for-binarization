//  A code for binarization using local thresholding algorithms
//  Copyright (C) 2021 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

/**

	Compilation:
		nvcc main.cu -o ltbin -O3 -std=c++11 -arch=sm_<xy> -I <includes>
	<xy> is the compute capability of the GPU (usually given in the form x.y),
	for example sm_21 corresponds to a compute capability of 2.1.
	<includes> are header files to be included

	example:
		nvcc main.cu -o ltbin -O3 -arch=sm_75

	Compilation with GCC (without CUDA code, extension must be changed to .cpp):
		g++ -std=c++11 -O3 -Wall main.cpp -o ltbincpu

*/

#ifdef __CUDACC__

#define ENABLE_CUDA_GPGPU // comment this line to use C++ code when compiling with CUDA compiler
						  // not needed if you are using GCC
#endif

#include "binarization.hpp"

#ifdef ENABLE_CUDA_GPGPU

#include "binarization_gpu.cuh"

#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <string>
#include <chrono>

#define W 31		// window size (must be odd)
#define R (W/2)		// filter radius

#if W % 2 == 0
#error Value of W: NO BUONO!
#endif

int main(const int narg, const char** args)
{
#ifdef ENABLE_CUDA_GPGPU
	gpuErrchk(cudaFree(nullptr)); // force to create the CUDA context
	std::cout << "CUDA enabled!" << std::endl;
#else
	std::cout << "CUDA NOT enabled!" << std::endl;
#endif

	std::string *img_paths;
	unsigned int Nimages;
	if (narg < 2)
	{
		std::cout << "No input image. Nothing to be done. Add the paths/names of the images as arguments to this program." << std::endl;
		return 0;
	}
	else
	{
		Nimages = narg-1;
		img_paths = new std::string[Nimages];
		for (int i = 1; i < narg; ++i)
			img_paths[i-1] = std::string(args[i]);
	}

	namespace chrono = std::chrono;

	for (int i = 0; i < Nimages; ++i)
	{
		int width, height, ch;
		unsigned char *raw = stbi_load(img_paths[i].c_str(), &width, &height, &ch, 1);

		if (!raw)
		{
			std::cerr << "Error: Cannot find " << img_paths[i] << '.' << std::endl;
			continue;
		}

		auto begin = chrono::steady_clock::now();

#ifdef ENABLE_CUDA_GPGPU
		imgproc::singh_gpu(raw, width, height);
#else
		imgproc::global(raw, width, height);
#endif

		auto end = chrono::steady_clock::now();

		std::cout << "Time elapsed for image " << i << ": "
				  << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6
				  << " [s]" << std::endl;

		std::string out_path("image");
		out_path += std::to_string(i);
		out_path += ".png";
		stbi_write_png(out_path.c_str(), width, height, 1, raw, width);

		delete[] raw;
	}

	return 0;
}



























