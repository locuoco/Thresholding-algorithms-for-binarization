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

#include <iostream>
#include <string>
#include <chrono>
#include <type_traits>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#ifdef __CUDACC__

#define ENABLE_CUDA_GPGPU	// comment this line to use C++ code when compiling with CUDA compiler
							// not needed if you are using GCC
#endif

#include "binarization.hpp"

#ifdef ENABLE_CUDA_GPGPU

#include "binarization_gpu.cuh"

#endif

/* available functions for thresholding:
	== CPU ==
	sauvola
	niblack
	bernsen
	singh
	sauvola_chan
	niblack_chan
	singh_chan
	global

	== GPU ==
	sauvola_gpu
	sauvola_gpu2
	niblack_gpu
	niblack_gpu2
	bernsen_gpu
	bernsen_gpu2
	singh_gpu
	singh_gpu2
	singh_gpu3
	global_gpu
*/

#define FUNC_TO_TEST bernsen
#define FUNC_ARGUMENTS , 15, 100

#define IS_GLOBAL 0 // if you want to test global or global_gpu, please set this to 1
#define IS_SINGH_GPU3 0 // if you want to test singh_gpu3, please set this to 1

#define TO_STR(...) #__VA_ARGS__
#define FUNC_STR(...) TO_STR(__VA_ARGS__)
#define FUNC_NAME FUNC_STR(FUNC_TO_TEST FUNC_ARGUMENTS)

void save_single_image(unsigned char *raw, unsigned int width, unsigned int height, int i, std::string name = std::string("image"))
{
	name += '_';
	name += std::to_string(i);
	name += ".png";
	stbi_write_png(name.c_str(), width, height, 1, raw, width);
}

template <int R = 6>
void simple_test(unsigned char *raw, unsigned int w, unsigned int h)
{
#if defined(ENABLE_CUDA_GPGPU) && IS_GLOBAL == 0 && IS_SINGH_GPU3 == 0
	imgproc::FUNC_TO_TEST<R>(raw, w, h FUNC_ARGUMENTS);
#elif IS_SINGH_GPU3 == 0
	imgproc::FUNC_TO_TEST(raw, w, h, R FUNC_ARGUMENTS);
#else
	imgproc::FUNC_TO_TEST(raw, w, h FUNC_ARGUMENTS);
#endif
}

template <int R = 6>
void prealloc_test(unsigned char *raw, unsigned int w, unsigned int h, int idx)
{
	namespace chrono = std::chrono;

	auto begin = chrono::steady_clock::now();

	simple_test<R>(raw, w, h);

	auto end = chrono::steady_clock::now();

	std::cout << "Time elapsed for processing image " << idx << ": "
			  << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6
			  << " [s]\n";

	save_single_image(raw, w, h, idx, FUNC_NAME);

	begin = chrono::steady_clock::now();

	simple_test<R>(raw, w, h);

	end = chrono::steady_clock::now();

	std::cout << "Time elapsed for processing image " << idx << " with preallocation: "
			  << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6
			  << " [s]\n";
}

template <int R = 6>
void prealloc_test2(unsigned char *raw, unsigned int w, unsigned int h, int idx)
{
	namespace chrono = std::chrono;

	simple_test<R>(raw, w, h);

	auto begin = chrono::steady_clock::now();

	simple_test<R>(raw, w, h);

	auto end = chrono::steady_clock::now();

	std::cout << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6 << ' ';
}

template <int R, int Step, int maxR, std::enable_if<(R < 0)>::type* = nullptr>
void window_test(unsigned char*, unsigned int, unsigned int, int)
{
	std::cout << '\n' << FUNC_NAME << ' ';
}

template <int R = 20, int Step = 2, int maxR = R, std::enable_if<(R >= 0)>::type* = nullptr>
void window_test(unsigned char *raw, unsigned int w, unsigned int h, int idx)
{
	std::cout << '\t' << (maxR % Step + maxR - R)*2+1; // :)
	window_test<R-Step, Step, maxR>(raw, w, h, idx);
	prealloc_test2<R>(raw, w, h, idx);
}

int main(const int narg, const char** args)
{
	namespace chrono = std::chrono;

	auto begin = chrono::steady_clock::now();

#ifdef ENABLE_CUDA_GPGPU
	gpuErrchk(cudaFree(nullptr)); // force to create the CUDA context
	std::cout << "CUDA enabled!\n";
#else
	std::cout << "CUDA NOT enabled!\n";
#endif

	auto end = chrono::steady_clock::now();

	std::cout << "Time elapsed for creating CUDA context: "
				  << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6
				  << " [s]\n";

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

	for (unsigned int i = 0; i < Nimages; ++i)
	{
		int width, height, ch;

		begin = chrono::steady_clock::now();

		unsigned char *raw = stbi_load(img_paths[i].c_str(), &width, &height, &ch, 1);

		if (!raw)
		{
			std::cerr << "Error: Cannot find " << img_paths[i] << '.' << std::endl;
			continue;
		}

		end = chrono::steady_clock::now();

		std::cout << "Time elapsed for loading image " << i << ": "
				  << chrono::duration_cast<chrono::microseconds>(end - begin).count() * 1.e-6
				  << " [s]\n";

		window_test(raw, width, height, i);

		delete[] raw;
	}

	return 0;
}



























