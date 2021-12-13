//  Common utility functions used for this project
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

#ifndef IMGPROC_COMMON_HPP
#define IMGPROC_COMMON_HPP

#ifdef __CUDACC__
#include "cuda_runtime.h"
#define IMGPROC_DEVICE_HOST __device__ __host__

#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUassert: " << cudaGetErrorString(code) << ' ' << file << ' ' << line << std::endl;
		if (abort)
		{
			cudaDeviceReset();
			exit(code);
		}
	}
}

#else
#define IMGPROC_DEVICE_HOST
#endif // __CUDACC__

#define IMGPROC_EPS 1.e-5f

#include <algorithm> // min, max

namespace imgproc
{

	template <typename T>
	IMGPROC_DEVICE_HOST T clamp(T x, T a, T b)
	{
#ifndef __CUDA_ARCH__
		using std::max;
		using std::min;
#endif
		return min(max(a, x), b);
		//return (x < a) ? a : (x > b) ? b : x;
	}

	template <typename T>
	IMGPROC_DEVICE_HOST T srgb2linear(T x)
	{
#ifndef __CUDA_ARCH__
		using std::max;
#endif
		x = max(x, (T)0);
		return (x <= (T)0.04045) ? x / (T)12.92 : pow((x + (T)0.055) / (T)1.055, (T)2.4);
	}

	template <typename T>
	IMGPROC_DEVICE_HOST T linear2srgb(T x)
	{
#ifndef __CUDA_ARCH__
		using std::max;
#endif
		x = max(x, (T)0);
		return (x <= (T)0.0031308) ? x * (T)12.92 : max(pow(x * (T)1.055, 1/(T)2.4) - (T)0.055, (T)0);
	}

	inline unsigned char* simple_alloc(unsigned long long needed_bytes, bool b_free = false)
	// reallocates only when needed
	{
		static unsigned char *buf = nullptr;
		static unsigned long long bytes = 0;

		if (b_free)
		{
			delete[] buf;
			buf = nullptr;
			bytes = 0;
		}

		if (needed_bytes > bytes)
		{
			delete[] buf;
			bytes = bytes*2 > needed_bytes ? bytes*2 : needed_bytes;
			buf = new unsigned char[bytes];
		}

		return buf;
	}

#ifdef __CUDACC__
	inline unsigned char* simple_alloc_gpu(unsigned long long needed_bytes, bool b_free = false)
	// reallocates only when needed
	{
		static unsigned char *d_buf = nullptr;
		static unsigned long long bytes = 0;

		if (b_free)
		{
			gpuErrchk(cudaFree(d_buf));
			d_buf = nullptr;
			bytes = 0;
		}

		if (needed_bytes > bytes)
		{
			gpuErrchk(cudaFree(d_buf));
			bytes = bytes*2 > needed_bytes ? bytes*2 : needed_bytes;
			gpuErrchk(cudaMalloc((void**)&d_buf, bytes));
		}

		return d_buf;
	}
#endif

	IMGPROC_DEVICE_HOST inline unsigned int clamp_coord(int x, int y, int w, int h)
	{
		x = clamp(x, 0, w-1);
		y = clamp(y, 0, h-1);
		return y * w + x;
	}

	IMGPROC_DEVICE_HOST inline unsigned int integ_coord(int x, int y, int w, int R)
	{
		return (y+R+1) * (w+2*R+1) + x+R+1;
	}

	IMGPROC_DEVICE_HOST inline long long take_local_var(const long long *integ, int x, int y, int w, int R)
	{
		return integ[integ_coord(x+R, y+R, w, R)] + integ[integ_coord(x-R-1, y-R-1, w, R)]
			 - integ[integ_coord(x-R-1, y+R, w, R)] - integ[integ_coord(x+R, y-R-1, w, R)];
	}

	inline void init_integral(long long *integ, int w, int h, int R)
	{
		for (int i = -R-1; i < w+R; ++i)
			integ[integ_coord(i, -R-1, w, R)] = 0;
		for (int i = -R; i < h+R; ++i)
			integ[integ_coord(-R-1, i, w, R)] = 0;
	}

	inline void integral(long long *integ, const unsigned char *grey, int w, int h, int R)
	// integral sum of GL
	// it takes care of boundary conditions (clamp to edge)
	{
		init_integral(integ, w, h, R);
		for (int i = -R; i < h+R; ++i)
			for(int j = -R; j < w+R; ++j)
				integ[integ_coord(j, i, w, R)] = grey[clamp_coord(j, i, w, h)]
											   + integ[integ_coord(j, i-1, w, R)] + integ[integ_coord(j-1, i, w, R)] - integ[integ_coord(j-1, i-1, w, R)];
	}

	inline void sq_integral(long long *integ, const unsigned char *grey, int w, int h, int R)
	// integral sum of GL^2
	{
		init_integral(integ, w, h, R);
		long long g;
		for (int i = -R; i < h+R; ++i)
			for(int j = -R; j < w+R; ++j)
			{
				g = grey[clamp_coord(j, i, w, h)];
				integ[integ_coord(j, i, w, R)] = g*g
											   + integ[integ_coord(j, i-1, w, R)] + integ[integ_coord(j-1, i, w, R)] - integ[integ_coord(j-1, i-1, w, R)];
			}
	}

	inline void func_integral(long long *integ, const unsigned char *grey, int w, int h, int R, long long (*f)(long long))
	// integral sum of any function f(GL)
	{
		init_integral(integ, w, h, R);
		for (int i = -R; i < h+R; ++i)
			for(int j = -R; j < w+R; ++j)
				integ[integ_coord(j, i, w, R)] = f(grey[clamp_coord(j, i, w, h)])
											   + integ[integ_coord(j, i-1, w, R)] + integ[integ_coord(j-1, i, w, R)] - integ[integ_coord(j-1, i-1, w, R)];
	}

	IMGPROC_DEVICE_HOST inline bool niblack_threshold(float I, float m, float d2, float K)
	{
		float a = I - m + IMGPROC_EPS;
		return a < 0 && a*a > K*K*d2; // assume K < 0
	}
	IMGPROC_DEVICE_HOST inline bool sauvola_threshold(float I, float m, float d2, float K)
	{
		float a = I + m*(K - 1);
		float b = m*K;
		return a < 0 || a*a < 4*b*b*d2; // assume K > 0
	}

} // namespace imgproc

#endif // IMGPROC_COMMON_HPP

































