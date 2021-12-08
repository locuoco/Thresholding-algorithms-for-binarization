//  CUDA kernels for GPU acceleration and C++ wrappers
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

#ifndef IMGPROC_BINARIZATION_GPU_CUH
#define IMGPROC_BINARIZATION_GPU_CUH

#include "common.hpp"

#define IMGPROC_BLOCK_SIZE		256
#define IMGPROC_BLOCK_SIZE_2D	16
#define IMGPROC_TILE_W			256	// tile width
#define IMGPROC_TILE_H			256	// tile height

namespace imgproc
{

	enum mdk_threshold {mdk_niblack, mdk_sauvola};

	template <mdk_threshold T>
	IMGPROC_DEVICE_HOST float choose_threshold(float m, float d, float K)
	{
		switch (T)
		{
			case mdk_niblack:
				return niblack_threshold(m, d, K);
			case mdk_sauvola:
				return sauvola_threshold(m, d, K);
		}
	}

	__global__ static void rgb2float_krnl(float *f_grey, const unsigned int *b_rgb, unsigned int n)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			unsigned int rgbi = b_rgb[idx];
			unsigned char *rgb = (unsigned char*)&rgbi;
			float r = rgb[0] / 255.f,
				  g = rgb[1] / 255.f,
				  b = rgb[2] / 255.f;
			g = clamp(0.2126f*srgb2linear(r) + 0.7152f*srgb2linear(g) + 0.0722f*srgb2linear(b), 0.f, 1.f); // gamma decompression & linear combination
			f_grey[idx] = linear2srgb(g); // gamma compression
			//f_grey[idx] = clamp((r + g + b) / 3, 0.f, 1.f);
		}
	}

	__global__ static void byte2float_krnl(float *f_grey, const unsigned char *b_grey, unsigned int n)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			f_grey[idx] = b_grey[idx] / 255.f;
		}
	}

	__global__ static void float2byte_krnl(unsigned char *b_grey, const float *f_grey, unsigned int n)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			b_grey[idx] = f_grey[idx] * 255.f;
		}
	}

	template <int R>
	__global__ void hor_mean_filter_krnl(float *mean, const unsigned char *grey, int w)
	{
		__shared__ unsigned char smem[IMGPROC_TILE_W + 2*R];
		int x = blockIdx.x * IMGPROC_TILE_W;
		int y = blockIdx.y;
		unsigned int bindex = threadIdx.x + R;

		for (int i = threadIdx.x; i < IMGPROC_TILE_W + 2*R; i += IMGPROC_TILE_W)
		{
			int ind = clamp(x + i - R, 0, w-1);
			smem[i] = grey[ind];
		}
		x += threadIdx.x;
		if (x >= w)
			return;
		__syncthreads();

		int sum = 0;
#pragma unroll
		for (int dx = -R; dx <= R; ++dx)
			sum += smem[bindex + dx];
		mean[y*w + x] = min(sum / ((R*2+1)*255.f), 1.f);
	}

	template <int R>
	__global__ void vert_mean_filter_krnl(float *__restrict__ mean, const float *__restrict__ grey, int w, int h)
	{
		__shared__ float smem[IMGPROC_TILE_H + 2*R];
		int x = blockIdx.x;
		int y = blockIdx.y * IMGPROC_TILE_H;
		unsigned int bindex = threadIdx.y + R;

		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[i] = grey[ind];
		}
		y += threadIdx.y;
		if (y >= h)
			return;
		__syncthreads();

		float sum = 0;
#pragma unroll
		for (int dy = -R; dy <= R; ++dy)
			sum += smem[bindex + dy];
		mean[y*w + x] = min(sum / (R*2+1), 1.f);
	}

	__global__ static void singh_binarize_krnl(unsigned char *grey, const float *mean, unsigned int n, float K)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			float I = grey[idx]/255.f;
			float m = mean[idx];
			float dev = I - m;
			float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
			grey[idx] = (I <= threshold) ? 0 : 255;
		}
	}

	__global__ static void singh_binarize_krnl2(unsigned char *grey, const long long *integ, int w, int h, int R, float K)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < w*h)
		{
			int i = idx / w, j = idx % w;
			float I = grey[idx]/255.f;
			float den = (2*R+1)*(2*R+1)*255;
			float m = min(take_local_var(integ, j, i, w, R)/den, 1.f);
			float dev = I - m;
			float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
			grey[idx] = (I <= threshold) ? 0 : 255;
		}
	}

	template <int R, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void singh_binarize_krnl3(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
	{
		__shared__ unsigned char smem[TILE*TILE];
		int x = blockIdx.x * blockDim.x;
		int y = blockIdx.y * blockDim.y;
		int bindex = (threadIdx.y+R) * TILE + threadIdx.x+R;

		for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < TILE*TILE; i += blockDim.x*blockDim.y)
		{
			int dx = i % TILE - R, dy = i / TILE - R;
			smem[i] = grey[clamp_coord(x+dx, y+dy, w, h)];
		}
		x += threadIdx.x;
		y += threadIdx.y;
		if (x >= w || y >= h)
			return;
		__syncthreads();
		int sum = 0;
		for (int dy = -R; dy <= R; ++dy)
#pragma unroll
			for (int dx = -R; dx <= R; ++dx)
				sum += smem[bindex + dy*TILE + dx];
		float I = smem[bindex]/255.f;
		float den = (2*R+1)*(2*R+1)*255.f;
		float m = sum / den;
		float dev = I - m;
		float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
		bin[y*w + x] = (I <= threshold) ? 0 : 255;
	}

	template <int R, mdk_threshold T, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void mdk_binarize_krnl(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
	{
		__shared__ unsigned char smem[TILE*TILE];
		int x = blockIdx.x * blockDim.x;
		int y = blockIdx.y * blockDim.y;
		int bindex = (threadIdx.y+R) * TILE + threadIdx.x+R;

		for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < TILE*TILE; i += blockDim.x*blockDim.y)
		{
			int dx = i % TILE - R, dy = i / TILE - R;
			smem[i] = grey[clamp_coord(x+dx, y+dy, w, h)];
		}
		x += threadIdx.x;
		y += threadIdx.y;
		if (x >= w || y >= h)
			return;
		__syncthreads();
		int sum = 0, sq = 0, g;
		for (int dy = -R; dy <= R; ++dy)
#pragma unroll
			for (int dx = -R; dx <= R; ++dx)
			{
				g = smem[bindex + dy*TILE + dx];
				sum += g;
				sq += g*g;
			}
		float I = smem[bindex]/255.f;
		float den = (2*R+1)*(2*R+1)*255.f;
		float den2 = den*255;
		float m = sum / den;
		float d = min(sqrtf(sq/den2 - m*m), .5f);
		float threshold = choose_threshold<T>(m, d, K);
		bin[y*w + x] = (I <= threshold) ? 0 : 255;
	}

	template <int R, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void bernsen_binarize_krnl(unsigned char *bin, const unsigned char *grey, int w, int h, unsigned char L, unsigned char T)
	{
		__shared__ unsigned char smem[TILE*TILE];
		int x = blockIdx.x * blockDim.x;
		int y = blockIdx.y * blockDim.y;
		int bindex = (threadIdx.y+R) * TILE + threadIdx.x+R;

		for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < TILE*TILE; i += blockDim.x*blockDim.y)
		{
			int dx = i % TILE - R, dy = i / TILE - R;
			smem[i] = grey[clamp_coord(x+dx, y+dy, w, h)];
		}
		x += threadIdx.x;
		y += threadIdx.y;
		if (x >= w || y >= h)
			return;
		__syncthreads();
		int ming = 255, maxg = 0, g;
		for (int dy = -R; dy <= R; ++dy)
#pragma unroll
			for (int dx = -R; dx <= R; ++dx)
			{
				g = smem[bindex + dy*TILE + dx];
				ming = min(ming, g);
				maxg = max(maxg, g);
			}
		int I = smem[bindex];
		int mid = (maxg + ming)/2;
		if (maxg - ming < L)
			bin[y*w + x] = (mid <= T) ? 0 : 255;
		else
			bin[y*w + x] = (I <= mid) ? 0 : 255;
	}

	__global__ static void global_binarize_krnl(unsigned char *grey, unsigned int n, unsigned char threshold)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			grey[idx] = (grey[idx] <= threshold) ? 0 : 255;
		}
	}

	inline void rgb2float_gpu(float *f_grey, const unsigned int *b_rgb, unsigned int n)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		rgb2float_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (f_grey, b_rgb, n);
	}
	inline void byte2float_gpu(float *f_grey, const unsigned char *b_grey, unsigned int n)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		byte2float_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (f_grey, b_grey, n);
	}
	inline void float2byte_gpu(unsigned char *b_grey, const float *f_grey, unsigned int n)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		float2byte_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (b_grey, f_grey, n);
	}
	template <int R>
	void hor_mean_filter_gpu(float *mean, const unsigned char *grey, unsigned int w, unsigned int h)
	{
		int nBlocksW = (w + IMGPROC_TILE_W - 1) / IMGPROC_TILE_W;
		hor_mean_filter_krnl<R> <<< dim3(nBlocksW, h), dim3(IMGPROC_TILE_W, 1) >>> (mean, grey, w);
	}
	template <int R>
	void vert_mean_filter_gpu(float *mean, const float *grey, unsigned int w, unsigned int h)
	{
		int nBlocksH = (h + IMGPROC_TILE_H - 1) / IMGPROC_TILE_H;
		vert_mean_filter_krnl<R> <<< dim3(w, nBlocksH), dim3(1, IMGPROC_TILE_H) >>> (mean, grey, w, h);
	}
	inline void singh_binarize_gpu(unsigned char *grey, const float *mean, unsigned int n, float K)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		singh_binarize_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, mean, n, K);
	}
	inline void singh_binarize_gpu2(unsigned char *grey, const long long *integ, int w, int h, int R, float K)
	{
		int nBlocks = (w*h + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		singh_binarize_krnl2 <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, integ, w, h, R, K);
	}
	template <int R>
	void singh_binarize_gpu3(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		singh_binarize_krnl3<R> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, K);
	}
	template <int R, mdk_threshold T>
	void mdk_binarize_gpu(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		mdk_binarize_krnl<R, T> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, K);
	}
	template <int R>
	void bernsen_binarize_gpu(unsigned char *bin, const unsigned char *grey, int w, int h, unsigned char L, unsigned char T)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		bernsen_binarize_krnl<R> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, L, T);
	}
	inline void global_binarize_gpu(unsigned char *grey, unsigned int n, unsigned char threshold)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		global_binarize_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, n, threshold);
	}

	template <int R = 15>
	extern void singh_gpu(unsigned char *raw, unsigned int w, unsigned int h, float K = 0.06f)
	{
		unsigned int n = w * h;
		// input/output (unsigned char), horizontal mean (float), mean (float)
		auto d_buf = simple_alloc_gpu(n*(2*sizeof(float) + sizeof(unsigned char)));
		float *d_hor_mean =(float*)d_buf;
		float *d_mean = d_hor_mean + n;
		unsigned char *d_raw = (unsigned char*)(d_mean + n);

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		hor_mean_filter_gpu<R>(d_hor_mean, d_raw, w, h);
		vert_mean_filter_gpu<R>(d_mean, d_hor_mean, w, h);

		singh_binarize_gpu(d_raw, d_mean, n, K);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	extern inline void singh_gpu2(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = 0.06f)
	{
		auto buf = simple_alloc((w + 2*R+1)*(h + 2*R+1)*sizeof(long long));
		long long *integ = (long long*)buf;

		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu((w + 2*R+1)*(h + 2*R+1)*sizeof(long long) + n*sizeof(unsigned char));
		long long *d_integ = (long long*)d_buf;
		unsigned char *d_raw = (unsigned char*)(d_integ + (w + 2*R+1)*(h + 2*R+1));

		integral(integ, raw, w, h, R);

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU
		gpuErrchk(cudaMemcpy(d_integ, integ, (w + 2*R+1)*(h + 2*R+1)*sizeof(long long), cudaMemcpyHostToDevice));

		singh_binarize_gpu2(d_raw, d_integ, w, h, R, K);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	template <int R = 15>
	extern void singh_gpu3(unsigned char *raw, unsigned int w, unsigned int h, float K = 0.06f)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		singh_binarize_gpu3<R>(d_bin, d_raw, w, h, K);

		gpuErrchk(cudaMemcpy(raw, d_bin, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	template <int R = 15, mdk_threshold T>
	extern void mdk_gpu(unsigned char *raw, unsigned int w, unsigned int h, float K)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		mdk_binarize_gpu<R, T>(d_bin, d_raw, w, h, K);

		gpuErrchk(cudaMemcpy(raw, d_bin, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}
	template <int R = 15>
	extern void niblack_gpu(unsigned char *raw, unsigned int w, unsigned int h, float K = -0.2f)
	{
		mdk_gpu<R, mdk_niblack>(raw, w, h, K);
	}
	template <int R = 15>
	extern void sauvola_gpu(unsigned char *raw, unsigned int w, unsigned int h, float K = 0.06f)
	{
		mdk_gpu<R, mdk_sauvola>(raw, w, h, K);
	}
	template <int R = 15>
	extern void bernsen_gpu(unsigned char *raw, unsigned int w, unsigned int h, unsigned char L = 25, unsigned char T = 127)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		bernsen_binarize_gpu<R>(d_bin, d_raw, w, h, L, T);

		gpuErrchk(cudaMemcpy(raw, d_bin, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	extern inline void global_gpu(unsigned char *raw, unsigned int w, unsigned int h, unsigned char threshold = 127)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		global_binarize_gpu(d_raw, n, threshold);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}
	
} // namespace imgproc

#endif


























