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
	IMGPROC_DEVICE_HOST bool choose_threshold(float I, float m, float d, float K)
	{
		switch (T)
		{
			case mdk_niblack:
				return niblack_threshold(I, m, d, K);
			case mdk_sauvola:
				return sauvola_threshold(I, m, d, K);
		}
	}

	// CONVERSION FUNCTIONS

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

	// SEPARABLE FILTERS

	template <int R>
	__global__ void hor_mean_filter_krnl(float *mean, const unsigned char *grey, int w)
	{
		__shared__ unsigned char smem[IMGPROC_TILE_W + 2*R];
		int x = blockIdx.x * IMGPROC_TILE_W;
		int y = blockIdx.y;
		int bindex = threadIdx.x + R;

		for (int i = threadIdx.x; i < IMGPROC_TILE_W + 2*R; i += IMGPROC_TILE_W)
		{
			int ind = clamp(x + i - R, 0, w-1);
			smem[i] = grey[y*w + ind];
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
		int bindex = threadIdx.y + R;

		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[i] = grey[ind*w + x];
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

	template <int R>
	__global__ void hor_mean_msq_filter_krnl(float *mean, float *msq, const unsigned char *grey, int w)
	{
		__shared__ unsigned char smem[IMGPROC_TILE_W + 2*R];
		int x = blockIdx.x * IMGPROC_TILE_W;
		int y = blockIdx.y;
		int bindex = threadIdx.x + R;

		for (int i = threadIdx.x; i < IMGPROC_TILE_W + 2*R; i += IMGPROC_TILE_W)
		{
			int ind = clamp(x + i - R, 0, w-1);
			smem[i] = grey[y*w + ind];
		}
		x += threadIdx.x;
		if (x >= w)
			return;
		__syncthreads();

		int sum = 0, sq = 0, g;
#pragma unroll
		for (int dx = -R; dx <= R; ++dx)
		{
			g = smem[bindex + dx];
			sum += g;
			sq += g*g;
		}
		const float den = (R*2+1)*255.f;
		mean[y*w + x] = min(sum / den, 1.f);
		msq[y*w + x] = min(sq / (den*255), 1.f);
	}

	template <int R>
	__global__ void vert_mean_msq_filter_krnl(float *__restrict__ mean, float *__restrict__ msq,
											  const float *__restrict__ mean_in, const float *__restrict__ msq_in, int w, int h)
	{
		__shared__ float smem[2][IMGPROC_TILE_H + 2*R];
		int x = blockIdx.x;
		int y = blockIdx.y * IMGPROC_TILE_H;
		int bindex = threadIdx.y + R;

		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[0][i] = mean_in[ind*w + x];
		}
		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[1][i] = msq_in[ind*w + x];
		}
		y += threadIdx.y;
		if (y >= h)
			return;
		__syncthreads();

		float sum = 0;
#pragma unroll
		for (int dy = -R; dy <= R; ++dy)
			sum += smem[0][bindex + dy];
		mean[y*w + x] = min(sum / (R*2+1), 1.f);

		sum = 0;
#pragma unroll
		for (int dy = -R; dy <= R; ++dy)
			sum += smem[1][bindex + dy];
		msq[y*w + x] = min(sum / (R*2+1), 1.f);
	}

	template <int R>
	__global__ void hor_min_max_filter_krnl(unsigned char *__restrict__ ming, unsigned char *__restrict__ maxg,
											const unsigned char *__restrict__ grey, int w)
	{
		__shared__ unsigned char smem[IMGPROC_TILE_W + 2*R];
		int x = blockIdx.x * IMGPROC_TILE_W;
		int y = blockIdx.y;
		int bindex = threadIdx.x + R;

		for (int i = threadIdx.x; i < IMGPROC_TILE_W + 2*R; i += IMGPROC_TILE_W)
		{
			int ind = clamp(x + i - R, 0, w-1);
			smem[i] = grey[y*w + ind];
		}
		x += threadIdx.x;
		if (x >= w)
			return;
		__syncthreads();

		int cmin = 255, cmax = 0, g;
#pragma unroll
		for (int dx = -R; dx <= R; ++dx)
		{
			g = smem[bindex + dx];
			cmin = min(cmin, g);
			cmax = max(cmax, g);
		}
		ming[y*w + x] = cmin;
		maxg[y*w + x] = cmax;
	}

	template <int R>
	__global__ void vert_min_max_filter_krnl(unsigned char *__restrict__ ming, unsigned char *__restrict__ maxg,
											 const unsigned char *__restrict__ ming_in, const unsigned char *__restrict__ maxg_in, int w, int h)
	{
		__shared__ unsigned char smem[2][IMGPROC_TILE_H + 2*R];
		int x = blockIdx.x;
		int y = blockIdx.y * IMGPROC_TILE_H;
		int bindex = threadIdx.y + R;

		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[0][i] = ming_in[ind*w + x];
		}
		for (int i = threadIdx.y; i < IMGPROC_TILE_H + 2*R; i += IMGPROC_TILE_H)
		{
			int ind = clamp(y + i - R, 0, h-1);
			smem[1][i] = maxg_in[ind*w + x];
		}
		y += threadIdx.y;
		if (y >= h)
			return;
		__syncthreads();

		unsigned char c = 255;
#pragma unroll
		for (int dy = -R; dy <= R; ++dy)
			c = min(c, smem[0][bindex + dy]);
		ming[y*w + x] = c;

		c = 0;
#pragma unroll
		for (int dy = -R; dy <= R; ++dy)
			c = max(c, smem[1][bindex + dy]);
		maxg[y*w + x] = c;
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
	template <int R>
	void hor_mean_msq_filter_gpu(float *mean, float *msq, const unsigned char *grey, unsigned int w, unsigned int h)
	{
		int nBlocksW = (w + IMGPROC_TILE_W - 1) / IMGPROC_TILE_W;
		hor_mean_msq_filter_krnl<R> <<< dim3(nBlocksW, h), dim3(IMGPROC_TILE_W, 1) >>> (mean, msq, grey, w);
	}
	template <int R>
	void vert_mean_msq_filter_gpu(float *mean, float *msq, const float *mean_in, const float *msq_in, unsigned int w, unsigned int h)
	{
		int nBlocksH = (h + IMGPROC_TILE_H - 1) / IMGPROC_TILE_H;
		vert_mean_msq_filter_krnl<R> <<< dim3(w, nBlocksH), dim3(1, IMGPROC_TILE_H) >>> (mean, msq, mean_in, msq_in, w, h);
	}
	template <int R>
	void hor_min_max_filter_gpu(unsigned char *ming, unsigned char *maxg, const unsigned char *grey, unsigned int w, unsigned int h)
	{
		int nBlocksW = (w + IMGPROC_TILE_W - 1) / IMGPROC_TILE_W;
		hor_min_max_filter_krnl<R> <<< dim3(nBlocksW, h), dim3(IMGPROC_TILE_W, 1) >>> (ming, maxg, grey, w);
	}
	template <int R>
	void vert_min_max_filter_gpu(unsigned char *ming, unsigned char *maxg, const unsigned char *ming_in, const unsigned char *maxg_in,
								 unsigned int w, unsigned int h)
	{
		int nBlocksH = (h + IMGPROC_TILE_H - 1) / IMGPROC_TILE_H;
		vert_min_max_filter_krnl<R> <<< dim3(w, nBlocksH), dim3(1, IMGPROC_TILE_H) >>> (ming, maxg, ming_in, maxg_in, w, h);
	}

	// THRESHOLDING METHODS

	__global__ static void singh_binarize_krnl(unsigned char *grey, const float *mean, unsigned int n, float K)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			float I = grey[idx]/255.f;
			float m = mean[idx];
			float dev = I - m;
			float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
			grey[idx] = (I < threshold) ? 0 : 255;
		}
	}

	template <int R, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void singh_binarize_krnl2(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
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
		bin[y*w + x] = (I < threshold) ? 0 : 255;
	}

	__global__ static void singh_binarize_krnl3(unsigned char *grey, const long long *integ, int w, int h, int R, float K)
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
			grey[idx] = (I < threshold) ? 0 : 255;
		}
	}

	template <mdk_threshold T>
	__global__ void mdk_binarize_krnl(unsigned char *grey, const float *mean, const float *msq, unsigned int n, float K)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			float I = grey[idx]/255.f;
			float m = mean[idx];
			float d2 = min(msq[idx] - m*m, 0.25f);
			grey[idx] = choose_threshold<T>(I, m, d2, K) ? 0 : 255;
		}
	}

	template <int R, mdk_threshold T, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void mdk_binarize_krnl2(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
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
		float d2 = min(sq/den2 - m*m, 0.25f);
		bin[y*w + x] = choose_threshold<T>(I, m, d2, K) ? 0 : 255;
	}

	__global__ static void bernsen_binarize_krnl(unsigned char *grey, const unsigned char *ming, const unsigned char *maxg,
												 unsigned int n, unsigned char L, unsigned char T)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			int I = grey[idx];
			int mini = ming[idx], maxi = maxg[idx];
			int mid = (maxi + mini)/2;
			if (maxi - mini < L)
				grey[idx] = (mid < T) ? 0 : 255;
			else
				grey[idx] = (I < mid) ? 0 : 255;
		}
	}

	template <int R, int TILE = IMGPROC_BLOCK_SIZE_2D + 2*R>
	__global__ void bernsen_binarize_krnl2(unsigned char *bin, const unsigned char *grey, int w, int h, unsigned char L, unsigned char T)
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
			bin[y*w + x] = (mid < T) ? 0 : 255;
		else
			bin[y*w + x] = (I < mid) ? 0 : 255;
	}

	__global__ static void global_binarize_krnl(unsigned char *grey, unsigned int n, unsigned char threshold)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			grey[idx] = (grey[idx] < threshold) ? 0 : 255;
		}
	}

	inline void singh_binarize_gpu(unsigned char *grey, const float *mean, unsigned int n, float K)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		singh_binarize_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, mean, n, K);
	}
	template <int R>
	void singh_binarize_gpu2(unsigned char *bin, const unsigned char *grey, unsigned int w, unsigned int h, float K)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		singh_binarize_krnl2<R> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, K);
	}
	inline void singh_binarize_gpu3(unsigned char *grey, const long long *integ, unsigned int w, unsigned int h, int R, float K)
	{
		int nBlocks = (w*h + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		singh_binarize_krnl3 <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, integ, w, h, R, K);
	}
	template <mdk_threshold T>
	void mdk_binarize_gpu(unsigned char *grey, const float *mean, const float *msq, unsigned int n, float K)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		mdk_binarize_krnl<T> <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, mean, msq, n, K);
	}
	template <int R, mdk_threshold T>
	void mdk_binarize_gpu2(unsigned char *bin, const unsigned char *grey, int w, int h, float K)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		mdk_binarize_krnl2<R, T> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, K);
	}
	inline void bernsen_binarize_gpu(unsigned char *grey, const unsigned char *ming, const unsigned char* maxg,
									 unsigned int n, unsigned char L, unsigned char T)
	{
		int nBlocks = (n + IMGPROC_BLOCK_SIZE - 1) / IMGPROC_BLOCK_SIZE;
		bernsen_binarize_krnl <<< nBlocks, IMGPROC_BLOCK_SIZE >>> (grey, ming, maxg, n, L, T);
	}
	template <int R>
	void bernsen_binarize_gpu2(unsigned char *bin, const unsigned char *grey, int w, int h, unsigned char L, unsigned char T)
	{
		int nBlocksW = (w + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		int nBlocksH = (h + IMGPROC_BLOCK_SIZE_2D - 1) / IMGPROC_BLOCK_SIZE_2D;
		bernsen_binarize_krnl2<R> <<< dim3(nBlocksW, nBlocksH), dim3(IMGPROC_BLOCK_SIZE_2D, IMGPROC_BLOCK_SIZE_2D) >>> (bin, grey, w, h, L, T);
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

	template <int R = 15>
	extern void singh_gpu2(unsigned char *raw, unsigned int w, unsigned int h, float K = 0.06f)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		singh_binarize_gpu2<R>(d_bin, d_raw, w, h, K);

		gpuErrchk(cudaMemcpy(raw, d_bin, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	extern inline void singh_gpu3(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = 0.06f)
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

		singh_binarize_gpu3(d_raw, d_integ, w, h, R, K);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}

	template <int R = 15, mdk_threshold T>
	extern void mdk_gpu(unsigned char *raw, unsigned int w, unsigned int h, float K)
	{
		unsigned int n = w * h;
		// input/output (unsigned char), horizontal mean (float), mean (float), horizontal msq (float), msq (float)
		auto d_buf = simple_alloc_gpu(n*(4*sizeof(float) + sizeof(unsigned char)));
		float *d_hor_mean =(float*)d_buf;
		float *d_mean = d_hor_mean + n;
		float *d_hor_msq = d_mean + n;
		float *d_msq = d_hor_msq + n;
		unsigned char *d_raw = (unsigned char*)(d_msq + n);

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		hor_mean_msq_filter_gpu<R>(d_hor_mean, d_hor_msq, d_raw, w, h);
		vert_mean_msq_filter_gpu<R>(d_mean, d_msq, d_hor_mean, d_hor_msq, w, h);

		mdk_binarize_gpu<T>(d_raw, d_mean, d_msq, n, K);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}
	template <int R = 15, mdk_threshold T>
	extern void mdk_gpu2(unsigned char *raw, unsigned int w, unsigned int h, float K)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		mdk_binarize_gpu2<R, T>(d_bin, d_raw, w, h, K);

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
	extern void niblack_gpu2(unsigned char *raw, unsigned int w, unsigned int h, float K = -0.2f)
	{
		mdk_gpu2<R, mdk_niblack>(raw, w, h, K);
	}
	template <int R = 15>
	extern void sauvola_gpu2(unsigned char *raw, unsigned int w, unsigned int h, float K = 0.06f)
	{
		mdk_gpu2<R, mdk_sauvola>(raw, w, h, K);
	}
	template <int R = 15>
	extern void bernsen_gpu(unsigned char *raw, unsigned int w, unsigned int h, unsigned char L = 25, unsigned char T = 127)
	{
		unsigned int n = w * h;
		// input/output (unsigned char), horizontal min/max (unsigned char*2), min/max (unsigned char*2)
		auto d_buf = simple_alloc_gpu(n*(5*sizeof(unsigned char)));
		unsigned char *d_hor_min = (unsigned char*)d_buf;
		unsigned char *d_hor_max = d_hor_min + n;
		unsigned char *d_min = d_hor_max + n;
		unsigned char *d_max = d_min + n;
		unsigned char *d_raw = (unsigned char*)(d_max + n);

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		hor_min_max_filter_gpu<R>(d_hor_min, d_hor_max, d_raw, w, h);
		vert_min_max_filter_gpu<R>(d_min, d_max, d_hor_min, d_hor_max, w, h);

		bernsen_binarize_gpu(d_raw, d_min, d_max, n, L, T);

		gpuErrchk(cudaMemcpy(raw, d_raw, n, cudaMemcpyDeviceToHost)); // copy data from GPU to CPU
	}
	template <int R = 15>
	extern void bernsen_gpu2(unsigned char *raw, unsigned int w, unsigned int h, unsigned char L = 25, unsigned char T = 127)
	{
		unsigned int n = w * h;
		auto d_buf = simple_alloc_gpu(2*n*sizeof(unsigned char));
		unsigned char *d_raw = (unsigned char*)d_buf;
		unsigned char *d_bin = d_raw + n;

		gpuErrchk(cudaMemcpy(d_raw, raw, n, cudaMemcpyHostToDevice)); // copy data from CPU to GPU

		bernsen_binarize_gpu2<R>(d_bin, d_raw, w, h, L, T);

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


























