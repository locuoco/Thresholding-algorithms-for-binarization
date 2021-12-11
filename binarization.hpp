//  C++ code for some thresholding/binarization algorithms
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

#ifndef IMGPROC_BINARIZATION_HPP
#define IMGPROC_BINARIZATION_HPP

#include "common.hpp"

#include <algorithm> // min
#include <cmath> // sqrtf

namespace imgproc
{

	inline void singh_binarize(unsigned char *grey, const long long *integ, int w, int h, int R, float K)
	// grey is both the input greyscale image and the output binarized image (inplace algorithm)
	{
		float den = (2*R+1)*(2*R+1)*255.f;
		for (int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j)
			{
				float I = grey[i*w + j]/255.f;
				float m = std::min(take_local_var(integ, j, i, w, R)/den, 1.f);
				float dev = I - m;
				float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
				grey[i*w + j] = (I < threshold) ? 0 : 255;
			}
	}
	inline void mdk_binarize(unsigned char *grey, const long long *integ, const long long *sq_integ, int w, int h, int R, float K,
							 bool (*t)(float I, float m, float d, float K))
	// grey is both the input greyscale image and the output binarized image (inplace algorithm)
	{
		float den = (2*R+1)*(2*R+1)*255;
		float den2 = den*255;
		for (int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j)
			{
				float I = grey[i*w + j]/255.f;
				float m = std::min(take_local_var(integ, j, i, w, R)/den, 1.f);
				float s = take_local_var(sq_integ, j, i, w, R)/den2;
				float d2 = std::min(s - m*m, 0.25f); // maximum std.dev. is 0.5
				grey[i*w + j] = t(I, m, d2, K) ? 0 : 255;
			}
	}
	inline void hor_min_max_filter(unsigned char *ming, unsigned char *maxg, const unsigned char *grey, int w, int h, int R)
	{
		for (int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j)
			{
				int cmin = 255, cmax = 0, g;
				for (int dx = -R; dx <= R; ++dx)
				{
					g = grey[i*w + clamp(j+dx, 0, w-1)];
					cmin = std::min(cmin, g);
					cmax = std::max(cmax, g);
				}
				ming[i*w + j] = cmin;
				maxg[i*w + j] = cmax;
			}
	}
	inline void vert_min_max_filter(unsigned char *ming, unsigned char *maxg, const unsigned char *ming_in, const unsigned char *maxg_in,
									int w, int h, int R)
	{
		for (int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j)
			{
				unsigned char c = 255;
				for (int dy = -R; dy <= R; ++dy)
					c = std::min(c, ming_in[clamp(i+dy, 0, h-1)*w + j]);
				ming[i*w + j] = c;
			}
		for (int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j)
			{
				unsigned char c = 0;
				for (int dy = -R; dy <= R; ++dy)
					c = std::max(c, maxg_in[clamp(i+dy, 0, h-1)*w + j]);
				maxg[i*w + j] = c;
			}
	}
	inline void bernsen_binarize(unsigned char *grey, const unsigned char *ming, const unsigned char *maxg, int n, unsigned char L, unsigned char T)
	{
		for (int idx = 0; idx < n; ++idx)
		{
			int I = grey[idx];
			int mid = (maxg[idx] + ming[idx])/2;
			if (maxg[idx] - ming[idx] < L)
				grey[idx] = (mid < T) ? 0 : 255;
			else
				grey[idx] = (I < mid) ? 0 : 255;
		}
	}

	inline void singh(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = 0.06f)
	{
		auto buf = simple_alloc((w + 2*R+1)*(h + 2*R+1)*sizeof(long long));
		long long *integ = (long long*)buf;

		integral(integ, raw, w, h, R);

		singh_binarize(raw, integ, w, h, R, K);
	}
	inline void mdk(unsigned char *raw, unsigned int w, unsigned int h, int R, float K, bool (*t)(float I, float m, float d, float K))
	{
		auto buf = simple_alloc(2*(w + 2*R+1)*(h + 2*R+1)*sizeof(long long));
		long long *integ = (long long*)buf;
		long long *sq_integ = integ + (w + 2*R+1)*(h + 2*R+1);

		integral(integ, raw, w, h, R);
		sq_integral(sq_integ, raw, w, h, R);

		mdk_binarize(raw, integ, sq_integ, w, h, R, K, t);
	}
	inline void niblack(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = -0.2f)
	{
		mdk(raw, w, h, R, K, niblack_threshold);
	}
	inline void sauvola(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = 0.06f)
	{
		mdk(raw, w, h, R, K, sauvola_threshold);
	}
	inline void bernsen(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, unsigned char L = 15, unsigned char T = 127)
	{
		unsigned int n = w*h;
		auto buf = simple_alloc(4*n*sizeof(unsigned char));
		unsigned char *hor_min = (unsigned char*)buf;
		unsigned char *hor_max = hor_min + n;
		unsigned char *ming = hor_max + n;
		unsigned char *maxg = ming + n;

		hor_min_max_filter(hor_min, hor_max, raw, w, h, R);
		vert_min_max_filter(ming, maxg, hor_min, hor_max, w, h, R);

		bernsen_binarize(raw, ming, maxg, n, L, T);
	}

	// CHAN algorithm

	inline unsigned char clamp_to_border(unsigned char *grey, int x, int y, int w, int h)
	{
		return (x >= 0 && y >= 0 && x < w && y < h) ? grey[y*w + x] : 0;
	}
	inline int clamp_to_border(int *C, int x, int w)
	{
		return (x >= 0 && x < w) ? C[x] : 0;
	}

	inline void singh_chan(unsigned char *raw, int w, int h, int R = 15, float K = 0.06f)
	{
		auto buf = simple_alloc(w*sizeof(int) + w*h*sizeof(unsigned char));
		int *C = (int*)buf;
		unsigned char *grey = (unsigned char*)(C + w);

		std::copy(raw, raw + w*h, grey);

		for (int j = 0; j < w; ++j)
			C[j] = 0;
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < w; ++j)
				C[j] += clamp_to_border(grey, j, i, w, h);
		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
				C[j] += clamp_to_border(grey, j, i+R, w, h) - clamp_to_border(grey, j, i-R-1, w, h);
			int c = 0;
			for (int j = 0; j < R; ++j)
				c += clamp_to_border(C, j, w);
			for (int j = 0; j < w; ++j)
			{
				c += clamp_to_border(C, j+R, w) - clamp_to_border(C, j-R-1, w);
				float I = grey[i*w + j]/255.f;
				float n = (std::min(j+R, w-1) - std::max(j-R-1, -1))*(std::min(i+R, h-1) - std::max(i-R-1, -1));
				float m = c/(n*255);
				float dev = I - m;
				float threshold = m * (1 + K * ( dev / (1 - dev + IMGPROC_EPS) - 1));
				raw[i*w + j] = (I < threshold) ? 0 : 255;
			}
		}
	}

	inline void mdk_chan(unsigned char *raw, int w, int h, int R, float K, bool (*t)(float I, float m, float d, float K))
	{
		auto buf = simple_alloc(2*w*sizeof(int) + w*h*sizeof(unsigned char));
		int *C = (int*)buf;
		int *S = C + w;
		unsigned char *grey = (unsigned char*)(S + w);

		std::copy(raw, raw + w*h, grey);
		int g, g2;

		for (int j = 0; j < w; ++j)
		{
			C[j] = 0;
			S[j] = 0;
		}
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < w; ++j)
			{
				g = clamp_to_border(grey, j, i, w, h);
				C[j] += g;
				S[j] += g*g;
			}
		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
			{
				g = clamp_to_border(grey, j, i+R, w, h);
				g2 = clamp_to_border(grey, j, i-R-1, w, h);
				C[j] += g - g2;
				S[j] += g*g - g2*g2;
			}
			int c = 0;
			int s = 0;
			for (int j = 0; j < R; ++j)
			{
				c += clamp_to_border(C, j, w);
				s += clamp_to_border(S, j, w);
			}
			for (int j = 0; j < w; ++j)
			{
				c += clamp_to_border(C, j+R, w) - clamp_to_border(C, j-R-1, w);
				s += clamp_to_border(S, j+R, w) - clamp_to_border(S, j-R-1, w);
				float I = grey[i*w + j]/255.f;
				float n = (std::min(j+R, w-1) - std::max(j-R-1, -1))*(std::min(i+R, h-1) - std::max(i-R-1, -1));
				float m = c/(n*255);
				float d2 = std::min(s/(255*255*n) - m*m, 0.25f);
				raw[i*w + j] = t(I, m, d2, K) ? 0 : 255;
			}
		}
	}
	inline void niblack_chan(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = -0.2f)
	{
		mdk_chan(raw, w, h, R, K, niblack_threshold);
	}
	inline void sauvola_chan(unsigned char *raw, unsigned int w, unsigned int h, int R = 15, float K = 0.06f)
	{
		mdk_chan(raw, w, h, R, K, sauvola_threshold);
	}

	// simple global thresholding

	inline void global(unsigned char *raw, unsigned int w, unsigned int h, unsigned char threshold = 127)
	{
		unsigned int n = w*h;
		for (unsigned int i = 0; i < n; ++i)
			raw[i] = (raw[i] < threshold) ? 0 : 255;
	}

} // namespace imgproc

#endif // IMGPROC_BINARIZATION_HPP

























