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

#define IMGPROC_CACHELINE 64

namespace imgproc
{

	inline void singh_binarize(unsigned char *grey, const long long *integ, int w, int h, int R, float K)
	// grey is both the input greyscale image and the output binarized image (inplace algorithm)
	{
		float den = (2*R+1)*(2*R+1)*255.f;
		for (int i = 0; i < h; ++i)
			for (int j = 0; j < w; ++j)
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
			for (int j = 0; j < w; ++j)
			{
				float I = grey[i*w + j]/255.f;
				float m = std::min(take_local_var(integ, j, i, w, R)/den, 1.f);
				float s = take_local_var(sq_integ, j, i, w, R)/den2;
				float d2 = clamp(s - m*m, 0.f, 0.25f); // maximum std.dev. is 0.5
				grey[i*w + j] = t(I, m, d2, K) ? 0 : 255;
			}
	}

	inline void hor_min_max_filter(unsigned char *ming, unsigned char *maxg, unsigned char *buf, const unsigned char *grey, int w, int h, int R)
	// buf must be 4*w bytes
	{
		unsigned char *minf = buf;
		unsigned char *minh = minf + w;
		unsigned char *maxf = minh + w;
		unsigned char *maxh = maxf + w;
		int W = 2*R + 1;
		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
			{
				unsigned char g = grey[i*w + j];
				if (j % W == 0)
				{
					minf[j] = g;
					maxf[j] = g;
				}
				else
				{
					minf[j] = std::min(g, minf[j-1]);
					maxf[j] = std::max(g, maxf[j-1]);
				}
			}
			unsigned char g = grey[i*w + w-1];
			minh[w-1] = g;
			maxh[w-1] = g;
			for (int j = w-2; j >= 0; --j)
			{
				unsigned char g = grey[i*w + j];
				if (j % W == W-1)
				{
					minh[j] = g;
					maxh[j] = g;
				}
				else
				{
					minh[j] = std::min(g, minh[j+1]);
					maxh[j] = std::max(g, maxh[j+1]);
				}
			}
			for (int j = 0; j < w; ++j)
			{
				ming[i*w + j] = std::min(minf[std::min(j+R, w-1)], minh[std::max(j-R, 0)]);
				maxg[i*w + j] = std::max(maxf[std::min(j+R, w-1)], maxh[std::max(j-R, 0)]);
			}
		}
	}
	inline void vert_min_max_filter(unsigned char *ming, unsigned char *maxg, unsigned char *buf, const unsigned char *ming_in, const unsigned char *maxg_in,
									int w, int h, int R)
	// buf must be 2*h*IMGPROC_CACHELINE bytes
	{
		unsigned char *mf = buf;
		unsigned char *mh = mf + h*IMGPROC_CACHELINE;
		int W = 2*R + 1;
		for (int j = 0; j < w; j += IMGPROC_CACHELINE)
		{
			int maxk = std::min(j + IMGPROC_CACHELINE, w) - j;
			for (int i = 0; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char g = ming_in[i*w + j+k];
					mf[i*maxk + k] = (i % W == 0) ? g : std::min(g, mf[(i-1)*maxk + k]);
				}

			for (int k = 0; k < maxk; ++k)
				mh[(h-1)*maxk + k] = ming_in[(h-1)*w + j+k];
			for (int i = h-2; i >= 0; --i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char g = ming_in[i*w + j+k];
					mh[i*maxk + k] = (i % W == W-1) ? g : std::min(g, mh[(i+1)*maxk + k]);
				}

			for (int i = 0; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
					ming[i*w + j+k] = std::min(mf[std::min(i+R, h-1)*maxk + k], mh[std::max(i-R, 0)*maxk + k]);
		}
		for (int j = 0; j < w; j += IMGPROC_CACHELINE)
		{
			int maxk = std::min(j + IMGPROC_CACHELINE, w) - j;
			for(int i = 0; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char g = maxg_in[i*w + j+k];
					mf[i*maxk + k] = (i % W == 0) ? g : std::max(g, mf[(i-1)*maxk + k]);
				}

			for (int k = 0; k < maxk; ++k)
				mh[(h-1)*maxk + k] = maxg_in[(h-1)*w + j+k];
			for (int i = h-2; i >= 0; --i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char g = maxg_in[i*w + j+k];
					mh[i*maxk + k] = (i % W == W-1) ? g : std::max(g, mh[(i+1)*maxk + k]);
				}

			for (int i = 0; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
					maxg[i*w + j+k] = std::max(mf[std::min(i+R, h-1)*maxk + k], mh[std::max(i-R, 0)*maxk + k]);
		}
	}

	inline void hor_min_max_filter2(unsigned char *ming, unsigned char *maxg, const unsigned char *grey, int w, int h, int R)
	{
		for (int i = 0; i < h; ++i)
		{
			unsigned char last_min = 255, last_max = 0;
			for (int dx = 0; dx <= R; ++dx)
				last_min = std::min(last_min, grey[i*w + std::min(dx, w-1)]);
			for (int dx = 0; dx <= R; ++dx)
				last_max = std::max(last_max, grey[i*w + std::min(dx, w-1)]);
			ming[i*w] = last_min;
			maxg[i*w] = last_max;
			for (int j = 1; j < w; ++j)
			{
				unsigned char old_g = grey[i*w + std::max(j-R-1, 0)];
				unsigned char new_g = grey[i*w + std::min(j+R, w-1)];
				last_min = std::min(last_min, new_g);
				last_max = std::max(last_max, new_g);
				if (j > R && old_g != new_g) // if oldest grey level is equal to the newest, then min/max is the same as the last one
				{
					if (last_min == old_g) // if the oldest grey level is equal to the previous local min, then current local min might be different
					{
						last_min = new_g;
						for (int dx = -R; dx < R; ++dx)
							last_min = std::min(last_min, grey[i*w + clamp(j+dx, 0, w-1)]);
					}
					if (last_max == old_g) // same with max
					{
						last_max = new_g;
						for (int dx = -R; dx < R; ++dx)
							last_max = std::max(last_max, grey[i*w + clamp(j+dx, 0, w-1)]);
					}
				}
				ming[i*w + j] = last_min;
				maxg[i*w + j] = last_max;
			}
		}
	}
	inline void vert_min_max_filter2(unsigned char *ming, unsigned char *maxg, const unsigned char *ming_in, const unsigned char *maxg_in,
									 int w, int h, int R)
	{
		static unsigned char last_m[IMGPROC_CACHELINE];
		for (int j = 0; j < w; j += IMGPROC_CACHELINE)
		{
			int maxk = std::min(j + IMGPROC_CACHELINE, w) - j;
			for (int k = 0; k < maxk; ++k)
				last_m[k] = 255;
			for (int dy = 0; dy <= R; ++dy)
				for (int k = 0; k < maxk; ++k)
					last_m[k] = std::min(last_m[k], ming_in[std::min(dy, h-1)*w + j+k]);
			for (int k = 0; k < maxk; ++k)
				ming[j+k] = last_m[k];
			for (int i = 1; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char old_g = ming_in[std::max(i-R-1, 0)*w + j+k];
					unsigned char new_g = ming_in[std::min(i+R, h-1)*w + j+k];
					last_m[k] = std::min(last_m[k], new_g);
					if (i > R && old_g != new_g) // same as before
					{
						if (last_m[k] == old_g)
						{
							last_m[k] = new_g;
							for (int dy = -R; dy < R; ++dy)
								last_m[k] = std::min(last_m[k], ming_in[clamp(i+dy, 0, h-1)*w + j+k]);
						}
					}
					ming[i*w + j+k] = last_m[k];
				}
		}

		for (int j = 0; j < w; j += IMGPROC_CACHELINE)
		{
			int maxk = std::min(j + IMGPROC_CACHELINE, w) - j;
			for (int k = 0; k < maxk; ++k)
				last_m[k] = 0;
			for (int dy = 0; dy <= R; ++dy)
				for (int k = 0; k < maxk; ++k)
					last_m[k] = std::max(last_m[k], maxg_in[std::min(dy, h-1)*w + j+k]);
			for (int k = 0; k < maxk; ++k)
				maxg[j+k] = last_m[k];
			for (int i = 1; i < h; ++i)
				for (int k = 0; k < maxk; ++k)
				{
					unsigned char old_g = maxg_in[std::max(i-R-1, 0)*w + j+k];
					unsigned char new_g = maxg_in[std::min(i+R, h-1)*w + j+k];
					last_m[k] = std::max(last_m[k], new_g);
					if (i > R && old_g != new_g) // same as before
					{
						if (last_m[k] == old_g)
						{
							last_m[k] = new_g;
							for (int dy = -R; dy < R; ++dy)
								last_m[k] = std::max(last_m[k], maxg_in[clamp(i+dy, 0, h-1)*w + j+k]);
						}
					}
					maxg[i*w + j+k] = last_m[k];
				}
		}
	}

	inline void bernsen_binarize(unsigned char *grey, const unsigned char *ming, const unsigned char *maxg, unsigned int n, int L, int T)
	{
		for (unsigned int idx = 0; idx < n; ++idx)
		{
			int I = grey[idx];
			int mid = maxg[idx] + ming[idx];
			if (maxg[idx] - ming[idx] < L)
				grey[idx] = (mid < 2*T) ? 0 : 255;
			else
				grey[idx] = (2*I < mid) ? 0 : 255;
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
		auto buf = simple_alloc((4*n + std::max(4*w, 2*h*IMGPROC_CACHELINE))*sizeof(unsigned char));
		unsigned char *hor_min = (unsigned char*)buf;
		unsigned char *hor_max = hor_min + n;
		unsigned char *ming = hor_max + n;
		unsigned char *maxg = ming + n;
		unsigned char *fil_buf = maxg + n;

		hor_min_max_filter(hor_min, hor_max, fil_buf, raw, w, h, R);
		vert_min_max_filter(ming, maxg, fil_buf, hor_min, hor_max, w, h, R);

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
				float d2 = clamp(s/(255*255*n) - m*m, 0.f, 0.25f);
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

























