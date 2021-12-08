# Thresholding algorithms for binarization

This repository contains a C++/CUDA implementation of some local thresholding algorithms for binarization:
- Singh's thresholding (T.R. Singh, S. Roy, O.I. Singh, T. Sinam, K.M. Singh, *A New Local Adaptive Thresholding Technique in Binarization*, IJCSI, Vol. 8, Issue 6, No 2, November 2011)
- Niblack's thresholding (W. Niblack, *An introduction to digital image processing*, Prentice-Hall, Englewood Cliffs, NJ, 1986)
- Sauvola's thresholding (J. Sauvola, M. Pietikainen, *Adaptive document image binarization*, Pattern Recognition 33(2), 2000
- Bernsen's thresholding (J. Bernsen, *Dynamic thresholding of gray-level images*, Proc. 8th Int. Conf. on Pattern Recognition, Paris, 1986)
- Chan's algorithm for memory-efficient implementation of previous thresholding techniques in CPU code (C. Chan, *Memory-efficient and fast implementation of local adaptive binarization methods*, School of Mathematics, Sun Yat-Sen University, China, 2019)

This program uses the STB library for image loading and writing.

## Compilation

  `nvcc main.cu -o ltbin -O3 -std=c++11 -arch=sm_<xy>`

`<xy>` is the compute capability of the GPU (usually given in the form x.y),
for example `sm_21` corresponds to a compute capability of 2.1.

It's also possible to compile the code in non-CUDA-capable systems, for which the GPU-acceleration will not be available, by changing the extension of `main.cu` to `main.cpp`. 

Compilation with GCC:

  `g++ -std=c++11 -O3 -Wall main.cpp -o ltbincpu`
