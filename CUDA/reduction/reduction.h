#ifndef __REDUCTION_H__
#define __REDUCTION_H__

template <class T>
void reduce(int size, int threads, int blocks, int smemSize,
            int whichKernel, T *d_in, T *d_out);

#endif