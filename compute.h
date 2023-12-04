#include <cuda_runtime.h>
void compute();
__global__ void cuda_compute(vector3* hVel_d, vector3* hPos_d, double* mass_d, vector3** accels_d, int n);
void nothing_test();
