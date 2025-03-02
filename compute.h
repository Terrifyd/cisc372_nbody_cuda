#include <cuda.h>
#include <cuda_runtime.h>
void compute();
__global__ void cuda_compute(vector3* hVel_d, vector3* hPos_d, double* mass_d, vector3** accels_d, int n);
__global__ void cuda_init_accels(vector3* values_d, vector3** accels_d, int numObjects);
__global__ void cuda_summation(vector3* hVel_d, vector3* hPos_d, vector3** accels_d); 
__global__ void cuda_reduction(vector3* hVel_d, vector3* hPos_d, vector3** accels_d); 
void nothing_test();
