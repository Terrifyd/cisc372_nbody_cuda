#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

//cuda_compute: Calculates the accels of object in the system on the gpu using cuda
//Parameters:
//	vector3* hVel_d - pointer to an array on the device that holds vector3's for each objects velocity
//	vector3* hPos_d - pointer to an array on the device that holds vector3's for each objects position
//	double* mass_d - pointer to an array on the device that holds the mass of each object
//	vector3** accels_d - a pointer to a 2D array on the device that will be used to store the pairwise acceleration 
//		of each object
//	int n - dimmensions of the section of accels_d that each thread will compute, for example if n is 3 then each thread
//		will calculate a 3x3 box of the accels_d matrix. Each thread does nxn computations 
//Returns: none 
//Side Effect: modifies the accels_d array and fills it with pairwise accelerations
__global__ void cuda_compute(vector3* hVel_d,
		vector3* hPos_d,
		double* mass_d,
		vector3** accels_d, 
		int n) {

	// Unique (x,y) coordinate of the thread
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	// x and y coordinates in accels where the thread will start and stop computing (based on n)
	int start_x = x * n; 
	int start_y = y * n;	
	int end_x = start_x + n;
	int end_y = start_y + n;
	
	int i,j,k;
	for (i = start_x; i < end_x; i++){
		for (j = start_y; j < end_y; j++){
			if (i==j && i < NUMENTITIES && j < NUMENTITIES) {
				FILL_VECTOR(accels_d[i][j],0,0,0);
			}
			else if(i < NUMENTITIES && j < NUMENTITIES) {
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos_d[i][k]-hPos_d[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass_d[j]/magnitude_sq;
				FILL_VECTOR(accels_d[i][j],
					accelmag*distance[0]/magnitude,
					accelmag*distance[1]/magnitude,
					accelmag*distance[2]/magnitude);

			}
		}
	}
}

__global__ void cuda_init_accels(vector3* values_d, vector3** accels_d, int numObjects) {
	for (int i = 0; i < numObjects; i++) {
		accels_d[i] = &values_d[i * numObjects];
	}
}
	
//cuda_reduction: Uses rucuction to sum the accels of each object and calculate new hVel and hPos
//Parameters:
//	vector3* hVel_d - pointer to an array on the device that holds vector3's for each objects velocity
//	vector3* hPos_d - pointer to an array on the device that holds vector3's for each objects position
//	double* mass_d - pointer to an array on the device that holds the mass of each object
//	vector3** accels_d - a pointer to a 2D array on the device that will be used to store the pairwise acceleration 
//		of each object 
//Returns: none 
//Side Effect: modifies the hVel_d and hPos_d arrays with new velocities and positions after 1 interval
__global__ void cuda_reduction(vector3* hVel_d, vector3* hPos_d, vector3** accels_d) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int k;
	//bool overload = false;
		
	int numThreads = blockDim.x;
	int numElements = NUMENTITIES;

	// for if there are too many entities for the number of threads
	if (numElements > (numThreads * 2)) {
		//overload = true;
		int jump;
		int n = 1;
		while (numElements > (numThreads * n)) {
			n++;
		}

		while (n > 2) {
			jump = (n - 1) * 1024;
			if ((j + jump) < numElements) {
				for (k = 0; k < 3; k++) {  	
					accels_d[i][j][k] += accels_d[i][j+jump][k];		
				}
			}
			n--;
			numElements = jump;
			__syncthreads();
		}
	}
	
	// thread j will add the elements of index j and j + stride together, dividing stride by 2 each time
	// because of this stride should not be greater than the number of threads 
	int stride = 1;
	while ((stride * 2) < numElements) {
		stride *= 2;
	}
	//if (i == 0 && j == 0) {printf("STRIDE IS %d\n\n", stride);}
	while (stride > 0) {
		//printf("in loop");
		//if (i == 0 && j == 0) {printf("\nstride is %d\n", stride);}
		if (j < stride) {
			if ((j + stride) < numElements) {
				for (k = 0; k < 3; k++) {
					//if (i == 0 && j == 0) {printf("(%f in %d added to %f in ind %d)\n", accels_d[i][j+stride][k], j+stride, accels_d[i][j][k], j);}
					accels_d[i][j][k] += accels_d[i][j+stride][k];
				}
			}
		}	
		__syncthreads();
		stride >>= 1;
	}
	
	if (i == 9 && j == 0) {
		//printf("accel sums of %d are (%f, %f, %f)\n", i, accels_d[i][0][0], accels_d[i][0][1], accels_d[i][0][2]);
	}
	//if (i == 0 && j ==0) {printf("accel red for %d is"
	if (j == 0) {
		for (k = 0; k < 3; k++) {
			//if ( i == 0 && j == 0) {printf("accel sum of %d is %f\n", k, accels_d[i][0][k]);}
			hVel_d[i][k] += accels_d[i][0][k] * INTERVAL;
			hPos_d[i][k] += hVel_d[i][k] * INTERVAL;
		}	
	}
	if (i == 9 && j == 0) {
		//printf("hPos is (%f, %f, %f)\n", hPos_d[i][0], hPos_d[i][1], hPos_d[i][2]);
	}
}

// serial summation for testing (still on device to avoid unnecessary memory transfers)
__global__ void cuda_summation(vector3* hVel_d, vector3* hPos_d, vector3** accels_d) {
	int i, j, k;

	// sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// want to make kernal call here?
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		// sum up column
		for (j=0;j<NUMENTITIES;j++){
/*
			if (i == 1) {
				printf("accels[%d][%d] is (%.10f, %.10f, %.10f)\n", i, j, accels_d[i][j][0], accels_d[i][j][1], accels_d[i][j][2]);
			}
*/
			for (k=0;k<3;k++)
				accel_sum[k]+=accels_d[i][j][k];
		}
/*
		if (i == 1) {
			printf("~accel_sum[%d] is (%.10f, %.10f, %.10f)\n", i, accel_sum[0], accel_sum[1], accel_sum[2]);
		}
*/
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel_d[i][k]+=accel_sum[k]*INTERVAL;
			hPos_d[i][k]+=hVel_d[i][k]*INTERVAL;
		}
	}
/*
	int  x = 1;
	printf("hVel_d[%d] holds (%.10f, %.10f, %.10f)\n", x, hVel_d[x][0], hVel_d[x][1], hVel_d[x][2]);
	printf("hPos_d[%d] holds (%.10f, %.10f, %.10f)\n\n", x, hPos_d[x][0], hPos_d[x][1], hPos_d[x][2]);
*/
}


