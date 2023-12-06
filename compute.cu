#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#ifdef __cplusplus
//extern "C" {
#endif

//cuda_compute: Calculates the accels of object in the system on the gpu using cuda
//Parameters:
//	vector3* hVel_d - pointer to an array on the device that holds vector3's for each objects velocity
//	vector3* hPos_d - pointer to an array on the device that holds vector3's for each objects position
//	double* mass_d - pointer to an array on the device that holds the mass of each object
//	vector3** accels_d - a pointer to a 2D array on the device that will be used to store the pairwise acceleration 
//		of each object
//Returns: none 
//Side Effect: modifies the hVel_d and hPos_d arrays with new velocities and positions after 1 interval
__global__ void cuda_compute(vector3* hVel_d,
		vector3* hPos_d,
		double* mass_d,
		vector3** accels_d, 
		int n) {
	//int thread_x = threadIdx.x;
	//int thread_y = threadIdx.y;
	//printf("thread (%i, %i)", thread_x, thread_x);
	//d_arr[thread_x] = d_arr[thread_x] * n;
	//printf("TEST\n");	
	printf("hPos_d[1][0] holds %lf\n", hPos_d[1][0]);
	FILL_VECTOR(accels_d[0][0], 1.0, 2.0, 3.0);
	printf("%lf\n", accels_d[0][0][2]);	
	printf("%d", NUMENTITIES);	
}

__global__ void cuda_init_accels(vector3* values_d, vector3** accels_d, int numObjects) {
	for (int i = 0; i < numObjects; i++) {
		accels_d[i] = &values_d[i * numObjects];
	}
}	

void nothing_test() {
	printf("HELLO");
}

#ifdef __cplusplus
//}
#endif

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	// values is 1d array and accels is 2d array
	// values is a pointer to start of an array and accels is a pointer to the pointer of values?
	// need both on device memory or maybe redo into one memory allocation
	// accels is the array that matters, it is a 2d array of vectors and each vector has 3 elements
	// accels[i][j][k] points to element [k] of the vector at [i][j]
	int i,j,k;
	
/*
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
*/

	// first compute the pairwise accelerations.  Effect is on the first argument.
	// want to make a kernal call here?
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}

	// sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// want to make kernal call here?
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	//free(accels);
	//free(values);
}


