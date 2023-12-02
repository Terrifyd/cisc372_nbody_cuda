#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void cuda_compute(int n) {
	int thread_x = blockDim.x;
	int thread_y = threadIdx.y;
	
	if (thread_x == 1 && thread_y == 1) {
		printf("--n is %i\n", n);
	}
	
	printf("thread (%i, %i)", thread_x, thread_y);
}

#ifdef __cplusplus
}
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
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];

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
	free(accels);
	free(values);
}


