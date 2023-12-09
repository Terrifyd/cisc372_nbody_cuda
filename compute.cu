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
//	int n - dimmensions of the section of accels_d that each thread will compute, for example if n is 3 then each thread
//		will calculate a 3x3 box of the accels_d matrix. Each thread does nxn computations 
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
	//printf("hPos_d[1][0] holds %lf\n", hPos_d[1][0]);
	//FILL_VECTOR(accels_d[0][0], 1.0, 2.0, 3.0);
	//printf("%lf\n", accels_d[0][0][2]);	
	//printf("%d", NUMENTITIES);	
	
	int x = threadIdx.x; // x coordinate of thread
	int y = threadIdx.y; // y coordinate of threa
	int start_x = x * n;
	int start_y = y * n;	
	int end_x = start_x + n;
	int end_y = start_y + n;
	
	int i,j,k;
	i = start_x;
	j = start_y;
	// first compute the pairwise accelerations.  Effect is on the first argument.
	// want to make a kernal call here?
	//if (x == 0 && y == 0) {printf("~~~ n=%d, i=%d, j=%d\n", n, i, j);} 
	for (i; i < end_x; i++){
		for (j; j < end_y; j++){
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
/*
				if (i < 10 && j < 10) {
					printf("accels_d[%d][%d] = (%lf,%lf,%lf)\n", 
						i, 
						j, 
						accels_d[i][j][0], 
						accels_d[i][j][1], 
						accels_d[i][j][2]);
				}	
*/
			}
		}
	}
	__syncthreads();
	int a = 8;
	int b = 2;
	if (x==0 && y==0){
		//printf("accels_d holds (%lf, %lf, %lf)\n", accels_d[a][b][0], accels_d[a][b][1], accels_d[a][b][2]);
	}
}

__global__ void cuda_init_accels(vector3* values_d, vector3** accels_d, int numObjects) {
	for (int i = 0; i < numObjects; i++) {
		accels_d[i] = &values_d[i * numObjects];
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
			if (i == 0) {
				printf("accels[%d][%d] is (%.10f, %.10f, %.10f)\n", i, j, accels_d[i][j][0], accels_d[i][j][1], accels_d[i][j][2]);
			}
			for (k=0;k<3;k++)
				accel_sum[k]+=accels_d[i][j][k];
		}
		if (i == 0) {
			printf("~accel_sum[%d] is (%.10f, %.10f, %.10f)\n", i, accel_sum[0], accel_sum[1], accel_sum[2]);
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel_d[i][k]+=accel_sum[k]*INTERVAL;
			hPos_d[i][k]+=hVel_d[i][k]*INTERVAL;
		}
	}
	int  x = 0;
	printf("hVel_d[%d] holds (%.10f, %.10f, %.10f)\n", x, hVel_d[x][0], hVel_d[x][1], hVel_d[x][2]);
	printf("hPos_d[%d] holds (%.10f, %.10f, %.10f)\n\n", x, hPos_d[x][0], hPos_d[x][1], hPos_d[x][2]);
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
/*
				if (i < 10 && j < 10) {
					printf("accels[%d][%d] = (%lf,%lf,%lf)\n", 
						i, 
						j, 
						accels[i][j][0], 
						accels[i][j][1], 
						accels[i][j][2]);
				}	
*/
			}
		}
	}
	//printf("accels holds (%lf, %lf, %lf)\n", accels[10][15][0], accels[10][15][1], accels[10][15][2]);

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


