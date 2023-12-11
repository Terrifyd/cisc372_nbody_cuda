#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"


// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass;
vector3* values;
vector3** accels;
vector3* hVel_d;
vector3* hPos_d;
double* mass_d;
vector3* values_d;
vector3** accels_d;


//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
	
	values = (vector3*)malloc(sizeof(vector3) * numObjects * numObjects);
	accels = (vector3**)malloc(sizeof(vector3*) * numObjects);
	for (int i = 0; i < numObjects; i++) {
		accels[i] = &values[i * numObjects];
	}
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&hVel_d, (sizeof(vector3) * numObjects));	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMalloc((void**)&hPos_d, (sizeof(vector3) * numObjects));	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMalloc((void**)&mass_d, (sizeof(double) * numObjects));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMalloc((void**)&values_d, (sizeof(vector3) * numObjects * numObjects));
	cudaMalloc((void**)&accels_d, (sizeof(vector3*) * numObjects));	
	cuda_init_accels<<<1, 1>>>(values_d, accels_d, numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
	
	free(values);
	free(accels);

	cudaFree(hVel_d);
	cudaFree(hPos_d);
	cudaFree(mass_d);
	
	cudaFree(values_d);
	cudaFree(accels_d);
}


//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

void copy_to_device(int numObjects) {
	cudaError_t cudaStatus;

	cudaMemcpy(hVel_d, hVel, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(hPos_d, hPos, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(mass_d, mass, sizeof(double) * numObjects, cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

}

__global__ void cuda_test(int* deviceArray) {
	int thread_x = threadIdx.x;
	printf("thread %d working\n", thread_x);
	deviceArray[thread_x] = deviceArray[thread_x] * 2;
	printf("thread %d placed %d in deviceArray\n", thread_x, deviceArray[thread_x]);
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	//clock_t t9=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
#ifdef DEBUG
	//printSystem(stdout);
#endif

	copy_to_device(NUMENTITIES);
	
	int n = 1;
	int dims = 1;
	while ((dims * dims * n * n * 1024) < (NUMENTITIES * NUMENTITIES)) {
		dims++;
	} 
	dim3 gridDim(dims, dims);
	dim3 blockDim(32, 32);
	
	clock_t c0, s0;
	clock_t c1 = 0;
	clock_t s1 = 0;
	for (t_now=0;t_now<(INTERVAL*5);t_now+=INTERVAL) {
		c0 = clock();
		cuda_compute<<<gridDim, blockDim>>>(hVel_d, hPos_d, mass_d, accels_d, n);
		c1 += clock() - c0;
		s0 = clock();
		//cuda_summation<<<1, 1>>>(hVel_d, hPos_d, accels_d);
		cuda_reduction<<<NUMENTITIES, 1024>>>(hVel_d, hPos_d, accels_d);
		s1 += clock() - s0;
	}

	// copying back for serial summation (need to switch for reduction later)
	vector3* hPos_dth = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES);
	vector3* hVel_dth = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES);
	cudaDeviceSynchronize();
	cudaMemcpy(hPos, hPos_d, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, hVel_d, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
/*
	for (int a = 0; a < 100; a++) {
		for (int b = 0; b < 3; b++) {

			//printf("hPos[%d][%d] holds %lf\n", a, b, hPos[a][b]);
			//printf("hPos[%d][%d] holds %lf\n", a, b, hPos_dth[a][b]); 
		}
	}
*/


	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("Computation spent %f seconds in compute and %f seconds in reduction\n", 
		(double)c1/CLOCKS_PER_SEC, (double)s1/CLOCKS_PER_SEC);
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}


