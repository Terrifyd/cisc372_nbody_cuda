#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
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
	
	cudaMalloc((void**)&hVel_d, (sizeof(vector3) * numObjects));	
	cudaMalloc((void**)&hPos_d, (sizeof(vector3) * numObjects));	
	cudaMalloc((void**)&mass_d, (sizeof(double) * numObjects));

	cudaMalloc((void**)&values_d, (sizeof(vector3) * numObjects * numObjects));
	cudaMalloc((void**)&accels_d, (sizeof(vector3*) * numObjects));	
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

__global__ void cuda_test(int* deviceArray) {
	int thread_x = threadIdx.x;
	printf("thread %d working\n", thread_x);
	deviceArray[thread_x] = deviceArray[thread_x] * 2;
	printf("thread %d placed %d in deviceArray\n", thread_x, deviceArray[thread_x]);
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
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
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}


/*
	int one = 1;
	dim3 dimGrid, dimBlock;
	dimGrid.x = 1;
	dimGrid.y = 1;
	dimGrid.z = 1;
	dimBlock.x = 32;
	dimBlock.y = 32;
	dimBlock.z = 1;
	//cuda_compute<<<dimGrid, dimBlock>>>(4);	
	
	printf("start test\n");
	int* h_arr;
	int* d_arr;

	h_arr = (int*)malloc(20 * sizeof(int));
	for (int i=0; i < 20; i++) {
		h_arr[i] = i;
	}
	
	cudaMalloc((void**)&d_arr, 20 * sizeof(int));
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(d_arr, h_arr, 20 * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	cuda_compute<<<1,20>>>(d_arr, 4);
	//nothing_test();

	cudaMemcpy(h_arr, d_arr, 20 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}

	for (int i = 0; i < 20; i++) {
		printf("Post kernal value at h_arr[%d] is %d\n", i, h_arr[i]);
	}

	free(h_arr);
	cudaFree(d_arr);

	printf("done test\n");		
	return 0;
*/

/*
	size_t size_c = 256 * sizeof(int);
	int size = 256 * sizeof(int);
	//int *hostArray = new int[10];
	int* hostArray = (int*)malloc(2 * size);

	int* deviceArray;
	cudaMalloc((void**)&deviceArray, size_c);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}


	hostArray[0] = 1;
	printf("~~~Host has %d at 0~~~\n", hostArray[0]);	
	cuda_test<<<1, 256>>>(deviceArray);
	cudaDeviceSynchronize();
	
	printf("before memcpy\n");
	cudaMemcpy(hostArray, deviceArray, size_c, cudaMemcpyDeviceToHost);
	printf("after memcpy\n");
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}
	printf("~~~Host recived %d at 0~~~\n", hostArray[0]);	
*/

	clock_t t1=clock()-t0;
#ifdef DEBUG
	//printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}


