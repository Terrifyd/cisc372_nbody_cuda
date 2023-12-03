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
vector3 *hVel_p;
vector3 *hPos_p;
double *mass_p;
vector3* values_p;
vector3** accels_p;


//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
	

	cudaMalloc((void**)&hVel_p, (sizeof(vector3) * numObjects));	
	cudaMalloc((void**)&hPos_p, (sizeof(vector3) * numObjects));	
	cudaMalloc((void**)&mass_p, (sizeof(double) * numObjects));

	cudaMalloc((void**)&values_p, (sizeof(vector3) * NUMENTITIES * NUMENTITIES));
	cudaMalloc((void**)&accels_p, (sizeof(vector3) * NUMENTITIES * NUMENTITIES));	
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

	free(hVel_p);
	free(hPos_p);
	free(mass_p);
	
	free(values_p);
	free(accels_p);
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

__global__ void cuda_test(int* hostArray) {
	int thread_x = threadIdx.x;
	printf("thread %d working\n", thread_x);
	hostArray[thread_x] = thread_x;
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
		//compute();
	}

	int one = 1;
	dim3 dimGrid, dimBlock;
	dimGrid.x = 1;
	dimGrid.y = 1;
	dimGrid.z = 1;
	dimBlock.x = 32;
	dimBlock.y = 32;
	dimBlock.z = 1;
	//cuda_compute<<<dimGrid, dimBlock>>>(4);	
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
	
	printf("before memcpy\n");
	cudaMemcpy(hostArray, deviceArray, size_c, cudaMemcpyDeviceToHost);
	printf("after memcpy\n");
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed : %s\n", cudaGetErrorString(cudaStatus));
	}
	printf("~~~Host recived %d at 0~~~\n", hostArray[0]);	

	clock_t t1=clock()-t0;
#ifdef DEBUG
	//printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}


