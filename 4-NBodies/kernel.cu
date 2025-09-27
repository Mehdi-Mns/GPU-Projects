/**
 * @file kernel.cu
 * @brief CPU and GPU implementations of N-body simulation visualized with OpenGL.
 *
 * This file implements an N-body gravitational simulation using both CPU and GPU.
 * Several GPU modes are available, including standard, double buffer, and shared memory.
 * OpenGL is used to render the bodies in 3D space with a trackball camera.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern "C" {
#include "camera.h"
}

// OpenGL Graphics includes
#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

// Screen dimensions and constants
#define SCREEN_X 800
#define SCREEN_Y 800
#define FPS_UPDATE 200 
#define TITLE "N-Body"

#define MASSMAX 5.0f

#define EPSILON 0.1						// Softening factor to avoid division by zero
//#define G (6.67 / 100000000000)
#define G 0.0000001						// Gravitational constant (scaled down for simulation)

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU_BUFFER_MODE 3
#define GPU_SM_MODE 4					// GPU with shared memory optimization

#define NBTHREADS 512					// Threads per block for CUDA kernels


// -------------------- Global Variables --------------------

int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4 *pos = NULL, *vel = NULL;			// Host arrays for positions and velocities
float* mass = NULL;							// Host array for masses

// Device arrays (GPU memory)
float4* d_pos_i = NULL, * d_pos_o = NULL; 
float4* d_vel_i = NULL, * d_vel_o = NULL;
float* d_mass = NULL;

int nbBodies = 1024;						// Number of simulated bodies
int buffer = 0;								// Double-buffer toggle


// -------------------- Initialization Helpers --------------------

/**
 * Generate random float4 array (positions or velocities).
 * @param n Number of elements.
 * @param d Scaling factor for radius.
 * @return Allocated array of float4 with w = 1.0.
 */
float4* randomArrayFloat4(int n, float d)
{
	float4* a = (float4*)malloc(n*sizeof(float4));
	float x, y, z, r;
	int i;
	for (i = 0; i < n; i++)
	{
		x = (2 * ((rand() % 1000) / 1000.0f) - 1);
		y = (2 * ((rand() % 1000) / 1000.0f) - 1);
		z = (2 * ((rand() % 1000) / 1000.0f) - 1);
		r = (rand() % 1000) / 1000.0f / sqrt(x*x + y*y + z*z);

		a[i].x = r*d*x;
		a[i].y = r*d*y;
		a[i].z = r*d*z;
		a[i].w = 1.0f; // must be 1.0
	}
	return a;
}

/**
 * Generate random float array (masses).
 * @param n Number of elements.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Allocated array of floats.
 */
float* randomArrayFloat(int n, float min, float max)
{
	float* a = (float*)malloc(n*sizeof(float));
	int i;
	for (i = 0; i < n; i++)
		a[i] = min + (max - min)*((rand() % 1000) / 1000.0f);
	return a;
}


// -------------------- CPU/GPU Initialization & Cleanup --------------------

/** Initializes bodies for CPU execution */
void initCPU()
{
	pos = randomArrayFloat4(nbBodies, 1.0f);
	vel = randomArrayFloat4(nbBodies, 0.0001f);
	mass = randomArrayFloat(nbBodies, 1.0f, MASSMAX);
}

/** Initializes bodies and allocates GPU memory */
void initGPU()
{
	pos = randomArrayFloat4(nbBodies, 1.0f);
	vel = randomArrayFloat4(nbBodies, 0.0001f);
	mass = randomArrayFloat(nbBodies, 1.0f, MASSMAX);

	cudaMalloc((void **)&d_pos_i, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_pos_o, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_vel_i, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_vel_o, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_mass, nbBodies*sizeof(float));

	cudaMemcpy(d_mass, mass, nbBodies*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_o, pos, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_o, vel, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
}

/** Initializes bodies and GPU memory for double-buffered GPU execution */
void initGPUBuffer()
{
	pos = randomArrayFloat4(nbBodies, 1.0f);
	vel = randomArrayFloat4(nbBodies, 0.0001f);
	mass = randomArrayFloat(nbBodies, 1.0f, MASSMAX);

	cudaMalloc((void **)&d_pos_i, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_pos_o, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_vel_i, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_vel_o, nbBodies*sizeof(float4));
	cudaMalloc((void **)&d_mass, nbBodies*sizeof(float));

	cudaMemcpy(d_mass, mass, nbBodies*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_i, pos, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_i, vel, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_o, pos, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_o, vel, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	buffer = 0;
}

/** Frees host memory for CPU execution */
void cleanCPU()
{
	if (pos) { free(pos);	pos = NULL; }
	if (vel) { free(vel);	vel = NULL; }
	if (mass) { free(mass);	mass = NULL; }
}

/** Frees host and device memory for GPU execution */
void cleanGPU()
{
	if (pos) { free(pos);	pos = NULL; }
	if (vel) { free(vel);	vel = NULL; }
	if (mass) { free(mass);	mass = NULL; }
	cudaFree(d_pos_i); cudaFree(d_pos_o); cudaFree(d_vel_i); cudaFree(d_vel_o); cudaFree(d_mass);
}

/** Frees host and device memory for double-buffered GPU execution */
void cleanGPUBuffer()
{
	if (pos) { free(pos);	pos = NULL; }
	if (vel) { free(vel);	vel = NULL; }
	if (mass) { free(mass);	mass = NULL; }
	buffer = 0;
	cudaFree(d_pos_i); cudaFree(d_pos_o); cudaFree(d_vel_i); cudaFree(d_vel_o); cudaFree(d_mass);
}


// -------------------- CUDA KERNELS --------------------

/**
 * GPU kernel to compute N-body interactions and update positions/velocities.
 *
 * @param d_pos_i Input positions
 * @param d_pos_o Output positions
 * @param d_vel_i Input velocities
 * @param d_vel_o Output velocities
 * @param d_mass Mass array
 * @param n Number of bodies
 */
__global__ void computeBodyGPU(float4* d_pos_i, float4* d_pos_o, float4* d_vel_i, float4* d_vel_o, float* d_mass, int n){
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= 0 && i < n){
		float4 acc, r;
		float tmp, d;
		acc.x = 0.0f;
		acc.y = 0.0f;
		acc.z = 0.0f;
		acc.w = 1.0f;
		for (int j = 0; j < n; j++) {
			r.x = d_pos_i[j].x - d_pos_i[i].x;
			r.y = d_pos_i[j].y - d_pos_i[i].y;
			r.z = d_pos_i[j].z - d_pos_i[i].z;
			tmp = (r.x * r.x) + (r.y * r.y) + (r.z * r.z);
			d = tmp + EPSILON * EPSILON;
			acc.x += G * r.x * d_mass[j] / sqrt(d*d*d);
			acc.y += G * r.y * d_mass[j] / sqrt(d*d*d);
			acc.z += G * r.z * d_mass[j] / sqrt(d*d*d);
		}
		d_pos_o[i].x = d_pos_i[i].x + d_vel_i[i].x;
		d_pos_o[i].y = d_pos_i[i].y + d_vel_i[i].y;
		d_pos_o[i].z = d_pos_i[i].z + d_vel_i[i].z;

		d_vel_o[i].x = d_vel_i[i].x + acc.x;
		d_vel_o[i].y = d_vel_i[i].y + acc.y;
		d_vel_o[i].z = d_vel_i[i].z + acc.z;
	}
}

/**
 * GPU kernel for double-buffered N-body computation.
 * Similar to computeBodyGPU but supports ping-pong buffers.
 *
 * @param d_pos_i Input positions
 * @param d_pos_o Output positions
 * @param d_vel_i Input velocities
 * @param d_vel_o Output velocities
 * @param d_mass Mass array
 * @param n Number of bodies
 */
__global__ void computeBodyGPUBuffer(float4* d_pos_i, float4* d_pos_o, float4* d_vel_i, float4* d_vel_o, float* d_mass, int n){
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= 0 && i < n){
		float d;
		float4 acc, r;
		acc.x = 0.0f;
		acc.y = 0.0f;
		acc.z = 0.0f;
		acc.w = 1.0f;

		for (int j = 0; j < n; j++) {
			r.x = d_pos_i[j].x - d_pos_i[i].x;
			r.y = d_pos_i[j].y - d_pos_i[i].y;
			r.z = d_pos_i[j].z - d_pos_i[i].z;
			d = (r.x * r.x) + (r.y * r.y) + (r.z * r.z) + EPSILON * EPSILON;
			acc.x += G * r.x * d_mass[j] / sqrt(d*d*d);
			acc.y += G * r.y * d_mass[j] / sqrt(d*d*d);
			acc.z += G * r.z * d_mass[j] / sqrt(d*d*d);
		}
		
		d_pos_o[i].x = d_pos_i[i].x + d_vel_i[i].x;
		d_pos_o[i].y = d_pos_i[i].y + d_vel_i[i].y;
		d_pos_o[i].z = d_pos_i[i].z + d_vel_i[i].z;

		d_vel_o[i].x = d_vel_i[i].x + acc.x;
		d_vel_o[i].y = d_vel_i[i].y + acc.y;
		d_vel_o[i].z = d_vel_i[i].z + acc.z;
	}
}

/**
 * GPU kernel that uses shared memory to accelerate N-body computation.
 * Each block loads a chunk of positions into shared memory to reduce global memory accesses.
 *
 * @param d_pos_i Input positions
 * @param d_pos_o Output positions
 * @param d_vel_i Input velocities
 * @param d_vel_o Output velocities
 * @param d_mass Mass array
 * @param n Number of bodies
 */
__global__ void computeBodyGPU_sm(float4* d_pos_i, float4* d_pos_o, float4* d_vel_i, float4* d_vel_o, float* d_mass, int n){

	__shared__ float4 pos_sm[NBTHREADS];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	float d;
	float4 acc, r;
	acc.x = 0.0f;
	acc.y = 0.0f;
	acc.z = 0.0f;
	acc.w = 1.0f;

	if (index < n){

			float4 pos_index = d_pos_i[index];
			float4 vel_index = d_vel_i[index];

			for (int m = 0; m < (n + NBTHREADS - 1) / NBTHREADS; m++){
				if (m * NBTHREADS + tx < n){
					pos_sm[tx] = d_pos_i[m * NBTHREADS + tx];
				}
				__syncthreads();

				for (int k = 0; k < blockDim.x; k++){

					r.x = pos_sm[k].x - pos_index.x;
					r.y = pos_sm[k].y - pos_index.y;
					r.z = pos_sm[k].z - pos_index.z;
					d = (r.x * r.x) + (r.y * r.y) + (r.z * r.z) + EPSILON * EPSILON;

					acc.x += G * r.x * d_mass[k] / sqrt(d*d*d);
					acc.y += G * r.y * d_mass[k] / sqrt(d*d*d);
					acc.z += G * r.z * d_mass[k] / sqrt(d*d*d);
				}
			}

			d_pos_o[index].x = pos_index.x + vel_index.x;
			d_pos_o[index].y = pos_index.y + vel_index.y;
			d_pos_o[index].z = pos_index.z + vel_index.z;

			d_vel_o[index].x = vel_index.x + acc.x;
			d_vel_o[index].y = vel_index.y + acc.y;
			d_vel_o[index].z = vel_index.z + acc.z;

	}
}

/**
 * Alternative GPU kernel for N-body computation using shared memory.
 * This kernel demonstrates a slightly different approach to update velocities and positions.
 * Each block copies positions into shared memory, then computes acceleration for each body.
 *
 * @param pos Input positions
 * @param vel Input velocities
 * @param pos2 Output positions
 * @param vel2 Output velocities
 * @param mass Mass array
 * @param nbBodies Number of bodies
 */
__global__ void kernel2(float4* pos, float4* vel, float4* pos2, float4* vel2, float* mass, int nbBodies) {

	__shared__ float4 sm[512];
	float4 r;
	float4 acc;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nbBodies) {
		vel2[i].x = vel[i].x;
		vel2[i].y = vel[i].y;
		vel2[i].z = vel[i].z;
		acc.x = 0;
		acc.y = 0;
		acc.z = 0;
		for (int k = 0; k < (nbBodies + 512 - 1) / 512; k++) {
			sm[threadIdx.x] = pos[i];
			__syncthreads();

			for (int j = 0; j < blockDim.x; j++) {
				r.x = sm[j].x - pos[i].x;
				r.y = sm[j].y - pos[i].y;
				r.z = sm[j].z - pos[i].z;
				float d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z) * sqrt(r.x * r.x + r.y * r.y + r.z * r.z) + EPSILON * EPSILON;
				acc.x += G * r.x * mass[j] / sqrt(d * d * d);
				acc.y += G * r.y * mass[j] / sqrt(d * d * d);
				acc.z += G * r.z * mass[j] / sqrt(d * d * d);
			}


		}
		vel2[i].x += acc.x;
		vel2[i].y += acc.y;
		vel2[i].z += acc.z;
		vel2[i].w = 1;
		pos2[i].x = pos[i].x + vel[i].x;
		pos2[i].y = pos[i].y + vel[i].y;
		pos2[i].z = pos[i].z + vel[i].z;
		pos2[i].w = 1;
	}
}

// -------------------- SIMULATION FUNCTIONS (CPU + GPU) --------------------

/**
 * CPU version of N-body simulation.
 * Computes all interactions and updates host arrays.
 */
void exampleCPU()
{
	float4 *new_pos = NULL;
	float4 *new_vel = NULL;
	new_pos = (float4*)malloc(nbBodies*sizeof(float4));
	new_vel = (float4*)malloc(nbBodies*sizeof(float4));

	for (int i = 0; i < nbBodies; i++)
	{

		float4 acc, r;
		acc.x = 0.0f;
		acc.y = 0.0f;
		acc.z = 0.0f;
		acc.w = 1.0f;
		for (int j = 0; j < nbBodies; j++) {
			r.x = pos[j].x - pos[i].x;
			r.y = pos[j].y - pos[i].y;
			r.z = pos[j].z - pos[i].z;
			float tmp = (r.x * r.x) + (r.y * r.y) + (r.z * r.z);
			float d = tmp + EPSILON * EPSILON;
			acc.x += G * r.x * mass[j] / sqrt(d*d*d);
			acc.y += G * r.y * mass[j] / sqrt(d*d*d);
			acc.z += G * r.z * mass[j] / sqrt(d*d*d);
		}
		new_pos[i].x = pos[i].x + vel[i].x;
		new_pos[i].y = pos[i].y + vel[i].y;
		new_pos[i].z = pos[i].z + vel[i].z;

		new_vel[i].x = vel[i].x + acc.x;
		new_vel[i].y = vel[i].y + acc.y;
		new_vel[i].z = vel[i].z + acc.z;
	}


	for (int i = 0; i < nbBodies; i++){
		pos[i].x = new_pos[i].x;
		pos[i].y = new_pos[i].y;
		pos[i].z = new_pos[i].z;

		vel[i].x = new_vel[i].x;
		vel[i].y = new_vel[i].y;
		vel[i].z = new_vel[i].z;
	}

	free(new_pos); free(new_vel);
}

/**
 * Executes N-body simulation on GPU using simple global memory kernel.
 */
void exampleGPU() {
	int nbBlocks = (nbBodies / NBTHREADS);

	cudaMemcpy(d_pos_i, pos, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_i, vel, nbBodies*sizeof(float4), cudaMemcpyHostToDevice);

	computeBodyGPU << < nbBlocks, NBTHREADS >> >(d_pos_i, d_pos_o, d_vel_i, d_vel_o, d_mass, nbBodies);

	cudaMemcpy(pos, d_pos_o, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(vel, d_vel_o, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
}

/**
 * Executes N-body simulation on GPU using double-buffered kernel.
 * Alternates input/output buffers each frame to avoid overwriting data.
 */
void exampleGPUBuffer() {
	int nbBlocks = (nbBodies + NBTHREADS - 1) / NBTHREADS;

	if (buffer == 0){
		computeBodyGPUBuffer << < nbBlocks, NBTHREADS >> >(d_pos_i, d_pos_o, d_vel_i, d_vel_o, d_mass, nbBodies);
		cudaMemcpy(pos, d_pos_o, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
		buffer = 1;
	}
	else{
		computeBodyGPUBuffer << < nbBlocks, NBTHREADS >> >(d_pos_o, d_pos_i, d_vel_o, d_vel_i, d_mass, nbBodies);
		cudaMemcpy(pos, d_pos_i, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
		buffer = 0;
	}
}

/**
 * Executes N-body simulation on GPU using shared memory kernel for acceleration.
 * Alternates buffers like the double-buffered version.
 */
void exampleGPU_sm() {
	int nbBlocks = (nbBodies + NBTHREADS - 1) / NBTHREADS;

	if (buffer == 0){
		computeBodyGPU_sm << < nbBlocks, NBTHREADS >> >(d_pos_i, d_pos_o, d_vel_i, d_vel_o, d_mass, nbBodies);
		//kernel2 << < nbBlocks, NBTHREADS >> >(d_pos_i, d_vel_i, d_pos_o, d_vel_o, d_mass, nbBodies);
		cudaMemcpy(pos, d_pos_o, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
		buffer = 1;
	}
	else{
		computeBodyGPU_sm << < nbBlocks, NBTHREADS >> >(d_pos_o, d_pos_i, d_vel_o, d_vel_i, d_mass, nbBodies);
		//kernel2 << < nbBlocks, NBTHREADS >> >(d_pos_o, d_vel_o, d_pos_i, d_vel_i, d_mass, nbBodies);
		cudaMemcpy(pos, d_pos_i, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost);
		buffer = 0;
	}
}


// -------------------- Rendering and Interaction --------------------

/**
 * Updates the simulation and FPS counter based on current mode.
 */
void calcNbodies() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU"; break;
		case GPU_MODE: m = "GPU"; break;
		case GPU_BUFFER_MODE: m = "GPU double buffer"; break;
		case GPU_SM_MODE: m = "GPU shared memory"; break;
		}
		sprintf(t, "%s (mode: %s, bodies: %i, FPS: %.2f)", TITLE, m, nbBodies, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE: exampleGPU(); break;
	case GPU_BUFFER_MODE: exampleGPUBuffer(); break;
	case GPU_SM_MODE: exampleGPU_sm(); break;
	}
}

/**
 * GLUT idle callback to continuously refresh the simulation.
 */
void idleNbodies()
{
	glutPostRedisplay();
}

/**
 * Renders all bodies in the simulation using OpenGL vertex arrays.
 * Applies camera transformations before drawing.
 */
void renderNbodies(void)
{
	calcNbodies();
	cameraApply();

	glClear(GL_COLOR_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, pos);
	glDrawArrays(GL_POINTS, 0, nbBodies);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

/**
 * Cleans resources depending on execution mode.
 */
void clean()
{
	switch (mode)
	{
	case CPU_MODE: cleanCPU(); break;
	case GPU_MODE: cleanGPU(); break;
	case GPU_BUFFER_MODE: cleanGPUBuffer(); break;
	case GPU_SM_MODE: cleanGPUBuffer(); break;
	}
}

/**
 * Initializes resources depending on execution mode.
 */
void init()
{
	switch (mode)
	{
	case CPU_MODE: initCPU(); break;
	case GPU_MODE: initGPU(); break;
	case GPU_BUFFER_MODE: initGPUBuffer(); break;
	case GPU_SM_MODE: initGPUBuffer(); break;
	}

}

/**
 * Switches between execution modes.
 */
void toggleMode(int m)
{
	clean();
	mode = m;
	init();
}


// -------------------- Input Handling --------------------

/**
 * Normal key handler (ASCII keys).
 *
 * @param key ASCII code of the pressed key.
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 */
void processNormalKeys(unsigned char key, int x, int y) {
	if (key == 27) exit(0);
	else if (key == '&') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
	else if (key == '"') toggleMode(GPU_BUFFER_MODE);
	else if (key == '\'') toggleMode(GPU_SM_MODE);
}

/**
 * Special key handler (non-ASCII keys).
 *
 * @param key GLUT key code (GLUT_KEY_UP, GLUT_KEY_DOWN, arrows, etc.)
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 * UP ARROW: increases the number of bodies in multiple steps (+ 1, 16, 128, 512).
 * DOWN ARROW: decreases the number of bodies in reverse steps (- 1, 16, 128, 512).
 */
void processSpecialKeys(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_UP:
		if (nbBodies <16) nbBodies++;
		else if (nbBodies <128) nbBodies += 16;
		else if (nbBodies <1024) nbBodies += 128;
		else nbBodies += 512;
		toggleMode(mode);
		break;
	case GLUT_KEY_DOWN:
		if (nbBodies>1024) nbBodies -= 512;
		else if (nbBodies>128) nbBodies -= 128;
		else if (nbBodies>16) nbBodies -= 16;
		else if (nbBodies > 1) nbBodies--;
		toggleMode(mode);
		break;
	}

}


// -------------------- OpenGL Setup --------------------

/**
 * Initialize OpenGL and GLUT
 * @param argc, argv: command line arguments
 */
void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_X, SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glColor4f(1.0, 1.0, 1.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	glPointSize(2.0f);
}


// -------------------- Main Entry Point --------------------

int main(int argc, char **argv) {

	srand(time(NULL));
	initGL(argc, argv);

	toggleMode(CPU_MODE);

	glutDisplayFunc(renderNbodies);
	glutIdleFunc(idleNbodies);
	glutMouseFunc(trackballMouseFunction);
	glutMotionFunc(trackballMotionFunction);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	GLint GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK) {
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
	}

	glutMainLoop();

	clean();

	return 1;
}
