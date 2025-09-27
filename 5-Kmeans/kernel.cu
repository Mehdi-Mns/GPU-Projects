/**
 * @file kernel.cu
 * @brief GPU-accelerated and CPU implementations of K-means clustering visualized with OpenGL.
 *
 * This file implements a K-means clustering algorithm for a large number of points.
 * Points are assigned to clusters either on CPU or GPU. OpenGL is used for visualization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cfloat>

extern "C" {
#include "camera.h"
}

// Screen dimensions and constants
#define SCREEN_X 800
#define SCREEN_Y 800
#define FPS_UPDATE 200 
#define TITLE "K-means"

#define CPU_MODE 1
#define GPU_MODE 2

int nbPoints = 2 * 1024 * 1024;				// Total number of data points
//int nbPoints = 2 * 512 * 512;
#define CLUSTERS 128						// Number of clusters


// -------------------- Global Variables --------------------

int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

// Points et cluster variables
float4 *points = NULL, *centroids = NULL, *pointColors = NULL, *centroidColors = NULL;
unsigned int* pointLabel = NULL;

// Device points et cluster varaibles
float4 *d_points = NULL, *d_centroids = NULL, *d_newCentroids = NULL, *d_pointColors = NULL, *d_centroidColors = NULL;
unsigned int* d_pointLabel = NULL;
int *d_newCentroidSize = NULL;						// Device array storing number of points per centroid

__constant__ float4 centroids_cm[CLUSTERS];			// Constant memory for centroids
__constant__ float4 centroidColors_cm[CLUSTERS];	// Constant memory for centroid colors


// -------------------- Utility Functions --------------------

/**
 * @brief Creates random 3D points forming artificial clusters.
 *
 * Each point is assigned to a cluster and positioned around its centroid.
 *
 * @param n Number of points to generate.
 * @param d Scaling factor for cluster spread.
 * @return Pointer to dynamically allocated array of float4 points.
 */
float4* createRandomData(int n, float d)
{
	// create artificial clusters
	int nbClusters = CLUSTERS;
	int i = 0;
	float x, y, z, r;
	float4* c = (float4*)malloc(nbClusters*sizeof(float4));
	float* s = (float*)malloc(nbClusters*sizeof(float));
	for (i = 0; i<nbClusters; i++)
	{
		x = (2 * ((rand() % 1000) / 1000.0f) - 1);
		y = (2 * ((rand() % 1000) / 1000.0f) - 1);
		z = (2 * ((rand() % 1000) / 1000.0f) - 1);
		r = powf(3 * (rand() % 1000) / 1000.0f, 4);

		c[i].x = r*d*x;
		c[i].y = r*d*y;
		c[i].z = r*d*z;
		c[i].w = 1.0f; // must be 1.0

		s[i] = (rand() % 1000) / 1000.0f + 0.5f;
	}

	float4* a = (float4*)malloc(n*sizeof(float4));
	for (i = 0; i<n; i++)
	{
		int cl = rand() % CLUSTERS;
		x = (2 * ((rand() % 1000) / 1000.0f) - 1);
		y = (2 * ((rand() % 1000) / 1000.0f) - 1);
		z = (2 * ((rand() % 1000) / 1000.0f) - 1);
		r = powf(2 * (rand() % 1000) / 1000.0f / sqrt(x*x + y*y + z*z), 2.5);

		a[i].x = c[cl].x + s[cl] * s[cl] * r*d*x;
		a[i].y = c[cl].y + s[cl] * s[cl] * r*d*y;
		a[i].z = c[cl].z + s[cl] * s[cl] * r*d*z;
		a[i].w = 1.0f; // must be 1.0
	}
	free(c);
	free(s);
	return a;
}

/**
 * @brief Generates a random RGB color (float4 with alpha=1).
 *
 * @return Random float4 color.
 */
float4 randomColor()
{
	float4 color;
	color.x = (rand() % 1000) / 1000.0f;
	color.y = (rand() % 1000) / 1000.0f;
	color.z = (rand() % 1000) / 1000.0f;
	color.w = 1.0f;
	return color;
}


// -------------------- CPU & GPU Implementation --------------------

/**
 * @brief Initialize all data structures for CPU execution.
 *
 * Allocates memory for points, point colors, cluster centroids, and assigns
 * initial labels and colors.
 */
void initCPU()
{
	points = createRandomData(nbPoints, 1.0f);
	pointColors = (float4*)malloc(nbPoints*sizeof(float4));
	pointLabel = (unsigned int*)malloc(nbPoints*sizeof(unsigned int));

	centroids = (float4*)malloc(CLUSTERS*sizeof(float4));
	centroidColors = (float4*)malloc(CLUSTERS*sizeof(float4));
	int i;
	for (i = 0; i<CLUSTERS; i++)
	{
		centroids[i] = points[i];  // Forgy method initialisation
		centroidColors[i] = randomColor();
	}
	for (i = 0; i<nbPoints; i++)
	{
		pointLabel[i] = 0;
	}
}

/**
 * @brief Free all allocated memory for CPU execution.
 */
void cleanCPU()
{
	if (points) { free(points);	points = NULL; }
	if (pointLabel) { free(pointLabel);	pointLabel = NULL; }
	if (pointColors) { free(pointColors); pointColors = NULL; }
	if (centroids) { free(centroids); centroids = NULL; }
	if (centroidColors) { free(centroidColors); centroidColors = NULL; }
}

/**
 * @brief Initialize all data structures and allocate memory for GPU execution.
 */
void initGPU()
{
	points = createRandomData(nbPoints, 1.0f);
	pointColors = (float4*)malloc(nbPoints*sizeof(float4));
	pointLabel = (unsigned int*)malloc(nbPoints*sizeof(unsigned int));

	centroids = (float4*)malloc(CLUSTERS*sizeof(float4));
	centroidColors = (float4*)malloc(CLUSTERS*sizeof(float4));
	//float4 *d_points = NULL, *d_centroids = NULL, *d_pointColors = NULL, *d_centroidColors = NULL;
	cudaMalloc((void **)&d_points, nbPoints*sizeof(float4));
	cudaMalloc((void **)&d_pointColors, nbPoints*sizeof(float4));
	cudaMalloc((void **)&d_pointLabel, nbPoints*sizeof(unsigned int));

	cudaMalloc((void **)&d_centroids, CLUSTERS*sizeof(float4));
	cudaMalloc((void **)&d_newCentroids, CLUSTERS*sizeof(float4));
	cudaMalloc((void **)&d_newCentroidSize, CLUSTERS*sizeof(int));
	cudaMalloc((void **)&d_centroidColors, CLUSTERS*sizeof(float4));

	int i;
	for (i = 0; i<CLUSTERS; i++)
	{
		centroids[i] = points[i];  // Forgy method initialisation
		centroidColors[i] = randomColor();
	}
	for (i = 0; i<nbPoints; i++)
	{
		pointLabel[i] = 0;
	}

	cudaMemcpy(d_points, points, nbPoints*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pointColors, pointColors, nbPoints*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pointLabel, pointLabel, nbPoints*sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_centroids, centroids, CLUSTERS*sizeof(float4), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_centroidColors, centroidColors, CLUSTERS*sizeof(float4), cudaMemcpyHostToDevice);

	//cudaMemcpyToSymbol(centroids_cm, centroids, CLUSTERS*sizeof(float4));
	cudaMemcpyToSymbol(centroidColors_cm, centroidColors, CLUSTERS*sizeof(float4));
}

/**
 * @brief Free all allocated memory for GPU execution.
 */
void cleanGPU()
{
	if (points) { free(points);	points = NULL; }
	if (pointLabel) { free(pointLabel);	pointLabel = NULL; }
	if (pointColors) { free(pointColors); pointColors = NULL; }
	if (centroids) { free(centroids); centroids = NULL; }
	if (centroidColors) { free(centroidColors); centroidColors = NULL; }

	cudaFree(d_points); cudaFree(d_pointColors); cudaFree(d_pointLabel); cudaFree(d_centroids); cudaFree(d_centroidColors);
	cudaFree(d_newCentroids); cudaFree(d_newCentroidSize);
}


// -------------------- CUDA KERNELS --------------------

//float4 *d_points = NULL, *d_centroids = NULL, *d_pointColors = NULL, *d_centroidColors = NULL;

/**
 * @brief GPU kernel: assign each point to the closest centroid.
 *
 * @param d_points Device array of point positions.
 * @param d_centroids Device array of centroids.
 * @param d_pointColors Device array of point colors.
 * @param d_centroidColors Device array of centroid colors.
 * @param d_pointLabel Device array storing point cluster assignments.
 * @param n Number of points.
 */
__global__ void kernelAssign(float4 *d_points, float4 *d_centroids, float4 *d_pointColors, float4 *d_centroidColors, unsigned int * d_pointLabel, int n){

	int index = threadIdx.x + blockIdx.x * blockDim.x;


	if (index < n){
		float4 tmp;
		int closestCluster;
		float distance, dmin;
		dmin = 0;

		for (int j = 0; j < CLUSTERS - 1; j++){
			tmp.x = (d_points[index].x - d_centroids[j].x);
			tmp.y = (d_points[index].y - d_centroids[j].y);
			tmp.z = (d_points[index].z - d_centroids[j].z);
			distance = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z); //| points[index] - centroids[j] |

			if (distance < dmin || j == 0){
				dmin = distance;
				closestCluster = j;
			}
		}
		d_pointLabel[index] = closestCluster; // point i assigned to closest cluster 
		d_pointColors[index].x = d_centroidColors[closestCluster].x;
		d_pointColors[index].y = d_centroidColors[closestCluster].y;
		d_pointColors[index].z = d_centroidColors[closestCluster].z;
	}
}

/**
 * @brief GPU kernel: assign points using constant memory centroids.
 *
 * @param d_points Device array of point positions.
 * @param d_pointColors Device array of point colors.
 * @param d_pointLabel Device array storing point cluster assignments.
 * @param n Number of points.
 */
__global__ void kernelAssignCM(float4 *d_points, float4 *d_pointColors, unsigned int * d_pointLabel, int n){

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < n){
		float4 tmp;
		int closestCluster;
		float distance, dmin;
		dmin = 0;

		for (int j = 0; j < CLUSTERS - 1; j++){
			tmp.x = (d_points[index].x - centroids_cm[j].x);
			tmp.y = (d_points[index].y - centroids_cm[j].y);
			tmp.z = (d_points[index].z - centroids_cm[j].z);
			distance = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z); //| points[index] - centroids[j] |

			if (distance < dmin || j == 0){
				dmin = distance;
				closestCluster = j;
			}
		}
		d_pointLabel[index] = closestCluster; // point i assigned to closest cluster 
		d_pointColors[index].x = centroidColors_cm[closestCluster].x;
		d_pointColors[index].y = centroidColors_cm[closestCluster].y;
		d_pointColors[index].z = centroidColors_cm[closestCluster].z;
	}
}

/**
 * @brief GPU kernel: sum points per centroid to compute new centroid positions.
 *
 * @param d_newCentroids Device array to store updated centroids.
 * @param d_newCentroidSize Device array to store number of points per centroid.
 * @param d_points Device array of point positions.
 * @param d_pointLabel Device array storing point cluster assignments.
 * @param n Number of points.
 */
__global__ void kernelReduce(float4 *d_newCentroids, int *d_newCentroidSize, float4 *d_points, unsigned int *d_pointLabel, int n){

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index == 0){
		for (int k = 0; k < CLUSTERS - 1; k++){
			d_newCentroids[k].x = 0;
			d_newCentroids[k].y = 0;
			d_newCentroids[k].z = 0;
			d_newCentroids[k].w = 1.0f;
			d_newCentroidSize[k] = 0;
		}
	}

	if (index < n){
		atomicAdd(&d_newCentroids[d_pointLabel[index]].x, d_points[index].x);
		atomicAdd(&d_newCentroids[d_pointLabel[index]].y, d_points[index].y);
		atomicAdd(&d_newCentroids[d_pointLabel[index]].z, d_points[index].z);
		atomicAdd(&d_newCentroidSize[d_pointLabel[index]], 1);
	}
}


// -------------------- SIMULATION FUNCTIONS (CPU + GPU) --------------------

/**
 * @brief Executes one iteration of K-means clustering on the CPU.
 *
 * Implements the "assignment" phase (assign points to nearest centroid)
 * and the "reduction" phase (recompute centroid positions).
 */
void exampleCPU()
{
	float4 tmp;
	float4* newCentroids;
	int* newCentroidSize;
	int closestCluster;
	float distance, dmin;

	newCentroids = (float4*)malloc(CLUSTERS * sizeof(float4));
	newCentroidSize = (int*)malloc(CLUSTERS * sizeof(int));

	// phase 1 (“assignment”):
	// assign each data point to the closest centroid

	dmin = 0;

	for (int i = 0; i < nbPoints - 1; i++) {
		for (int j = 0; j < CLUSTERS - 1; j++) {
			tmp.x = (points[i].x - centroids[j].x);
			tmp.y = (points[i].y - centroids[j].y);
			tmp.z = (points[i].z - centroids[j].z);
			distance = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z); //| points[i] - centroids[j] |

			if (distance < dmin || j == 0) {
				dmin = distance;
				closestCluster = j;
			}
		}
		pointLabel[i] = closestCluster; // point i assigned to closest cluster 
		pointColors[i].x = centroidColors[closestCluster].x;
		pointColors[i].y = centroidColors[closestCluster].y;
		pointColors[i].z = centroidColors[closestCluster].z;
	}

	// phase 2 (“reduction”): recompute centroids
	for (int k = 0; k < CLUSTERS - 1; k++) {
		newCentroids[k].x = 0;
		newCentroids[k].y = 0;
		newCentroids[k].z = 0;
		newCentroids[k].w = 1.0f;
		newCentroidSize[k] = 0;
	}

	for (int l = 0; l < nbPoints - 1; l++) {
		newCentroids[pointLabel[l]].x = newCentroids[pointLabel[l]].x + points[l].x;
		newCentroids[pointLabel[l]].y = newCentroids[pointLabel[l]].y + points[l].y;
		newCentroids[pointLabel[l]].z = newCentroids[pointLabel[l]].z + points[l].z;
		newCentroidSize[pointLabel[l]]++;
	}

	for (int m = 0; m < CLUSTERS - 1; m++) {
		centroids[m].x = newCentroids[m].x / newCentroidSize[m];
		centroids[m].y = newCentroids[m].y / newCentroidSize[m];
		centroids[m].z = newCentroids[m].z / newCentroidSize[m];
	}

	free(newCentroids); free(newCentroidSize);
}

/**
 * @brief Executes one iteration of K-means clustering on the GPU.
 */
void exampleGPU(){	
	float4 *newCentroids;
	int *newCentroidSize;

	newCentroids = (float4*)malloc(CLUSTERS*sizeof(float4));
	newCentroidSize = (int*)malloc(CLUSTERS*sizeof(int));

	int nbThreads = 512;
	int nbBlocks = (nbPoints + nbThreads - 1) / nbThreads;
	// phase 1 (“assignment”):
	// assign each data point to the closest centroid

	//cudaMemcpy(d_centroids, centroids, CLUSTERS*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(centroids_cm, centroids, CLUSTERS*sizeof(float4));

	//kernelAssign << <nbBlocks, nbThreads >> >(d_points, d_centroids, d_pointColors, d_centroidColors, d_pointLabel, nbPoints);
	kernelAssignCM << <nbBlocks, nbThreads >> >(d_points, d_pointColors, d_pointLabel, nbPoints);

	cudaMemcpy(pointLabel, d_pointLabel, nbPoints*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pointColors, d_pointColors, nbPoints*sizeof(float4), cudaMemcpyDeviceToHost);

	// phase 2 (“reduction”): recompute centroids

	//for (int k = 0; k < CLUSTERS - 1; k++){
	//	newCentroids[k].x = 0;
	//	newCentroids[k].y = 0;
	//	newCentroids[k].z = 0;
	//	newCentroids[k].w = 1.0f;
	//	newCentroidSize[k] = 0;
	//}

	kernelReduce << <nbBlocks, nbThreads >> >(d_newCentroids, d_newCentroidSize, d_points, d_pointLabel, nbPoints);

	cudaMemcpy(newCentroids, d_newCentroids, CLUSTERS*sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(newCentroidSize, d_newCentroidSize, CLUSTERS*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(centroids, d_centroids, CLUSTERS*sizeof(float4), cudaMemcpyDeviceToHost);

	//for (int l = 0; l < nbPoints - 1; l++){
	//	newCentroids[pointLabel[l]].x = newCentroids[pointLabel[l]].x + points[l].x;
	//	newCentroids[pointLabel[l]].y = newCentroids[pointLabel[l]].y + points[l].y;
	//	newCentroids[pointLabel[l]].z = newCentroids[pointLabel[l]].z + points[l].z;
	//	newCentroidSize[pointLabel[l]]++;
	//}

	for (int m = 0; m < CLUSTERS - 1; m++){
		centroids[m].x = newCentroids[m].x / newCentroidSize[m];
		centroids[m].y = newCentroids[m].y / newCentroidSize[m];
		centroids[m].z = newCentroids[m].z / newCentroidSize[m];
	}

	free(newCentroids); free(newCentroidSize);
}


// -------------------- Rendering and Interaction --------------------

/**
 * @brief Calculate clusters and update FPS information.
 */
void calcClusters() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU"; break;
		case GPU_MODE: m = "GPU"; break;
		}
		sprintf(t, "%s: %s, %i points, %.2f FPS", TITLE, m, nbPoints, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE: exampleGPU(); break;
	}
}

/**
 * @brief GLUT idle function, triggers redraw.
 */
void idleKmeans()
{
	glutPostRedisplay();
}

/**
 * @brief Render points and centroids using OpenGL.
 */
void renderKmeans(void)
{
	calcClusters();
	cameraApply();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPointSize(1.0f);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, points);
	glColorPointer(4, GL_FLOAT, 0, pointColors);
	glDrawArrays(GL_POINTS, 0, nbPoints);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glPointSize(5.0f);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, centroids);
	glDrawArrays(GL_POINTS, 0, CLUSTERS);
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
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
}

/**
 * Special key handler (non-ASCII keys).
 *
 * @param key GLUT key code (GLUT_KEY_UP, GLUT_KEY_DOWN, arrows, etc.)
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 * UP ARROW: doubles the number of points used for clustering.
 * DOWN ARROW: halves the number of points used, but never below twice the number of clusters.
 */
void processSpecialKeys(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_UP:
		nbPoints *= 2;
		toggleMode(mode);
		break;
	case GLUT_KEY_DOWN:
		if (nbPoints>2 * CLUSTERS) nbPoints /= 2;
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
	glEnable(GL_DEPTH_TEST);
}


// -------------------- Main Entry Point --------------------

int main(int argc, char **argv) {

	srand(time(NULL));
	initGL(argc, argv);


	toggleMode(CPU_MODE);

	glutDisplayFunc(renderKmeans);
	glutIdleFunc(idleKmeans);
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
