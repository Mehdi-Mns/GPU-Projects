/**
 * @file kernel.cu
 * @brief GPU-accelerated and CPU implementations of Conway's Game of Life.
 *
 * This file implements Conway's Game of Life simulation on a 2D grid.
 * Each cell evolves according to classic rules, with GPU acceleration for large grids.
 * OpenGL is used for visualization of the evolving cell states.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex>

#include <math.h>
#include <time.h>
#include <list>

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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// Screen dimensions and constants
#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "GameOfLife"

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU_MODE2 3

// CUDA helper for error checking
#define F "kernel.cu"
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors
(cudaError err, const char *file, const int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		system("pause");
		exit(1);
	}
}

#define N SCREEN_X*SCREEN_Y			// Total number of pixels
#define M 512						// Number of threads per CUDA block


// -------------------- OpenGL & Global Variables --------------------

// OpenGL texture et buffer
GLuint imageTex;
GLuint imageBuffer;
float* debug;

// Game of Life variables
int range = 5;          // Neighborhood radius for the Game of Life
int surviveLo = 34;     // Minimum neighbors to survive
int surviveHi = 58;     // Maximum neighbors to survive
int birthLo = 34;       // Minimum neighbors to spawn a new cell
int birthHi = 45;       // Maximum neighbors to spawn a new cell
int chanceOfLife = 25;	// % probability of a pixel being alive at initialization

// Globals 
float scale = 0.003f;
float mx, my;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

// Buffers for pixel data
float4* pixels;         // Host pixel buffer (displayed image)
float4* previousPixels; // Host buffer for previous frame (used in CPU mode)
float4* dpixels2;       // Device buffer (ping-pong rendering)
float4* dpixels;        // Device buffer (ping-pong rendering)

bool usePixel1 = true;	// Flag for alternating between GPU buffers


// -------------------- Initialization and Cleanup --------------------

/**
 * Initializes the pixel buffer with random cells (alive or dead).
 * Alive cells are red (1,0,0), dead cells are black (0,0,0).
 */
void initCells()
{
	srand(time(NULL));
	int i, j;
	for (i = 0; i < SCREEN_Y; i++)
		for (j = 0; j < SCREEN_X; j++)
		{
			float x = (float)(scale*(j - SCREEN_X / 2));
			float y = (float)(scale*(i - SCREEN_Y / 2));
			float4* p = pixels + (i*SCREEN_X + j);
			// dead: black
			p->x = 0.0f;
			p->y = 0.0f;
			p->z = 0.0f;
			p->w = 1.0f;
			if (rand() % 100 < chanceOfLife)
			{	// alive cell (purple)
				p->x = 1.0f;
				p->y = 0.0f;
				p->z = 1.0f;
			}
		}
}

/**
 * Allocates host memory (page-locked for faster CUDA transfer) and initializes cells.
 */
void initCPU()
{
	//pixels = (float4*)malloc(SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMallocHost((void**)&pixels, SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMallocHost((void**)&previousPixels, SCREEN_X*SCREEN_Y * sizeof(float4));
	initCells();
}

/**
 * Frees host memory allocated in CPU mode.
 */
void cleanCPU()
{
	//free(pixels);
	cudaFreeHost(pixels);
	cudaFreeHost(previousPixels);
}

/**
 * Allocates GPU and host memory, initializes cells, and copies initial state to device.
 */
void initGPU()
{
	//pixels = (float4*)malloc(SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMallocHost((void**)&pixels, SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMalloc((void**)&dpixels, SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMallocHost((void**)&previousPixels, SCREEN_X*SCREEN_Y * sizeof(float4));
	cudaMalloc((void**)&dpixels2, SCREEN_X*SCREEN_Y * sizeof(float4));
	initCells();
	cudaMemcpy(usePixel1 ? dpixels : dpixels2, pixels, N * sizeof(float4), cudaMemcpyHostToDevice);
}

/**
 * Frees GPU and host memory allocated in GPU mode.
 */
void cleanGPU()
{
	//free(pixels);
	cudaFreeHost(pixels);
	cudaFreeHost(previousPixels);
	cudaFree(dpixels);
	cudaFree(dpixels2);
}


// -------------------- Utility Functions --------------------

/**
 * Checks if a pixel is alive.
 * @param p Pixel (float4).
 * @return true if RGB components are > 0, false otherwise.
 */
__host__ __device__ bool isAlive(float4* p)
{
	return p->x > 0 || p->y > 0 || p->z > 0;
}

/**
 * Safe modulo operation (handles negative values).
 * @param a Value.
 * @param b Modulus.
 * @return Result of (a mod b).
 */
__host__ __device__ int mod(int a, int b)
{
	return (a % b + b) % b;
}


// -------------------- CPU & GPU Implementation of Game of Life --------------------

/**
 * Executes one iteration of the Game of Life on the CPU.
 * Uses `previousPixels` as input and writes the new state into `pixels`.
 */
void exampleCPU()
{
	int i, j;
	for (i = 0; i < SCREEN_Y; i++)
		for (j = 0; j < SCREEN_X; j++)
		{
			float4* pp = previousPixels + (i*SCREEN_X + j);
			float4* p = pixels + (i*SCREEN_X + j);

			int alives = 0;
			for (int k = i - range; k < i + range + 1; k++)
			{
				for (int l = j - range; l < j + range + 1; l++)
				{
					int k1 = mod(k, SCREEN_Y);
					int l1 = mod(l, SCREEN_X);
					float4* neighbour = previousPixels + (k1*SCREEN_X + l1);
					if (isAlive(neighbour))
					{
						alives++;
					}
				}
			}

			if (isAlive(pp) && (alives < surviveLo || alives > surviveHi))
			{
				p->x = 0.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
			else if (!isAlive(pp) && alives >= birthLo && alives <= birthHi)
			{
				p->x = 1.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
		}
}

/**
 * Executes one iteration of the Game of Life on the GPU.
 * Each thread computes the new state of one cell (pixel).
 *
 * @param pixels        Output buffer (next generation of cells).
 * @param previousPixels Input buffer (previous generation of cells).
 * @param screen_x      Width of the simulation grid (number of columns).
 * @param screen_y      Height of the simulation grid (number of rows).
 * @param range         Neighborhood radius (e.g., 1 for Moore neighborhood).
 * @param surviveLo     Minimum number of alive neighbors for a live cell to survive.
 * @param surviveHi     Maximum number of alive neighbors for a live cell to survive.
 * @param birthLo       Minimum number of alive neighbors for a dead cell to become alive.
 * @param birthHi       Maximum number of alive neighbors for a dead cell to become alive.
 */
__global__ void jeuVie(
	float4* pixels, float4* previousPixels, int screen_x, int screen_y,
	int range, int surviveLo, int surviveHi, int birthLo, int birthHi)
{
	int size = screen_x * screen_y;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) {
		int i, j;
		j = index % screen_x;
		i = index / screen_x;

		float4* pp = previousPixels + (i*screen_x + j);
		float4* p = pixels + (i*screen_x + j);

		int alives = 0;
		for (int k = i - range; k < i + range + 1; k++)
		{
			for (int l = j - range; l < j + range + 1; l++)
			{
				int k1 = mod(k, screen_y);
				int l1 = mod(l, screen_x);
				float4* neighbour = previousPixels + (k1*screen_x + l1);
				if (isAlive(neighbour))
				{
					alives++;
				}
			}
		}

		// Apply rules
		if (isAlive(pp))
		{
			if (alives < surviveLo || alives > surviveHi) 
			{	// death
				p->x = 0.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
			else
			{	// survive
				p->x = 1.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
		}
		else
		{
			if (alives >= birthLo && alives <= birthHi)
			{	// birth
				p->x = 1.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
			else
			{	// stay dead
				p->x = 0.0f;
				p->y = 0.0f;
				p->z = 0.0f;
			}
		}
	}
}

/**
 * GPU wrapper: launches `jeuVie` kernel and performs ping-pong buffer swap.
 */
void exampleGPU()
{
	jeuVie << <((N + M - 1) / M), M >> > (usePixel1 ? dpixels2 : dpixels, usePixel1 ? dpixels : dpixels2,
		SCREEN_X, SCREEN_Y, range, surviveLo, surviveHi, birthLo, birthHi);
	cudaMemcpy(pixels, usePixel1 ? dpixels2 : dpixels, N * sizeof(float4), cudaMemcpyDeviceToHost);
	usePixel1 = !usePixel1;
}


// -------------------- Rendering and Interaction --------------------

/**
 * Updates FPS counter and executes CPU/GPU iteration depending on mode.
 */
void calculate() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU mode"; break;
		case GPU_MODE: m = "GPU mode"; break;
		case GPU_MODE2: m = "GPU mode 2"; break;
		}
		sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE: exampleGPU(); break;
	//case GPU_MODE2: exampleGPU2(); break;
	}
}

/**
 * Idle callback: requests redisplay.
 */
void idle()
{
	glutPostRedisplay();
}

/**
 * Display callback: renders pixels to the screen.
 */
void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE2: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	}
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
	//case GPU_MODE2: cleanGPU2(); break;
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
	//case GPU_MODE2: initGPU2(); break;
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
 * Mouse click handler.
 *
 * @param button  Mouse button pressed (0 = left, 1 = right, 2 = middle, 3 = wheel up, 4 = wheel down).
 * @param state   Button state (GLUT_DOWN / GLUT_UP).
 * @param x       Mouse cursor X position in window coordinates.
 * @param y       Mouse cursor Y position in window coordinates.
 */
void mouse(int button, int state, int x, int y)
{
	if (button <= 2)
	{
		mx = (float)(scale*(x - SCREEN_X / 2));
		my = -(float)(scale*(y - SCREEN_Y / 2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

/**
 * Mouse drag handler (while holding a button).
 *
 * @param x  New mouse X coordinate.
 * @param y  New mouse Y coordinate.
 */
void mouseMotion(int x, int y)
{
	mx = (float)(scale*(x - SCREEN_X / 2));
	my = -(float)(scale*(y - SCREEN_Y / 2));
}

/**
 * Normal key handler (ASCII keys).
 *
 * @param key ASCII code of the pressed key.
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 */
void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) { clean(); exit(0); }
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
	else if (key == '3') toggleMode(GPU_MODE2);
}

/**
 * Special key handler (non-ASCII keys).
 *
 * @param key GLUT key code (GLUT_KEY_UP, GLUT_KEY_DOWN, arrows, etc.)
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 */
void processSpecialKeys(int key, int x, int y) {
	// other keys (F1, F2, arrows, home, etc.)
	switch (key) {
	case GLUT_KEY_UP: break;
	case GLUT_KEY_DOWN: break;
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
	glDisable(GL_DEPTH_TEST);

	// View Ortho
	// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
	// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


// -------------------- Main Entry Point --------------------

int main(int argc, char **argv) {

	initGL(argc, argv);

	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	GLint GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK) {
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
	}

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}

