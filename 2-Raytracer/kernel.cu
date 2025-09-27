/**
 * @file kernel.cu
 * @brief CPU and GPU implementations of N-body simulation visualized with OpenGL.
 *
 * This file implements an N-body gravitational simulation using both CPU and GPU.
 * Several GPU modes are available, including standard, double buffer, and shared memory.
 * OpenGL is used to render the bodies in 3D space with a trackball camera.
 */

#include <stdio.h>
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
#include "device_launch_parameters.h"

// Screen dimensions and constants
#define SCREEN_X 1024
#define SCREEN_Y 768
#define SIZE SCREEN_X*SCREEN_Y
#define FPS_UPDATE 500
#define TITLE "Raytracer"

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU_MODE_CM 3			// GPU with constant memory optimization

#define NB_THREADS 256
#define INF 2e10f 


// -------------------- Sphere definition --------------------

struct Sphere {
	/** Sphere color components */
	float r, g, b;

	/** Sphere radius */
	float radius;

	/** Sphere position (center) */
	float x, y, z;

	/**
	 * Compute whether a ray cast from (cx, cy) intersects the sphere.
	 *
	 * @param cx - X coordinate of the ray
	 * @param cy - Y coordinate of the ray
	 * @param sh - [output] shading factor based on hit position
	 *
	 * @return intersection depth along Z axis if hit, otherwise -INF
	 */
	__host__ __device__ float hit(float cx, float cy, float *sh) {
		float dx = cx - x;
		float dy = cy - y;
		float dz2 = radius*radius - dx*dx - dy*dy;
		if (dz2>0) {
			float dz = sqrtf(dz2);
			*sh = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};


// -------------------- Global Variables --------------------

// OpenGL texture et buffer
GLuint imageTex;
GLuint imageBuffer;
float* debug;

// Globals 
float scale = 1.0f;
float mx, my;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;
int nb_spheres = 20;

// Pixel buffers
float4 *pixels;			// Host-side pixel buffer
float4 *pixels_gpu;		// Device-side pixel buffer

// Host-side sphere list
Sphere *spheres;

// GPU constant memory for spheres
__constant__ Sphere spheresCM[20];


// -------------------- Sphere Generation --------------------

/**
 * Generate random spheres with random colors, sizes, and positions.
 */
void generateSpheres()
{
	Sphere *s;
	for (int i = 0; i < nb_spheres; i++){
		s = spheres + i;
		s->x = (float)(rand() % SCREEN_X);
		s->y = (float)(rand() % SCREEN_Y);
		s->z = (float)(rand() % 200);

		s->r = (float) rand() / RAND_MAX;
		s->g = (float) rand() / RAND_MAX;
		s->b = (float) rand() / RAND_MAX;

		s->radius = (float)((rand() % 150) + 15);

	}
}


// -------------------- CPU & GPU Initialization & Cleanup --------------------

void initCPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
	spheres = (Sphere*)malloc(nb_spheres*sizeof(Sphere));
	
	generateSpheres();
}

void cleanCPU()
{
	free(pixels);
	free(spheres);
}

void initGPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
	spheres = (Sphere*)malloc(nb_spheres*sizeof(Sphere));
	cudaMalloc((void **)&pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4));
	generateSpheres();
}

void initGPUCM()
{
	nb_spheres = 20;
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
	spheres = (Sphere*)malloc(nb_spheres*sizeof(Sphere));
	cudaMalloc((void **)&pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4));
	generateSpheres();
	cudaMemcpyToSymbol(spheresCM, spheres, nb_spheres*sizeof(Sphere));
}

void cleanGPU()
{
	free(pixels);
	free(spheres);
	cudaFree(pixels_gpu);
}


// -------------------- CPU Raytracing Example --------------------

/**
 * CPU-based pixel rendering.
 * Iterates through all pixels and computes the color based on the closest sphere hit.
 */
void exampleCPU()
{
	int i, j, k;
	float sh;
	float hit, tmp;
	Sphere *s;

	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		float x = -mx + (float)(scale*(j - SCREEN_X / 2));
		float y = -my + (float)(scale*(i - SCREEN_Y / 2));
		float4* p = pixels + (i*SCREEN_X + j);
		float closest = -INF;

		p->x = 0.00f;
		p->y = 0.00f;
		p->z = 0.00f;

		for (k = 0; k < nb_spheres; k++){
			
			s = spheres + k;

			hit = s->hit(x, y, &sh);
			tmp = sh * s->z;
			if (hit > -INF && tmp > closest){

				closest = tmp;
				p->x = s->r * sh;
				p->y = s->g * sh;
				p->z = s->b * sh;
				p->w = 1.0f;

			}
		}
	}
}


// -------------------- GPU Kernels --------------------

/**
 * Computes the color of each pixel using ray-sphere intersection tests.
 * Uses spheres stored in global memory.
 *
 * @param pixels_gpu    Device buffer of size SCREEN_X * SCREEN_Y storing RGBA float4 pixels.
 * @param spheres       Device array of Sphere structs (sphere geometry + color).
 * @param dx            X translation (camera offset, usually from mouse input).
 * @param dy            Y translation (camera offset, usually from mouse input).
 * @param d_scale       Scale factor controlling zoom (smaller = zoom in, larger = zoom out).
 * @param spheres_count Number of spheres to test for intersection.
 */
__global__ void computeGPUPixel(float4* pixels_gpu, Sphere* spheres, float dx, float dy, float d_scale, int spheres_count)
{
	Sphere *s;
	float hit, tmp, sh;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / SCREEN_X;
	int j = index % SCREEN_X;

	// Compute ray position in world space
	float x = - dx + (float)(d_scale*(j - SCREEN_X / 2));
	float y = - dy + (float)(d_scale*(i - SCREEN_Y / 2));
	float4* p = pixels_gpu + (i*SCREEN_X + j);
	float closest = -INF;

	p->x = 0.00f;
	p->y = 0.00f;
	p->z = 0.00f;

	// Test intersection with all spheres
	for (int k = 0; k < spheres_count ; k++){

		s = spheres + k;

		hit = s->hit(x, y, &sh);
		tmp = sh * s->z;
		if (hit > -INF && tmp > closest){

			closest = tmp;
			p->x = s->r * sh;
			p->y = s->g * sh;
			p->z = s->b * sh;
			p->w = 1.0f;

		}
	}
}

/**
 * Same as computeGPUPixel, but spheres are stored in **constant memory**
 * (faster access than global memory for small arrays).
 *
 * @param pixels_gpu    Device buffer of pixels (RGBA float4).
 * @param dx            X translation (camera offset).
 * @param dy            Y translation (camera offset).
 * @param d_scale       Scale factor controlling zoom.
 * @param spheres_count Number of spheres in the constant memory array.
 */
__global__ void computeGPUPixelCM(float4* pixels_gpu, float dx, float dy, float d_scale, int spheres_count)
{
	Sphere *s;
	float hit, tmp, sh;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / SCREEN_X;
	int j = index % SCREEN_X;

	float x = -dx + (float)(d_scale*(j - SCREEN_X / 2));
	float y = -dy + (float)(d_scale*(i - SCREEN_Y / 2));
	float4* p = pixels_gpu + (i*SCREEN_X + j);
	float closest = -INF;

	p->x = 0.00f;
	p->y = 0.00f;
	p->z = 0.00f;

	for (int k = 0; k < spheres_count; k++){

		s = spheresCM + k;

		hit = s->hit(x, y, &sh);
		tmp = sh * s->z;
		if (hit > -INF && tmp > closest){

			closest = tmp;
			p->x = s->r * sh;
			p->y = s->g * sh;
			p->z = s->b * sh;
			p->w = 1.0f;

		}
	}
}


// -------------------- GPU Execution Wrappers --------------------

void exampleGPU()
{
	Sphere *d_spheres;

	cudaMalloc((void **) &d_spheres, nb_spheres*sizeof(Sphere));

	cudaMemcpy(d_spheres, spheres, nb_spheres*sizeof(Sphere), cudaMemcpyHostToDevice);

	computeGPUPixel <<<SIZE/512,512>>>(pixels_gpu, d_spheres, mx, my, scale, nb_spheres);

	cudaMemcpy(pixels, pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4), cudaMemcpyDeviceToHost);

	cudaFree(d_spheres);

}

void exampleGPUCM()
{

	computeGPUPixelCM << <SIZE / 512, 512 >> >(pixels_gpu, mx, my, scale, nb_spheres);

	cudaMemcpy(pixels, pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4), cudaMemcpyDeviceToHost);

}


// -------------------- Rendering and Interaction --------------------

/**
 * Called every frame: updates FPS, executes CPU/GPU render depending on mode.
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
		case GPU_MODE_CM: m = "GPU mode - constant memory"; break;
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
	case GPU_MODE_CM: exampleGPUCM(); break;
	}
}

/**
 * Idle callback: triggers continuous rendering.
 */
void idle()
{
	glutPostRedisplay();
}

/**
 * Render callback: draws the computed pixels to the screen.
 */
void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE_CM: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
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
	case GPU_MODE_CM: cleanGPU(); break;
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
	case GPU_MODE_CM: initGPUCM(); break;
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

	if (key == 27) exit(0);
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
	else if (key == '3') toggleMode(GPU_MODE_CM);
}

/**
 * Special key handler (non-ASCII keys).
 *
 * @param key GLUT key code (GLUT_KEY_UP, GLUT_KEY_DOWN, arrows, etc.)
 * @param x   Mouse X position at time of key press.
 * @param y   Mouse Y position at time of key press.
 * UP ARROW: add a new sphere with random position, color, and radius (max 100 spheres).
 * DOWN ARROW: remove the last sphere if at least one exists.
 */
void processSpecialKeys(int key, int x, int y) {
	// other keys (F1, F2, arrows, home, etc.)
	switch (key) {
	case GLUT_KEY_UP: 
		if (nb_spheres < 101)
		{
			nb_spheres++;
			spheres = (Sphere*)realloc(spheres, nb_spheres*sizeof(Sphere));

			Sphere *s;
			s = spheres + nb_spheres - 1;
			s->x = (float)(rand() % SCREEN_X);
			s->y = (float)(rand() % SCREEN_Y);
			s->z = (float)(rand() % 200);

			s->r = (float)rand() / RAND_MAX;
			s->g = (float)rand() / RAND_MAX;
			s->b = (float)rand() / RAND_MAX;

			s->radius = (float)((rand() % 150) + 15);
		}
		break;
	case GLUT_KEY_DOWN: 
		if (nb_spheres > 0)
		{
			nb_spheres--;
			spheres = (Sphere*)realloc(spheres, nb_spheres*sizeof(Sphere));
		}
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

	srand(time(NULL));

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
