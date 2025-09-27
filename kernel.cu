#include <stdio.h>
#include <math.h>

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

// Screen dimensions and constants
#define SCREEN_X 1024
#define SCREEN_Y 768
#define SIZE SCREEN_X*SCREEN_Y
#define FPS_UPDATE 500
#define TITLE "Julia Fractals"

#define CPU_MODE 1
#define GPU_MODE 2

// Number of CUDA threads per block
#define NB_THREADS 256

// OpenGL texture et buffer
GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals for fractal parameters */
float scale = 0.003f;
float mx, my;
int prec = 15;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4 *pixels;				// CPU-side pixel buffer
float4 *pixels_gpu;			// GPU-side pixel buffer

// Complex number struct
typedef struct {
	float reel;
	float im;
} Complex;


// -------------------- Complex number operations (host + device) -------------------- 

/**
 * Multiply two complex numbers: c3 = c1 * c2
 * @param c1: first complex number (real + imag)
 * @param c2: second complex number (real + imag)
 * @return c3: resulting complex number (real + imag)
 */
__device__ __host__ Complex mul_complex(Complex c1, Complex c2){
	Complex c3;
	c3.reel = c1.reel * c2.reel - c1.im * c2.im;
	c3.im = c1.reel * c2.im + c2.reel * c1.im;
	return c3;
}

/**
 * Add two complex numbers: c3 = c1 + c2
 * @param c1: first complex number
 * @param c2: second complex number
 * @return c3: resulting complex number
 */
__device__ __host__ Complex add_complex(Complex c1, Complex c2){
	Complex c3;
	c3.reel = c1.reel + c2.reel;
	c3.im = c1.im + c2.im;
	return c3;
}

/**
 * Compute squared magnitude of a complex number: |a|^2
 * @param a: complex number
 * @return squared magnitude = a.reel^2 + a.im^2
 */
__device__ __host__ float squaredMagnitude(Complex a){
	return a.reel*a.reel + a.im*a.im;
}

/**
 * Compute the Julia fractal color for a point in the complex plane
 * @param x: real coordinate in fractal space
 * @param y: imaginary coordinate in fractal space
 * @param sx: real part of Julia seed
 * @param sy: imaginary part of Julia seed
 * @param p: maximum iteration count (precision)
 * @return grayscale value in [0,1]; 1 = escaped early, 0 = inside Julia set
 */
__device__ __host__ float juliaColor(float x, float y, float sx, float sy, int p){
	Complex a = { x, y };
	Complex seed = { sx, sy };
	for (int i = 0; i < p; i++){
		a = add_complex(mul_complex(a, a), seed);
		if (squaredMagnitude(a) > 4) return 1 - i / (float)p;
	}

	return 0;
}


// -------------------- CUDA kernel to compute Julia set pixels -------------------- 

/**
 * Compute Julia set in parallel on GPU
 * @param pixels_gpu: output buffer on GPU (float4 RGBA per pixel)
 * @param scale: mapping scale from pixels to fractal coordinates
 * @param mx: Julia seed real component (from mouse)
 * @param my: Julia seed imaginary component
 * @param prec: iteration precision
 * Each thread computes one pixel using its global thread index.
 */
__global__ void juliaColorGPU(float4* pixels_gpu, float scale, float mx, float my, int prec)
{
	// 1D thread index
	int index = threadIdx.x + blockIdx.x * blockDim.x ;

	if (index < SIZE) {
		int i = index / SCREEN_X;
		int j = index % SCREEN_X;

		// Map pixel to fractal coordinates
		float x = (float)(scale*(j - SCREEN_X / 2));
		float y = (float)(scale*(i - SCREEN_Y / 2));

		float4* p_g = pixels_gpu + (i*SCREEN_X + j);

		float grey = juliaColor(x, y, mx, my, prec);

		p_g->x = 1.0f;
		p_g->y = grey;
		p_g->z = 1.0f;
		p_g->w = 1.0f;

		// Highlight pixels near Julia seed
		if (sqrt((x - mx)*(x - mx) + (y - my)*(y - my)) < 0.01)
			p_g->x = 0.0f;
	}
}


// -------------------- CPU / GPU memory management -------------------- 

/**
 * Initialize CPU pixel buffer
 * Allocates memory for all pixels (RGBA floats)
 */
void initCPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

/**
 * Free CPU pixel buffer
 */
void cleanCPU()
{
	free(pixels);
}

/**
 * Initialize GPU pixel buffer and CPU readback buffer
 * Allocates GPU memory and CPU memory for copying results
 */
void initGPU()
{
	cudaMalloc((void **)&pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4));
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

/**
 * Free GPU memory
 */
void cleanGPU()
{
	cudaFree(pixels_gpu);
}


// -------------------- Julia fractal computation (CPU) -------------------- 

/**
 * Compute Julia set on CPU
 * Fills the global 'pixels' buffer
 */
void exampleCPU()
{
	int i, j;
	float grey;
	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		float x = (float)(scale*(j - SCREEN_X / 2));
		float y = (float)(scale*(i - SCREEN_Y / 2));
		float4* p = pixels + (i*SCREEN_X + j);

		grey = juliaColor(x, y, mx, my, prec);

		p->x = 1.0f;
		p->y = grey;
		p->z = 1.0f;
		p->w = 1.0f;

		if (sqrt((x - mx)*(x - mx) + (y - my)*(y - my))<0.01)
			p->x = 0.0f;
	}
}

/**
 * Compute Julia set on GPU
 * Launches CUDA kernel and copies result to CPU buffer
 */
void exampleGPU()
{
	int nbBlocks = ((SCREEN_X*SCREEN_Y) / 512);

	juliaColorGPU<<<nbBlocks,512>>>(pixels_gpu, scale, mx, my, prec);
	cudaMemcpy(pixels, pixels_gpu, SCREEN_X*SCREEN_Y*sizeof(float4),cudaMemcpyDeviceToHost);

}


// -------------------- Julia fractal computation (CPU) -------------------- 

/**
 * Update fractal pixels and FPS counter
 * Chooses CPU or GPU computation based on 'mode'
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
	}
}

/**
 * GLUT idle callback
 * Marks the window for redraw
 */
void idle()
{
	glutPostRedisplay();
}


// -------------------- Rendering function (OpenGL) --------------------

/**
 * Render Julia fractal to OpenGL window
 * Uses glDrawPixels with RGBA float buffer
 */
void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	}
	glutSwapBuffers();
}

/**
 * Clean allocated memory depending on mode
 * Side effect: frees CPU or GPU buffers
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
 * Initialize memory buffers depending on mode
 * Side effect: allocates CPU and/or GPU buffers
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
 * Toggle between CPU and GPU computation modes
 * @param m: new mode (CPU_MODE or GPU_MODE)
 * Side effect: cleans previous buffers and initializes new ones
 */
void toggleMode(int m)
{
	clean();
	mode = m;
	init();
}


// -------------------- Mouse input handling --------------------

/**
 * Mouse button callback
 * @param button: which mouse button was pressed/released
 * @param state: button state (pressed/released)
 * @param x, y: mouse coordinates in window pixels
 * Side effect: updates Julia seed (mx, my) and scale if wheel scrolled
 */
void mouse(int button, int state, int x, int y)
{
	if (button <= 2)		// left, middle, right buttons
	{
		mx = (float)(scale*(x - SCREEN_X / 2));
		my = -(float)(scale*(y - SCREEN_Y / 2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

/**
 * Mouse motion callback
 * @param x, y: current mouse coordinates
 * Side effect: updates Julia seed (mx, my)
 */
void mouseMotion(int x, int y)
{
	mx = (float)(scale*(x - SCREEN_X / 2));
	my = -(float)(scale*(y - SCREEN_Y / 2));
}


// -------------------- Mouse input handling --------------------

/**
 * Normal key press callback
 * @param key: ASCII key pressed
 * @param x, y: mouse coordinates (unused)
 * Side effect: exit program or toggle CPU/GPU mode
 */
void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) exit(0);			// ESC to quit
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
}

/**
 * Special key press callback (arrows, function keys)
 * @param key: GLUT key code (e.g., GLUT_KEY_UP)
 * @param x, y: mouse coordinates (unused)
 * Side effect: increase/decrease fractal iteration precision
 */
void processSpecialKeys(int key, int x, int y) {
	// other keys (F1, F2, arrows, home, etc.)
	switch (key) {
	case GLUT_KEY_UP: 
		if (prec < 50)		// increase iterations
			prec++;
		break;
	case GLUT_KEY_DOWN: 
		if (prec > 1)		// decrease iterations
			prec--;
		break;
	}
}


// -------------------- Mouse input handling --------------------

/**
 * Initialize OpenGL and GLUT
 * @param argc, argv: command line arguments
 * Side effect: creates window, sets projection, disables depth testing
 */
void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_X, SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0, 0.0, 0.0, 0.0);			// black background
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


// -------------------- Mouse input handling --------------------

/**
 * Main function
 * @param argc: argument count
 * @param argv: argument values
 * Side effect: initializes OpenGL, CUDA buffers, registers callbacks, enters GLUT main loop
 * @return int: always 1 (never reached because GLUT loop is infinite)
 */
int main(int argc, char **argv) {

	initGL(argc, argv);

	// allocate CPU/GPU buffers
	init();

	// Register GLUT callbacks
	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	// Initialize GLEW for OpenGL extensions
	GLint GlewInitResult = glewInit();
	if (GlewInitResult != GLEW_OK) {
		printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
	}

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
