/*
 * This code was modified from the NVIDIA CUDA examples
 *     specifically the simpleGL and nbody and ...
 *
 * S. James Lee, 2008 Fall
 */

#include <GL/glew.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <paramgl.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <cutil.h>
#include <cuda_runtime_api.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include "ParticleRenderer.h"
#include "wing.h"
#include "framerate.h"

////////////////////////////////////////////////////////////////////////////////
// simulation parameter
float 	scaleFactor = 1.5f;		// 10.0f, 50
float 	velFactor = 8.0f;			// 15.0f, 100
float	massFactor = 120000.0f;	// 50000000.0,
float 	gStep = 0.001f;				// 0.005f
int 	gOffset = 0;
int 	gApprx = 4;

// GL drah_wing object
ParticleRenderer* renderer = 0;
int 	numBodies = 16384;
int 	gDrawMode = 1;
float 	gPointSize = 3.0f; //1.0f;
float	gSpriteSize = 0.8f;//scaleFactor*0.25f;

// simulation data storage
float* 	gPos = 0;
float* 	gVel = 0;
GLuint	gVBO = 0;				// 8 float (4 position, 4 color)
float*	d_particleData = 0;		// device side particle data storage
float*	h_particleData = 0;		// host side particle data storage

// view params
int 	ox = 0, oy = 0;
int 	buttonState        = 0;
float 	camera_trans[]     = {0, 6*scaleFactor, -45*scaleFactor};
float 	camera_rot[]       = {0, 0, 0};
float 	camera_trans_lag[] = {0, 6*scaleFactor, -45*scaleFactor};
float 	camera_rot_lag[]   = {0, 0, 0};
const float inertia        = 0.1;

float   sw = 1024.0f;
float	sh = 768.0f;

// cuda related...
int 	numBlocks = 1;
int 	numThreadsPerBlock = 256;

// simulation data
#define box 100
Wing* h_wing;
Wing* d_wing;

bool pauze = false;

// useful clamp macro
#define LIMIT(x,min,max) { if ((x)>(max)) (x)=(max); if ((x)<(min)) (x)=(min);}

////////////////////////////////////////////////////////////////////////////////
// forward declaration
void init(int bodies);
void reset(void);
void initGL(void);
void runCuda(void);
void display(void);
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void special(int key, int x, int y);
void idle(void);
void loadData(char* filename, int bodies);
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);

////////////////////////////////////////////////////////////////////////////////
void init(int bodies)
{
	
	// blocks per grid
	numBlocks = bodies / numThreadsPerBlock;
	
	// host particle data (position, velocity)
	h_particleData = (float*) malloc(8 * bodies * sizeof(float));
	h_wing = (Wing*) malloc(sizeof(Wing));
	
    // device particle data
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_particleData, 8 * bodies * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_wing, sizeof(Wing)));

	// load inital data set
    int idx = 0;
    int vidx = 0;
    int offset = 0;
    for (int i = 0; i < bodies; i++)
    {
        // float array index
        idx = i * 4;
        vidx = bodies*4 + idx;
        
        if ((i % 2) == 0)
            offset = idx;
        else
            offset = (idx + (bodies / 2)) % (bodies * 4);
        
        // set value from global data storage
        h_particleData[idx + 0]     = gPos[offset + 0]; // x
        h_particleData[idx + 1]     = gPos[offset + 1]; // y
        h_particleData[idx + 2]     = gPos[offset + 2]; // z
        h_particleData[idx + 3]     = gPos[offset + 3]; // mass
        h_particleData[vidx + 0]    = gVel[offset + 0]; // vx
        h_particleData[vidx + 1]    = gVel[offset + 1]; // vy
        h_particleData[vidx + 2]    = gVel[offset + 2]; // vz
        h_particleData[vidx + 3]    = gVel[offset + 3]; // padding    
    }
    
    //initializing wing
    h_wing = new Wing();
	
	// copy initial value to GPU memory
	CUDA_SAFE_CALL( cudaMemcpy(d_particleData, h_particleData,
							   8 * bodies * sizeof(float), 
							   cudaMemcpyHostToDevice) );

    CUDA_SAFE_CALL( cudaMemcpy(d_wing, h_wing,
                               sizeof(Wing), 
                               cudaMemcpyHostToDevice) );
}

////////////////////////////////////////////////////////////////////////////////
void reset(void)
{
   	// reset camera
   	camera_trans[0] = 0; 
   	camera_trans[1] = 6 * scaleFactor; 
   	camera_trans[2] = -45 * scaleFactor;
	camera_rot[0] = camera_rot[1] = camera_rot[2] = 0;
	camera_trans_lag[0] = 0; 
	camera_trans_lag[1] = 6 * scaleFactor; 
	camera_trans_lag[2] = -45 * scaleFactor;
	camera_rot_lag[0] = camera_rot_lag[1] = camera_rot_lag[2] = 0;
	
    pauze = false;

	// reset dataset
	CUDA_SAFE_CALL( cudaMemcpy(d_particleData, h_particleData,
							   8 * numBodies * sizeof(float), 
							   cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_wing, h_wing,
                               sizeof(Wing), 
                               cudaMemcpyHostToDevice) );
}

////////////////////////////////////////////////////////////////////////////////
void initGL(void)
{
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_VERSION_1_5 "
			             "GL_ARB_multitexture "
                         "GL_ARB_vertex_buffer_object")) 
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
	
	// particle renderer
    renderer = new ParticleRenderer(numBodies);
    createVBO((GLuint*)&gVBO);
    renderer->setVBO(gVBO, numBodies);
    renderer->setSpriteSize(0.4f);
    renderer->setShaders((char*)"../../../src/wing/data/sprite.vert", 
    					 (char*)"../../../src/wing/data/sprite.frag");
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel interface
extern "C" 
void cudaComputeGalaxy(float4* pos, float4* pdata, int width, int height, 
					   float step, int apprx, int offset, Wing* wing);

////////////////////////////////////////////////////////////////////////////////
void runCuda(void)
{
    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    CUDA_SAFE_CALL( cudaGLMapBufferObject( (void**)&dptr, gVBO) );

	// only compute 1/16 at one time
	gOffset = (gOffset+1) % (gApprx);
	
    // execute the kernel
    // each block has 16x16 threads, grid 16xX: X will be decided by the # of bodies
    if(!pauze)
        cudaComputeGalaxy(dptr, (float4*)d_particleData, 256, numBodies / 256, 
    				      gStep, gApprx, gOffset, d_wing);

    // unmap buffer object
    CUDA_SAFE_CALL( cudaGLUnmapBufferObject(gVBO) );  
}

////////////////////////////////////////////////////////////////////////////////
void display(void)
{
    // update simulation
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
    
    //box
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(box);
    
    //axes
    glColor3f(1.0, .0, .0);
    glBegin(GL_LINES);
       glVertex3f(.0f, .0f, .0f);  //x
       glVertex3f(box/2 + 20.0f, .0f, .0f);
       
       glVertex3f(.0f, .0f, .0f);  //y
       glVertex3f(.0f, box/2 + 20.0f, .0f);
       
       glVertex3f(.0f, .0f, .0f);  //x
       glVertex3f(.0f, .0f, box/2 + 20.0f);
    glEnd();

    //h_wing
    glColor4f(0.0f, 0.9f, 0.0f, 1.0);

    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    
    
    GLUquadric * o = gluNewQuadric();
    gluQuadricNormals(o, GLU_SMOOTH);
    
    glPushMatrix();
    glTranslatef(h_wing->pos.x, h_wing->pos.y, -box/2);
    gluCylinder(o, h_wing->radius, h_wing->radius, box, 20, 2); // o, r_top, r_bot, wys, ile katow, ?
    glPopMatrix();
    gluDeleteQuadric(o);

    
    glBegin(GL_QUADS );
       glVertex3f(h_wing->pos.x, h_wing->pos.y + h_wing->radius, -box/2); //gora 
       glVertex3f(h_wing->pos.x + h_wing->length, 0, -box/2);
       glVertex3f(h_wing->pos.x + h_wing->length, 0, box/2);  
       glVertex3f(h_wing->pos.x, h_wing->pos.y + h_wing->radius, box/2);

       glVertex3f(h_wing->pos.x, h_wing->pos.y - h_wing->radius, -box/2);  //dol
       glVertex3f(h_wing->pos.x + h_wing->length, 0, -box/2);
       glVertex3f(h_wing->pos.x + h_wing->length, 0, box/2);  
       glVertex3f(h_wing->pos.x, h_wing->pos.y - h_wing->radius, box/2);
    glEnd();


    glBegin(GL_TRIANGLES);
       glVertex3f(h_wing->pos.x, h_wing->pos.y + h_wing->radius, -box/2); //gora 
       glVertex3f(h_wing->pos.x, h_wing->pos.y - h_wing->radius, -box/2);
       glVertex3f(h_wing->pos.x + h_wing->length, 0, -box/2);  

       glVertex3f(h_wing->pos.x, h_wing->pos.y + h_wing->radius, box/2);  //dol
       glVertex3f(h_wing->pos.x, h_wing->pos.y - h_wing->radius, box/2);
       glVertex3f(h_wing->pos.x + h_wing->length, 0, box/2); 
    glEnd();


    glBegin(GL_TRIANGLE_FAN);
    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
    {
        float x = h_wing->radius*sin(kat);
        float y = h_wing->radius*cos(kat);
        glVertex3f(x + h_wing->pos.x, y + h_wing->pos.y, -box/2);
    }
    glEnd();    

    glBegin(GL_TRIANGLE_FAN);
    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
    {
        float x = h_wing->radius*sin(kat);
        float y = h_wing->radius*cos(kat);
        glVertex3f(x + h_wing->pos.x, y + h_wing->pos.y, box/2);
    }
    glEnd();


    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
    glTranslatef(camera_trans_lag[0], 
		     camera_trans_lag[1], 
		     camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
	
	// render bodies
    renderer->setPointSize(gPointSize);
    renderer->setSpriteSize(gSpriteSize);
    renderer->display(gDrawMode);
	
    // update frame rate
	framerateUpdate();
	
    glutSwapBuffers();

    glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
    
    //
    gSpriteSize *= (float)w / sw;
    gPointSize *= (float)w / sw;
    
    //
    sw = w;
    sh = h;
}

////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState = button + 1;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        //buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        //buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
void motion(int x, int y)
{
    float dx = x - ox;
    float dy = y - oy;

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += 0.005f * fabs(camera_trans[2]) * dx * (1024.0f/sw) / 2.0;
        camera_trans[1] -= 0.005f * fabs(camera_trans[2]) * dy * (768.0f/sh) / 2.0;
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += dy / 5.0;
        camera_rot[1] += dx / 5.0;
    }
    
    ox = x; oy = y;
	glutPostRedisplay();	
}

////////////////////////////////////////////////////////////////////////////////
void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
    	case '\033':
        case 'q':
            exit(0);
            break;
        case 'r':
        	// reset configuration
    		reset();
            //h_wing->resetWingPosition();
        	break;
        case 'd':
        	// change rendering mode
        	gDrawMode = (gDrawMode+1) % 3;
        	break;
        case 'p':
            pauze = !pauze;
            break;
        case '=':
        	// increase point size
        	h_wing->increase();
        	break;
        case '-':
        	// decrese point size
        	h_wing->decrease();
        	break;
    }
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
void special(int /*key*/, int /*x*/, int /*y*/)
{
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
void idle(void)
{
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//float mx,my,mz,Mx,My,Mz;
void loadData(char* filename, int bodies)
{
    int skip = 49152 / bodies;
    
    FILE *fin;
    
    if ((fin = fopen(filename, "r")))
    {
        const int maxStringLenght = 256;
        char buf[maxStringLenght];
        float v[7];
        int idx = 0;
        
        // allocate memory
        gPos = (float*)malloc(sizeof(float)*bodies*4);
        gVel = (float*)malloc(sizeof(float)*bodies*4);
         
    	for (int i=0; i< bodies; i++)
    	{
    		// depend on input size...
    		for (int j=0; j < skip; j++)
    			fgets(buf, maxStringLenght, fin);	// load line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		// update index
    		idx = i * 4;
    		
    		// position
    		gPos[idx+0] = v[1]*scaleFactor;
    		gPos[idx+1] = v[2]*scaleFactor;
    		gPos[idx+2] = v[3]*scaleFactor;
    		
    		// mass
    		gPos[idx+3] = v[0]*massFactor;
    		//printf("mass : %f\n", gPos[idx+3]);
    		
    		// velocity
    		gVel[idx+0] = v[4]*velFactor;
    		gVel[idx+1] = v[5]*velFactor;
    		gVel[idx+2] = v[6]*velFactor;
    		gVel[idx+3] = 1.0f;
    	}   
    }
    else
    {
    	printf("cannot find file...: %s\n", filename);
    	exit(0);
    }
	//printf("bulge min,max: %f %f %f %f %f %f\n", mx, my, mz, Mx, My, Mz);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = numBodies * 8 * sizeof( float); //4
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL( cudaGLRegisterBufferObject(*vbo) );

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(*vbo) );

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // get number of SMs on this GPU
    int devID;
    cudaDeviceProp props;
    CUDA_SAFE_CALL( cudaGetDevice(&devID) );
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, devID) );

	// thread block size
    int p = 256;	// width  (number of threads in col within block)
    int q = 1;		// height (number of threads in row within block)
	
	// get total number of bodies
    if (!cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &numBodies))
    	// default number of bodies is #SMs * 4 * CTA size
    	//numBodies = p * q * 4 * props.multiProcessorCount;
    	numBodies = 8192;

	if (numBodies > 49152)
	{
		numBodies = 49152;
		printf("maximun number of bodies is 49152.\n");
	}
		
	// keep num of threads per block to 256
    if (q * p > 256)
    {
        p = 256 / q;
        printf("Setting p=%d, q=%d to maintain %d threads per block\n", p, q, 256);
    }

    if (q == 1 && numBodies < p)
    {
        p = numBodies;
    }

	// Data loading
	if (numBodies % 4096 != 0)
	{
		printf("number of body must be mulples of 4096\n");
		exit(0);
	}
	loadData((char*)"../../../src/wing/data/dubinski.tab", numBodies);
		
	// OpenGL: create app window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(1024, 768);
	char wtitle[256];
	sprintf(wtitle, "CUDA Galaxy Simulation (%d bodies)", numBodies); 
	glutCreateWindow(wtitle);
    
    // GL setup	
	initGL();
	
	// Initialize nbody system...	
    init(numBodies);
    
    // GL callback function
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

	//
	framerateTitle(wtitle);
	
	// let's start main loop
    glutMainLoop();

	deleteVBO((GLuint*)&gVBO);

	// clean up memory stuff
    if (gPos)
        free(gPos);
    if (gVel)
        free(gVel);
	
	if (renderer)
		delete renderer;
	
    return 0;
}




