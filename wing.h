#ifndef WING_H
#define WING_H

#include <cutil_inline.h>

#define box 100

class Wing
{
	public:
		float3 pos;
		const float radius;
		const float length;
		const float mass;
		float wspol_unoszenia;
		float kat_natarcia;
		float3 sila_nosna;
		float move;

		Wing(float3 pos = make_float3(-10, 0, 4), float radius = 4, float length = 30)
			: pos(pos), 
			radius(radius),  
			length(length),
			mass(10),
			wspol_unoszenia(1),
			kat_natarcia(0.f),
			sila_nosna(make_float3(0.f, 0.f, 0.f)),
			move(0.2)
		{};

		void increase()
		{
			if(pos.y <  2.6)
				pos.y += move;
		}

		void decrease()
		{
			if(pos.y > -2.6)
				pos.y -= move;
		}

		void resetWingPosition()
		{
			pos.y = 0;
		}
/*
		void print()
		{
		    glColor4f(0.0f, 0.9f, 0.0f, 1.0);

		    glDisable(GL_BLEND);
		    glEnable(GL_LIGHTING);
		    glEnable(GL_LIGHT0);
		    glEnable(GL_COLOR_MATERIAL);
		    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
		    
		    
		    GLUquadric * o = gluNewQuadric();
		    gluQuadricNormals(o, GLU_SMOOTH);
		    

		    glPushMatrix();
		    glTranslatef(pos.x, pos.y, -box/2);
		    gluCylinder(o, radius, radius, box, 20, 2); // o, r_top, r_bot, wys, ile katow, ?
		    glPopMatrix();
		    gluDeleteQuadric(o);

		    
		    glBegin(GL_QUADS);
		       glVertex3f(pos.x, pos.y + radius, -box/2); //gora 
		       glVertex3f(pos.x + length, 0, -box/2);
		       glVertex3f(pos.x + length, 0, box/2);  
		       glVertex3f(pos.x, pos.y + radius, box/2);

		       glVertex3f(pos.x, pos.y - radius, -box/2);  //dol
		       glVertex3f(pos.x + length, 0, -box/2);
		       glVertex3f(pos.x + length, 0, box/2);  
		       glVertex3f(pos.x, pos.y - radius, box/2);
		    glEnd();


		    glBegin(GL_TRIANGLES);
		       glVertex3f(pos.x, pos.y + radius, -box/2); //gora 
		       glVertex3f(pos.x, pos.y - radius, -box/2);
		       glVertex3f(pos.x + length, 0, -box/2);  

		       glVertex3f(pos.x, pos.y + radius, box/2);  //dol
		       glVertex3f(pos.x, pos.y - radius, box/2);
		       glVertex3f(pos.x + length, 0, box/2); 
		    glEnd();


		    glBegin(GL_TRIANGLE_FAN);
		    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
		    {
		        float xc = radius*sin(kat);
		        float yc = radius*cos(kat);
		        glVertex3f(xc + pos.x, yc + pos.y, -box/2);
		    }
		    glEnd();    

		    glBegin(GL_TRIANGLE_FAN);
		    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
		    {
		        float xc = radius*sin(kat);
		        float yc = radius*cos(kat);
		        glVertex3f(xc + pos.x, yc + pos.y, box/2);
		    }
		    glEnd();
		}
*/
};


#endif // WING_H