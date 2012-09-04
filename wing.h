#ifndef WING_H
#define WING_H

#define box 100

class Wing
{
	public:
		float x;	//wspol poczatku
		float y;	//wspol poczatku
		float z;	//wspol pocz
		float promien;
		float dl;
		float masa;
		float wspol_unoszenia;
		float gestosc_pow;
		float kat_natarcia;
		//float3 v_powietrza;
		//float3 sila_nosna;

		Wing(float xx = -10, float yy = 0, float pro = 4, 
			 float dll = 30, float mas = 10, float wspol = 1, 
			 float katt_natarcia = .0f)
			: x(xx), y(yy), promien(pro), 
			dl(dll),masa(mas), wspol_unoszenia(wspol), kat_natarcia(katt_natarcia)
		{
			//sila_nosna = make_float3(0.f, 0.f, 0.f);
		
		};

		void print()
		{
			//wing
		    glColor4f(0.0f, 0.9f, 0.0f, 1.0);

		    glDisable(GL_BLEND);
		    glEnable(GL_LIGHTING);
		    glEnable(GL_LIGHT0);
		    glEnable(GL_COLOR_MATERIAL);
		    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
		    
		    
		    GLUquadric * o = gluNewQuadric();
		    gluQuadricNormals(o, GLU_SMOOTH);
		    

		    glPushMatrix();
		    glTranslatef(x, y, -box/2);
		    gluCylinder(o, promien, promien, box, 20, 2); // o, r_top, r_bot, wys, ile katow, ?
		    glPopMatrix();
		    gluDeleteQuadric(o);

		    
		    glBegin(GL_QUADS);
		       glVertex3f(x, y + promien, -box/2); //gora 
		       glVertex3f(x + dl, 0, -box/2);
		       glVertex3f(x + dl, 0, box/2);  
		       glVertex3f(x, y + promien, box/2);

		       glVertex3f(x, y - promien, -box/2);  //dol
		       glVertex3f(x + dl, 0, -box/2);
		       glVertex3f(x + dl, 0, box/2);  
		       glVertex3f(x, y - promien, box/2);
		    glEnd();


		    glBegin(GL_TRIANGLES);
		       glVertex3f(x, y + promien, -box/2); //gora 
		       glVertex3f(x, y - promien, -box/2);
		       glVertex3f(x + dl, 0, -box/2);  

		       glVertex3f(x, y + promien, box/2);  //dol
		       glVertex3f(x, y - promien, box/2);
		       glVertex3f(x + dl, 0, box/2); 
		    glEnd();


		    glBegin(GL_TRIANGLE_FAN);
		    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
		    {
		        float xc = promien*sin(kat);
		        float yc = promien*cos(kat);
		        glVertex3f(xc + x, yc + y, -box/2);
		    }
		    glEnd();    

		    glBegin(GL_TRIANGLE_FAN);
		    for(float kat = 0.0f; kat < (2.0f*M_PI); kat += (M_PI/32.0f))
		    {
		        float xc = promien*sin(kat);
		        float yc = promien*cos(kat);
		        glVertex3f(xc + x, yc + y, box/2);
		    }
		    glEnd();
		}
};


#endif // WING_H