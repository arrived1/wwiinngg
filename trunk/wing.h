#ifndef WING_H
#define WING_H


class Skrzydlo
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

		Skrzydlo(float xx = -10, float yy = 0, float pro = 4, float dll = 30, float mas = 10, float wspol = 1, float katt_natarcia = .0f)
			:x(xx), y(yy), promien(pro), dl(dll),masa(mas), wspol_unoszenia(wspol), kat_natarcia(katt_natarcia)
		{
			//sila_nosna = make_float3(0.f, 0.f, 0.f);
		
		};

};


#endif // WING_H