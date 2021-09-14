#ifndef _VOLUMEN_KERNEL_H_
#define _VOLUMEN_KERNEL_H_

#include "Arista_kernel.cu"

// acum1 y acum2 contienen el nuevo estado del volumen para las capas 1 y 2, respectivamente
__device__ void coulomb(float4 *acum1, float4 *acum2, float r, float angulo1, float angulo2, float angulo3,
						float angulo4, float delta_T, float ccn, float gravedad, float epsilon_h, float L, float H)
{
	float fsc, muc, sc, normq, aux;
	float u1, u2;

	u1 = M_SQRT2*sqrtf(powf(acum1->y,2.0) + powf(acum1->z,2.0))*acum1->x/sqrtf(powf(acum1->x,4.0) + powf(fmaxf(acum1->x,epsilon_h),4.0));
	u2 = M_SQRT2*sqrtf(powf(acum2->y,2.0) + powf(acum2->z,2.0))*acum2->x/sqrtf(powf(acum2->x,4.0) + powf(fmaxf(acum2->x,epsilon_h),4.0));
	fsc = 1.0 - r*(1.0 - expf(-powf(10.0*acum1->x/epsilon_h,2.0)));
	muc = defTerminoFriccion(angulo1, angulo2, angulo3, angulo4, acum1->x, u1, acum2->x, u2,
			r, gravedad, epsilon_h, L, H);
	sc = fsc*muc*gravedad*acum2->x*delta_T*ccn;
	normq = sqrtf(powf(acum2->y,2.0) + powf(acum2->z,2.0));
	if ((normq+sc >= EPSILON) && (acum2->x > 0.0))
		aux = normq / (normq+sc)*(normq >= sc);
	else
		aux = 0.0;
	acum2->y *= aux;
	acum2->z *= aux;
}

// d_datosVolumenes contiene el estado anterior. Want1 y Want2 contienen el estado anterior
// del volumen para las capas 1 y 2, respectivamente. acum1 y acum2 contienen el nuevo estado
// del volumen para las capas 1 y 2, respectivamente
__device__ void disImplicita(float4 Want1, float4 Want2, float4 *acum1, float4 *acum2, float r, float delta_T,
						float mfc, float mf0, float mfs, float gravedad, float epsilon_h)
{
	float h1, h1m, h2, h2m;
	float u1, u2, u1x, u1y, u2x, u2y;
	float uo1x, uo1y, uo2x, uo2y, du;
	float hmod, hmodm;

	if (acum1->x < 0.0)  acum1->x = 0.0;
	if (acum2->x < 0.0)  acum2->x = 0.0;

	// complejos
	h1 = acum1->x;
	h1m = sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	h2 = acum2->x;
	h2m = sqrtf(powf(h2,4.0) + powf(fmaxf(h2,epsilon_h),4.0));

	u1x = M_SQRT2*acum1->y*h1/h1m;
	u1y = M_SQRT2*acum1->z*h1/h1m;
	u2x = M_SQRT2*acum2->y*h2/h2m;
	u2y = M_SQRT2*acum2->z*h2/h2m;
	hmod = sqrtf(powf(Want1.x,4.0) + powf(fmaxf(Want1.x,epsilon_h),4.0));
	hmodm = sqrtf(powf(Want2.x,4.0) + powf(fmaxf(Want2.x,epsilon_h),4.0));

	uo1x = M_SQRT2*Want1.y*Want1.x/hmod;
	uo1y = M_SQRT2*Want1.z*Want1.x/hmod;
	uo2x = M_SQRT2*Want2.y*Want2.x/hmodm;
	uo2y = M_SQRT2*Want2.z*Want2.x/hmodm;
	du = sqrtf(powf(uo1x-uo2x,2.0) + powf(uo1y-uo2y,2.0));
	u1 = sqrtf(uo1x*uo1x + uo1y*uo1y);
	u2 = sqrtf(uo2x*uo2x + uo2y*uo2y);
	hmod = h2 + r*h1;
	hmodm = sqrtf(powf(hmod,4.0) + powf(fmaxf(hmod,epsilon_h*(1.0+r)),4.0));
	if ((h1 > 0) && (h2 > 0)) {
		// Fricción entre capas
		float c1 = delta_T*M_SQRT2*h2*hmod/hmodm*mfc*du;
		float c2 = delta_T*r*M_SQRT2*h1*hmod/hmodm*mfc*du;
		float c3 = delta_T*gravedad*mfs*mfs*u2/(powf(h2,4.0/3.0)+EPSILON);
		float det = 1.0 / ((1.0+c1)*(1.0+c2+c3)-c1*c2);
		acum1->y = h1*(u1x*(1.0+c2+c3)+c1*u2x)*det;
		acum2->y = h2*(u2x*(1.0+c1)+c2*u1x)*det;
		acum1->z = h1*(u1y*(1.0+c2+c3)+c1*u2y)*det;
		acum2->z = h2*(u2y*(1.0+c1)+c2*u1y)*det;
	}
	if ((h1 > 0) && (h2 < epsilon_h)) {
		// Fricción con el fondo
		u1x = M_SQRT2*acum1->y*h1/h1m;
		u1y = M_SQRT2*acum1->z*h1/h1m;
		float c1 = delta_T*gravedad*mf0*mf0*u1/(powf(h1,4.0/3.0)+EPSILON);
		acum1->y = h1*u1x/(1.0+c1);
		acum1->z = h1*u1y/(1.0+c1);
	}
	if ((h2 > 0) &&  (h1 < epsilon_h)) {
		u2x = M_SQRT2*acum2->y*h2/h2m;
		u2y = M_SQRT2*acum2->z*h2/h2m;
		float c1 = delta_T*gravedad*mfs*mfs*u2/(powf(h2,4.0/3.0)+EPSILON);
		acum2->y = h2*u2x/(1.0+c1);
		acum2->z = h2*u2y/(1.0+c1);
	}
}

__device__ void filtroEstado(float4 *acum1, float4 *acum2, float r, float vmax1, float vmax2,
						float delta_T, float gravedad, float epsilon_h)
{
	float aux, aux0, u1x, u1y, u2x, u2y;
	float h1, h1m, h2, h2m, u1, u2;
	float hmod, hmodm, cf;
//	float du, gp;

	if (acum1->x < 0.0)  acum1->x = 0.0;
	if (acum2->x < 0.0)  acum2->x = 0.0;

	aux0 = 1.0/(powf(acum1->x/epsilon_h,4.0) + EPSILON);
	aux = expf(-delta_T*aux0);
	acum1->y *= aux;
	acum1->z *= aux;
	aux0 = 1.0/(powf(acum2->x/epsilon_h,4.0) + EPSILON);
	aux = expf(-delta_T*aux0);
	acum2->y *= aux;
	acum2->z *= aux;

	// Complejos
	h1 = acum1->x;
	h1m = sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	h2 = acum2->x;
	h2m = sqrtf(powf(h2,4.0) + powf(fmaxf(h2,epsilon_h),4.0));

	u1x = M_SQRT2*acum1->y*h1/h1m;
	u1y = M_SQRT2*acum1->z*h1/h1m;
	u2x = M_SQRT2*acum2->y*h2/h2m;
	u2y = M_SQRT2*acum2->z*h2/h2m;

	u1 = sqrtf(powf(u1x,2.0) + powf(u1y,2.0));
	u2 = sqrtf(powf(u2x,2.0) + powf(u2y,2.0));
/*du = sqrtf(powf(u1x-u2x,2.0) + powf(u1y-u2y,2.0));
gp = gravedad*(1.0 - r);*/
	hmod = h2 + r*h1;
	hmodm = sqrtf(powf(hmod,4.0) + powf(fmaxf(hmod,epsilon_h*(1.0 + r)),4.0));
//	cf = sqrtf(M_SQRT2*powf(du,2.0)*(h1+h2) / (gp*sqrtf(powf(h1+h2,4.0) + powf(fmaxf(h1+h2,2*epsilon_h),4.0))));
cf = 0.0;
	if ((cf > 1) && (h1 > 0) && (h2 > 0)) {
		// cout << "Atencion: " << cf << endl;
		float c1 = M_SQRT2*h2*hmod/hmodm*fmaxf(cf-1.0,0.0);
		float c2 = r*M_SQRT2*h1*hmod/hmodm*fmaxf(cf-1.0,0.0);
		float det = (1+c1)*(1+c2) - c1*c2;
		float u1n = (u1*(1+c2) + c1*u2)/det;
		float u2n = (u2*(1+c1) + c2*u1)/det;

		u1x *= u1n/(u1 + EPSILON);
		u1y *= u1n/(u1 + EPSILON);
		acum1->y = u1x*h1;
		acum1->z = u1y*h1;
		u2x *= u2n/(u2 + EPSILON);
		u2y *= u2n/(u2 + EPSILON);
		acum2->y = u2x*h2;
		acum2->z = u2y*h2;
	}
	// Fin complejos

	if (vmax1 > 0.0) {
		float h, hm, ux, uy, u;

		h = acum1->x;
		hm = sqrtf(powf(h,4.0) + powf(fmaxf(h,epsilon_h),4.0));
		ux = M_SQRT2*acum1->y*h/hm;
		uy = M_SQRT2*acum1->z*h/hm;
		u = sqrtf(ux*ux+uy*uy);
		if (u > vmax1) {
			ux *= vmax1/u;
			uy *= vmax1/u;
			acum1->y = ux*h;
			acum1->z = uy*h;
		}
	}
	if (vmax2 > 0.0) {
		float h, hm, ux, uy, u;

		h = acum2->x;
		hm = sqrtf(powf(h,4.0) + powf(fmaxf(h,epsilon_h),4.0));
		ux = M_SQRT2*acum2->y*h/hm;
		uy = M_SQRT2*acum2->z*h/hm;
		u = sqrtf(ux*ux+uy*uy);
		if (u > vmax2) {
			ux *= vmax2/u; 
			uy *= vmax2/u;
			acum2->y = ux*h;
			acum2->z = uy*h;
		}
	}
}

__global__ void actualizarEta1MaximaGPU(float2 *d_eta1_maxima, int num_volx, int num_voly, float tiempo_act)
{
	float4 Wact1, Wact2;
	float2 val_eta1;
	float val;
	int pos, pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		// pos = posición en el vector d_eta1_maxima
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		// Sumamos 1 a la coordenada y de la textura porque la primera
		// fila de la textura corresponde a volúmenes de comunicación
		// de otro cluster
		pos_y_hebra++;

		// Actualizamos la eta1 máxima, si procede
		val_eta1 = d_eta1_maxima[pos];
		Wact1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
		Wact2 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
		//val = Wact1.x + Wact2.x - Wact1.w;
		val = Wact1.x - Wact1.w;
		if (val > val_eta1.x) {
			val_eta1.x = val;
			val_eta1.y = tiempo_act;
			d_eta1_maxima[pos] = val_eta1;
		}
	}
}

__global__ void obtenerDeltaTVolumenesGPU(float4 *d_acumulador_1, float *d_deltaTVolumenes,
										  int num_volumenes, float area, float CFL)
{
	float deltaT, paso;
	int pos_hebra;

	pos_hebra = blockIdx.x*NUM_HEBRAS_VOL + threadIdx.x;

	// Comprobamos si la hebra se sale del vector
	if (pos_hebra < num_volumenes) {
		deltaT = d_acumulador_1[pos_hebra].w;
		paso = ((deltaT < EPSILON) ? 1e30 : (2.0*CFL*area)/deltaT);
		d_deltaTVolumenes[pos_hebra] = paso;
	}
}

__global__ void obtenerEstadoYDeltaTVolumenesGPU(float4 *d_acumulador_1, float4 *d_acumulador_2,
			float *d_deltaTVolumenes, int num_volx, int num_voly, float area, float CFL, float r,
			float delta_T, float angulo1, float angulo2, float angulo3, float angulo4, float mfc,
			float mf0, float mfs, float vmax1, float vmax2, float gravedad, float epsilon_h, float L, float H)
{
	float4 Want1, Want2;
	float4 acum1, acum2;
	float val, paso;
	int pos, pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		// pos = posición en el acumulador
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		// Sumamos 1 a la coordenada y de la textura porque la primera
		// fila de la textura corresponde a volúmenes de comunicación
		// de otro cluster
		pos_y_hebra++;
		val = delta_T / area;

		// Contribución al delta T
		acum1 = d_acumulador_1[pos];
		paso = ((acum1.w < EPSILON) ? 1e30 : (2.0*CFL*area)/acum1.w);
		d_deltaTVolumenes[pos] = paso;

		// Actualizamos la capa 1
		Want1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
		// Ponemos el nuevo estado de la capa 1 en acum1
		acum1.x = Want1.x + val*acum1.x;
		acum1.y = Want1.y + val*acum1.y;
		acum1.z = Want1.z + val*acum1.z;
		acum1.w = Want1.w;

		// Actualizamos la capa 2
		acum2 = d_acumulador_2[pos];
		Want2 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
		// Ponemos el nuevo estado de la capa 2 en acum2
		acum2.x = Want2.x + val*acum2.x;
		acum2.y = Want2.y + val*acum2.y;
		acum2.z = Want2.z + val*acum2.z;
		acum2.w = Want2.w;

		filtroEstado(&acum1, &acum2, r, vmax1, vmax2, delta_T, gravedad, epsilon_h);
		disImplicita(Want1, Want2, &acum1, &acum2, r, delta_T, mfc, mf0, mfs, gravedad, epsilon_h);
		coulomb(&acum1, &acum2, r, angulo1, angulo2, angulo3, angulo4, delta_T, 1.0, gravedad,
			epsilon_h, L, H);

		d_acumulador_1[pos] = acum1;
		d_acumulador_2[pos] = acum2;
	}
}


#endif
