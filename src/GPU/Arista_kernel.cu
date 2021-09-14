#ifndef _ARISTA_KERNEL_H_
#define _ARISTA_KERNEL_H_

#include "Matriz.cu"
#define _USE_MATH_DEFINES
#include <math.h>

#ifdef COULOMB

// Ley de Coulomb
__device__ float defTerminoFriccion(float angulo1, float angulo2, float angulo3, float angulo4, float h1ij,
						float u1ij_n, float h2ij, float u2ij_n, float r, float gravedad, float epsilon_h,
						float L, float H)
{
	return fabsf(tanf(angulo1))*(L/H);
}

#else

// Ley de Pouliquen
__device__ float defTerminoFriccion(float angulo1, float angulo2, float angulo3, float angulo4, float h1ij,
						float u1ij_n, float h2ij, float u2ij_n, float r, float gravedad, float epsilon_h,
						float L, float H)
{
	float muf, fr1, fr2, fr;
	float delta1, delta2, delta3, delta4, beta, L2, chi;
	float mustart, mustop;
	float rr, gp;

	rr = 1.0 - r*(1.0 - expf(-powf(10.0*h1ij/epsilon_h,2.0)));
	gp = gravedad*rr;
	fr1 = M_SQRT2*powf(u1ij_n,2.0)*h1ij/(gp*sqrtf(powf(h1ij,4.0) + powf(fmaxf(h1ij,epsilon_h),4.0)));
	fr2 = M_SQRT2*powf(u2ij_n,2.0)*h2ij/(gp*sqrtf(powf(h2ij,4.0) + powf(fmaxf(h2ij,epsilon_h),4.0)));
	fr = sqrtf(fr1 + fr2 + (1.0 - rr)*fr1*fr2);

	beta = 0.136;
	L2 = 8.0e-4;
	chi = 1.0e-3;

	delta1 = angulo1;  // angulo1*PI/180.0;
	delta2 = angulo2;  // angulo2*PI/180.0;
	delta3 = angulo3;  // angulo3*PI/180.0;
	delta4 = angulo4;  // angulo4*PI/180.0;

	mustop = tanf(delta1) + (tanf(delta2) - tanf(delta1)) / (1.0 + h2ij/L2);
	mustart = tanf(delta3) + (tanf(delta4) - tanf(delta3)) / (1.0 + h2ij/L2);

	if (fr > beta)
		muf = tanf(delta1) + (tanf(delta2) - tanf(delta1)) / (1.0 + beta*h2ij/(fr*L2));
	else {
		if (fabsf(fr) < EPSILON)
			muf = tanf(delta3) + (tanf(delta4) - tanf(delta3)) / (1.0 + h2ij/L2);
		else
			muf = mustart + powf(fr/beta,chi)*(mustop-mustart);
	}

	return muf*L/H;
}

#endif

__device__ TVec4 getFlujo_1dC(TVec4 *W, float epsilon_h)
{
	float h, qn, u;
	TVec4 F;

	h = W->x;
	qn = W->y;
	if (h < EPSILON) {
		F.x = qn;
		F.y = 0.0;
	}
	else {
		u = M_SQRT2*h*qn / sqrtf(powf(h,4.0) + powf(fmaxf(h,epsilon_h),4.0));
		F.x = qn;
		F.y = u*qn;
	}

	h = W->z;
	qn = W->w;
	if (h < EPSILON) {
		F.z = qn;
		F.w = 0;
	}
	else {
		u = M_SQRT2*h*qn / sqrtf(powf(h,4.0) + powf(fmaxf(h,epsilon_h),4.0));
		F.z = qn;
		F.w = u*qn;
	}

	return F;
}

__device__ TVec4 terminosPresion1D(float h1ij, float h2ij, TVec4 *W0_rot, TVec4 *W1_rot,
								  float H0, float H1, float r, float gravedad)
{
	TVec4 tp;
	float Hm, h0, h1, deta1, deta2;

	Hm = fminf(H0,H1);
	h0 = fmaxf(W0_rot->x + W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->x + W1_rot->z - H1 + Hm, 0.0);
	deta1 = h1-h0;

	h0 = fmaxf(W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->z - H1 + Hm, 0.0);
	deta2 = h1-h0;

	tp.x = 0.0;
	tp.y = gravedad*h1ij*deta1;
	tp.z = 0.0;
	tp.w = gravedad*h2ij*((1-r)*deta2 + r*deta1);

	return tp;
}

__device__ TVec4 terminosPresion1DMod(float h1ij, float h2ij, float u1ij_n, float u2ij_n,
					TVec4 *W0_rot, TVec4 *W1_rot, float H0, float H1, float r, float delta_T,
					float angulo1, float angulo2, float angulo3, float angulo4, float peso,
					float gravedad, float epsilon_h, float L, float H)
{
	TVec4 tp;
	float Hm, h0, h1, deta1, deta2;
	float muc, fsc, sc;
	int coulomb;

	Hm = fminf(H0,H1);
	h0 = fmaxf(W0_rot->x + W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->x + W1_rot->z - H1 + Hm, 0.0);
	deta1 = h1-h0;

	h0 = fmaxf(W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->z - H1 + Hm, 0.0);
	muc = defTerminoFriccion(angulo1, angulo2, angulo3, angulo4, h1ij, u1ij_n, h2ij, u2ij_n,
			r, gravedad, epsilon_h, L, H);
	fsc = 1.0 - r*(1.0 - expf(-powf(10.0*h1ij/epsilon_h,2.0)));
	sc = fsc*muc*gravedad*h2ij;
	coulomb = (fabsf(h2ij*u2ij_n) < peso*sc*delta_T) ? 1 : 0;
	deta2 = h1-h0;

	tp.x = 0.0;
	tp.y = gravedad*h1ij*deta1;
	tp.z = 0.0;
	if (coulomb)
		tp.w = gravedad*h2ij*r*deta1;
	else
		tp.w = gravedad*h2ij*((1-r)*deta2 + r*deta1);

	return tp;
}

__device__ TVec4 aproximarAutovalores1D(TVec4 *f0, float r, float gravedad, float epsilon_h, int flag)
{
	TVec4 D;
	float aux, uu, u1, u2;
	float gp = gravedad*(1.0 - r);
	float q = (f0->y + f0->w)*(flag==1) + (f0->x*f0->y + f0->z*f0->w)*(flag==0);
	float h = f0->x + f0->z;
	float hd = sqrtf(powf(h,4.0) + powf(fmaxf(h,epsilon_h),4.0));
	float u = M_SQRT2*q*h/hd;
	float cg = sqrtf(gravedad*h);

	// Autovalores externos
	D.x = u - cg;
	D.w = u + cg;
	
	// Autovalores internos
	if (flag == 0) {
		uu = M_SQRT2*(f0->x*f0->w + f0->y*f0->z)*h/hd;
		u1 = f0->y;
		u2 = f0->w;
	}
	else {
		u1 = M_SQRT2*f0->x*f0->y / sqrtf(powf(f0->x,4.0) + powf(fmaxf(f0->x,epsilon_h),4.0));
		u2 = M_SQRT2*f0->z*f0->w / sqrtf(powf(f0->z,4.0) + powf(fmaxf(f0->z,epsilon_h),4.0));
		uu = M_SQRT2*(u1*f0->z + u2*f0->x)*h/hd;
	}
	aux = 1.0 - M_SQRT2*powf(u1-u2,2.0)*h/(gp*hd);
	cg = sqrtf(gp*f0->x*f0->z*M_SQRT2*h/hd*fabsf(aux));

	D.y = uu - cg;
	D.z = uu + cg;

	/*for (i=0; i<3; i++) {
		for (j=1; j<4; j++) {
			if (v_get_val(&D,i) > v_get_val(&D,j)) {
				aux = v_get_val(&D,j);
				v_set_val(&D, j, v_get_val(&D,i));
				v_set_val(&D, i, aux);
			}
		}
	}*/
	// i=0, j=1
	if (D.x > D.y) {
		aux = D.y;
		D.y = D.x;
		D.x = aux;
	}
	// i=0, j=2
	if (D.x > D.z) {
		aux = D.z;
		D.z = D.x;
		D.x = aux;
	}
	// i=0, j=3
	if (D.x > D.w) {
		aux = D.w;
		D.w = D.x;
		D.x = aux;
	}
	// i=1, j=2
	if (D.y > D.z) {
		aux = D.z;
		D.z = D.y;
		D.y = aux;
	}
	// i=1, j=3
	if (D.y > D.w) {
		aux = D.w;
		D.w = D.y;
		D.y = aux;
	}
	// i=2, j=1
	if (D.z > D.y) {
		aux = D.y;
		D.y = D.z;
		D.z = aux;
	}
	// i=2, j=3
	if (D.z > D.w) {
		aux = D.w;
		D.w = D.z;
		D.z = aux;
	}

	return D;
}

__device__ TVec4 identityModification(float h1ij, float h2ij, float u1ij_n, float u2ij_n, TVec4 *W0_rot,
					TVec4 *W1_rot, float H0, float H1, float dif_q1, float dif_q2, float r, float delta_T,
					float angulo1, float angulo2, float angulo3, float angulo4, float peso,
					float gravedad, float epsilon_h, float L, float H)
{
	TVec4 I2;
	float Hm, h0, h1;
	float deta1, deta2;
	float muc, fsc, sc;
	int coulomb;

	Hm = fminf(H0,H1);
	h0 = fmaxf(W0_rot->x + W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->x + W1_rot->z - H1 + Hm, 0.0);
	deta1 = h1-h0;

	h0 = fmaxf(W0_rot->z - H0 + Hm, 0.0);
	h1 = fmaxf(W1_rot->z - H1 + Hm, 0.0);
	deta2 = h1-h0;

	muc = defTerminoFriccion(angulo1, angulo2, angulo3, angulo4, h1ij, u1ij_n, h2ij, u2ij_n,
			r, gravedad, epsilon_h, L, H);
	fsc = 1.0 - r*(1.0 - expf(-powf(10.0*h1ij/epsilon_h,2.0)));
	sc = fsc*muc*gravedad*h2ij;

	coulomb = (fabsf(h2ij*u2ij_n) < peso*sc*delta_T) ? 1 : 0;
	I2.x = deta1;
	I2.y = dif_q1;
	I2.w = dif_q2;
	if (coulomb)
		I2.z = 0.0;
	else {
		I2.x -= deta2*((W0_rot->x > epsilon_h) && (W1_rot->x > epsilon_h));
		I2.z = deta2;
	}

    return I2;
}

// Tratamiento seco-mojado distinto
/*__device__ void tratamientoSecoMojado(TVec4 *W0_rot, TVec4 *W1_rot, float *H0, float *H1, float epsilon_h)
{
	if ((W0_rot->z < epsilon_h) && (*H1 - W1_rot->z > *H0))
		W1_rot->w *= 0.0*(W1_rot->w <= 0) + 1.0*(W1_rot->w > 0);
	if ((W1_rot->z < epsilon_h) && (*H0 - W0_rot->z > *H1))
		W0_rot->w *= 0.0*(W0_rot->w >= 0) + 1.0*(W0_rot->w < 0);

	if ((W0_rot->x < epsilon_h) && (*H1 - W1_rot->x - W1_rot->z > *H0 - W0_rot->z))
		W1_rot->y *= 0.0*(W1_rot->y <= 0) + 1.0*(W1_rot->y > 0);
	if ((W1_rot->x < epsilon_h) && (*H0 - W0_rot->x - W0_rot->z > *H1 - W1_rot->z))
		W0_rot->y *= 0.0*(W0_rot->y >= 0) + 1.0*(W0_rot->y < 0);

	if ((W0_rot->z + W0_rot->x < epsilon_h) && (*H1 - W1_rot->x - W1_rot->z > *H0)) {
		W1_rot->y *= 0.0*(W1_rot->y <= 0) + 1.0*(W1_rot->y > 0);
		W1_rot->w *= 0.0*(W1_rot->w <= 0) + 1.0*(W1_rot->w > 0);
	}
	if ((W1_rot->z + W1_rot->x < epsilon_h) && (*H0 - W0_rot->x - W0_rot->z > *H1)) {
		W0_rot->y *= 0.0*(W0_rot->y >= 0) + 1.0*(W0_rot->y < 0);
		W0_rot->w *= 0.0*(W0_rot->w >= 0) + 1.0*(W0_rot->w < 0);
	}
}*/

__device__ void tratamientoSecoMojado(TVec4 *W0_rot, TVec4 *W1_rot, float *H0, float *H1, float epsilon_h)
{
	if ((W0_rot->z < epsilon_h) && (*H1 - W1_rot->z > *H0))
		W1_rot->w = 0.0;
	if ((W1_rot->z < epsilon_h) && (*H0 - W0_rot->z > *H1))
		W0_rot->w = 0.0;

	if ((W0_rot->x < epsilon_h) && (*H1 - W1_rot->x - W1_rot->z > *H0 - W0_rot->z))
		W1_rot->y = 0.0;
	if ((W1_rot->x < epsilon_h) && (*H0 - W0_rot->x - W0_rot->z > *H1 - W1_rot->z))
		W0_rot->y = 0.0;

	if ((W0_rot->z + W0_rot->x < epsilon_h) && (*H1 - W1_rot->x - W1_rot->z > *H0)) {
		W1_rot->y = 0.0;
		W1_rot->w = 0.0;
	}
	if ((W1_rot->z + W1_rot->x < epsilon_h) && (*H0 - W0_rot->x - W0_rot->z > *H1)) {
		W0_rot->y = 0.0;
		W0_rot->w = 0.0;
	}
}

// pos_vol0 y pos_vol1 son las posiciones de los acumuladores donde la arista escribirá sus contribuciones
__device__ void procesarArista(TVec *W0, TVec *W1, float H0, float H1, float normal_x, float normal_y,
				float longitud, float area, float r, float delta_T, float angulo1, float angulo2,
				float angulo3, float angulo4, float peso, float beta, float4 *d_acumulador_1,
				float4 *d_acumulador_2, int pos_vol0, int pos_vol1, float gravedad, float epsilon_h,
				float L, float H, int num_volumenes)
{
	int i;
	TVec4 DES, tp, tp2;
	// Vectores Fij+ y Fij-
	TVec Fmas6, Fmenos6;
	TVec4 Fmas4, Fmenos4;
	// Vector normal unitario a la arista
	float2 normal1;
	float h1ij, h2ij;
	float u1ij_n, u2ij_n, u1ij_t, u2ij_t;
	float a, b, c, a0, a1, a2;
	// Autovalores
	float aut1, aut2, aut3;
	float max_autovalor;
	// Vectores de velocidad de los volúmenes 0 y 1 para las capas 1 y 2.
	// u<volumen>n
	float u0n, u1n;
	// Vectores de caudal tangenciales de los volúmenes 0 y 1 para las capas
	// 1 y 2. q<volumen>t
	float q0t, q1t;
	// Valores de h de los volúmenes 0 y 1 para las capas 1 y 2,
	// y sus raíces cuadradas
	// h<volumen>
	float h0, h1, sqrt_h0, sqrt_h1;
	// Estados rotados de los volúmenes 0 y 1
	TVec4 W0_rot, W1_rot;
	float4 acum0_1, acum0_2;
	float4 acum1_1, acum1_2;

	// Obtenemos el vector normal unitario a la arista
	normal1.x = normal_x/longitud;
	normal1.y = normal_y/longitud;

	// Obtenemos los estados rotados W0_rot y W1_rot
	W0_rot.x = v_get_val(W0,0);
	W0_rot.y = v_get_val(W0,1)*normal1.x + v_get_val(W0,2)*normal1.y;
	W0_rot.z = v_get_val(W0,3);
	W0_rot.w = v_get_val(W0,4)*normal1.x + v_get_val(W0,5)*normal1.y;

	W1_rot.x = v_get_val(W1,0);
	W1_rot.y = v_get_val(W1,1)*normal1.x + v_get_val(W1,2)*normal1.y;
	W1_rot.z = v_get_val(W1,3);
	W1_rot.w = v_get_val(W1,4)*normal1.x + v_get_val(W1,5)*normal1.y;

	tratamientoSecoMojado(&W0_rot, &W1_rot, &H0, &H1, epsilon_h);

	// Capa 1
	h0 = W0_rot.x;
	h1 = W1_rot.x;
	h1ij = 0.5*(h0 + h1);

	u0n = M_SQRT2*h0*W0_rot.y / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
	u1n = M_SQRT2*h1*W1_rot.y / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	sqrt_h0 = sqrtf(h0);
	sqrt_h1 = sqrtf(h1);
	u1ij_n = (sqrt_h0*u0n + sqrt_h1*u1n) / (sqrt_h0 + sqrt_h1 + EPSILON);

	// Capa 2
	h0 = W0_rot.z;
	h1 = W1_rot.z;
	h2ij = 0.5*(h0 + h1);

	u0n = M_SQRT2*h0*W0_rot.w / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
	u1n = M_SQRT2*h1*W1_rot.w / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	sqrt_h0 = sqrtf(h0);
	sqrt_h1 = sqrtf(h1);
	u2ij_n = (sqrt_h0*u0n + sqrt_h1*u1n) / (sqrt_h0 + sqrt_h1 + EPSILON);

	if ((h1ij >= EPSILON) || (h2ij >= EPSILON)) {
		// Hay agua
		// Leemos los valores de los acumuladores
		if (pos_vol0 >= 0) {
			acum0_1 = d_acumulador_1[pos_vol0];
			acum0_2 = d_acumulador_2[pos_vol0];
		}
		else {
			// El volumen 0 es un volumen del cluster adyacente superior.
			// El acumulador es cero porque estamos en el procesamiento de aristas horizontales de tipo 3
			acum0_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			acum0_2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		if ((pos_vol1 != -1) && (pos_vol1 < num_volumenes)) {
			acum1_1 = d_acumulador_1[pos_vol1];
			acum1_2 = d_acumulador_2[pos_vol1];
		}
		else {
			// Si pos_vol1 == -1, es una arista frontera y no se utiliza acum1_1 ni acum1_2.
			// Si pos_vol1 >= num_volumenes, el acumulador es cero porque estamos en el procesamiento
			// de aristas horizontales de tipo 3
			acum1_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			acum1_2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}

		// Obtenemos los términos de presión
		tp = terminosPresion1D(h1ij, h2ij, &W0_rot, &W1_rot, H0, H1, r, gravedad);
		tp2 = terminosPresion1DMod(h1ij, h2ij, u1ij_n, u2ij_n, &W0_rot, &W1_rot, H0, H1, r,
				delta_T, angulo1, angulo2, angulo3, angulo4, peso, gravedad, epsilon_h, L, H);

		// Obtenemos los autovalores de A
		DES.x = h1ij;
		DES.y = u1ij_n;
		DES.z = h2ij;
		DES.w = u2ij_n;
		DES = aproximarAutovalores1D(&DES, r, gravedad, epsilon_h, 0);
		Fmas4 = aproximarAutovalores1D(&W0_rot, r, gravedad, epsilon_h, 1);
		Fmenos4 = aproximarAutovalores1D(&W1_rot, r, gravedad, epsilon_h, 1);

		aut1 = fminf(DES.x, fminf(Fmas4.x,Fmenos4.x));
		aut3 = fmaxf(DES.w, fmaxf(Fmas4.w,Fmenos4.w));
		max_autovalor = fmaxf(fabsf(aut3), fabsf(aut1));
		a = fmaxf(fabsf(u1ij_n), fabsf(u2ij_n));
		if (a > max_autovalor)
			max_autovalor = a;
		aut2 = fmaxf(fabsf(DES.y), fabsf(DES.z));
		if (SGN(aut1 + aut3) < 0)
			aut2 = -aut2;

		if ( (fabsf(aut1-aut3) >= EPSILON) && (fabsf(aut1-aut2) >= EPSILON) && (fabsf(aut2-aut3) >= EPSILON) ) {
			if ((h1ij >= epsilon_h) && (h2ij >= epsilon_h) ) {
				a = fabsf(aut1) / ((aut1-aut3)*(aut1-aut2));
				b = fabsf(aut2) / ((aut2-aut1)*(aut2-aut3));
				c = fabsf(aut3) / ((aut3-aut1)*(aut3-aut2));
				a0 = a*aut2*aut3 + b*aut1*aut3 + c*aut1*aut2;
				a1 = -a*(aut2 + aut3) - b*(aut1 + aut3) - c*(aut1 + aut2);
				a2 = a + b + c;
			}
			else {
				a0 = (aut3*fabsf(aut1) - aut1*fabsf(aut3)) / (aut3 - aut1);
				a1 = (fabsf(aut3) - fabsf(aut1)) / (aut3 - aut1);
				a2 = 0.0;
			}
		}
		else {
			a0 = max_autovalor;
			a1 = 0.0;
			a2 = 0.0;
		}

		// Fmas4 = getFlujo_1dC(&W1_rot) - getFlujo_1dC(&W0_rot);
		Fmas4 = getFlujo_1dC(&W1_rot, epsilon_h);
		Fmenos4 = getFlujo_1dC(&W0_rot, epsilon_h);
		v_sub4(&Fmas4, &Fmenos4, &Fmas4);

		v_add4(&tp2, &Fmas4, &tp2);
		v_add4(&Fmas4, &tp, &Fmenos4);
		sv_mlt4(0.5, &Fmenos4, &Fmenos4);

		// Fmas4 = A*tp2;
		a = gravedad*h1ij;
		b = gravedad*h2ij;
		Fmas4.x = tp2.y;
		Fmas4.y = (a - u1ij_n*u1ij_n)*tp2.x + 2*u1ij_n*tp2.y + a*tp2.z;
		Fmas4.z = tp2.w;
		Fmas4.w = r*b*tp2.x + (b - u2ij_n*u2ij_n)*tp2.z + 2*u2ij_n*tp2.w;

		// DES = I2
		DES = identityModification(h1ij, h2ij, u1ij_n, u2ij_n, &W0_rot, &W1_rot, H0, H1, W1_rot.y-W0_rot.y,
			W1_rot.w-W0_rot.w, r, delta_T, angulo1, angulo2, angulo3, angulo4, peso, gravedad, epsilon_h, L, H);

		// DES = 0.5*(a0*DES + a1*tp2 + a2*Fmas4);
		DES.x = a0*DES.x + a1*tp2.x + a2*Fmas4.x;
		DES.y = a0*DES.y + a1*tp2.y + a2*Fmas4.y;
		DES.z = a0*DES.z + a1*tp2.z + a2*Fmas4.z;
		DES.w = a0*DES.w + a1*tp2.w + a2*Fmas4.w;
		sv_mlt4(0.5, &DES, &DES);

		// Obtenemos Fij+ y Fij- de 4 componentes
		v_copy4(&Fmenos4, &Fmas4);
		// Fmenos4 += getFlujo1d(&W0_rot) - DES;
		tp = getFlujo_1dC(&W0_rot, epsilon_h);
		v_sub4(&tp, &DES, &tp);
		v_add4(&Fmenos4, &tp, &Fmenos4);
		// Fmas4   += DES - getFlujo1d(&W1_rot);
		tp = getFlujo_1dC(&W1_rot, epsilon_h);
		v_sub4(&DES, &tp, &tp);
		v_add4(&Fmas4, &tp, &Fmas4);

		// Calculamos u1ij_t
		h0 = W0_rot.x;
		h1 = W1_rot.x;
		q0t = v_get_val(W0,2)*normal1.x - v_get_val(W0,1)*normal1.y;
		q1t = v_get_val(W1,2)*normal1.x - v_get_val(W1,1)*normal1.y;
		u0n = M_SQRT2*h0*q0t / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
		u1n = M_SQRT2*h1*q1t / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
		if (fabsf(Fmenos4.x) < EPSILON)
			u1ij_t = 0.0;
		else if (Fmenos4.x > 0)
			u1ij_t = u0n;
		else
			u1ij_t = u1n;

		// Calculamos u2ij_t
		h0 = W0_rot.z;
		h1 = W1_rot.z;
		q0t = v_get_val(W0,5)*normal1.x - v_get_val(W0,4)*normal1.y;
		q1t = v_get_val(W1,5)*normal1.x - v_get_val(W1,4)*normal1.y;
		u0n = M_SQRT2*h0*q0t / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
		u1n = M_SQRT2*h1*q1t / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
		if (fabsf(Fmenos4.z) < EPSILON)
			u2ij_t = 0.0;
		else if (Fmenos4.z > 0)
			u2ij_t = u0n;
		else
			u2ij_t = u1n;

		// Obtenemos Fij+ y Fij-
		h1ij = Fmenos4.x*u1ij_t;
		v_set_val(&Fmas6, 0, Fmas4.x);
		v_set_val(&Fmas6, 1, Fmas4.y*normal1.x + h1ij*normal1.y);
		v_set_val(&Fmas6, 2, Fmas4.y*normal1.y - h1ij*normal1.x);
		v_set_val(&Fmenos6, 0, Fmenos4.x);
		v_set_val(&Fmenos6, 1, Fmenos4.y*normal1.x - h1ij*normal1.y);
		v_set_val(&Fmenos6, 2, Fmenos4.y*normal1.y + h1ij*normal1.x);

		h2ij = Fmenos4.z*u2ij_t;
		v_set_val(&Fmas6, 3, Fmas4.z);
		v_set_val(&Fmas6, 4, Fmas4.w*normal1.x + h2ij*normal1.y);
		v_set_val(&Fmas6, 5, Fmas4.w*normal1.y - h2ij*normal1.x);
		v_set_val(&Fmenos6, 3, Fmenos4.z);
		v_set_val(&Fmenos6, 4, Fmenos4.w*normal1.x - h2ij*normal1.y);
		v_set_val(&Fmenos6, 5, Fmenos4.w*normal1.y + h2ij*normal1.x);

		sv_mlt6(longitud, &Fmas6, &Fmas6);
		sv_mlt6(longitud, &Fmenos6, &Fmenos6);

		// Inicio positividad
		float b, dt0, dt1;
		float dta1, dta2;
		float factor = 1.0*beta;
		float alpha;
		// hp<volumen>_<capa>
		float hp0_0, hp0_1, hp1_0, hp1_1;

		// Asignamos hp0_0, hp0_1, hp1_0 y hp1_1
		b = delta_T/area;
		hp0_0 = v_get_val(W0,0) + b*acum0_1.x;
		hp0_1 = v_get_val(W0,3) + b*acum0_2.x;
		if (pos_vol1 != -1) {
			// Es una arista interna
			hp1_0 = v_get_val(W1,0) + b*acum1_1.x;
			hp1_1 = v_get_val(W1,3) + b*acum1_2.x;
		}

		dta1 = dta2 = 1e30;
		if (v_get_val(&Fmenos6,0) > 0.0)
			dta1 = hp0_0*area/(factor*v_get_val(&Fmenos6,0) + EPSILON);
		if (pos_vol1 != -1) {
			// Es una arista interna
			if (v_get_val(&Fmas6,0) > 0.0)
				dta2 = hp1_0*area/(factor*v_get_val(&Fmas6,0) + EPSILON);
		}
		dt0 = fminf(dta1,dta2);

		dta1 = dta2 = 1e30;
		if (v_get_val(&Fmenos6,3) > 0.0)
			dta1 = hp0_1*area/(factor*v_get_val(&Fmenos6,3) + EPSILON);
		if (pos_vol1 != -1) {
			// Es una arista interna
			if (v_get_val(&Fmas6,3) > 0.0)
				dta2 = hp1_1*area/(factor*v_get_val(&Fmas6,3) + EPSILON);
		}
		dt1 = fminf(dta1,dta2);

		if (delta_T <= dt0)
			alpha = 1.0;
		else
			alpha = dt0/(delta_T + EPSILON);
		for (i=0; i<3; i++) {
			v_set_val(&Fmenos6, i, alpha*v_get_val(&Fmenos6,i));
			v_set_val(&Fmas6, i, alpha*v_get_val(&Fmas6,i));
		}

		if (delta_T <= dt1)
			alpha = 1.0;
		else
			alpha = dt1/(delta_T + EPSILON);
		for (i=3; i<6; i++) {
			v_set_val(&Fmenos6, i, alpha*v_get_val(&Fmenos6,i));
			v_set_val(&Fmas6, i, alpha*v_get_val(&Fmas6,i));
		}
		// Fin positividad

		if (max_autovalor < epsilon_h)
			max_autovalor += epsilon_h;

		c = longitud*max_autovalor/peso;
		if (pos_vol0 >= 0) {
			// Actualizamos el valor del acumulador del volumen 0 para la capa 1
			acum0_1.x -= peso*v_get_val(&Fmenos6,0);
			acum0_1.y -= peso*v_get_val(&Fmenos6,1);
			acum0_1.z -= peso*v_get_val(&Fmenos6,2);
			acum0_1.w += c;
			d_acumulador_1[pos_vol0] = acum0_1;
			// Actualizamos el valor del acumulador del volumen 0 para la capa 2
			acum0_2.x -= peso*v_get_val(&Fmenos6,3);
			acum0_2.y -= peso*v_get_val(&Fmenos6,4);
			acum0_2.z -= peso*v_get_val(&Fmenos6,5);
			acum0_2.w += c;
			d_acumulador_2[pos_vol0] = acum0_2;
		}

		if ((pos_vol1 != -1) && (pos_vol1 < num_volumenes)) {
			// Actualizamos el valor del acumulador del volumen 1 para la capa 1
			acum1_1.x -= peso*v_get_val(&Fmas6,0);
			acum1_1.y -= peso*v_get_val(&Fmas6,1);
			acum1_1.z -= peso*v_get_val(&Fmas6,2);
			acum1_1.w += c;
			d_acumulador_1[pos_vol1] = acum1_1;
			// Actualizamos el valor del acumulador del volumen 1 para la capa 2
			acum1_2.x -= peso*v_get_val(&Fmas6,3);
			acum1_2.y -= peso*v_get_val(&Fmas6,4);
			acum1_2.z -= peso*v_get_val(&Fmas6,5);
			acum1_2.w += c;
			d_acumulador_2[pos_vol1] = acum1_2;
		}
	}
}

// tipo indica el tipo de aristas que se están procesando.
// tipo=1 => aristas_ver1, tipo=2 => aristas_ver2, tipo=3 => aristas_hor1, tipo=4 => aristas_hor2
// Si es una arista vertical => borde1 = borde_izq, borde2 = borde_der
// Si es una arista horizontal => borde1 = borde_sup, borde2 = borde_inf
__global__ void procesarAristasGPU(int num_volx, int num_voly, int num_volumenes, float borde1, float borde2, float longitud,
				float area, float r, float delta_T, float angulo1, float angulo2, float angulo3, float angulo4,
				float peso, float beta, float4 *d_acumulador_1, float4 *d_acumulador_2, float gravedad,
				float epsilon_h, float L, float H, int tipo, int id_hebra, int ultima_hebra)
{
	float4 datos_vol0, datos_vol1;
	// Posición (x,y) de la malla asociada a la hebra
	int pos_x_hebra, pos_y_hebra;
	int pos_vol0, pos_vol1;
	// W0 y W1 tienen forma [h1, q1x, q1y, h2, q2x, q2y]
	TVec W0, W1;

	if (tipo < 3) {
		// Arista vertical
		// Multiplicamos pos_x_hebra por 2 porque se procesan aristas alternas
		pos_x_hebra = 2*(blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x);
		pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y;
		if (tipo == 2) pos_x_hebra++;

		// Comprobamos si la hebra (arista) está dentro de los límites de la malla
		if ((pos_x_hebra <= num_volx) && (pos_y_hebra < num_voly)) {
			// Procesamos la arista vertical
			// Obtenemos los datos de los volúmenes 0 y 1
			if (pos_x_hebra == 0) {
				// Frontera izquierda
				// normal_x = -longitud
				// El volumen 0 de la arista está situado a la derecha de la arista
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);

				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 es fantasma
				v_set_val(&W1, 0, v_get_val(&W0,0));
				v_set_val(&W1, 1, v_get_val(&W0,1)*borde1);
				v_set_val(&W1, 2, v_get_val(&W0,2));
				v_set_val(&W1, 3, v_get_val(&W0,3));
				v_set_val(&W1, 4, v_get_val(&W0,4)*borde1);
				v_set_val(&W1, 5, v_get_val(&W0,5));

				pos_vol0 = pos_y_hebra*num_volx;
				procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, -longitud, 0.0, longitud, area, r,
					delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
					pos_vol0, -1, gravedad, epsilon_h, L, H, num_volumenes);
			}
			else {
				// normal_x = longitud
				// El volumen 0 de la arista está situado a la izquierda de la arista
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra-1, pos_y_hebra+1);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra-1, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				if (pos_x_hebra == num_volx) {
					// Frontera derecha. El volumen 1 es fantasma
					v_set_val(&W1, 0, v_get_val(&W0,0));
					v_set_val(&W1, 1, v_get_val(&W0,1)*borde2);
					v_set_val(&W1, 2, v_get_val(&W0,2));
					v_set_val(&W1, 3, v_get_val(&W0,3));
					v_set_val(&W1, 4, v_get_val(&W0,4)*borde2);
					v_set_val(&W1, 5, v_get_val(&W0,5));

					pos_vol0 = (pos_y_hebra+1)*num_volx - 1;
					procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, longitud, 0.0, longitud, area, r,
						delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
						pos_vol0, -1, gravedad, epsilon_h, L, H, num_volumenes);
				}
				else {
					// Arista interna
					// El volumen 1 de la arista está situado a la derecha de la arista
					// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
					// corresponde a volúmenes de comunicación de otro cluster
					datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 0, datos_vol1.x);
					v_set_val(&W1, 1, datos_vol1.y);
					v_set_val(&W1, 2, datos_vol1.z);
					datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 3, datos_vol1.x);
					v_set_val(&W1, 4, datos_vol1.y);
					v_set_val(&W1, 5, datos_vol1.z);

					pos_vol0 = pos_y_hebra*num_volx + pos_x_hebra-1;
					procesarArista(&W0, &W1, datos_vol0.w, datos_vol1.w, longitud, 0.0, longitud, area, r,
						delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
						pos_vol0, pos_vol0+1, gravedad, epsilon_h, L, H, num_volumenes);
				}
			}
		}
	}
	else {
		// Arista horizontal
		// Multiplicamos pos_y_hebra por 2 porque se procesan aristas alternas
		pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x;
		pos_y_hebra = 2*(blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y);
		if (tipo == 4) pos_y_hebra++;

		// Comprobamos si la hebra (arista) está dentro de los límites de la malla
		if ((pos_x_hebra < num_volx) && (pos_y_hebra <= num_voly)) {
			// Procesamos la arista horizontal
			// Obtenemos los datos de los volúmenes 0 y 1
			if ((pos_y_hebra == 0) && (id_hebra == 0)) {
				// Frontera superior
				// normal_y = -longitud
				// El volumen 0 de la arista está situado debajo de la arista.
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 es fantasma
				v_set_val(&W1, 0, v_get_val(&W0,0));
				v_set_val(&W1, 1, v_get_val(&W0,1));
				v_set_val(&W1, 2, v_get_val(&W0,2)*borde1);
				v_set_val(&W1, 3, v_get_val(&W0,3));
				v_set_val(&W1, 4, v_get_val(&W0,4));
				v_set_val(&W1, 5, v_get_val(&W0,5)*borde1);

				procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, -longitud, longitud, area, r,
					delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
					pos_x_hebra, -1, gravedad, epsilon_h, L, H, num_volumenes);
			}
			else {
				// normal_y = longitud
				// El volumen 0 de la arista está situado arriba de la arista.
				// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
				// de la textura corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				if ((pos_y_hebra == num_voly) && ultima_hebra) {
					// Frontera inferior. El volumen 1 es fantasma
					v_set_val(&W1, 0, v_get_val(&W0,0));
					v_set_val(&W1, 1, v_get_val(&W0,1));
					v_set_val(&W1, 2, v_get_val(&W0,2)*borde2);
					v_set_val(&W1, 3, v_get_val(&W0,3));
					v_set_val(&W1, 4, v_get_val(&W0,4));
					v_set_val(&W1, 5, v_get_val(&W0,5)*borde2);

					pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
					procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, longitud, longitud, area, r,
						delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
						pos_vol0, -1, gravedad, epsilon_h, L, H, num_volumenes);
				}
				else {
					// Arista interna
					// El volumen 1 de la arista está situado debajo de la arista.
					// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
					// corresponde a volúmenes de comunicación de otro cluster
					datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 0, datos_vol1.x);
					v_set_val(&W1, 1, datos_vol1.y);
					v_set_val(&W1, 2, datos_vol1.z);
					datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 3, datos_vol1.x);
					v_set_val(&W1, 4, datos_vol1.y);
					v_set_val(&W1, 5, datos_vol1.z);

					pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
					pos_vol1 = pos_vol0 + num_volx;
					procesarArista(&W0, &W1, datos_vol0.w, datos_vol1.w, 0.0, longitud, longitud, area, r,
						delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
						pos_vol0, pos_vol1, gravedad, epsilon_h, L, H, num_volumenes);
				}
			}
		}
	}
}

// tipo debe ser 3 (primer grupo de aristas horizontales)
__global__ void procesarAristasNoComGPU(int num_volx, int num_voly, int num_volumenes, float borde1, float borde2,
				float longitud, float area, float r, float delta_T, float angulo1, float angulo2, float angulo3,
				float angulo4, float peso, float beta, float4 *d_acumulador_1, float4 *d_acumulador_2, float gravedad,
				float epsilon_h, float L, float H, int tipo, int id_hebra, int ultima_hebra)
{
	float4 datos_vol0, datos_vol1;
	// Posición (x,y) de la malla asociada a la hebra
	int pos_x_hebra, pos_y_hebra;
	int pos_vol0, pos_vol1;
	// W0 y W1 tienen forma [h1, q1x, q1y, h2, q2x, q2y]
	TVec W0, W1;

	// Arista horizontal
	// Multiplicamos pos_y_hebra por 2 porque se procesan aristas alternas
	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x;
	pos_y_hebra = 2*(blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y);

	// Comprobamos si la hebra (arista) está dentro de los límites de la malla
	if ((pos_x_hebra < num_volx) && (pos_y_hebra <= num_voly)) {
		// Procesamos la arista horizontal
		if ((pos_y_hebra == 0) && (id_hebra == 0)) {
			// Frontera superior
			// normal_y = -longitud
			// El volumen 0 de la arista está situado debajo de la arista.
			// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
			// corresponde a volúmenes de comunicación de otro cluster
			datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
			v_set_val(&W0, 0, datos_vol0.x);
			v_set_val(&W0, 1, datos_vol0.y);
			v_set_val(&W0, 2, datos_vol0.z);
			datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
			v_set_val(&W0, 3, datos_vol0.x);
			v_set_val(&W0, 4, datos_vol0.y);
			v_set_val(&W0, 5, datos_vol0.z);
			// El volumen 1 es fantasma
			v_set_val(&W1, 0, v_get_val(&W0,0));
			v_set_val(&W1, 1, v_get_val(&W0,1));
			v_set_val(&W1, 2, v_get_val(&W0,2)*borde1);
			v_set_val(&W1, 3, v_get_val(&W0,3));
			v_set_val(&W1, 4, v_get_val(&W0,4));
			v_set_val(&W1, 5, v_get_val(&W0,5)*borde1);

			procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, -longitud, longitud, area, r,
				delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
				pos_x_hebra, -1, gravedad, epsilon_h, L, H, num_volumenes);
		}
		else {
			// normal_y = longitud
			if ((pos_y_hebra == num_voly) && ultima_hebra) {
				// Frontera inferior.
				// El volumen 0 de la arista está situado arriba de la arista.
				// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
				// de la textura corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 es fantasma
				v_set_val(&W1, 0, v_get_val(&W0,0));
				v_set_val(&W1, 1, v_get_val(&W0,1));

				v_set_val(&W1, 2, v_get_val(&W0,2)*borde2);
				v_set_val(&W1, 3, v_get_val(&W0,3));
				v_set_val(&W1, 4, v_get_val(&W0,4));
				v_set_val(&W1, 5, v_get_val(&W0,5)*borde2);

				pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
				procesarArista(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, longitud, longitud, area, r,
					delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
					pos_vol0, -1, gravedad, epsilon_h, L, H, num_volumenes);
			}
			else if ((pos_y_hebra > 0) && (pos_y_hebra < num_voly)) {
				// Arista interna que no es de comunicación.
				// El volumen 0 de la arista está situado arriba de la arista.
				// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
				// de la textura corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 de la arista está situado debajo de la arista.
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W1, 0, datos_vol1.x);
				v_set_val(&W1, 1, datos_vol1.y);
				v_set_val(&W1, 2, datos_vol1.z);
				datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W1, 3, datos_vol1.x);
				v_set_val(&W1, 4, datos_vol1.y);
				v_set_val(&W1, 5, datos_vol1.z);

				pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
				pos_vol1 = pos_vol0 + num_volx;
				procesarArista(&W0, &W1, datos_vol0.w, datos_vol1.w, 0.0, longitud, longitud, area, r,
					delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
					pos_vol0, pos_vol1, gravedad, epsilon_h, L, H, num_volumenes);
			}
		}
	}
}

// tipo debe ser 3 (primer grupo de aristas horizontales)
__global__ void procesarAristasComGPU(int num_volx, int num_voly, int num_volumenes, float borde1, float borde2,
				float longitud, float area, float r, float delta_T, float angulo1, float angulo2, float angulo3,
				float angulo4, float peso, float beta, float4 *d_acumulador_1, float4 *d_acumulador_2, float gravedad,
				float epsilon_h, float L, float H, int tipo, int id_hebra, int ultima_hebra)
{
	float4 datos_vol0, datos_vol1;
	// Posición x de la malla asociada a la hebra
	int pos_x_hebra;
	int pos_vol0, pos_vol1;
	// W0 y W1 tienen forma [h1, q1x, q1y, h2, q2x, q2y]
	TVec W0, W1;

	// Arista horizontal
	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_ARI_COM + threadIdx.x;
	// threadIdx.y es 0 o 1 (la altura del bloque de hebras es 2)

	// Comprobamos si la hebra (arista) está dentro de los límites de la malla
	if (pos_x_hebra < num_volx) {
		if ((id_hebra != 0) && (threadIdx.y == 0)) {
			// Procesamos la arista de comunicación superior.
			// El volumen 0 de la arista está situado arriba de la arista.
			// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
			// de la textura corresponde a volúmenes de comunicación de otro cluster
			datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, 0);
			v_set_val(&W0, 0, datos_vol0.x);
			v_set_val(&W0, 1, datos_vol0.y);
			v_set_val(&W0, 2, datos_vol0.z);
			datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, 0);
			v_set_val(&W0, 3, datos_vol0.x);
			v_set_val(&W0, 4, datos_vol0.y);
			v_set_val(&W0, 5, datos_vol0.z);
			// El volumen 1 de la arista está situado debajo de la arista.
			// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
			// corresponde a volúmenes de comunicación de otro cluster
			datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, 1);
			v_set_val(&W1, 0, datos_vol1.x);
			v_set_val(&W1, 1, datos_vol1.y);
			v_set_val(&W1, 2, datos_vol1.z);
			datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, 1);
			v_set_val(&W1, 3, datos_vol1.x);
			v_set_val(&W1, 4, datos_vol1.y);
			v_set_val(&W1, 5, datos_vol1.z);

			pos_vol0 = -num_volx + pos_x_hebra;
			pos_vol1 = pos_vol0 + num_volx;
			procesarArista(&W0, &W1, datos_vol0.w, datos_vol1.w, 0.0, longitud, longitud, area, r,
				delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
				pos_vol0, pos_vol1, gravedad, epsilon_h, L, H, num_volumenes);
		}
		else if ((! ultima_hebra) && (threadIdx.y == 1)) {
			// Procesamos la arista de comunicación inferior.
			// El volumen 0 de la arista está situado arriba de la arista.
			// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
			// de la textura corresponde a volúmenes de comunicación de otro cluster
			datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, num_voly);
			v_set_val(&W0, 0, datos_vol0.x);
			v_set_val(&W0, 1, datos_vol0.y);
			v_set_val(&W0, 2, datos_vol0.z);
			datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, num_voly);
			v_set_val(&W0, 3, datos_vol0.x);
			v_set_val(&W0, 4, datos_vol0.y);
			v_set_val(&W0, 5, datos_vol0.z);
			// El volumen 1 de la arista está situado debajo de la arista.
			// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
			// corresponde a volúmenes de comunicación de otro cluster
			datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, num_voly+1);
			v_set_val(&W1, 0, datos_vol1.x);
			v_set_val(&W1, 1, datos_vol1.y);
			v_set_val(&W1, 2, datos_vol1.z);
			datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, num_voly+1);
			v_set_val(&W1, 3, datos_vol1.x);
			v_set_val(&W1, 4, datos_vol1.y);
			v_set_val(&W1, 5, datos_vol1.z);

			pos_vol0 = (num_voly-1)*num_volx + pos_x_hebra;
			pos_vol1 = pos_vol0 + num_volx;
			procesarArista(&W0, &W1, datos_vol0.w, datos_vol1.w, 0.0, longitud, longitud, area, r,
				delta_T, angulo1, angulo2, angulo3, angulo4, peso, beta, d_acumulador_1, d_acumulador_2,
				pos_vol0, pos_vol1, gravedad, epsilon_h, L, H, num_volumenes);
		}
	}
}

/************************************************/
/* Funciones para el cálculo del deltaT inicial */
/************************************************/

// pos_vol0 y pos_vol1 son las posiciones de los acumuladores donde la arista escribirá sus contribuciones
__device__ void procesarAristaDeltaTInicial(TVec *W0, TVec *W1, float H0, float H1,
				float normal_x, float normal_y, float longitud, float r, float4 *d_acumulador_1,
				int pos_vol0, int pos_vol1, float gravedad, float epsilon_h, int num_volumenes)
{
	TVec4 DES;
	// Vector normal unitario a la arista
	float2 normal1;
	float h1ij, h2ij;
	float u1ij_n, u2ij_n;
	// Autovalores
	float b, max_autovalor;
	float u0n, u1n;
	float h0, h1, sqrt_h0, sqrt_h1;
	// Estados rotados de los volúmenes 0 y 1
	TVec4 W0_rot, W1_rot;
	float4 acum0_1, acum1_1;

	// Obtenemos el vector normal unitario a la arista
	normal1.x = normal_x/longitud;
	normal1.y = normal_y/longitud;

	// Leemos los valores de la capa 1 de los acumuladores
	if (pos_vol0 >= 0)
		acum0_1 = d_acumulador_1[pos_vol0];
	if ((pos_vol1 != -1) && (pos_vol1 < num_volumenes))
		acum1_1 = d_acumulador_1[pos_vol1];

	// Obtenemos los estados rotados W0_rot y W1_rot
	W0_rot.x = v_get_val(W0,0);
	W0_rot.y = v_get_val(W0,1)*normal1.x + v_get_val(W0,2)*normal1.y;
	W0_rot.z = v_get_val(W0,3);
	W0_rot.w = v_get_val(W0,4)*normal1.x + v_get_val(W0,5)*normal1.y;

	W1_rot.x = v_get_val(W1,0);
	W1_rot.y = v_get_val(W1,1)*normal1.x + v_get_val(W1,2)*normal1.y;
	W1_rot.z = v_get_val(W1,3);
	W1_rot.w = v_get_val(W1,4)*normal1.x + v_get_val(W1,5)*normal1.y;

	// Capa 1
	h0 = W0_rot.x;
	h1 = W1_rot.x;
	h1ij = 0.5*(h0 + h1);

	u0n = M_SQRT2*h0*W0_rot.y / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
	u1n = M_SQRT2*h1*W1_rot.y / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	sqrt_h0 = sqrtf(h0);
	sqrt_h1 = sqrtf(h1);
	u1ij_n = (sqrt_h0*u0n + sqrt_h1*u1n) / (sqrt_h0 + sqrt_h1 + EPSILON);

	// Capa 2
	h0 = W0_rot.z;
	h1 = W1_rot.z;
	h2ij = 0.5*(h0 + h1);

	u0n = M_SQRT2*h0*W0_rot.w / sqrtf(powf(h0,4.0) + powf(fmaxf(h0,epsilon_h),4.0));
	u1n = M_SQRT2*h1*W1_rot.w / sqrtf(powf(h1,4.0) + powf(fmaxf(h1,epsilon_h),4.0));
	sqrt_h0 = sqrtf(h0);
	sqrt_h1 = sqrtf(h1);
	u2ij_n = (sqrt_h0*u0n + sqrt_h1*u1n) / (sqrt_h0 + sqrt_h1 + EPSILON);

	// Obtenemos los autovalores de A
	DES.x = h1ij;
	DES.y = u1ij_n;
	DES.z = h2ij;
	DES.w = u2ij_n;
	DES = aproximarAutovalores1D(&DES, r, gravedad, epsilon_h, 0);

	max_autovalor = fmaxf(DES.x, DES.w);
	b = fmaxf(fabsf(u1ij_n), fabsf(u2ij_n));
	if (b > max_autovalor)
		max_autovalor = b;

	if (max_autovalor < epsilon_h)
		max_autovalor += epsilon_h;

	b = longitud*max_autovalor;
	if (pos_vol0 >= 0) {
		// Actualizamos el valor del acumulador del volumen 0 para la capa 1
		acum0_1.w += b;
		d_acumulador_1[pos_vol0] = acum0_1;
	}

	if ((pos_vol1 != -1) && (pos_vol1 < num_volumenes)) {
		// Actualizamos el valor del acumulador del volumen 1 para la capa 1
		acum1_1.w += b;
		d_acumulador_1[pos_vol1] = acum1_1;
	}
}

__global__ void procesarAristasDeltaTInicialGPU(int num_volx, int num_voly, int num_volumenes, float borde1, float borde2,
				float longitud, float r, float4 *d_acumulador_1, float gravedad, float epsilon_h, int tipo,
				int id_hebra, int ultima_hebra)
{
	float4 datos_vol0, datos_vol1;
	// Posición (x,y) de la malla asociada a la hebra
	int pos_x_hebra, pos_y_hebra;
	int pos_vol0, pos_vol1;
	// W0 y W1 tienen forma [h1, q1x, q1y, h2, q2x, q2y]
	TVec W0, W1;

	if (tipo < 3) {
		// Arista vertical
		// Multiplicamos pos_x_hebra por 2 porque se procesan aristas alternas
		pos_x_hebra = 2*(blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x);
		pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y;
		if (tipo == 2) pos_x_hebra++;

		// Comprobamos si la hebra (arista) está dentro de los límites de la malla
		if ((pos_x_hebra <= num_volx) && (pos_y_hebra < num_voly)) {
			// Procesamos la arista vertical.
			// Obtenemos los datos de los volúmenes 0 y 1
			if (pos_x_hebra == 0) {
				// Frontera izquierda
				// normal_x = -longitud
				// El volumen 0 de la arista está situado a la derecha de la arista.
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 es fantasma
				v_set_val(&W1, 0, v_get_val(&W0,0));
				v_set_val(&W1, 1, v_get_val(&W0,1)*borde1);
				v_set_val(&W1, 2, v_get_val(&W0,2));
				v_set_val(&W1, 3, v_get_val(&W0,3));
				v_set_val(&W1, 4, v_get_val(&W0,4)*borde1);
				v_set_val(&W1, 5, v_get_val(&W0,5));

				pos_vol0 = pos_y_hebra*num_volx;
				procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol0.w, -longitud, 0.0,
					longitud, r, d_acumulador_1, pos_vol0, -1, gravedad, epsilon_h, num_volumenes);
			}
			else {
				// normal_x = longitud
				// El volumen 0 de la arista está situado a la izquierda de la arista.
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra-1, pos_y_hebra+1);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra-1, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				if (pos_x_hebra == num_volx) {
					// Frontera derecha. El volumen 1 es fantasma
					v_set_val(&W1, 0, v_get_val(&W0,0));
					v_set_val(&W1, 1, v_get_val(&W0,1)*borde2);
					v_set_val(&W1, 2, v_get_val(&W0,2));
					v_set_val(&W1, 3, v_get_val(&W0,3));
					v_set_val(&W1, 4, v_get_val(&W0,4)*borde2);
					v_set_val(&W1, 5, v_get_val(&W0,5));

					pos_vol0 = (pos_y_hebra+1)*num_volx - 1;
					procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol0.w, longitud, 0.0,
						longitud, r, d_acumulador_1, pos_vol0, -1, gravedad, epsilon_h, num_volumenes);
				}
				else {
					// Arista interna
					// El volumen 1 de la arista está situado a la derecha de la arista.
					// Sumamos 1 a la coordenada y porque la primera fila de la textura corresponde
					// a volúmenes de comunicación de otro cluster
					datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 0, datos_vol1.x);
					v_set_val(&W1, 1, datos_vol1.y);
					v_set_val(&W1, 2, datos_vol1.z);
					datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 3, datos_vol1.x);
					v_set_val(&W1, 4, datos_vol1.y);
					v_set_val(&W1, 5, datos_vol1.z);

					pos_vol0 = pos_y_hebra*num_volx + pos_x_hebra-1;
					procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol1.w, longitud, 0.0,
						longitud, r, d_acumulador_1, pos_vol0, pos_vol0+1, gravedad, epsilon_h, num_volumenes);
				}
			}
		}
	}
	else {
		// Arista horizontal
		// Multiplicamos pos_y_hebra por 2 porque se procesan aristas alternas
		pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x;
		pos_y_hebra = 2*(blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y);
		if (tipo == 4) pos_y_hebra++;

		// Comprobamos si la hebra (arista) está dentro de los límites de la malla
		if ((pos_x_hebra < num_volx) && (pos_y_hebra <= num_voly)) {
			// Procesamos la arista horizontal.
			// Obtenemos los datos de los volúmenes 0 y 1
			if ((pos_y_hebra == 0) && (id_hebra == 0)) {
				// Frontera superior
				// normal_y = -longitud
				// El volumen 0 de la arista está situado debajo de la arista
				// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
				// corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				// El volumen 1 es fantasma
				v_set_val(&W1, 0, v_get_val(&W0,0));
				v_set_val(&W1, 1, v_get_val(&W0,1));
				v_set_val(&W1, 2, v_get_val(&W0,2)*borde1);
				v_set_val(&W1, 3, v_get_val(&W0,3));
				v_set_val(&W1, 4, v_get_val(&W0,4));
				v_set_val(&W1, 5, v_get_val(&W0,5)*borde1);

				procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, -longitud,
					longitud, r, d_acumulador_1, pos_x_hebra, -1, gravedad, epsilon_h, num_volumenes);
			}
			else {
				// normal_y = longitud
				// El volumen 0 de la arista está situado arriba de la arista
				// Dejamos igual la coordenada y de la textura (en vez de restar 1) porque la primera fila
				// de la textura corresponde a volúmenes de comunicación de otro cluster
				datos_vol0 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 0, datos_vol0.x);
				v_set_val(&W0, 1, datos_vol0.y);
				v_set_val(&W0, 2, datos_vol0.z);
				datos_vol0 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra);
				v_set_val(&W0, 3, datos_vol0.x);
				v_set_val(&W0, 4, datos_vol0.y);
				v_set_val(&W0, 5, datos_vol0.z);
				if ((pos_y_hebra == num_voly) && ultima_hebra) {
					// Frontera inferior. El volumen 1 es fantasma
					v_set_val(&W1, 0, v_get_val(&W0,0));
					v_set_val(&W1, 1, v_get_val(&W0,1));
					v_set_val(&W1, 2, v_get_val(&W0,2)*borde2);
					v_set_val(&W1, 3, v_get_val(&W0,3));
					v_set_val(&W1, 4, v_get_val(&W0,4));
					v_set_val(&W1, 5, v_get_val(&W0,5)*borde2);

					pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
					procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol0.w, 0.0, longitud,
						longitud, r, d_acumulador_1, pos_vol0, -1, gravedad, epsilon_h, num_volumenes);
				}
				else {
					// Arista interna
					// El volumen 1 de la arista está situado debajo de la arista
					// Sumamos 1 a la coordenada y de la textura porque la primera fila de la textura
					// corresponde a volúmenes de comunicación de otro cluster
					datos_vol1 = tex2D(texDatosVolumenes_1, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 0, datos_vol1.x);
					v_set_val(&W1, 1, datos_vol1.y);
					v_set_val(&W1, 2, datos_vol1.z);
					datos_vol1 = tex2D(texDatosVolumenes_2, pos_x_hebra, pos_y_hebra+1);
					v_set_val(&W1, 3, datos_vol1.x);
					v_set_val(&W1, 4, datos_vol1.y);
					v_set_val(&W1, 5, datos_vol1.z);

					pos_vol0 = (pos_y_hebra-1)*num_volx + pos_x_hebra;
					pos_vol1 = pos_vol0 + num_volx;
					procesarAristaDeltaTInicial(&W0, &W1, datos_vol0.w, datos_vol1.w, 0.0, longitud,
						longitud, r, d_acumulador_1, pos_vol0, pos_vol1, gravedad, epsilon_h, num_volumenes);
				}
			}
		}
	}
}

#endif
