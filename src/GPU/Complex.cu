#ifndef _COMPLEX_H_
#define _COMPLEX_H_

#define CONSTANTES_GPU
#include "Constantes.hxx"
#undef  CONSTANTES_GPU

// Almacenamos un número complejo en un float2.
// x es la parte real e y es la parte imaginaria.
typedef float2 fcomplex;

// Suma un complejo y un real
__device__ fcomplex cradd(fcomplex a, float b) {
	fcomplex res;
	res.x = a.x + b;
	res.y = a.y;
	return res;
}

// Suma dos complejos
__device__ fcomplex ccadd(fcomplex a, fcomplex b) {
	fcomplex res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	return res;
}

// Resta un complejo y un real
__device__ fcomplex crsub(fcomplex a, float b) {
	fcomplex res;
	res.x = a.x - b;
	res.y = a.y;
	return res;
}

// Resta un real y un complejo
__device__ fcomplex rcsub(float a, fcomplex b) {
	fcomplex res;
	res.x = a - b.x;
	res.y = -b.y;
	return res;
}

// Resta dos complejos
__device__ fcomplex ccsub(fcomplex a, fcomplex b) {
	fcomplex res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	return res;
}

// Multiplica un complejo y un real
__device__ fcomplex crmul(fcomplex a, float b) {
	fcomplex res;
	res.x = a.x*b;
	res.y = a.y*b;
	return res;
}

// Multiplica dos complejos
__device__ fcomplex ccmul(fcomplex a, fcomplex b) {
	fcomplex res;
	res.x = a.x*b.x - a.y*b.y;
	res.y = a.x*b.y + a.y*b.x;
	return res;
}

// Divide un complejo por un real
__device__ fcomplex crdiv(fcomplex a, float b) {
	fcomplex res;
	res.x = a.x/b;
	res.y = a.y/b;
	return res;
}

// Divide un real por un complejo
__device__ fcomplex rcdiv(float a, fcomplex b) {
	fcomplex res;
	float denominador = b.x*b.x + b.y*b.y;
	res.x = a*b.x/denominador;
	res.y = -a*b.y/denominador;
	return res;
}

// Divide dos complejos
__device__ fcomplex ccdiv(fcomplex a, fcomplex b) {
	fcomplex res;
	float denominador = b.x*b.x + b.y*b.y;
	res.x = (a.x*b.x + a.y*b.y)/denominador;
	res.y = (a.y*b.x - a.x*b.y)/denominador;
	return res;
}

// Raíz cuadrada de un complejo
__device__ fcomplex sqrtc(fcomplex a) {
	fcomplex res;
	float p1, p2;

	if ((a.x < 0.0) && (fabsf(a.y) < EPSILON)) {
		// Raíz cuadrada de un real negativo (sqrt(-x) = i*sqrt(x))
		res.x = 0.0;
		res.y = sqrtf(-a.x);
	}
	else {
		p1 = sqrtf(sqrtf(a.x*a.x + a.y*a.y));
		p2 = 0.5*atan2f(a.y, a.x);
		res.x = p1*cosf(p2);
		res.y = p1*sinf(p2);
	}
	return res;
}

// Raíz cuadrada de un real
__device__ fcomplex sqrtr(float a) {
	fcomplex res;

	if (a < 0.0) {
		// Raíz cuadrada de un real negativo (sqrt(-x) = i*sqrt(x))
		res.x = 0.0;
		res.y = sqrtf(-a);
	}
	else {
		res.x = sqrtf(a);
		res.y = 0.0;
	}
	return res;
}

// Eleva un complejo a un real (devuelve a^b)
__device__ fcomplex crpow(fcomplex a, float b) {
	fcomplex res;
	float p1 = powf(sqrtf(a.x*a.x + a.y*a.y), b);
	float p2 = b*atan2(a.y, a.x);

	res.x = p1*cosf(p2);
	res.y = p1*sinf(p2);
	return res;
}

#endif
