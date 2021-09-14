#include "Constantes.hxx"

// TEST 1: capa 1 en 100 % y capa 2 en 15 % del dominio
// (la capa 2 al final ocupa el 80 % del dominio)
/*
Scalar topografia(Scalar x, Scalar y, Scalar L, Scalar H) {
	Scalar prof;
	x*=L;
	y*=L;

	prof = 1.0*(x<-3.0) + (5.0 - exp(-x*x-y*y))*(x>=-3.0);
	prof /= H;
	return prof;
}

Scalar cini_h1(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h;
	Scalar h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-x*x-y*y))*((x>=-3.0) && (x<=-1.5));
	h = 1.0*(x<-3.0) + (5.0 - exp(-x*x-y*y))*(x>=-3.0) - h2;
	h /= H;

	return h;
}

Scalar cini_q1x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q1y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}

Scalar cini_h2(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h1, h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-x*x-y*y))*((x>=-3.0) && (x<=-1.5));
	h2 /= H;

	return h2;
}

Scalar cini_q2x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q2y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}
*/

// TEST 2: capa 1 en 60 % y capa 2 en 15 % del dominio
// (la capa 2 al final ocupa el 60 % del dominio)

Scalar topografia(Scalar x, Scalar y, Scalar L, Scalar H) {
	Scalar prof;
	x*=L;
	y*=L;

	prof = 1.0*(x<-1.0) + (6.0 - exp(-(x-2)*(x-2)-y*y))*(x>=-1.0);
	prof /= H;
	return prof;
}

Scalar cini_h1(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h;
	Scalar h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-(x-2)*(x-2)-y*y))*((x>=-1.0) && (x<=0.5));
	h = 0.0*(x<-1.0) + (4.0 - exp(-(x-2)*(x-2)-y*y))*(x>=-1.0) - h2;
	h /= H;

	return h;
}

Scalar cini_q1x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q1y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}

Scalar cini_h2(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h1, h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-(x-2)*(x-2)-y*y))*((x>=-1.0) && (x<=0.5));
	h2 /= H;

	return h2;
}

Scalar cini_q2x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q2y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}


// TEST 3: capa 1 en 20 % y capa 2 en 15 % del dominio
// (la capa 2 al final ocupa el 20 % del dominio)

/*Scalar topografia(Scalar x, Scalar y, Scalar L, Scalar H) {
	Scalar prof;
	x*=L;
	y*=L;

	prof = 1.0*(x<3.0) + (6.0 - exp(-(x-3)*(x-3)-y*y))*(x>=3.0);
	prof /= H;
	return prof;
}

Scalar cini_h1(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h;
	Scalar h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-(x-3)*(x-3)-y*y))*((x>=3.0) && (x<=4.5));
	h = 0.0*(x<3.0) + (4.0 - exp(-(x-3)*(x-3)-y*y))*(x>=3.0) - h2;
	h /= H;

	return h;
}

Scalar cini_q1x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q1y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}

Scalar cini_h2(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h1, h2;
	x *= L;
	y *= L;

	h2 = (2.0 - exp(-(x-3)*(x-3)-y*y))*((x>=3.0) && (x<=4.5));
	h2 /= H;

	return h2;
}

Scalar cini_q2x(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qx;
	qx = 0.0;
	return qx;
}

Scalar cini_q2y(Scalar x, Scalar y, Scalar L, Scalar H, Scalar Q, Scalar prof) {
	Scalar qy;
	qy = 0.0;
	return qy;
}*/
