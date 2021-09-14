#include "Constantes.hxx"

// Presa circular interna

Scalar topografia(Scalar x, Scalar y, Scalar L, Scalar H) {
	Scalar prof;
	prof = 5.0;
	prof /= H;
	return prof;
}

Scalar cini_h1(Scalar x, Scalar y, Scalar L, Scalar H, Scalar prof) {
	Scalar h;
	x *= L;
	y *= L;

	h = (sqrt(x*x + y*y) > 1.5) ? 4.0 : 0.5;
h=0.0;
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
	prof *= H;

	h1 = (sqrt(x*x + y*y) > 1.5) ? 4.0 : 0.5;
	h2 = prof - h1;
h2=0.0;
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
