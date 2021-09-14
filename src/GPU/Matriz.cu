#ifndef _MATRIZ_H_
#define _MATRIZ_H_

#define CONSTANTES_GPU
#include "Constantes.hxx"
#undef  CONSTANTES_GPU
#include <float.h>

/***********************/
/* Definición de tipos */
/***********************/

// Tipo matriz
typedef struct {
	float mat[NUM_VARIABLES][NUM_VARIABLES];
} TMat;

// Tipo vector
typedef struct {
	float vec[NUM_VARIABLES];
} TVec;

// Tipo matriz 4x4
typedef struct {
	float mat[4][4];
} TMat4;

// Tipo vector 4x1
typedef float4 TVec4;

/********************/
/* Macros de acceso */
/********************/

#define	m_set_val(A,i,j,val)	((A)->mat[(i)][(j)] = (val))
#define	m_add_val(A,i,j,val)	((A)->mat[(i)][(j)] += (val))
#define	m_sub_val(A,i,j,val)	((A)->mat[(i)][(j)] -= (val))
#define	m_get_val(A,i,j)		((A)->mat[(i)][(j)])
#define	v_set_val(v,i,val)		((v)->vec[(i)] = (val))
#define	v_add_val(v,i,val)		((v)->vec[(i)] += (val))
#define	v_sub_val(v,i,val)		((v)->vec[(i)] -= (val))
#define	v_get_val(v,i)			((v)->vec[(i)])

/******************************/
/* Inicialización de vectores */
/******************************/

// Copia el vector in en out
__device__ void v_copy4(TVec4 *in, TVec4 *out) {
	out->x = in->x;
	out->y = in->y;
	out->z = in->z;
	out->w = in->w;
}

// Pone todos los elementos del vector v a cero
__device__ void v_zero4(TVec4 *v) {
	v->x = 0.0;
	v->y = 0.0;
	v->z = 0.0;
	v->w = 0.0;
}

// Copia el vector in en out
__device__ void v_copy6(TVec *in, TVec *out) {
	v_set_val(out, 0, v_get_val(in,0));
	v_set_val(out, 1, v_get_val(in,1));
	v_set_val(out, 2, v_get_val(in,2));
	v_set_val(out, 3, v_get_val(in,3));
	v_set_val(out, 4, v_get_val(in,4));
	v_set_val(out, 5, v_get_val(in,5));
}

// Pone todos los elementos del vector v a cero
__device__ void v_zero6(TVec *v) {
	v_set_val(v, 0, 0.0);
	v_set_val(v, 1, 0.0);
	v_set_val(v, 2, 0.0);
	v_set_val(v, 3, 0.0);
	v_set_val(v, 4, 0.0);
	v_set_val(v, 5, 0.0);
}

// Pone todos los elementos del vector v a uno
__device__ void v_ones6(TVec *v) {
	v_set_val(v, 0, 1.0);
	v_set_val(v, 1, 1.0);
	v_set_val(v, 2, 1.0);
	v_set_val(v, 3, 1.0);
	v_set_val(v, 4, 1.0);
	v_set_val(v, 5, 1.0);
}

// Pone todos los elementos del vector v a uno
__device__ void v_ones4(TVec4 *v) {
	v->x = 1.0;
	v->y = 1.0;
	v->z = 1.0;
	v->w = 1.0;
}

/************************************/
/* Operaciones básicas con vectores */
/************************************/

// out <- s*v
__device__ void sv_mlt4(float s, TVec4 *v, TVec4 *out) {
	out->x = s*v->x;
	out->y = s*v->y;
	out->z = s*v->z;
	out->w = s*v->w;
}

// out <- s*v
__device__ void sv_mlt6(float s, TVec *v, TVec *out) {
	v_set_val(out, 0, s*v_get_val(v,0));
	v_set_val(out, 1, s*v_get_val(v,1));
	v_set_val(out, 2, s*v_get_val(v,2));
	v_set_val(out, 3, s*v_get_val(v,3));
	v_set_val(out, 4, s*v_get_val(v,4));
	v_set_val(out, 5, s*v_get_val(v,5));
}

// out <- A*v (v es un vector columna)
__device__ void mv_mlt4(TMat4 *A, TVec4 *v, TVec4 *out)
{
	out->x = m_get_val(A,0,0)*v->x + m_get_val(A,0,1)*v->y + m_get_val(A,0,2)*v->z + m_get_val(A,0,3)*v->w;
	out->y = m_get_val(A,1,0)*v->x + m_get_val(A,1,1)*v->y + m_get_val(A,1,2)*v->z + m_get_val(A,1,3)*v->w;
	out->z = m_get_val(A,2,0)*v->x + m_get_val(A,2,1)*v->y + m_get_val(A,2,2)*v->z + m_get_val(A,2,3)*v->w;
	out->w = m_get_val(A,3,0)*v->x + m_get_val(A,3,1)*v->y + m_get_val(A,3,2)*v->z + m_get_val(A,3,3)*v->w;
}

// out <- v1+v2
__device__ void v_add6(TVec *v1, TVec *v2, TVec *out) {
	float val;

	val = v_get_val(v1,0) + v_get_val(v2,0);	v_set_val(out, 0, val);
	val = v_get_val(v1,1) + v_get_val(v2,1);	v_set_val(out, 1, val);
	val = v_get_val(v1,2) + v_get_val(v2,2);	v_set_val(out, 2, val);
	val = v_get_val(v1,3) + v_get_val(v2,3);	v_set_val(out, 3, val);
	val = v_get_val(v1,4) + v_get_val(v2,4);	v_set_val(out, 4, val);
	val = v_get_val(v1,5) + v_get_val(v2,5);	v_set_val(out, 5, val);
}

// out <- v1+v2
__device__ void v_add4(TVec4 *v1, TVec4 *v2, TVec4 *out) {
	out->x = v1->x + v2->x;
	out->y = v1->y + v2->y;
	out->z = v1->z + v2->z;
	out->w = v1->w + v2->w;
}

// out <- v1-v2
__device__ void v_sub6(TVec *v1, TVec *v2, TVec *out) {
	float val;

	val = v_get_val(v1,0) - v_get_val(v2,0);	v_set_val(out, 0, val);
	val = v_get_val(v1,1) - v_get_val(v2,1);	v_set_val(out, 1, val);
	val = v_get_val(v1,2) - v_get_val(v2,2);	v_set_val(out, 2, val);
	val = v_get_val(v1,3) - v_get_val(v2,3);	v_set_val(out, 3, val);
	val = v_get_val(v1,4) - v_get_val(v2,4);	v_set_val(out, 4, val);
	val = v_get_val(v1,5) - v_get_val(v2,5);	v_set_val(out, 5, val);
}

// out <- v1-v2
__device__ void v_sub4(TVec4 *v1, TVec4 *v2, TVec4 *out) {
	out->x = v1->x - v2->x;
	out->y = v1->y - v2->y;
	out->z = v1->z - v2->z;
	out->w = v1->w - v2->w;
}

// out <- sgn(v)
__device__ void v_sgn4(TVec4 *v, TVec4 *out) {
	float val;

	val = v->x;
	if (fabsf(val) < EPSILON)	out->x = 0.0;
	else if (val > 0)			out->x = 1.0;
	else						out->x = -1.0;
	val = v->y;
	if (fabsf(val) < EPSILON)	out->y = 0.0;
	else if (val > 0)			out->y = 1.0;
	else						out->y = -1.0;
	val = v->z;
	if (fabsf(val) < EPSILON)	out->z = 0.0;
	else if (val > 0)			out->z = 1.0;
	else						out->z = -1.0;
	val = v->w;
	if (fabsf(val) < EPSILON)	out->w = 0.0;
	else if (val > 0)			out->w = 1.0;
	else						out->w = -1.0;
}

/************************************/
/* Operaciones básicas con matrices */
/************************************/

// out <- s*A
__device__ void sm_mlt4(float s, TMat4 *A, TMat4 *out) {
	m_set_val(out, 0, 0, s*m_get_val(A,0,0));
	m_set_val(out, 0, 1, s*m_get_val(A,0,1));
	m_set_val(out, 0, 2, s*m_get_val(A,0,2));
	m_set_val(out, 0, 3, s*m_get_val(A,0,3));
	m_set_val(out, 1, 0, s*m_get_val(A,1,0));
	m_set_val(out, 1, 1, s*m_get_val(A,1,1));
	m_set_val(out, 1, 2, s*m_get_val(A,1,2));
	m_set_val(out, 1, 3, s*m_get_val(A,1,3));
	m_set_val(out, 2, 0, s*m_get_val(A,2,0));
	m_set_val(out, 2, 1, s*m_get_val(A,2,1));
	m_set_val(out, 2, 2, s*m_get_val(A,2,2));
	m_set_val(out, 2, 3, s*m_get_val(A,2,3));
	m_set_val(out, 3, 0, s*m_get_val(A,3,0));
	m_set_val(out, 3, 1, s*m_get_val(A,3,1));
	m_set_val(out, 3, 2, s*m_get_val(A,3,2));
	m_set_val(out, 3, 3, s*m_get_val(A,3,3));
}

// out <- A*B
__device__ void mm_mlt4(TMat4 *A, TMat4 *B, TMat4 *out) {
	float val;

	// Fila 1
	val = m_get_val(A,0,0)*m_get_val(B,0,0) + m_get_val(A,0,1)*m_get_val(B,1,0) + m_get_val(A,0,2)*m_get_val(B,2,0) + m_get_val(A,0,3)*m_get_val(B,3,0);
	m_set_val(out, 0, 0, val);
	val = m_get_val(A,0,0)*m_get_val(B,0,1) + m_get_val(A,0,1)*m_get_val(B,1,1) + m_get_val(A,0,2)*m_get_val(B,2,1) + m_get_val(A,0,3)*m_get_val(B,3,1);
	m_set_val(out, 0, 1, val);
	val = m_get_val(A,0,0)*m_get_val(B,0,2) + m_get_val(A,0,1)*m_get_val(B,1,2) + m_get_val(A,0,2)*m_get_val(B,2,2) + m_get_val(A,0,3)*m_get_val(B,3,2);
	m_set_val(out, 0, 2, val);
	val = m_get_val(A,0,0)*m_get_val(B,0,3) + m_get_val(A,0,1)*m_get_val(B,1,3) + m_get_val(A,0,2)*m_get_val(B,2,3) + m_get_val(A,0,3)*m_get_val(B,3,3);
	m_set_val(out, 0, 3, val);
	// Fila 2
	val = m_get_val(A,1,0)*m_get_val(B,0,0) + m_get_val(A,1,1)*m_get_val(B,1,0) + m_get_val(A,1,2)*m_get_val(B,2,0) + m_get_val(A,1,3)*m_get_val(B,3,0);
	m_set_val(out, 1, 0, val);
	val = m_get_val(A,1,0)*m_get_val(B,0,1) + m_get_val(A,1,1)*m_get_val(B,1,1) + m_get_val(A,1,2)*m_get_val(B,2,1) + m_get_val(A,1,3)*m_get_val(B,3,1);
	m_set_val(out, 1, 1, val);
	val = m_get_val(A,1,0)*m_get_val(B,0,2) + m_get_val(A,1,1)*m_get_val(B,1,2) + m_get_val(A,1,2)*m_get_val(B,2,2) + m_get_val(A,1,3)*m_get_val(B,3,2);
	m_set_val(out, 1, 2, val);
	val = m_get_val(A,1,0)*m_get_val(B,0,3) + m_get_val(A,1,1)*m_get_val(B,1,3) + m_get_val(A,1,2)*m_get_val(B,2,3) + m_get_val(A,1,3)*m_get_val(B,3,3);
	m_set_val(out, 1, 3, val);
	// Fila 3
	val = m_get_val(A,2,0)*m_get_val(B,0,0) + m_get_val(A,2,1)*m_get_val(B,1,0) + m_get_val(A,2,2)*m_get_val(B,2,0) + m_get_val(A,2,3)*m_get_val(B,3,0);
	m_set_val(out, 2, 0, val);
	val = m_get_val(A,2,0)*m_get_val(B,0,1) + m_get_val(A,2,1)*m_get_val(B,1,1) + m_get_val(A,2,2)*m_get_val(B,2,1) + m_get_val(A,2,3)*m_get_val(B,3,1);
	m_set_val(out, 2, 1, val);
	val = m_get_val(A,2,0)*m_get_val(B,0,2) + m_get_val(A,2,1)*m_get_val(B,1,2) + m_get_val(A,2,2)*m_get_val(B,2,2) + m_get_val(A,2,3)*m_get_val(B,3,2);
	m_set_val(out, 2, 2, val);
	val = m_get_val(A,2,0)*m_get_val(B,0,3) + m_get_val(A,2,1)*m_get_val(B,1,3) + m_get_val(A,2,2)*m_get_val(B,2,3) + m_get_val(A,2,3)*m_get_val(B,3,3);
	m_set_val(out, 2, 3, val);
	// Fila 4
	val = m_get_val(A,3,0)*m_get_val(B,0,0) + m_get_val(A,3,1)*m_get_val(B,1,0) + m_get_val(A,3,2)*m_get_val(B,2,0) + m_get_val(A,3,3)*m_get_val(B,3,0);
	m_set_val(out, 3, 0, val);
	val = m_get_val(A,3,0)*m_get_val(B,0,1) + m_get_val(A,3,1)*m_get_val(B,1,1) + m_get_val(A,3,2)*m_get_val(B,2,1) + m_get_val(A,3,3)*m_get_val(B,3,1);
	m_set_val(out, 3, 1, val);
	val = m_get_val(A,3,0)*m_get_val(B,0,2) + m_get_val(A,3,1)*m_get_val(B,1,2) + m_get_val(A,3,2)*m_get_val(B,2,2) + m_get_val(A,3,3)*m_get_val(B,3,2);
	m_set_val(out, 3, 2, val);
	val = m_get_val(A,3,0)*m_get_val(B,0,3) + m_get_val(A,3,1)*m_get_val(B,1,3) + m_get_val(A,3,2)*m_get_val(B,2,3) + m_get_val(A,3,3)*m_get_val(B,3,3);
	m_set_val(out, 3, 3, val);
}

// out <- A*D
// D es una matriz diagonal representada por su diagonal principal
__device__ void md_mlt4(TMat4 *A, TVec4 *D, TMat4 *out) {
	float val;

	val = D->x;
	m_set_val(out, 0, 0, m_get_val(A,0,0)*val);
	m_set_val(out, 1, 0, m_get_val(A,1,0)*val);
	m_set_val(out, 2, 0, m_get_val(A,2,0)*val);
	m_set_val(out, 3, 0, m_get_val(A,3,0)*val);
	val = D->y;
	m_set_val(out, 0, 1, m_get_val(A,0,1)*val);
	m_set_val(out, 1, 1, m_get_val(A,1,1)*val);
	m_set_val(out, 2, 1, m_get_val(A,2,1)*val);
	m_set_val(out, 3, 1, m_get_val(A,3,1)*val);
	val = D->z;
	m_set_val(out, 0, 2, m_get_val(A,0,2)*val);
	m_set_val(out, 1, 2, m_get_val(A,1,2)*val);
	m_set_val(out, 2, 2, m_get_val(A,2,2)*val);
	m_set_val(out, 3, 2, m_get_val(A,3,2)*val);
	val = D->w;
	m_set_val(out, 0, 3, m_get_val(A,0,3)*val);
	m_set_val(out, 1, 3, m_get_val(A,1,3)*val);
	m_set_val(out, 2, 3, m_get_val(A,2,3)*val);
	m_set_val(out, 3, 3, m_get_val(A,3,3)*val);
}

#endif
