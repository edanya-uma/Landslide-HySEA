#include "Constantes.hxx"
#include <sys/stat.h> 
#include <fstream>
#include <cmath>
#include "cond_ini.cxx"
#include "mpi.h"

int obtenerIndicePunto(float *longitud, float *latitud, float lon, float lat, int num_volx, int num_voly)
{
        int i, j;
        int pos = -1;
        bool encontrado;

        // Buscamos el ?ndice en longitud
        i = 0;
        encontrado = false;

        while ((i < num_volx) && (! encontrado))  {
                if (longitud[i] >= lon) {
                        encontrado = true;
                        if (fabs(lon - longitud[max(i-1,0)]) < fabs(lon - longitud[i]))
                                i = max(i-1,0);
                }
                else i++;
        }
        if (encontrado) {
                // Si la longitud est? dentro del dominio, buscamos el ?ndice en latitud
                j = 0;
                encontrado = false;
                while ((j < num_voly) && (! encontrado))  {
                        if (latitud[j] >= lat) {
                                encontrado = true;
                                if (fabs(lat - latitud[max(j-1,0)]) < fabs(lat - latitud[j]))
                                        j = max(j-1,0);
                                pos = j*num_volx + i;
                        }
                        else j++;
                }
        }

        return pos;
}

// Devuelve true si existe el fichero, false en otro caso
bool existeFichero(char *fichero)
{
	struct stat stFichInfo;
	bool existe;
	int intStat;

	// Obtenemos los atributos del fichero
	intStat = stat(fichero, &stFichInfo);
	if (intStat == 0) {
		// Hemos obtenido los atributos del fichero. Por tanto, el fichero existe.
		existe = true;
	}
	else {
		// No hemos obtenido los atributos del fichero. Notar que esto puede
		// significar que no tenemos permiso para acceder al fichero. Para
		// hacer esta comprobación, comprobar los valores de intStat.
		existe = false;
	}

	return existe;
}

void asignarVariables(Scalar x, Scalar y, Scalar *prof, Scalar *h1, Scalar *q1x, Scalar *q1y, Scalar *h2,
			Scalar *q2x, Scalar *q2y, Scalar L, Scalar H, Scalar Q)
{
	*prof = topografia(x, y, L, H);
	*h1 = cini_h1(x, y, L, H, *prof);
	*q1x = cini_q1x(x, y, L, H, Q, *prof);
	*q1y = cini_q1y(x, y, L, H, Q, *prof);
	*h2 = cini_h2(x, y, L, H, *prof);
	*q2x = cini_q2x(x, y, L, H, Q, *prof);
	*q2y = cini_q2y(x, y, L, H, Q, *prof);
}

// num_volx y num_voly se refieren al cluster, no a toda la malla
void setCondicionesIniciales(TDatoCluster *datos_cluster, Scalar xmin, Scalar xmax, Scalar ymin, Scalar ymax,
				Scalar ancho_vol, Scalar alto_vol, int num_volx, int num_voly, int num_vols_saltar,
				int num_vols_leer, Scalar L, Scalar H, Scalar Q, int id_hebra)
{
	int fila_inicial = num_vols_saltar/num_volx;
	int num_filas_leer = num_vols_leer/num_volx;
	int i, j, pos;
	Scalar prof, h1, q1x, q1y, h2, q2x, q2y;
	Scalar x, y;

	if (id_hebra == 0) {
		// La primera hebra no tiene cluster adyacente superior.
		// No escribimos nada en la primera fila de datosVolumenes
		pos = num_volx;
		for (j=fila_inicial; j<fila_inicial+num_filas_leer; j++) {
			for (i=0; i<num_volx; i++) {
				x = xmin + (i+0.5)*ancho_vol;
				y = ymin + (j+0.5)*alto_vol;
				asignarVariables(x, y, &prof, &h1, &q1x, &q1y, &h2, &q2x, &q2y, L, H, Q);

				datos_cluster->datosVolumenes_1[pos].x = h1;
				datos_cluster->datosVolumenes_1[pos].y = q1x;
				datos_cluster->datosVolumenes_1[pos].z = q1y;
				datos_cluster->datosVolumenes_1[pos].w = prof;
				datos_cluster->datosVolumenes_2[pos].x = h2;
				datos_cluster->datosVolumenes_2[pos].y = q2x;
				datos_cluster->datosVolumenes_2[pos].z = q2y;
				datos_cluster->datosVolumenes_2[pos].w = prof;
				pos++;
			}
		}

	}
	else {
		// Hay cluster adyacente superior.
		// Los datos incluyen los volúmenes de comunicación inferiores
		// del cluster adyacente superior (la primera fila de datosVolumenes)
		pos = 0;
		for (j=fila_inicial; j<fila_inicial+num_filas_leer; j++) {
			for (i=0; i<num_volx; i++) {
				x = xmin + (i+0.5)*ancho_vol;
				y = ymin + (j+0.5)*alto_vol;
				asignarVariables(x, y, &prof, &h1, &q1x, &q1y, &h2, &q2x, &q2y, L, H, Q);

				datos_cluster->datosVolumenes_1[pos].x = h1;
				datos_cluster->datosVolumenes_1[pos].y = q1x;
				datos_cluster->datosVolumenes_1[pos].z = q1y;
				datos_cluster->datosVolumenes_1[pos].w = prof;
				datos_cluster->datosVolumenes_2[pos].x = h2;
				datos_cluster->datosVolumenes_2[pos].y = q2x;
				datos_cluster->datosVolumenes_2[pos].z = q2y;
				datos_cluster->datosVolumenes_2[pos].w = prof;
				pos++;
			}
		}
	}
}

// Devuelve 0 si todo ha ido bien, 1 si ha habido algún error (no existe algún fichero)
int cargarDatosProblema(string fich_ent, TDatoCluster *datos_cluster, string &nombre_bati, string &prefijo,
				int *num_voly_otros, int *num_voly_total, Scalar *xmin, Scalar *xmax, Scalar *ymin, Scalar *ymax,
				Scalar *Hmin_global, Scalar *borde_sup, Scalar *borde_inf, Scalar *borde_izq, Scalar *borde_der,
				Scalar *ancho_vol, Scalar *alto_vol, Scalar *area, Scalar *tiempo_tot, Scalar *tiempo_guardar,
				Scalar *CFL, Scalar *r, Scalar *angulo1, Scalar *angulo2, Scalar *angulo3, Scalar *angulo4,
				Scalar *mfc, Scalar *mf0, Scalar *mfs, Scalar *vmax1, Scalar *vmax2, Scalar *gravedad,
				Scalar *epsilon_h, Scalar *L, Scalar *H, Scalar *Q, Scalar *T, int num_procs, int id_hebra,
				int *leer_fichero_puntos, int **indiceVolumenesGuardado, 
				int **posicionesVolumenesGuardado, int *num_puntos_guardar)
{
	// num_voly_otros es el número de filas de volúmenes de todos los procesos menos el último
	int i, indice1, num_vols_leer;
	int num_volumenes, num_volx;
	int leerDeFichero, normalizar;
	// Variables de un volumen
	Scalar mitad_ancho, mitad_alto;
	Scalar val;
	bool frontera;
	Scalar centro_x, centro_y;
	// Valor mínimo de H de la submalla (el mínimo global es Hmin_global). Si es negativo,
	// se corrigen todas las H para que sean todas mayores o iguales que 0
	Scalar Hmin;
	// Stream para leer los ficheros de datos topográficos
	// y el estado inicial (si procede)
	ifstream fich2;
	// Directorio donde se encuentran los ficheros de datos
	string directorio;
	string fich_topo, fich_est;
	string fich_puntos;
	Scalar W[6];

	// Ponemos en directorio el directorio donde están los ficheros de datos
	i = fich_ent.find_last_of("/");
	if (i > -1) {
		// Directorios indicados con '/' (S.O. distinto de windows)
		directorio = fich_ent.substr(0,i)+"/";
	}
	else {
		i = fich_ent.find_last_of("\\");
		if (i > -1) {
			// Directorios indicados con '\' (S.O. Windows)
			directorio = fich_ent.substr(0,i)+"\\";
		}
		else {
			// No se ha especificado ningún directorio para los ficheros de datos
			directorio = "";
		}
	}

	// Leemos los datos del problema del fichero de entrada
	ifstream fich(fich_ent.c_str());
	fich >> nombre_bati;
	fich >> leerDeFichero;
	if (leerDeFichero == 0) {
		fich >> *xmin;
		fich >> *xmax;
		fich >> *ymin;
		fich >> *ymax;
		fich >> datos_cluster->num_volx;
		fich >> *num_voly_total;
		datos_cluster->num_voly = *num_voly_total;
	}
	else {
		// Leemos el fichero con los datos de la topografía
		fich >> fich_topo;
		fich >> fich_est;
		fich_topo = directorio+fich_topo;
		fich_est = directorio+fich_est;
		if (! existeFichero((char *) fich_topo.c_str())) {
			cerr << "Error: No se ha encontrado el fichero '" << fich_topo << "'" << endl;
			return 1;
		}
		if (! existeFichero((char *) fich_est.c_str())) {
			cerr << "Error: No se ha encontrado el fichero '" << fich_est << "'" << endl;
			return 1;
		}
		fich2.open(fich_topo.c_str());
		fich2 >> *xmin;
		fich2 >> *xmax;
		fich2 >> *ymin;
		fich2 >> *ymax;
		fich2 >> datos_cluster->num_volx;
		fich2 >> *num_voly_total;
		datos_cluster->num_voly = *num_voly_total;
		// Los datos topográficos y el estado inicial se leerán al final,
		// cuando estén creados todos los volúmenes
	}
	fich >> *borde_sup;
	fich >> *borde_inf;
	fich >> *borde_izq;
	fich >> *borde_der;
	fich >> *tiempo_tot;
	fich >> *tiempo_guardar;
	fich >> *leer_fichero_puntos;
	if (*leer_fichero_puntos == 1)
                fich >> fich_puntos;
	fich_puntos=directorio+fich_puntos;
	if (! existeFichero((char *) fich_topo.c_str())) {
                        cerr << "Error: No se ha encontrado el fichero '" << fich_puntos << "'" << endl;
                        return 1;
                }
	fich >> *CFL;
	fich >> *r;
#ifdef COULOMB
	// Ley de Coulomb
	fich >> *angulo1;
	*angulo1 *= M_PI/180.0;
#else
	// Ley de Pouliquen
	fich >> *angulo1;
	fich >> *angulo2;
	fich >> *angulo3;
	fich >> *angulo4;
	*angulo1 *= M_PI/180.0;
	*angulo2 *= M_PI/180.0;
	*angulo3 *= M_PI/180.0;
	*angulo4 *= M_PI/180.0;
#endif
	fich >> *mfc;
	fich >> *mf0;
	fich >> *mfs;
	fich >> *vmax1;
	fich >> *vmax2;
	fich >> normalizar;
	if (normalizar == 0) {
		*gravedad = 9.81;
		*L = *H = *Q = *T = 1.0;
	}
	else {
		fich >> *L;
		fich >> *H;
		*Q = sqrt(9.81*pow(*H,3.0));
		*T = (*L)*(*H)/(*Q);
		*gravedad = 1.0;

		*xmin /= *L;
		*xmax /= *L;
		*ymin /= *L;
		*ymax /= *L;
		*tiempo_tot /= *T;
		*tiempo_guardar /= *T;
		*mfc *= *L;
		*mf0 *= ((*Q)/(*H))*sqrt(*L)/pow(*H,7.0/6.0);
		*mfs *= ((*Q)/(*H))*sqrt(*L)/pow(*H,7.0/6.0);
		*vmax1 /= (*Q)/(*H);
		*vmax2 /= (*Q)/(*H);
	}
	fich >> prefijo;
	*epsilon_h = 5e-3/(*H);
	fich.close();


        // Obtenemos los par?metros comunes a todos los vol?menes
        *ancho_vol = ((*xmax)-(*xmin)) / datos_cluster->num_volx;
        *alto_vol = ((*ymax)-(*ymin)) / datos_cluster->num_voly;
        mitad_ancho = 0.5*(*ancho_vol);
        mitad_alto = 0.5*(*alto_vol);
        *area = (*ancho_vol)*(*alto_vol);

        // Leemos los puntos de guardado
        if (*leer_fichero_puntos == 1) {

		*posicionesVolumenesGuardado = (int *) malloc(num_volumenes*sizeof(int));
                for (i=0; i<num_volumenes; i++)
                        (*posicionesVolumenesGuardado)[i] = -1;
                fich.open(fich_puntos.c_str());
                fich >> *num_puntos_guardar;

                *indiceVolumenesGuardado = (int *) malloc((*num_puntos_guardar)*sizeof(int));
                if (*indiceVolumenesGuardado == NULL) {
                        cerr << "Error: Not enough CPU memory" << endl;
                        return 1;
                }

		float X[datos_cluster->num_volx];
		float Y[datos_cluster->num_voly];
		for (i=0; i<datos_cluster->num_volx; i++)
			X[i]=(*xmin+i*(*ancho_vol))*(*L);
		for (i=0; i<datos_cluster->num_voly; i++)
			Y[i]=(*ymin+i*(*alto_vol))*(*L);

		float lon,lat;
		int pos;

                for (i=0; i<(*num_puntos_guardar); i++) {
                        fich >> lon;
                        fich >> lat;
			lon = 63259*(lon+67.7)/0.6;
			lat = 55604*(lat-18.3)/0.5;
			//if((lon>=X[0]) && (lon<=X[datos_cluster->num_volx-1]) && (lat>=Y[0]) && (lat<=Y[datos_cluster->num_voly-1]))
	                        pos = obtenerIndicePunto(X,Y,lon,lat,datos_cluster->num_volx,datos_cluster->num_voly);
			//else
			//	pos = -1;
                        if (pos >= 0) {
                                if ((*posicionesVolumenesGuardado)[pos] == -1) {
                                        // Posici?n pos nueva
                                        (*posicionesVolumenesGuardado)[pos] = i;
                                        (*indiceVolumenesGuardado)[i] = pos;
                                }
                                else {
                                        // Posici?n pos repetida
                                        (*indiceVolumenesGuardado)[i] = (*posicionesVolumenesGuardado)[pos];
                                }
                        }
                        else {
                                (*indiceVolumenesGuardado)[i] = -1;
                        }
                }
                fich.close();
        }


	// Obtenemos las dimensiones del subproblema
	num_volx = datos_cluster->num_volx;
	// indice1: num_voly de todo el dominio
	indice1 = datos_cluster->num_voly;
	// Obtenemos el número de filas del subproblema y lo ponemos en datos_cluster->num_voly
	// (forzamos a que sea un número par para no tener que procesar dos filas adicionales
	// inferiores debido a la positividad. Si es par sólo es necesario procesar una).
	datos_cluster->num_voly = (int) ceil(datos_cluster->num_voly/num_procs);
	if (datos_cluster->num_voly % 2 != 0) {
		(datos_cluster->num_voly)++;
	}
	*num_voly_otros = datos_cluster->num_voly;
	if (id_hebra == num_procs-1) {
		// Es la última hebra
		datos_cluster->num_voly = indice1 - (num_procs-1)*(*num_voly_otros);
	}
	// num_volumenes = número de volúmenes de la submalla asociada a la hebra
	num_volumenes = datos_cluster->num_volx * datos_cluster->num_voly;

	// Reservamos memoria para los datos de los volúmenes.
	// Para mantener la coherencia, todas las hebras reservan una fila de volúmenes
	// al principio y otra al final para los volúmenes de comunicación de los
	// clusters adyacentes, aunque la primera hebra no usará la primera fila
	// (al no tener cluster adyacente superior) y la última hebra no usará las
	// últimas filas (al no tener cluster adyacente inferior).
	// En el procesamiento de las aristas, es necesario procesar estos volúmenes
	// adicionales debido a la positividad
	datos_cluster->datosVolumenes_1 = new float4[num_volumenes + 2*num_volx];
	datos_cluster->datosVolumenes_2 = new float4[num_volumenes + 2*num_volx];
	datos_cluster->eta1_maxima = new float2[num_volumenes];
	// Asignamos los punteros a los volúmenes de comunicación del cluster
	// y de los clusters adyacentes
	datos_cluster->puntero_datosVolumenesComClusterSup_1 = datos_cluster->datosVolumenes_1 + num_volx;
	datos_cluster->puntero_datosVolumenesComClusterSup_2 = datos_cluster->datosVolumenes_2 + num_volx;
	datos_cluster->puntero_datosVolumenesComClusterInf_1 = datos_cluster->datosVolumenes_1 + num_volumenes;
	datos_cluster->puntero_datosVolumenesComClusterInf_2 = datos_cluster->datosVolumenes_2 + num_volumenes;
	datos_cluster->puntero_datosVolumenesComOtroClusterSup_1 = datos_cluster->datosVolumenes_1 + num_volumenes + num_volx;
	datos_cluster->puntero_datosVolumenesComOtroClusterSup_2 = datos_cluster->datosVolumenes_2 + num_volumenes + num_volx;
	datos_cluster->puntero_datosVolumenesComOtroClusterInf_1 = datos_cluster->datosVolumenes_1;
	datos_cluster->puntero_datosVolumenesComOtroClusterInf_2 = datos_cluster->datosVolumenes_2;
	// Ponemos en indice1 el índice inicial de los volúmenes que hay que leer en los ficheros de datos
	// (considerando los volúmenes de comunicación inferiores del cluster adyacente superior, que también se almacenan).
	// Ponemos en num_vols_leer el número de volúmenes que hay que leer en los ficheros de datos.
	// Estos volúmenes se almacenarán en datos_cluster->datosVolumenes_[1|2]
	if (id_hebra == 0) {
		// Es la primera hebra
		indice1 = 0;
		// num_vols_leer = Los volúmenes de la submalla más los volúmenes de comunicación
		// superiores del cluster adyacente inferior
		if (num_procs == 1)
			num_vols_leer = num_volumenes;
		else
			num_vols_leer = num_volumenes + num_volx;
	}
	else if (id_hebra == num_procs-1) {
		// Es la última hebra
		indice1 = id_hebra*num_volx*(*num_voly_otros) - num_volx;
		// num_vols_leer = Los volúmenes de la submalla más los volúmenes de comunicación
		// inferiores del cluster adyacente superior
		num_vols_leer = num_volx + num_volumenes;
	}
	else {
		// Es una hebra intermedia
		indice1 = id_hebra*num_volx*(*num_voly_otros) - num_volx;
		// num_vols_leer = Los volúmenes de la submalla más los volúmenes de comunicación
		// de los dos clústers adyacentes
		num_vols_leer = num_volumenes + 2*num_volx;
	}

	if (leerDeFichero == 0) {
		setCondicionesIniciales(datos_cluster, *xmin, *xmax, *ymin, *ymax, *ancho_vol, *alto_vol,
			datos_cluster->num_volx, datos_cluster->num_voly, indice1, num_vols_leer, *L, *H, *Q, id_hebra);
		*Hmin_global = 0.0;
	}
	else {
		// LECTURA DE DATOS DE LA TOPOGRAFIA
		// Primero saltamos las profundidades de los volúmenes de clusters anteriores a id_hebra,
		// excepto los volúmenes de comunicación inferiores del cluster adyacente superior
		for (i=0; i<indice1; i++)
			fich2 >> val;
		Hmin = 1e30;
		if (id_hebra == 0) {
			// La primera hebra no tiene cluster adyacente superior.
			// No escribimos nada en la primera fila de datosVolumenes
			for (i=num_volx; i<num_volx+num_vols_leer; i++) {
				fich2 >> val;
				val /= *H;
				datos_cluster->datosVolumenes_1[i].w = val;
				datos_cluster->datosVolumenes_2[i].w = val;
				if (val < Hmin)
					Hmin = val;
			}
		}
		else {
			// Hay cluster adyacente superior.
			// Los datos a leer del fichero incluyen los volúmenes de comunicación
			// inferiores del cluster adyacente superior (la primera fila de datosVolumenes)
			for (i=0; i<num_vols_leer; i++) {
				fich2 >> val;
				val /= *H;
				datos_cluster->datosVolumenes_1[i].w = val;
				datos_cluster->datosVolumenes_2[i].w = val;
				if (val < Hmin)
					Hmin = val;
			}
		}
		fich2.close();

		// Obtenemos el mínimo Hmin de todos los clusters por reducción
		MPI_Allreduce (&Hmin, Hmin_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

		// Corregimos los valores de profundidad, si hay alguna negativa
		if (*Hmin_global >= 0.0)
			*Hmin_global = 0.0;
		else {
			if (id_hebra == 0) {
				// La primera hebra no tiene cluster adyacente superior.
				// No escribimos nada en la primera fila de datosVolumenes
				for (i=num_volx; i<num_volx+num_vols_leer; i++) {
					datos_cluster->datosVolumenes_1[i].w -= *Hmin_global;
					datos_cluster->datosVolumenes_2[i].w -= *Hmin_global;
				}
			}
			else {
				// Hay cluster adyacente superior.
				for (i=0; i<num_vols_leer; i++) {
					datos_cluster->datosVolumenes_1[i].w -= *Hmin_global;
					datos_cluster->datosVolumenes_2[i].w -= *Hmin_global;
				}
			}
		}

		cout << "HMIN_GLOBAL = " << *Hmin_global << endl;

		// LECTURA DE DATOS DEL ESTADO INICIAL
		fich2.open(fich_est.c_str());
		// Primero saltamos los estados de los volúmenes de clusters anteriores a id_hebra,
		// excepto los volúmenes de comunicación inferiores del cluster adyacente superior
		for (i=0; i<indice1; i++) {
			fich2 >> val;
			fich2 >> val;
			fich2 >> val;
			fich2 >> val;
			fich2 >> val;
			fich2 >> val;
		}
		if (id_hebra == 0) {
			// La primera hebra no tiene cluster adyacente superior.
			// No escribimos nada en la primera fila de datosVolumenes
			for (i=num_volx; i<num_volx+num_vols_leer; i++) {
				fich2 >> W[0];  fich2 >> W[1];  fich2 >> W[2];
				fich2 >> W[3];  fich2 >> W[4];  fich2 >> W[5];
				W[0] /= *H;
				W[1] /= *Q;
				W[2] /= *Q;
				W[3] /= *H;
				W[4] /= *Q;
				W[5] /= *Q;
				datos_cluster->datosVolumenes_1[i].x = W[0];
				datos_cluster->datosVolumenes_1[i].y = W[1];
				datos_cluster->datosVolumenes_1[i].z = W[2];
				datos_cluster->datosVolumenes_2[i].x = W[3];
				datos_cluster->datosVolumenes_2[i].y = W[4];
				datos_cluster->datosVolumenes_2[i].z = W[5];
			}
		}
		else {
			// Hay cluster adyacente superior.
			// Los datos a leer del fichero incluyen los volúmenes de comunicación
			// inferiores del cluster adyacente superior (la primera fila de datosVolumenes)
			for (i=0; i<num_vols_leer; i++) {
				fich2 >> W[0];  fich2 >> W[1];  fich2 >> W[2];
				fich2 >> W[3];  fich2 >> W[4];  fich2 >> W[5];
				W[0] /= *H;
				W[1] /= *Q;
				W[2] /= *Q;
				W[3] /= *H;
				W[4] /= *Q;
				W[5] /= *Q;
				datos_cluster->datosVolumenes_1[i].x = W[0];
				datos_cluster->datosVolumenes_1[i].y = W[1];
				datos_cluster->datosVolumenes_1[i].z = W[2];
				datos_cluster->datosVolumenes_2[i].x = W[3];
				datos_cluster->datosVolumenes_2[i].y = W[4];
				datos_cluster->datosVolumenes_2[i].z = W[5];
			}
		}
		fich2.close();
	}

	// Asignamos los valores de eta1 máxima para cada volumen del cluster
	for (i=0; i<num_volumenes; i++) {
		indice1 = num_volx+i;
		datos_cluster->eta1_maxima[i].x = (datos_cluster->datosVolumenes_1[indice1].x + datos_cluster->datosVolumenes_2[indice1].x - datos_cluster->datosVolumenes_1[indice1].w);
		datos_cluster->eta1_maxima[i].y = 0.0;
	}

	return 0;
}

void liberarMemoria(TDatoCluster *dc) {
	delete [] (dc->datosVolumenes_1);
	delete [] (dc->datosVolumenes_2);
}

void mostrarDatosProblema(int num_volx, int num_voly, Scalar xmin, Scalar xmax, Scalar ymin, Scalar ymax, Scalar tiempo_tot,
				Scalar CFL, Scalar r, Scalar angulo1, Scalar angulo2, Scalar angulo3, Scalar angulo4, Scalar mfc,
				Scalar mf0, Scalar mfs, Scalar vmax1, Scalar vmax2, Scalar L, Scalar H, Scalar Q, Scalar T)
{
	cout << "Info Problema" << endl;
	cout << "Volumenes: " << num_volx << " x " << num_voly << ", Total: " << num_volx*num_voly << endl;
	cout << "X: [" << xmin*L << ", " << xmax*L << "]" << endl;
	cout << "Y: [" << ymin*L << ", " << ymax*L << "]" << endl;
	cout << "CFL: " << CFL << endl;
	cout << "r: " << r << endl;
#ifdef COULOMB
	cout << "Angulo de reposo: " << angulo1*180.0/M_PI << endl;
#else
	cout << "Angulos de reposo: " << angulo1*180.0/M_PI;
	cout << ", " << angulo2*180.0/M_PI;
	cout << ", " << angulo3*180.0/M_PI;
	cout << ", " << angulo4*180.0/M_PI << endl;
#endif
	cout << "Friccion entre capas: " << mfc/L << endl;
	cout << "Friccion agua-fondo: " << mf0/((Q/H)*sqrt(L)/pow(H,7.0/6.0)) << endl;
	cout << "Friccion sedimento-fondo: " << mfs/((Q/H)*sqrt(L)/pow(H,7.0/6.0)) << endl;
	cout << "Velocidad maxima agua: " << vmax1*(Q/H) << endl;
	cout << "Velocidad maxima sedimentos: " << vmax2*(Q/H) << endl;
	cout << "Tiempo de simulacion: " << tiempo_tot*T << " seg." << endl;
}

