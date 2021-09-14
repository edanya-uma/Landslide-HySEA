#include <vector_types.h>
#include <string.h>
#include <mpi.h>
#include "Constantes.hxx"
#include "Problema.cxx"

/*****************/
/* Funciones GPU */
/*****************/

extern "C" int comprobarSoporteCUDA();
extern "C" int shallowWater(TDatoCluster *datos_cluster, float xmin, float ymin, float HMin, char *nombre_bati,
		char *prefijo, int num_voly_otros, int num_voly_total, float borde_sup, float borde_inf, float borde_izq,
		float borde_der, float ancho_vol, float alto_vol, float area, float tiempo_tot, float tiempo_guardar,
		float CFL, float r, float angulo1, float angulo2, float angulo3, float angulo4, float peso, float beta,
		float mfc, float mf0, float mfs, float vmax1, float vmax2, float gravedad, float epsilon_h, float L, float H,
		float Q, float T, int num_procs, int id_hebra, double *tiempo, int leer_fichero_puntos, 
		int *indiceVolumenesGuardado, int *posicionesVolumenesGuardado, int num_puntos_guardar);

/*********************/
/* Fin funciones GPU */
/*********************/

void mostrarFormatoPrograma(char *argv[])
{
	cerr << "Uso: " << endl;
	cerr << argv[0] << " ficheroDatos" << endl << endl; 
	cerr << "Formato de ficheroDatos:" << endl;
	cerr << "\tNombre de la batimetria" << endl;
	cerr << "\tLeer condiciones iniciales de fichero (0|1)" << endl;
	cerr << "\tSi 0:" << endl;
	cerr << "\t\tXmin" << endl;
	cerr << "\t\tXmax" << endl;
	cerr << "\t\tYmin" << endl;
	cerr << "\t\tYmax" << endl;
	cerr << "\t\tVolumenes en x" << endl;
	cerr << "\t\tVolumenes en y" << endl;
	cerr << "\tSi 1:" << endl;
	cerr << "\t\tFichero de topografia" << endl;
	cerr << "\t\tFichero de estado inicial" << endl;
	cerr << "\tBorde superior  (1: abierto, 0 o -1: pared)" << endl;
	cerr << "\tBorde inferior  (1: abierto, 0 o -1: pared)" << endl;
	cerr << "\tBorde izquierdo (1: abierto, 0 o -1: pared)" << endl;
	cerr << "\tBorde derecho   (1: abierto, 0 o -1: pared)" << endl;
	cerr << "\tTiempo de simulacion" << endl;
	cerr << "\tTiempo de guardado (-1: no guardar, 0: todo)" << endl;
	cerr << "\tCFL" << endl;
	cerr << "\tRatio de densidades" << endl;
	cerr << "\tAngulos de reposo (1 si Coulomb, 4 si Pouliquen)" << endl;
	cerr << "\tFriccion entre capas" << endl;
	cerr << "\tFriccion agua-fondo" << endl;
	cerr << "\tFriccion sedimentos-fondo" << endl;
	cerr << "\tVelocidad maxima agua" << endl;
	cerr << "\tVelocidad maxima sedimentos" << endl;
	cerr << "\tNormalizar (0|1)" << endl;
	cerr << "\tSi 1:" << endl;
	cerr << "\t\tL" << endl;
	cerr << "\t\tH" << endl;
	cerr << "\tPrefijo de los ficheros de guardado" << endl;
}

int main(int argc, char *argv[])
{
	TDatoCluster datos_cluster;
	char fich_ent[256];
	int iter, soporteCUDA, err, err2 = 1;
	double tiempo_gpu, tiempo_multigpu;
	// Variables del problema
	int num_voly_otros, num_voly_total;
	Scalar xmin, xmax, ymin, ymax;
	Scalar borde_sup, borde_inf, borde_izq, borde_der;
	Scalar ancho_vol, alto_vol, area;
	Scalar tiempo_tot, tiempo_guardar;
	Scalar Hmin, CFL, r;
	Scalar angulo1, angulo2, angulo3, angulo4;
	Scalar mfc, mf0, mfs;
	Scalar vmax1, vmax2;
	Scalar gravedad, epsilon_h;
	Scalar L, H, Q, T;
	string nombre_bati, prefijo;
	// Variables para MPI
	MPI_Status status;
	int id_hebra, num_procs;
	int ultima_hebra;
	int *indiceVolumenesGuardado = NULL;
        int *posicionesVolumenesGuardado = NULL;
        int leer_fichero_puntos, num_puntos_guardar;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_hebra);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	ultima_hebra = (id_hebra == num_procs-1) ? 1 : 0;

	if (id_hebra == 0) {
		// El proceso 0 lee los datos de entrada
		err = 0;
		if (argc < 2) {
			mostrarFormatoPrograma(argv);
			err = 1;
		}
		else {
			// Fichero de datos
			strcpy(fich_ent, argv[1]);
			if (! existeFichero(fich_ent)) {
				cerr << "Error en hebra " << id_hebra << ": No se ha encontrado el fichero '" << fich_ent << "'" << endl;
				err = 1;
			}
		}
	}

	// El proceso 0 envía err y fich_prob al resto de procesos
	MPI_Bcast (&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (fich_ent, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
	string str_fich_ent(fich_ent);

	if (err == 0) {
		// No ha habido error
		// Todos los procesos ejecutan esto

		// Comprobamos si la tarjeta gráfica soporta CUDA
		soporteCUDA = comprobarSoporteCUDA();
		if (soporteCUDA == 1) {
			cerr << "Error en hebra " << id_hebra << ": No hay tarjeta grafica" << endl;
			err = 1;
		}
		else if (soporteCUDA == 2) {
				cerr << "Error en hebra " << id_hebra << ": No hay ninguna tarjeta grafica que soporte CUDA" << endl;
				err = 1;
		}

		cout << "Hebra " << id_hebra << " cargando datos" << endl;
		// Creamos una instancia de Problema
		err = cargarDatosProblema(str_fich_ent, &datos_cluster, nombre_bati, prefijo, &num_voly_otros, &num_voly_total,
				&xmin, &xmax, &ymin, &ymax, &Hmin, &borde_sup, &borde_inf, &borde_izq, &borde_der, &ancho_vol, &alto_vol,
				&area, &tiempo_tot, &tiempo_guardar, &CFL, &r, &angulo1, &angulo2, &angulo3, &angulo4, &mfc, &mf0, &mfs,
				&vmax1, &vmax2, &gravedad, &epsilon_h, &L, &H, &Q, &T, num_procs, id_hebra, &leer_fichero_puntos, 
				&indiceVolumenesGuardado, &posicionesVolumenesGuardado,
                        	&num_puntos_guardar);

		// Comprobamos si ha habido error en algún proceso
		MPI_Allreduce(&err, &err2, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

		if (id_hebra == 0) {
			mostrarDatosProblema(datos_cluster.num_volx, num_voly_total, xmin, xmax, ymin, ymax, tiempo_tot,
				CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, L, H, Q, T);
		}
	}

	cout << scientific;
	if (err2 == 0) {
		// MultiGPU
		if (id_hebra == 0) {
			cout << endl;
			cout << "MultiGPU" << endl;
			cout << "--------" << endl;
		}
		err = shallowWater(&datos_cluster, (float) xmin, (float) ymin, (float) Hmin, (char *) nombre_bati.c_str(),
				(char *) prefijo.c_str(), num_voly_otros, num_voly_total, (float) borde_sup, (float) borde_inf,
				(float) borde_izq, (float) borde_der, (float) ancho_vol, (float) alto_vol, (float) area, (float) tiempo_tot,
				(float) tiempo_guardar, (float) CFL, (float) r, (float) angulo1, (float) angulo2, (float) angulo3,
				(float) angulo4, (float) 1.0, (float) 1.0, (float) mfc, (float) mf0, (float) mfs, (float) vmax1,
				(float) vmax2, (float) gravedad, (float) epsilon_h, (float) L, (float) H, (float) Q, (float) T,
				num_procs, id_hebra, &tiempo_gpu, leer_fichero_puntos, indiceVolumenesGuardado, posicionesVolumenesGuardado,
                        	num_puntos_guardar);
		if (err > 0) {
			if (err == 1)
				cerr << "Error: No hay memoria GPU suficiente" << endl;
			else if (err == 2)
				cerr << "Error: No hay memoria CPU suficiente" << endl;
			return 1;
		}

		// El tiempo total es el máximo de los tiempos locales
		MPI_Reduce (&tiempo_gpu, &tiempo_multigpu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if (id_hebra == 0)
			cout << endl << "Tiempo: " << tiempo_multigpu << " seg" << endl;
	}

	MPI_Finalize();

	return 0;
}

