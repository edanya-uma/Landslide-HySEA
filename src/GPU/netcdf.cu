#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <netcdf.h>
#include "pnetcdf.h"

bool ErrorEnNetCDF;
// Ids de ficheros
int ncid_eta1, ncid_q1x, ncid_q1y;
int ncid_eta2, ncid_q2x, ncid_q2y;
// Ids de variables
int time_eta1_id, eta1_id;
int time_q1x_id, q1x_id;
int time_q1y_id, q1y_id;
int time_eta2_id, eta2_id;
int time_q2x_id, q2x_id;
int time_q2y_id, q2y_id;
int eta1_max_id;

void check_err(int iret)
{
	if ((iret != NC_NOERR) && (! ErrorEnNetCDF)) {
		fprintf(stderr, "%s\n", ncmpi_strerror(iret));
		ErrorEnNetCDF = true;
	}
}

void fgennc(int id_hebra, float *x_grid, float *y_grid, float *x, float *y, char *nombre_bati, char *prefijo, int nvar,
			int *p_ncid, int *time_id, int *var_id, int nx_nc, int ny_nc, int num_volx, int num_voly, int num_voly_otros,
			int num_voly_total, float xmin, float ymin, float ancho_vol, float alto_vol, float tiempo_tot, float CFL,
			float r, float angulo1, float angulo2, float angulo3, float angulo4, float mfc, float mf0, float mfs,
			float vmax1, float vmax2, float *bati)
{
	char nombre_fich[256];
	char cadena[256];
	// Dimensiones
	int grid_x_dim, grid_y_dim;
	int grid_dims[2];
	int x_dim, y_dim;
	int var_dims[3];
	int time_dim;
	// Ids
	int ncid;
	int grid_id, grid_x_id, grid_y_id;
	int x_id, y_id;
	float val_float, fill_float;
	struct timeval tv;
	char fecha_act[24];

	int iret;

	// Creamos el fichero y entramos en modo definición
	if (nvar == 1)       sprintf(nombre_fich, "%s_eta1.nc", prefijo);
	else if (nvar == 2)  sprintf(nombre_fich, "%s_q1x.nc", prefijo);
	else if (nvar == 3)  sprintf(nombre_fich, "%s_q1y.nc", prefijo);
	else if (nvar == 4)  sprintf(nombre_fich, "%s_eta2.nc", prefijo);
	else if (nvar == 5)  sprintf(nombre_fich, "%s_q2x.nc", prefijo);
	else if (nvar == 6)  sprintf(nombre_fich, "%s_q2y.nc", prefijo);
	iret = ncmpi_create(MPI_COMM_WORLD, nombre_fich, NC_CLOBBER, MPI_INFO_NULL, p_ncid);
	check_err(iret);
	ncid = *p_ncid;

	// Definimos dimensiones
	iret = ncmpi_def_dim(ncid, "lon", nx_nc, &x_dim);
	check_err(iret);
	iret = ncmpi_def_dim(ncid, "lat", ny_nc, &y_dim);
	check_err(iret);
	if (nvar == 1) {
		iret = ncmpi_def_dim(ncid, "grid_x", num_volx, &grid_x_dim);
		check_err(iret);
		iret = ncmpi_def_dim(ncid, "grid_y", num_voly_total, &grid_y_dim);
		check_err(iret);
	}
	iret = ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	check_err(iret);

	// Definimos variables
	iret = ncmpi_def_var(ncid, "lon", NC_FLOAT, 1, &x_dim, &x_id);
	check_err(iret);
	iret = ncmpi_def_var(ncid, "lat", NC_FLOAT, 1, &y_dim, &y_id);
	check_err(iret);
	if (nvar == 1) {
		iret = ncmpi_def_var(ncid, "grid_x", NC_FLOAT, 1, &grid_x_dim, &grid_x_id);
		check_err(iret);
		iret = ncmpi_def_var(ncid, "grid_y", NC_FLOAT, 1, &grid_y_dim, &grid_y_id);
		check_err(iret);
		grid_dims[0] = grid_y_dim;
		grid_dims[1] = grid_x_dim;
		iret = ncmpi_def_var(ncid, "bathymetry", NC_FLOAT, 2, grid_dims, &grid_id);
		check_err(iret);
	}
	var_dims[0] = time_dim;
	var_dims[1] = y_dim;
	var_dims[2] = x_dim;
	// Nota: reutilizamos el array grid_dims
	grid_dims[0] = y_dim;
	grid_dims[1] = x_dim;
	iret = ncmpi_def_var(ncid, "time", NC_FLOAT, 1, &time_dim, time_id);
	check_err(iret);
	if (nvar == 1) {
		iret = ncmpi_def_var(ncid, "max_height", NC_FLOAT, 2, grid_dims, &eta1_max_id);
		check_err(iret);
		iret = ncmpi_def_var(ncid, "eta1", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 6, "meters");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 14, "Wave amplitude");
		check_err(iret);
	}
	else if (nvar == 2) {
		iret = ncmpi_def_var(ncid, "q1x", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 13, "meters/second");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 26, "Mass flow of water along x");
		check_err(iret);
	}
	else if (nvar == 3) {
		iret = ncmpi_def_var(ncid, "q1y", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 13, "meters/second");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 26, "Mass flow of water along y");
		check_err(iret);
	}
	else if (nvar == 4) {
		iret = ncmpi_def_var(ncid, "eta2", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 6, "meters");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 9, "Sediments");
		check_err(iret);
	}
	else if (nvar == 5) {
		iret = ncmpi_def_var(ncid, "q2x", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 13, "meters/second");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 30, "Mass flow of sediments along x");
		check_err(iret);
	}
	else if (nvar == 6) {
		iret = ncmpi_def_var(ncid, "q2y", NC_FLOAT, 3, var_dims, var_id);
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "units", 13, "meters/second");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, *var_id, "long_name", 30, "Mass flow of sediments along y");
		check_err(iret);
	}

	// Asignamos attributos
	fill_float = -1e+30;
	if (nvar == 1) {
		iret = ncmpi_put_att_text(ncid, grid_id, "long_name", 15, "Grid bathymetry");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, grid_id, "standard_name", 5, "depth");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, grid_id, "units", 6, "meters");
		check_err(iret);
		iret = ncmpi_put_att_float(ncid, grid_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = ncmpi_put_att_float(ncid, grid_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);

		iret = ncmpi_put_att_text(ncid, grid_x_id, "long_name", 11, "Grid x axis");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, grid_x_id, "units", 6, "meters");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, grid_y_id, "long_name", 11, "Grid y axis");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, grid_y_id, "units", 6, "meters");
		check_err(iret);

		iret = ncmpi_put_att_text(ncid, eta1_max_id, "long_name", 22, "Maximum wave amplitude");
		check_err(iret);
		iret = ncmpi_put_att_text(ncid, eta1_max_id, "units", 6, "meters");
		check_err(iret);
		iret = ncmpi_put_att_float(ncid, eta1_max_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = ncmpi_put_att_float(ncid, eta1_max_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
	}

	iret = ncmpi_put_att_text(ncid, x_id, "long_name", 6, "x axis");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, x_id, "units", 6, "meters");
	check_err(iret);

	iret = ncmpi_put_att_text(ncid, y_id, "long_name", 6, "y axis");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, y_id, "units", 6, "meters");
	check_err(iret);

	iret = ncmpi_put_att_text(ncid, *time_id, "long_name", 4, "Time");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, *time_id, "units", 24, "seconds since 1970-01-01");
	check_err(iret);
	iret = ncmpi_put_att_float(ncid, *var_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = ncmpi_put_att_float(ncid, *var_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);

	// Atributos globales
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.0");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "title", 25, "TsunamiHySEA model output");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "creator_name", 12, "EDANYA Group");
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "institution", 20, "University of Malaga");
	check_err(iret);
	sprintf(cadena, "M. de la Asunción et al. Scalable simulation of tsunamis generated by submarine landslides on GPU clusters. Parallel Computing. Submitted");
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "comments", strlen(cadena), cadena);
	check_err(iret);
	sprintf(cadena, "http://path.to.paper/paper.pdf");
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "references", strlen(cadena), cadena);
	check_err(iret);

	gettimeofday(&tv, NULL);
	strftime(fecha_act, 24, "%Y-%m-%d %H:%M:%S", localtime(&(tv.tv_sec)));
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "history", strlen(fecha_act), fecha_act);
	check_err(iret);

	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "grid_name", strlen(nombre_bati), nombre_bati);
	check_err(iret);
	val_float = tiempo_tot;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "simulation_time", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = CFL;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "CFL", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = r;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "ratio_of_densities", NC_FLOAT, 1, &val_float);
	check_err(iret);
	if (angulo2 < 0.0) {
		sprintf(cadena, "%.4f", angulo1);
		iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "friction_law", 7, "Coulomb");
 	}
	else {
		sprintf(cadena, "%.4f, %.4f, %.4f, %.4f", angulo1, angulo2, angulo3, angulo4);
		iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "friction_law", 9, "Pouliquen");
	}
	check_err(iret);
	iret = ncmpi_put_att_text(ncid, NC_GLOBAL, "angle_of_repose", strlen(cadena), cadena);
	val_float = mfc;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "friction_between_layers", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = mf0;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "friction_water_bottom", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = mfs;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "friction_sediments_bottom", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = vmax1;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "max_speed_water", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = vmax2;
	iret = ncmpi_put_att_float(ncid, NC_GLOBAL, "max_speed_sediments", NC_FLOAT, 1, &val_float);
	check_err(iret);

	// Fin del modo definición
	iret = ncmpi_enddef(ncid);
	check_err(iret);

	// Guardamos x
	iret = ncmpi_put_var_float_all(ncid, x_id, x);
	check_err(iret);
	// Guardamos y
	iret = ncmpi_put_var_float_all(ncid, y_id, y);
	check_err(iret);

	// Guardamos la batimetría
	if (nvar == 1) {
		MPI_Offset start[] = {id_hebra*num_voly_otros, 0};
		MPI_Offset count[] = {num_voly, num_volx};
		iret = ncmpi_put_var_float_all(ncid, grid_x_id, x_grid);
		check_err(iret);
		iret = ncmpi_put_var_float_all(ncid, grid_y_id, y_grid);
		check_err(iret);
		iret = ncmpi_put_vara_float_all(ncid, grid_id, start, count, bati);
		check_err(iret);
	}
}

void initNC(int id_hebra, char *nombre_bati, char *prefijo, int num_volx, int num_voly, int num_voly_otros,
			int num_voly_total, int *nx_nc, int *ny_nc, int npics, float xmin, float ymin, float ancho_vol,
			float alto_vol, float tiempo_tot, float CFL, float r, float angulo1, float angulo2, float angulo3,
			float angulo4, float mfc, float mf0, float mfs, float vmax1, float vmax2, float *bati)
{
	float *x_grid, *y_grid;
	float *x, *y;
	int i;

	ErrorEnNetCDF = false;
	*nx_nc = (num_volx-1)/npics + 1;
	*ny_nc = (num_voly_total-1)/npics + 1;
	x_grid = (float *) malloc(num_volx*sizeof(float));
	y_grid = (float *) malloc(num_voly_total*sizeof(float));
	x = (float *) malloc((*nx_nc)*sizeof(float));
	y = (float *) malloc((*ny_nc)*sizeof(float));

	for (i=0; i<num_volx; i++)
		x_grid[i] = xmin + (i + 0.5)*ancho_vol;
	for (i=0; i<num_voly_total; i++)
		y_grid[i] = ymin + (i + 0.5)*alto_vol;
	for (i=0; i<(*nx_nc); i++)
		x[i] = xmin + (i*npics + 0.5)*ancho_vol;
	for (i=0; i<(*ny_nc); i++)
		y[i] = ymin + (i*npics + 0.5)*alto_vol;

	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 1, &ncid_eta1, &time_eta1_id, &eta1_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);
	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 2, &ncid_q1x, &time_q1x_id, &q1x_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);
	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 3, &ncid_q1y, &time_q1y_id, &q1y_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);
	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 4, &ncid_eta2, &time_eta2_id, &eta2_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);
	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 5, &ncid_q2x, &time_q2x_id, &q2x_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);
	fgennc(id_hebra, x_grid, y_grid, x, y, nombre_bati, prefijo, 6, &ncid_q2y, &time_q2y_id, &q2y_id, *nx_nc,
		*ny_nc, num_volx, num_voly, num_voly_otros, num_voly_total, xmin, ymin, ancho_vol, alto_vol, tiempo_tot,
		CFL, r, angulo1, angulo2, angulo3, angulo4, mfc, mf0, mfs, vmax1, vmax2, bati);

	free(x_grid);
	free(y_grid);
	free(x);
	free(y);
}

void writerecs(int nx_nc, int ny_nc, int iniy_nc, int ncid, int time_id, int var_id, int paso,
				float tiempo_act, float *var)
{
	int iret;
	float t_act = tiempo_act;
	MPI_Offset num = paso;
	MPI_Offset uno = 1;
	MPI_Offset start[] = {num, iniy_nc, 0};
	MPI_Offset count[] = {1, ny_nc, nx_nc};

	// Guardamos el tiempo
	iret = ncmpi_put_vara_float_all(ncid, time_id, &num, &uno, &t_act);
	check_err(iret);

	// Guardamos la variable var
	iret = ncmpi_put_vara_float_all(ncid, var_id, start, count, var);
	check_err(iret);

	iret = ncmpi_sync(ncid);
	check_err(iret);
}

void writeEta1NC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *eta1)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_eta1, time_eta1_id, eta1_id, num, tiempo_act, eta1);
}

void writeQ1xNC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *q1x)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_q1x, time_q1x_id, q1x_id, num, tiempo_act, q1x);
}

void writeQ1yNC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *q1y)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_q1y, time_q1y_id, q1y_id, num, tiempo_act, q1y);
}

void writeEta2NC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *eta2)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_eta2, time_eta2_id, eta2_id, num, tiempo_act, eta2);
}

void writeQ2xNC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *q2x)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_q2x, time_q2x_id, q2x_id, num, tiempo_act, q2x);
}

void writeQ2yNC(int nx_nc, int ny_nc, int iniy_nc, int num, float tiempo_act, float *q2y)
{
	writerecs(nx_nc, ny_nc, iniy_nc, ncid_q2y, time_q2y_id, q2y_id, num, tiempo_act, q2y);
}

void closeNC(int nx_nc, int ny_nc, int iniy_nc, float *eta1_max)
{
	MPI_Offset start[] = {iniy_nc, 0};
	MPI_Offset count[] = {ny_nc, nx_nc};
	int iret;

	// Guardamos eta1 máxima
	iret = ncmpi_put_vara_float_all(ncid_eta1, eta1_max_id, start, count, eta1_max);
	check_err(iret);
	// Cerramos los ficheros
	iret = ncmpi_close(ncid_eta1);
	check_err(iret);
	iret = ncmpi_close(ncid_q1x);
	check_err(iret);
	iret = ncmpi_close(ncid_q1y);
	check_err(iret);
	iret = ncmpi_close(ncid_eta2);
	check_err(iret);
	iret = ncmpi_close(ncid_q2x);
	check_err(iret);
	iret = ncmpi_close(ncid_q2y);
	check_err(iret);
}

