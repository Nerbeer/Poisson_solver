#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#define EASY 1
#define DEBUG 1
#define SQUARE 1 //SQUARE - включает квадрат для 4х 8x
#define N_ITER 1000
#define M 512 // A - NxMxK;
#define N 512 // A - NxMxK;
#define K 512 // A - NxMxK;

int main(int argc, char **argv)
{
	int rank, size;		// My rank and total # of proc
	int row_rank, col_rank; // My row and column rank
	int coord[2];		// My coords in grid
	int dimension;		// #of dimensions
	int dim[2], period[2], reorder; //variables for grid creation
	int local_N, local_M; // local sizes
	double *Ax, *Bx; // local matrices
	double *Sl, *Sr, *Su, *Sd;
	double hx, hy, hz; // variables 
	double nev = 0.;
	int i_start, i_end, j_start, j_end;
	int iter = 0; // iteration #
	hx = hy = hz = 1;
	i_start = j_start = 0;
	Sr = Sl = Su = Sd = NULL;
	MPI_Comm cart_comm;	// Grid comm
	MPI_Comm col_comm;	// My column comm
	MPI_Comm row_comm;  // My row comm
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	// Forming a grid
	switch (size)
	{
	case 1: // 1x1
		dim[0] = 1; dim[1] = 1;
		dimension = 2;
		break;
	case 2: // 1x2
		dim[0] = 1; dim[1] = 2;
		dimension = 2;
		break;
	case 4: // 2x2 or 4x1 or 1x4
		#ifdef SQUARE
		dim[0] = 2; dim[1] = 2;
		dimension = 2;
		#endif // SQUARE

		#ifndef SQUARE
		dim[0] = 1; dim[1] = 4;
		dimension = 2;
		#endif // ROW	
		break;
	case 8:
		#ifdef SQUARE
		dim[0] = 2; dim[1] = 4;
		dimension = 2;
		#endif // SQUARE

		#ifndef SQUARE
		dim[0] = 1; dim[1] = 8;
		dimension = 2;
		#endif // ROW	
		break;

	case 9:
		#ifdef SQUARE
		dim[0] = 3; dim[1] = 3;
		dimension = 2;
		#endif // SQUARE
		break;
	default:
		printf("Please run with 1, 2, 4 or 8 processes.\n"); fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
		break;
	}

	local_N = i_end = N / dim[0];
	local_M = j_end = K / dim[1];

	// No wrap around
	period[0] = 0; period[1] = 0;
	// Reordering ranks in grid comm
	reorder = 1;

	MPI_Cart_create(MPI_COMM_WORLD, dimension, dim, period, reorder, &cart_comm);

	// Get new rank and coords in new cartesian comm 
	MPI_Comm_rank(cart_comm, &rank);
	MPI_Cart_coords(cart_comm, rank, dimension, coord);

	// Create comms for rows and columns
	int var_coord[2];
	// Column comm
	var_coord[0] = 1; var_coord[1] = 0;
	MPI_Cart_sub(cart_comm, var_coord, &col_comm);
	MPI_Comm_rank(col_comm, &col_rank);
	//Row comm
	var_coord[0] = 0; var_coord[1] = 1;
	MPI_Cart_sub(cart_comm, var_coord, &row_comm);
	MPI_Comm_rank(row_comm, &row_rank);

	if (coord[0] == 0 || coord[0] == (dim[0] - 1))
		local_N += 1;
	else
		local_N += 2;

	if (coord[1] == 0 || coord[1] == (dim[1] - 1))
		local_M += 1;
	else
		local_M += 2;

	#ifdef DEBUG
	for (int i = 0; i < size; i++)
	{
		if (rank == i)
			printf("Rank = %d , Col_rank = %d, Row_rank = %d, coordinates are %d %d local N = %d  local M = %d \n ", rank, col_rank, row_rank, coord[0], coord[1], local_N, local_M); fflush(stdout);

	}
	MPI_Barrier(cart_comm);
	#endif // DEBUG

	//Init local A and B
	Ax = new double[local_N*local_M*K];
	Bx = new double[local_N*local_M*K];


	for (int k = 0; k < K; k++)
	{
		for (int i = 0; i < local_N; i++)
		{
			for (int j = 0; j < local_M; j++)
			{
				Ax[local_N*local_M*k + local_M*i + j] = 1;
				Bx[local_N*local_M*k + local_M*i + j] = 1;
			}
		}
	}

	// Create types for shadows
	MPI_Datatype Type_H, Type_lr, Type_ud;//Type_right,Type_down,Type_up;

	MPI_Type_vector(local_N*K, 1, local_M, MPI_DOUBLE, &Type_H);
	MPI_Type_create_resized(Type_H, 0, sizeof(MPI_DOUBLE) * 2, &Type_lr);
	MPI_Type_commit(&Type_lr);

	MPI_Type_vector(K, local_M, local_M*local_N, MPI_DOUBLE, &Type_H);
	MPI_Type_create_resized(Type_H, 0, local_M*sizeof(MPI_DOUBLE) * 2, &Type_ud);
	MPI_Type_commit(&Type_ud);

	//Need to start iterations
	double fx, fy, fz;
	fx = fy = fz = 0.;
	int ijk = 0;

	double start, finish;
	start = MPI_Wtime();

#ifdef EASY
	while (iter < N_ITER)
	{
		for (int k = 1; k < K - 1; k++)
		{
			for (int i = 1; i < local_N - 1; i++)
			{
				for (int j = 1; j < local_M - 1; j++)
				{
					fx = (Ax[local_N*local_M*k + local_M*(i + 1) + j] + Ax[local_N*local_M*k + local_M*(i - 1) + j]) / (hx*hx);
					fy = (Ax[local_N*local_M*k + local_M*i + j + 1] + Ax[local_N*local_M*k + local_M*i + j - 1]) / (hy*hy);
					fz = (Ax[local_N*local_M*(k + 1) + local_M*i + j] + Ax[local_N*local_M*(k - 1) + local_M*i + j]) / (hz*hz);
					Bx[local_N*local_M*k + local_M*i + j] = (fx + fy + fz) / (2 / (hx*hx) + 2 / (hy*hy) + 2 / (hz*hz));
					// Need to comp nev
				}
			}
		}
		for (int k = 1; k < K - 1; k++)
		{
			for (int i = 1; i < local_N - 1; i++)
			{
				for (int j = 1; j < local_M - 1; j++)
				{
					ijk = local_N*local_M*k + local_M*i + j;
					Ax[ijk] = Bx[ijk];
				}
			}
		}
		// Sending and recv slice in row to row + 1
		if (row_rank != dim[1] - 1)
			MPI_Sendrecv(&Bx[local_M - 2], 1, Type_lr, row_rank + 1, 0, &Ax[local_M - 1], 1, Type_lr, row_rank + 1, 0, row_comm, &status);
		
		// Sending and recv slice in row to row - 1
		if (row_rank != 0)
			MPI_Sendrecv(&Bx[1], 1, Type_lr, row_rank - 1, 0, Ax, 1, Type_lr, row_rank - 1, 0, row_comm, &status);
		
		// Sending and recv slice in column to column + 1
		if (col_rank != dim[0] - 1)
			MPI_Sendrecv(&Bx[(local_N - 2)*local_M], 1, Type_ud, col_rank + 1, 0, &Ax[(local_N - 1)*local_M], 1, Type_ud, col_rank + 1, 0, col_comm, &status);

		// Sending and recv slice in column to column - 1
		if (col_rank != 0)
			MPI_Sendrecv(&Bx[local_M], 1, Type_ud, col_rank - 1, 0, Ax, 1, Type_ud, col_rank - 1, 0, col_comm, &status);

		iter++;
	}
				#endif

#ifndef EASY
	while (iter < N_ITER)
	{
		// recv from i-1 and j-1
		if (coord[0] != 0 || coord[1] != 0)
		{ 
			if (col_rank > 0)
			{ 
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Waiting slice from COLUMN - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Recv(Ax, 1, Type_ud, col_rank - 1, 0, col_comm, &status);
				if (row_rank > 0)
				{
				#ifdef DEBUG
					printf("Rank %d coordinates are %d %d ------- Waiting slice from ROW - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

					MPI_Recv(Ax, 1, Type_lr, row_rank - 1, 0, row_comm, &status);
				}
			}
			else
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Waiting slice from ROW - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Recv(Ax, 1, Type_lr, row_rank - 1, 0, row_comm, &status);
			}
		}

				#ifdef DEBUG
		printf("Rank %d coordinates are %d %d ------- COMPUTING \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

		// computing
		for (int k = 1; k < K - 1; k++)
		{
			for (int i = 1; i < local_N - 1; i++)
			{
				for (int j = 1; j < local_M - 1; j++)
				{
					fx = (Ax[local_N*local_M*k + local_M*(i + 1) + j] + Ax[local_N*local_M*k + local_M*(i - 1) + j]) / (hx*hx);
					fy = (Ax[local_N*local_M*k + local_M*i + j + 1] + Ax[local_N*local_M*k + local_M*i + j - 1]) / (hy*hy);
					fz = (Ax[local_N*local_M*(k + 1) + local_M*i + j] + Ax[local_N*local_M*(k - 1) + local_M*i + j]) / (hz*hz);
					Ax[local_N*local_M*k + local_M*i + j] = (fx + fy + fz) / (2 / (hx*hx) + 2 / (hy*hy) + 2 / (hz*hz));
					// Need to comp nev
				}
			}
		}

		// send to i-1 j-1
		if (coord[0] != 0 || coord[1] != 0)
		{ 
			if (col_rank > 0)
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Sending slice to COLUMN - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Send(&Ax[local_M], 1, Type_ud, col_rank - 1, 0, col_comm);
				if (row_rank > 0)
				{
				#ifdef DEBUG
					printf("Rank %d coordinates are %d %d ------- Sending slice to ROW - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

					MPI_Send(&Ax[1], 1, Type_lr, row_rank - 1, 0, row_comm);
				}
			}
			else
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Sending slice to ROW - 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Send(&Ax[1], 1, Type_lr, row_rank - 1, 0, row_comm);
			}
		}


		
		//send and recv from i+1 j+1
		if (coord[0] != dim[0] - 1 || coord[1] != dim[1] - 1)
		{ 
			// Send
			if (col_rank < dim[0] - 1)
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Sending slice to COLUMN + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Send(&Ax[(local_N - 2)*local_M], 1, Type_ud, col_rank + 1, 0, col_comm);
				if (row_rank < dim[1] - 1)
				{
					#ifdef DEBUG
					printf("Rank %d coordinates are %d %d ------- Sending slice to ROW + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
					#endif // DEBUG

					MPI_Send(&Ax[local_M - 2], 1, Type_lr, row_rank + 1, 0, row_comm);
				}
			}
			else
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Sending slice to ROW + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Send(&Ax[local_M - 2], 1, Type_lr, row_rank + 1, 0, row_comm);
			}
			// Recv
			if (col_rank < dim[0] - 1)
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Waiting slice from COLUMN + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Recv(&Ax[(local_N - 1)*local_M], 1, Type_ud, col_rank + 1, 0, col_comm, &status);
				if (row_rank < dim[1] - 1)
				{
				#ifdef DEBUG
					printf("Rank %d coordinates are %d %d ------- Waiting slice from ROW + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

					MPI_Recv(&Ax[local_M - 1], 1, Type_lr, row_rank + 1, 0, row_comm, &status);
				}
			}
			else
			{
				#ifdef DEBUG
				printf("Rank %d coordinates are %d %d ------- Waiting slice from ROW + 1 \n ", rank, coord[0], coord[1]); fflush(stdout);
				#endif // DEBUG

				MPI_Recv(&Ax[local_M - 1], 1, Type_lr, row_rank + 1, 0, row_comm, &status);
			}
		}	
		//if (rank == 0)
			//printf("Iter %d done \n ",iter); fflush(stdout);
		iter++;
	}
				#endif

	finish = MPI_Wtime();

	double loc_comp_time = finish - start;
	double max_time;
	MPI_Allreduce(&loc_comp_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0)
		printf("NxMxK = %dx%dx%d \nGrid - %dx%d \nLongest time %g\n ",N,M,K,dim[0],dim[1], max_time); fflush(stdout);
	


	MPI_Finalize();
	return 0;
}