// C++ program to demonstrate working of Guassian Elimination
// method
#include "stdio.h"
//#include <math.h>
#include "cmath"
#include <omp.h>
using namespace std;

#define N 1000 // Number of unknowns
#define NUM_THREADS	8
#define NUM_TEST 5
// function to reduce matrix to r.e.f. Returns a value to
// indicate whether matrix is singular or not
int forwardElim(double** mat);

// function to calculate the values of the unknowns
void backSub(double** mat);

// function to get matrix content
void gaussianElimination(double** mat)
{
	/* reduction into r.e.f. */
	int singular_flag = forwardElim(mat);

	/* if matrix is singular */
	if (singular_flag != -1)
	{
		printf("Singular Matrix.\n");

		/* if the RHS of equation corresponding to
		zero row is 0, * system has infinitely
		many solutions, else inconsistent*/
		if (mat[singular_flag][N])
			printf("Inconsistent System.");
		else
			printf("May have infinitely many "
				"solutions.");

		return;
	}

	/* get solution to system and print it using
	backward substitution */

	backSub(mat);

	/*printf("Time measured for forward elimination: %0.7f seconds.\n", endForwardElim);
	printf("Time measured for back subtitution: %0.7f seconds.\n", endBackSub);
	printf("Total execution is %0.7f seconds elapsed with %d threads used.\n\n", endForwardElim + endBackSub, NUM_THREADS);*/
}

// function for elementary operation of swapping two rows
void swap_row(double** mat, int i, int j)
{
	/*printf("Swapped rows %d and %d\n", i, j);*/
	for (int k = 0; k <= N; k++)
	{
		double temp = mat[i][k];
		mat[i][k] = mat[j][k];
		mat[j][k] = temp;
	}
}

// function to print matrix content at any stage
void print(double** mat)
{

	for (int i = 0; i < N; i++, printf("\n"))
		for (int j = 0; j <= N; j++)
			printf("%lf ", mat[i][j]);

	printf("\n");
}

// function to reduce matrix to r.e.f.
int forwardElim(double** mat)
{
	for (int k = 0; k < N - 1; k++)
	{
		// Initialize maximum value and index for pivot
		int i_max = k;
		int v_max = mat[i_max][k];

		/* find greater amplitude for pivot if any */
		for (int i = k + 1; i < N; i++)
			if (abs(mat[i][k]) > v_max)
				v_max = mat[i][k], i_max = i;

		/* if a prinicipal diagonal element is zero,
		* it denotes that matrix is singular, and
		* will lead to a division-by-zero later. */
		if (!mat[k][i_max])
			return k; // Matrix is singular

		/* Swap the greatest value row with current row */ //PIVOTISATIOn
		if (i_max != k)
			swap_row(mat, k, i_max);

		int i, j;
		double f;
#pragma omp parallel for private(i, f, j)
		for (i = k + 1; i < N; i++)
		{
			/* factor f to set current row kth element to 0,
			* and subsequently remaining kth column to 0 */
			f = mat[i][k] / mat[k][k];

			/* subtract fth multiple of corresponding kth
			row element*/
			for (j = k + 1; j <= N; j++)
				mat[i][j] -= mat[k][j] * f;

			/* filling lower triangular matrix with zeros*/ //set first element of second/third row to 0
			mat[i][k] = 0;
		}
		//print(mat);	 //for matrix state
	}
	//print(mat);		 //for matrix state
	return -1;
}

// function to calculate the values of the unknowns
void backSub(double** mat)
{
	double x[N]; // An array to store solution
	int j;
	/* Start calculating from last equation up to the
	first */
	for (int i = N - 1; i >= 0; i--)
	{
		/* start with the RHS of the equation */
		double x_i = mat[i][N];

		/* Initialize j to i+1 since matrix is upper
		triangular*/
#pragma omp parallel for reduction(-:x_i)
		for (j = i + 1; j < N; j++)
		{
			/* subtract all the lhs values
			* except the coefficient of the variable
			* whose value is being calculated */
			x_i -= mat[i][j] * x[j];
		}

		/* divide the RHS by the coefficient of the
		unknown being calculated */
		x[i] = x_i / mat[i][i];
	}

	printf("\nSolution for the system:\n");
	for (int i = 0; i < N; i++)
		printf("%lf\n", x[i]);
}

void displayMatrix(double** mat)
{
	printf("Below is the matrix of linear equation: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			printf("%f ", mat[i][j]);
		}
		printf("\n");
	}

}

void fill2DArray(double** arr, const int x, const int y);

double** create2DArray(const int x, const int y);

// Driver program
int main()
{
	double startTime, endTime, totalTime = 0, avgTime = 0;

	for (int i = 0; i < NUM_TEST;i++)
	{
		srand(124);

		/* input matrix */ // [N + 1] is the RHS

		double** mat = create2DArray(N, N + 1);
		fill2DArray(mat, N, N + 1);

		displayMatrix(mat);

		omp_set_num_threads(NUM_THREADS);

		startTime = omp_get_wtime();

		gaussianElimination(mat);

		endTime = omp_get_wtime() - startTime;
		printf("Test %d: %f seconds\n", i + 1, endTime);
		totalTime += endTime;
	}

	avgTime = totalTime / NUM_TEST;
	printf("\nAverage time measured for gaussian elimination: %f seconds using %d threads.\n", avgTime, NUM_THREADS);
	return 0;
}


double** create2DArray(const int x, const int y)
{

	double** arr_2d = new double* [x];

	double* arr_1d = new double[x * y];

	for (int i = 0; i < x; i++)
	{
		arr_2d[i] = &arr_1d[i * y];
	}
	return arr_2d;
}

void fill2DArray(double** arr, const int x, const int y)
{
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			arr[i][j] = rand() % 5;

			if (i == j)
			{
				arr[i][j] *= -1;
			}
		}
	}
}
