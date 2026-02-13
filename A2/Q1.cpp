#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    long N = 100000000;
    double *A = (double*)malloc(N*sizeof(double));
    double *B = (double*)malloc(N*sizeof(double));
    double *C = (double*)malloc(N*sizeof(double));

	if (!A || !B || !C) {
		printf("Memory allocation failed\n");
		return 1;
	}
	
	int max_threads = omp_get_max_threads();
	double T1 = 0, Tp = 0, Sp = 0, Throughput = 0, Efficiency = 0;

    for (long i=0;i<N;i++) A[i]=B[i]=1.0;

	printf("Threads | Time (seconds) | Speed up (Sp) | Throughput         | Efficiency\n");
	for(int t = 1; t <= max_threads; t++){
		omp_set_num_threads(t);
		double start = omp_get_wtime();
		#pragma omp parallel for
		for (long i=0;i<N;i++)
			C[i] = A[i] + B[i];
		double end = omp_get_wtime();
		if(t == 1) T1 = end - start;
		Tp = end - start;
		Sp = T1/Tp;
		Throughput = N/Tp;
		Efficiency = Sp/t;
		printf(" %d      | %f       | %f      | %f    | %f\n", t, Sp, Tp, Throughput, Efficiency);
		 

	}
	free(A);
	free(B);
	free(C);
    return 0;
}
