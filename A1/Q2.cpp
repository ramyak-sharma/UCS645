#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int N = 600;

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    double t1 = omp_get_wtime();

    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i][j] += A[i][k] * B[k][j];

    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    std::cout << "Sequential Time = " << seq_time << " seconds\n\n";

    std::cout << "1D Parallel\n";

    for (int threads = 2; threads <= 12; ++threads) {
        omp_set_num_threads(threads);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i][j] = 0.0;

        t1 = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    C[i][j] += A[i][k] * B[k][j];

        t2 = omp_get_wtime();

        double par_time = t2 - t1;
        double speedup = seq_time / par_time;

        std::cout << "Threads = " << threads
                  << "  Time = " << par_time
                  << "  Speedup = " << speedup << "\n";
    }

    std::cout << "\n2D Parallel\n";

    for (int threads = 2; threads <= 12; ++threads) {
        omp_set_num_threads(threads);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i][j] = 0.0;

        t1 = omp_get_wtime();

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];

        t2 = omp_get_wtime();

        double par_time = t2 - t1;
        double speedup = seq_time / par_time;

        std::cout << "Threads = " << threads
                  << "  Time = " << par_time
                  << "  Speedup = " << speedup << "\n";
    }

    return 0;
}
