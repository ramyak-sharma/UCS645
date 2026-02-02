#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int n = 1 << 16;
    double a = 2.5;
    std::vector<double> X(n), Y(n);

    for (int i = 0; i < n; ++i) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double t1 = omp_get_wtime();
    for (int i = 0; i < n; ++i)
        X[i] = a * X[i] + Y[i];
    double t2 = omp_get_wtime();

    double seq_time = t2 - t1;
    std::cout << "Sequential Time = " << seq_time << " seconds\n\n";

    for (int threads = 2; threads <= 12; ++threads) {

        omp_set_num_threads(threads);

        for (int i = 0; i < n; ++i)
            X[i] = 1.0;

        t1 = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            X[i] = a * X[i] + Y[i];

        t2 = omp_get_wtime();

        double par_time = t2 - t1;
        double speedup = seq_time / par_time;

        std::cout << "Threads = " << threads
                  << "  Time = " << par_time
                  << "  Speedup = " << speedup << "\n";
    }

    return 0;
}
