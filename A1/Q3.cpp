
#include <iostream>
#include <omp.h>

int main() {
    long num_steps = 100000000;
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    double t1 = omp_get_wtime();

    for (long i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;

    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    std::cout << "Sequential:\n";
    std::cout << "Pi = " << pi << "\n";
    std::cout << "Time = " << seq_time << "\n\n";

    for (int threads = 2; threads <= 12; ++threads) {
        omp_set_num_threads(threads);

        sum = 0.0;

        t1 = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (long i = 0; i < num_steps; ++i) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        pi = step * sum;

        t2 = omp_get_wtime();

        double par_time = t2 - t1;
        double speedup = seq_time / par_time;

        std::cout << "Threads=" << threads
                  << "  Pi=" << pi
                  << "  Time=" << par_time
                  << "  Speedup=" << speedup << "\n";
    }

    return 0;
}
