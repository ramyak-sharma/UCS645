#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>

struct Particle {
    float x, y;
    float fx, fy;
    float vx, vy;
    float m;
};

float g_eps = 1.0f;
float g_sigma = 1.0f;
float g_dt = 0.01f;

void init_particles_grid(std::vector<Particle>& particles, int N) {
    int side = std::ceil(std::sqrt(N));
    float spacing = 1.5f;
    particles.resize(N);
    int idx = 0;
    for (int x = 0; x < side && idx < N; ++x)
    for (int y = 0; y < side && idx < N; ++y) {
        auto& p = particles[idx++];
        p.x = x * spacing;
        p.y = y * spacing;
        p.vx = p.vy = 0;
        p.fx = p.fy = 0;
        p.m = 1;
    }
}

void update_particles(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        float ax = p.fx / p.m;
        float ay = p.fy / p.m;
        p.vx += ax * g_dt;
        p.vy += ay * g_dt;
        p.x += p.vx * g_dt;
        p.y += p.vy * g_dt;
        if (p.x < 0 || p.x > 100) p.vx *= -1;
        if (p.y < 0 || p.y > 100) p.vy *= -1;
    }
}

void compute_forces_static(std::vector<Particle>& particles, int num_threads) {
    int n = particles.size();
    for (auto& p : particles)
        p.fx = p.fy = 0;

#pragma omp parallel num_threads(num_threads)
    {
        std::vector<float> fx_local(n, 0);
        std::vector<float> fy_local(n, 0);

#pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float dx = particles[i].x - particles[j].x;
                float dy = particles[i].y - particles[j].y;
                float r2 = dx*dx + dy*dy;
                if (r2 < 1e-6f) continue;
                float inv_r2 = 1.0f / r2;
                float sr2 = g_sigma * g_sigma * inv_r2;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float force = 24 * g_eps * inv_r2 * (2 * sr12 - sr6);
                float fx = force * dx;
                float fy = force * dy;
                fx_local[i] += fx;
                fy_local[i] += fy;
                fx_local[j] -= fx;
                fy_local[j] -= fy;
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                particles[i].fx += fx_local[i];
                particles[i].fy += fy_local[i];
            }
        }
    }
}

void compute_forces_dynamic(std::vector<Particle>& particles, int num_threads) {
    int n = particles.size();
    for (auto& p : particles)
        p.fx = p.fy = 0;

#pragma omp parallel num_threads(num_threads)
    {
        std::vector<float> fx_local(n, 0);
        std::vector<float> fy_local(n, 0);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float dx = particles[i].x - particles[j].x;
                float dy = particles[i].y - particles[j].y;
                float r2 = dx*dx + dy*dy;
                if (r2 < 1e-6f) continue;
                float inv_r2 = 1.0f / r2;
                float sr2 = g_sigma * g_sigma * inv_r2;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float force = 24 * g_eps * inv_r2 * (2 * sr12 - sr6);
                float fx = force * dx;
                float fy = force * dy;
                fx_local[i] += fx;
                fy_local[i] += fy;
                fx_local[j] -= fx;
                fy_local[j] -= fy;
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                particles[i].fx += fx_local[i];
                particles[i].fy += fy_local[i];
            }
        }
    }
}

void compute_forces_guided(std::vector<Particle>& particles, int num_threads) {
    int n = particles.size();
    for (auto& p : particles)
        p.fx = p.fy = 0;

#pragma omp parallel num_threads(num_threads)
    {
        std::vector<float> fx_local(n, 0);
        std::vector<float> fy_local(n, 0);

#pragma omp for schedule(guided)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float dx = particles[i].x - particles[j].x;
                float dy = particles[i].y - particles[j].y;
                float r2 = dx*dx + dy*dy;
                if (r2 < 1e-6f) continue;
                float inv_r2 = 1.0f / r2;
                float sr2 = g_sigma * g_sigma * inv_r2;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float force = 24 * g_eps * inv_r2 * (2 * sr12 - sr6);
                float fx = force * dx;
                float fy = force * dy;
                fx_local[i] += fx;
                fy_local[i] += fy;
                fx_local[j] -= fx;
                fy_local[j] -= fy;
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                particles[i].fx += fx_local[i];
                particles[i].fy += fy_local[i];
            }
        }
    }
}

void compute_forces_auto(std::vector<Particle>& particles, int num_threads) {
    int n = particles.size();
    for (auto& p : particles)
        p.fx = p.fy = 0;

#pragma omp parallel num_threads(num_threads)
    {
        std::vector<float> fx_local(n, 0);
        std::vector<float> fy_local(n, 0);

#pragma omp for schedule(auto)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float dx = particles[i].x - particles[j].x;
                float dy = particles[i].y - particles[j].y;
                float r2 = dx*dx + dy*dy;
                if (r2 < 1e-6f) continue;
                float inv_r2 = 1.0f / r2;
                float sr2 = g_sigma * g_sigma * inv_r2;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float force = 24 * g_eps * inv_r2 * (2 * sr12 - sr6);
                float fx = force * dx;
                float fy = force * dy;
                fx_local[i] += fx;
                fy_local[i] += fy;
                fx_local[j] -= fx;
                fy_local[j] -= fy;
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                particles[i].fx += fx_local[i];
                particles[i].fy += fy_local[i];
            }
        }
    }
}

struct BenchmarkResult {
    std::string schedule_type;
    int num_threads;
    int num_particles;
    int iterations;
    double total_time;
    double avg_iteration_time;
    double speedup;
    double efficiency;
    double iterations_per_second;
};

BenchmarkResult run_benchmark(const std::string& schedule_type, 
                               int num_threads, 
                               int num_particles, 
                               int iterations,
                               double baseline_time = 0.0) {
    std::vector<Particle> particles;
    init_particles_grid(particles, num_particles);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        if (schedule_type == "static") compute_forces_static(particles, num_threads);
        else if (schedule_type == "dynamic") compute_forces_dynamic(particles, num_threads);
        else if (schedule_type == "guided") compute_forces_guided(particles, num_threads);
        else if (schedule_type == "auto") compute_forces_auto(particles, num_threads);
        update_particles(particles);
    }
    
    // Actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        if (schedule_type == "static") compute_forces_static(particles, num_threads);
        else if (schedule_type == "dynamic") compute_forces_dynamic(particles, num_threads);
        else if (schedule_type == "guided") compute_forces_guided(particles, num_threads);
        else if (schedule_type == "auto") compute_forces_auto(particles, num_threads);
        update_particles(particles);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    BenchmarkResult result;
    result.schedule_type = schedule_type;
    result.num_threads = num_threads;
    result.num_particles = num_particles;
    result.iterations = iterations;
    result.total_time = duration.count();
    result.avg_iteration_time = duration.count() / iterations;
    result.iterations_per_second = iterations / duration.count();
    
    if (baseline_time > 0.0) {
        result.speedup = baseline_time / duration.count();
        result.efficiency = result.speedup / num_threads * 100.0;
    } else {
        result.speedup = 1.0;
        result.efficiency = 100.0;
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int num_particles = 500;
    int iterations = 1000;
    
    // Parse command line arguments
    if (argc > 1) num_particles = std::atoi(argv[1]);
    if (argc > 2) iterations = std::atoi(argv[2]);
    
    std::cout << "OpenMP Scheduling Performance Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Particles: " << num_particles << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n\n";
    
    std::vector<std::string> schedules = {"static", "dynamic", "guided", "auto"};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    std::vector<BenchmarkResult> all_results;
    
    // Run benchmarks
    std::cout << "Running benchmarks...\n\n";
    
    for (const auto& schedule : schedules) {
        std::cout << "Testing " << schedule << " scheduling:\n";
        
        double baseline_time = 0.0;
        
        for (int num_threads : thread_counts) {
            std::cout << "  " << num_threads << " thread(s)... " << std::flush;
            
            BenchmarkResult result = run_benchmark(schedule, num_threads, num_particles, iterations, baseline_time);
            
            if (num_threads == 1) {
                baseline_time = result.total_time;
            }
            
            all_results.push_back(result);
            
            std::cout << std::fixed << std::setprecision(3) 
                      << result.total_time << "s "
                      << "(" << result.speedup << "x speedup)\n";
        }
        std::cout << "\n";
    }
    
    // Write results to CSV
    std::string filename = "benchmark_results.csv";
    std::ofstream csv(filename);
    
    csv << "Schedule,Threads,Particles,Iterations,TotalTime(s),AvgIterTime(ms),Speedup,Efficiency(%),Iter/sec\n";
    
    for (const auto& result : all_results) {
        csv << result.schedule_type << ","
            << result.num_threads << ","
            << result.num_particles << ","
            << result.iterations << ","
            << std::fixed << std::setprecision(6) << result.total_time << ","
            << std::fixed << std::setprecision(6) << (result.avg_iteration_time * 1000) << ","
            << std::fixed << std::setprecision(3) << result.speedup << ","
            << std::fixed << std::setprecision(2) << result.efficiency << ","
            << std::fixed << std::setprecision(2) << result.iterations_per_second << "\n";
    }
    
    csv.close();
    
    // Print summary table
    std::cout << "Results Summary:\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(10) << "Schedule" 
              << std::setw(10) << "Threads"
              << std::setw(15) << "Time(s)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Efficiency(%)"
              << std::setw(15) << "Iter/sec\n";
    std::cout << "================================================================================\n";
    
    for (const auto& result : all_results) {
        std::cout << std::left << std::setw(10) << result.schedule_type
                  << std::setw(10) << result.num_threads
                  << std::fixed << std::setprecision(3) << std::setw(15) << result.total_time
                  << std::setw(12) << result.speedup
                  << std::fixed << std::setprecision(2) << std::setw(15) << result.efficiency
                  << std::fixed << std::setprecision(2) << std::setw(15) << result.iterations_per_second << "\n";
    }
    
    std::cout << "\nResults saved to: " << filename << "\n";
    
    // Find best performer
    auto best = std::min_element(all_results.begin(), all_results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.total_time < b.total_time;
        });
    
    std::cout << "\nBest Performance: " << best->schedule_type 
              << " with " << best->num_threads << " threads "
              << "(" << std::fixed << std::setprecision(3) << best->total_time << "s)\n";
    
    return 0;
}
