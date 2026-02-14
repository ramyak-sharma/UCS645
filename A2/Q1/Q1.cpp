#include <vector>
#include <cmath>
#include <omp.h>
#include <SFML/Graphics.hpp>
#include <iostream>

struct Particle {
    float x, y;
    float fx, fy;
    float vx, vy;
    float m;
};

float g_eps = 1.0f;
float g_sigma = 1.0f;
float g_dt = 0.01f;
int g_stepsPerFrame = 2;

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

void compute_forces(std::vector<Particle>& particles) {
    int n = particles.size();
    for (auto& p : particles)
        p.fx = p.fy = 0;

#pragma omp parallel
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
                float force =
                    24 * g_eps * inv_r2 * (2 * sr12 - sr6);
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

int main() {
    omp_set_num_threads(8);
    std::vector<Particle> particles;
    init_particles_grid(particles, 500);
    
    sf::RenderWindow window(sf::VideoMode({800u, 800u}), "MD Simulation");
    window.setFramerateLimit(60);
    
    sf::CircleShape circle(3.f);
    circle.setFillColor(sf::Color::White);
    
    float scale = 5.f;
    float offset = 50.f;
    
    std::cout << "Molecular Dynamics Simulation\n";
    std::cout << "Controls:\n";
    std::cout << "  1-5: Change epsilon (0.5, 1.0, 2.0, 3.0, 5.0)\n";
    std::cout << "  +/-: Increase/decrease timestep\n";
    std::cout << "  Space: Pause/unpause\n";
    std::cout << "  ESC: Exit\n\n";
    
    bool paused = false;
    
    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                switch (keyPressed->code) {
                    case sf::Keyboard::Key::Num1:
                        g_eps = 0.5f;
                        std::cout << "Epsilon = " << g_eps << "\n";
                        break;
                    case sf::Keyboard::Key::Num2:
                        g_eps = 1.0f;
                        std::cout << "Epsilon = " << g_eps << "\n";
                        break;
                    case sf::Keyboard::Key::Num3:
                        g_eps = 2.0f;
                        std::cout << "Epsilon = " << g_eps << "\n";
                        break;
                    case sf::Keyboard::Key::Num4:
                        g_eps = 3.0f;
                        std::cout << "Epsilon = " << g_eps << "\n";
                        break;
                    case sf::Keyboard::Key::Num5:
                        g_eps = 5.0f;
                        std::cout << "Epsilon = " << g_eps << "\n";
                        break;
                    case sf::Keyboard::Key::Equal:
                        g_dt *= 1.1f;
                        std::cout << "dt = " << g_dt << "\n";
                        break;
                    case sf::Keyboard::Key::Hyphen:
                        g_dt *= 0.9f;
                        std::cout << "dt = " << g_dt << "\n";
                        break;
                    case sf::Keyboard::Key::Space:
                        paused = !paused;
                        std::cout << (paused ? "Paused\n" : "Running\n");
                        break;
                    case sf::Keyboard::Key::Escape:
                        window.close();
                        break;
                    default:
                        break;
                }
            }
        }
        
        if (!paused) {
            for(int s = 0; s < g_stepsPerFrame; s++) {
                compute_forces(particles);
                update_particles(particles);
            }
        }
        
        window.clear();
        for (auto& p : particles) {
            circle.setPosition(sf::Vector2f(
                p.x * scale + offset,
                p.y * scale + offset
            ));
            window.draw(circle);
        }
        window.display();
    }
    
    return 0;
}
