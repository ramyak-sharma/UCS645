cmake_minimum_required(VERSION 3.20)
project(md_sim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/opt/homebrew")
set(SFML_DIR "/opt/homebrew/lib/cmake/SFML")

find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

# Just compile Q1.cpp - no ImGui sources
add_executable(sim Q1.cpp)

target_include_directories(sim BEFORE PRIVATE
    /opt/homebrew/include
)

target_link_libraries(sim PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
)

# OpenMP for macOS
target_compile_options(sim PRIVATE
    -Xpreprocessor -fopenmp
)

target_include_directories(sim PRIVATE
    /opt/homebrew/opt/libomp/include
)

target_link_directories(sim PRIVATE
    /opt/homebrew/opt/libomp/lib
)

target_link_libraries(sim PRIVATE omp)

if(APPLE)
    set_target_properties(sim PROPERTIES
        BUILD_RPATH "/opt/homebrew/lib"
        INSTALL_RPATH "/opt/homebrew/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
