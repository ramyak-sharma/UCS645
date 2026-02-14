#!/usr/bin/env python3
"""
Visualization script for OpenMP scheduling benchmark results.
Generates multiple graphs from the CSV output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_execution_time(df, output_dir='plots'):
    """Plot execution time for different schedules and thread counts."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for schedule in df['Schedule'].unique():
        schedule_data = df[df['Schedule'] == schedule]
        ax.plot(schedule_data['Threads'], schedule_data['TotalTime(s)'], 
                marker='o', linewidth=2, label=schedule)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Execution Time vs Thread Count', fontsize=14, fontweight='bold')
    ax.legend(title='Schedule Type')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['Threads'].unique())
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time.png', dpi=300)
    print(f"Saved: {output_dir}/execution_time.png")
    plt.close()

def plot_speedup(df, output_dir='plots'):
    """Plot speedup for different schedules."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for schedule in df['Schedule'].unique():
        schedule_data = df[df['Schedule'] == schedule]
        ax.plot(schedule_data['Threads'], schedule_data['Speedup'], 
                marker='o', linewidth=2, label=schedule)
    
    # Add ideal speedup line
    threads = df['Threads'].unique()
    ax.plot(threads, threads, 'k--', linewidth=2, label='Ideal (Linear)', alpha=0.5)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Thread Count', fontsize=14, fontweight='bold')
    ax.legend(title='Schedule Type')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(threads)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup.png', dpi=300)
    print(f"Saved: {output_dir}/speedup.png")
    plt.close()

def plot_efficiency(df, output_dir='plots'):
    """Plot parallel efficiency for different schedules."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for schedule in df['Schedule'].unique():
        schedule_data = df[df['Schedule'] == schedule]
        ax.plot(schedule_data['Threads'], schedule_data['Efficiency(%)'], 
                marker='o', linewidth=2, label=schedule)
    
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, label='100% Efficiency', alpha=0.5)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency vs Thread Count', fontsize=14, fontweight='bold')
    ax.legend(title='Schedule Type')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['Threads'].unique())
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency.png', dpi=300)
    print(f"Saved: {output_dir}/efficiency.png")
    plt.close()

def plot_throughput(df, output_dir='plots'):
    """Plot iterations per second."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for schedule in df['Schedule'].unique():
        schedule_data = df[df['Schedule'] == schedule]
        ax.plot(schedule_data['Threads'], schedule_data['Iter/sec'], 
                marker='o', linewidth=2, label=schedule)
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Iterations per Second', fontsize=12)
    ax.set_title('Throughput vs Thread Count', fontsize=14, fontweight='bold')
    ax.legend(title='Schedule Type')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['Threads'].unique())
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput.png', dpi=300)
    print(f"Saved: {output_dir}/throughput.png")
    plt.close()

def plot_comparison_heatmap(df, output_dir='plots'):
    """Create a heatmap comparing all schedules and thread counts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Pivot data for heatmap
    pivot = df.pivot(index='Schedule', columns='Threads', values='TotalTime(s)')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Execution Time (s)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Schedule Type', fontsize=12)
    ax.set_title('Execution Time Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap.png', dpi=300)
    print(f"Saved: {output_dir}/heatmap.png")
    plt.close()

def plot_bar_comparison(df, output_dir='plots'):
    """Bar chart comparing schedules at different thread counts."""
    os.makedirs(output_dir, exist_ok=True)
    
    thread_counts = df['Threads'].unique()
    n_threads = len(thread_counts)
    
    fig, axes = plt.subplots(1, n_threads, figsize=(4*n_threads, 6))
    if n_threads == 1:
        axes = [axes]
    
    for idx, threads in enumerate(thread_counts):
        ax = axes[idx]
        thread_data = df[df['Threads'] == threads]
        
        bars = ax.bar(thread_data['Schedule'], thread_data['TotalTime(s)'])
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Schedule Type', fontsize=10)
        ax.set_ylabel('Execution Time (s)', fontsize=10)
        ax.set_title(f'{threads} Thread(s)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bar_comparison.png', dpi=300)
    print(f"Saved: {output_dir}/bar_comparison.png")
    plt.close()

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'benchmark_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Run the benchmark first: ./benchmark")
        sys.exit(1)
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"\nData summary:")
    print(f"  Schedules: {', '.join(df['Schedule'].unique())}")
    print(f"  Thread counts: {', '.join(map(str, df['Threads'].unique()))}")
    print(f"  Particles: {df['Particles'].iloc[0]}")
    print(f"  Iterations: {df['Iterations'].iloc[0]}")
    
    print("\nGenerating plots...")
    plot_execution_time(df)
    plot_speedup(df)
    plot_efficiency(df)
    plot_throughput(df)
    plot_comparison_heatmap(df)
    plot_bar_comparison(df)
    
    print("\nAll plots generated successfully!")
    print("Check the 'plots/' directory for output files.")

if __name__ == '__main__':
    main()
