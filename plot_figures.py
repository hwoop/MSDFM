# plot_figures.py
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime

# === Timestamp 기반 디렉토리 생성 ===
BASE_DIR = 'output'
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # Windows 호환을 위해 콜론 대신 하이픈 사용 권장
OUTPUT_DIR = os.path.join(BASE_DIR, timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Figures will be saved to: {OUTPUT_DIR}")
# ====================================

def _save_and_close(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {filepath}")

def plot_fig4(sim_df_linear, sim_df_nonlinear, informative_idxs=[0,1,2], noninformative_idxs=[3,4,5]):
    """Reproduces Fig 4"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Dataset 1: Linear
    units = sim_df_linear['unit_nr'].unique()[:20] 
    for u in units:
        u_df = sim_df_linear[sim_df_linear['unit_nr'] == u]
        axes[0, 0].plot(u_df['time_cycles'], u_df['state'], 'gray', alpha=0.5)
        axes[0, 1].plot(u_df['time_cycles'], u_df[f's_{informative_idxs[0]+1}'], 'b', alpha=0.5)
        axes[0, 2].plot(u_df['time_cycles'], u_df[f's_{noninformative_idxs[0]+1}'], 'b', alpha=0.5)
        
    axes[0, 0].set_title('(a) State (Linear)')
    axes[0, 0].axhline(1.0, color='r', label='Threshold')
    axes[0, 1].set_title('(b) Informative Sensor')
    axes[0, 2].set_title('(c) Non-informative Sensor')
    
    # Dataset 2: Nonlinear
    units = sim_df_nonlinear['unit_nr'].unique()[:20]
    for u in units:
        u_df = sim_df_nonlinear[sim_df_nonlinear['unit_nr'] == u]
        axes[1, 0].plot(u_df['time_cycles'], u_df['state'], 'gray', alpha=0.5)
        axes[1, 1].plot(u_df['time_cycles'], u_df[f's_{informative_idxs[0]+1}'], 'b', alpha=0.5)
        axes[1, 2].plot(u_df['time_cycles'], u_df[f's_{noninformative_idxs[0]+1}'], 'b', alpha=0.5)
        
    axes[1, 0].set_title('(d) State (Nonlinear)')
    axes[1, 0].axhline(1.0, color='r')
    axes[1, 1].set_title('(e) Informative Sensor')
    axes[1, 2].set_title('(f) Non-informative Sensor')
    
    plt.tight_layout()
    # [Name Change]: Explicit Figure Number
    _save_and_close(fig, "Fig4_Simulated_Degradation_Trajectory.png")

def plot_fig5(estimated_c_linear, estimated_c_nonlinear, true_c):
    """Reproduces Fig 5"""
    fig = plt.figure(figsize=(10, 5))
    sensors = np.arange(1, len(true_c) + 1)
    
    plt.plot(sensors, true_c, 'ko', label='Prespecified value')
    plt.plot(sensors, estimated_c_linear, 'g^', label='Result of linear processes')
    plt.plot(sensors, estimated_c_nonlinear, 'rv', label='Result of nonlinear processes')
    
    plt.xlabel('No. of sensors')
    plt.ylabel('Parameter c_p')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    # [Name Change]: Explicit Figure Number
    _save_and_close(fig, "Fig5_Parameter_Estimation_Cp.png")

def plot_psgs_ware(ranked_sensors, group_scores, sensor_scores, title_suffix="", filename=None):
    """Reproduces Fig 6 & 9"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Individual Scores
    ordered_scores = [sensor_scores[s] for s in ranked_sensors]
    x_indices = np.arange(1, len(ranked_sensors) + 1)
    
    bars = axes[0].bar(x_indices, ordered_scores, width=0.4, color='tab:blue')
    axes[0].set_xlabel('Prioritized order')
    axes[0].set_ylabel('WARE score (%)')
    axes[0].set_title(f'Individual Sensor WARE {title_suffix}')
    axes[0].set_xticks(x_indices)
    
    for bar, sensor_name in zip(bars, ranked_sensors):
        height = bar.get_height()
        s_num = re.findall(r'\d+', sensor_name)[0]
        axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{s_num}',
                     ha='center', va='bottom', fontsize=9)
    axes[0].grid(True, axis='y', linestyle='--')

    # (b) Group Scores
    x = np.arange(1, len(group_scores) + 1)
    axes[1].plot(x, group_scores, 'b.-')
    
    min_idx = np.argmin(group_scores)
    axes[1].plot(min_idx + 1, group_scores[min_idx], 'ro', label='Optimal sensor group')
    axes[1].annotate('Optimal sensor group', 
                     xy=(min_idx+1, group_scores[min_idx]), 
                     xytext=(min_idx+1, group_scores[min_idx]*1.1),
                     arrowprops=dict(facecolor='red', shrink=0.05))
    
    axes[1].set_xlabel('No. of sensor groups')
    axes[1].set_ylabel('WARE score (%)')
    axes[1].set_title(f'Group WARE Scores {title_suffix}')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Use provided filename or fallback
    if filename:
        save_name = filename
    else:
        clean_suffix = re.sub(r'\W+', '_', title_suffix).strip('_')
        save_name = f"Fig_PSGS_{clean_suffix}.png"
        
    _save_and_close(fig, save_name)

def plot_are_comparison(results, percentiles, title_suffix="", filename=None):
    """Reproduces Fig 7 & 10"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    styles = {'No Fusion': 'go-', 'No Selection': 'ms-', 'Proposed': 'rv-'}
    pct_labels = [int(p*100) for p in percentiles]
    
    # (a) Mean
    for method, res in results.items():
        if method in styles:
            axes[0].plot(pct_labels, res['mean'], styles[method], label=method)
    axes[0].set_xlabel('Life percentile (%)')
    axes[0].set_ylabel('Mean of ARE (%)')
    axes[0].set_title(f'Mean of ARE {title_suffix}')
    axes[0].legend()
    axes[0].grid(True)
    
    # (b) Variance
    for method, res in results.items():
        if method in styles:
            axes[1].plot(pct_labels, res['var'], styles[method], label=method)
    axes[1].set_xlabel('Life percentile (%)')
    axes[1].set_ylabel('Variance of ARE')
    axes[1].set_title(f'Variance of ARE {title_suffix}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if filename:
        save_name = filename
    else:
        clean_suffix = re.sub(r'\W+', '_', title_suffix).strip('_')
        save_name = f"Fig_ARE_Comparison_{clean_suffix}.png"
        
    _save_and_close(fig, save_name)