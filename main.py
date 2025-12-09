# main.py
import numpy as np
import pandas as pd
from config import Config
from data_loader import load_data, get_lifetimes
from models import MSDFM_Parameters
from sensor_selection import psgs_algorithm, calculate_ware, run_prediction_for_unit
from simulation_data import SimulationGenerator
from experiments import run_percentile_experiment
from plot_figures import plot_fig4, plot_fig5, plot_psgs_ware, plot_are_comparison, plot_fig8

def run_pipeline(name, train_df, test_df, test_ruls, lifetimes, fig_psgs_name=None, fig_are_name=None):
    print(f"\n=== Running Pipeline for {name} ===")
    
    # 1. Parameter Estimation
    print("Estimating Parameters...")
    msdfm = MSDFM_Parameters()
    msdfm.estimate_state_params(lifetimes)
    sensors = [c for c in train_df.columns if c.startswith('s_')]
    msdfm.estimate_measurement_params(train_df, sensors)
    estimated_c = [msdfm.sensor_params[s]['c'] for s in sensors]
    
    # 2. PSGS
    print("Running PSGS...")
    best_sensors, ranked_order, group_scores, sensor_scores = psgs_algorithm(train_df, lifetimes, msdfm)
    print(f"Optimal Group: {best_sensors}")
    
    # Plot PSGS (Fig 6 / 9)
    # [FIX]: Pass specific filename for Figure 6 or 9
    plot_psgs_ware(ranked_order, group_scores, sensor_scores, title_suffix=f"({name})", filename=fig_psgs_name)
    
    # 3. RUL Experiments
    print("Running Percentile Experiments...")
    sensor_sets = {
        'No Fusion': [ranked_order[0]],
        'No Selection': sensors,
        'Proposed': best_sensors
    }
    
    results, percentiles = run_percentile_experiment(test_df, test_ruls, msdfm, sensor_sets)
    
    # Plot ARE (Fig 7 / 10)
    # [FIX]: Pass specific filename for Figure 7 or 10
    plot_are_comparison(results, percentiles, title_suffix=f"({name})", filename=fig_are_name)
    
    return estimated_c

def main():
    # --- PART 1: Simulation Study ---
    print("Generating Simulation Data...")
    gen = SimulationGenerator(n_units=100, n_sensors=10)
    
    # Linear Dataset
    sim_lin_train, sim_lin_lifetimes = gen.generate_dataset(mode='linear')
    units = sim_lin_train['unit_nr'].unique()
    train_units = units[:90]
    test_units = units[90:]
    lin_train_df = sim_lin_train[sim_lin_train['unit_nr'].isin(train_units)]
    lin_test_df = sim_lin_train[sim_lin_train['unit_nr'].isin(test_units)]
    lin_test_ruls = sim_lin_lifetimes[90:]
    lin_train_lifetimes = sim_lin_lifetimes[:90]
    
    # Nonlinear Dataset
    sim_non_train, sim_non_lifetimes = gen.generate_dataset(mode='nonlinear')
    non_train_df = sim_non_train[sim_non_train['unit_nr'].isin(train_units)]
    non_test_df = sim_non_train[sim_non_train['unit_nr'].isin(test_units)]
    non_test_ruls = sim_non_lifetimes[90:]
    non_train_lifetimes = sim_non_lifetimes[:90]
    
    # [Fig 4]: Generated automatically in function
    plot_fig4(lin_train_df, non_train_df)
    
    # Run Pipeline for Linear (Fig 6 & 7 related)
    est_c_lin = run_pipeline("Sim Linear", lin_train_df, lin_test_df, lin_test_ruls, lin_train_lifetimes,
                             fig_psgs_name="Fig6_Sim_Linear_Sensor_Selection.png",
                             fig_are_name="Fig7_Sim_Linear_RUL_Performance.png")
    
    # Run Pipeline for Nonlinear (Fig 6 & 7 related)
    est_c_non = run_pipeline("Sim Nonlinear", non_train_df, non_test_df, non_test_ruls, non_train_lifetimes,
                             fig_psgs_name="Fig6_Sim_Nonlinear_Sensor_Selection.png",
                             fig_are_name="Fig7_Sim_Nonlinear_RUL_Performance.png")
    
    # [Fig 5]
    plot_fig5(est_c_lin, est_c_non, gen.true_params['c'])
    
    # --- PART 2: NASA Experiment ---
    print("\nLoading NASA Data...")
    try:
        nasa_train, nasa_test, nasa_rul = load_data()
    except:
        return

    # NASA Validation Split
    units = nasa_train['unit_nr'].unique()
    split_idx = int(len(units) * 0.8)
    train_units = units[:split_idx]
    val_units = units[split_idx:]
    
    nasa_train_split = nasa_train[nasa_train['unit_nr'].isin(train_units)]
    nasa_val_split = nasa_train[nasa_train['unit_nr'].isin(val_units)]
    
    lifetimes = get_lifetimes(nasa_train)
    train_lifetimes = lifetimes[:split_idx]
    val_lifetimes = lifetimes[split_idx:]
    
    print("Generating Figure 8...")
    plot_fig8(nasa_train)  # 전체 Training set (100 units) 시각화
    
    # Run Pipeline for NASA (Fig 9 & 10)
    run_pipeline("NASA FD001", nasa_train_split, nasa_val_split, val_lifetimes, train_lifetimes,
                 fig_psgs_name="Fig9_NASA_Sensor_Selection.png",
                 fig_are_name="Fig10_NASA_RUL_Performance.png")

if __name__ == "__main__":
    main()