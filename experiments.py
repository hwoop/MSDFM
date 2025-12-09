# experiments.py
import numpy as np
import pandas as pd
from sensor_selection import run_prediction_for_unit
from config import Config

def run_percentile_experiment(test_df, test_ruls, msdfm, sensor_sets):
    """
    Runs RUL prediction at 10%...90% of lifetime.
    Returns Mean ARE and Variance of ARE for each method.
    
    sensor_sets: {'No Fusion': [s1], 'No Selection': [all], 'Proposed': [best]}
    """
    percentiles = np.arange(0.1, 1.0, 0.1) # 10% to 90%
    results = {method: {'mean': [], 'var': []} for method in sensor_sets.keys()}
    
    test_units = test_df['unit_nr'].unique()
    
    for pct in percentiles:
        print(f"  Running experiment at {pct*100:.0f}% lifetime...")
        
        # Temp results for this percentile
        pct_ares = {method: [] for method in sensor_sets.keys()}
        
        for u_idx, u in enumerate(test_units):
            # === [FIX START]: Robust handling for numpy array vs DataFrame ===
            if isinstance(test_ruls, (np.ndarray, list)):
                true_rul_total = test_ruls[u_idx]
            elif isinstance(test_ruls, pd.DataFrame):
                if 'RUL' in test_ruls.columns:
                    true_rul_total = test_ruls.iloc[u_idx]['RUL']
                else:
                    true_rul_total = test_ruls.iloc[u_idx, 0]
            elif isinstance(test_ruls, pd.Series):
                true_rul_total = test_ruls.iloc[u_idx]
            else:
                raise TypeError(f"Unsupported type for test_ruls: {type(test_ruls)}")

            # Ensure scalar value
            if hasattr(true_rul_total, 'item'):
                true_rul_total = true_rul_total.item()
            # === [FIX END] ===
            
            u_data = test_df[test_df['unit_nr'] == u]
            
            # For percentile cuts, we need the total lifetime.
            total_life = true_rul_total
            cutoff_time = int(total_life * pct)
            
            if cutoff_time < 2: continue # Need at least some data
            
            # Truncate Data
            truncated_df = u_data[u_data['time_cycles'] <= cutoff_time]
            
            # True RUL at cutoff
            actual_rul = total_life - cutoff_time
            
            for method, sensors in sensor_sets.items():
                # Predict
                preds = run_prediction_for_unit(truncated_df, sensors, msdfm)
                pred_rul = preds[-1]
                
                # ARE (Eq 23)
                if actual_rul <= 0:
                     # Avoid division by zero if actual_rul is 0 (unlikely with < 1.0 pct)
                     are = 0.0
                else:
                    are = np.abs(actual_rul - pred_rul) / actual_rul
                    
                pct_ares[method].append(are)
        
        # Aggregate results for this percentile
        for method in sensor_sets.keys():
            # Handle empty lists if filters skipped all units
            if len(pct_ares[method]) > 0:
                results[method]['mean'].append(np.mean(pct_ares[method]))
                results[method]['var'].append(np.var(pct_ares[method]))
            else:
                results[method]['mean'].append(0.0)
                results[method]['var'].append(0.0)
            
    return results, percentiles