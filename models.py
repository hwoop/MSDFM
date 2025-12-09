# models.py
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from config import Config

lowess = sm.nonparametric.lowess

class MSDFM_Parameters:
    def __init__(self):
        self.mu_eta = None
        self.sigma_eta = None
        self.sigma_B = None
        self.sensor_params = {} 
        self.Cov_matrix = None 
        self.sensor_list = []

    def estimate_state_params(self, lifetimes):
        N = len(lifetimes)
        D = Config.FAILURE_THRESHOLD
        T_k = lifetimes 

        def neg_log_likelihood(sigma_B_tilde):
            if sigma_B_tilde <= 0: return 1e9
            term1 = np.sum(D / (T_k + sigma_B_tilde))
            term2 = np.sum(T_k / (T_k + sigma_B_tilde))
            mu_eta_hat = term1 / term2 if term2 != 0 else 0
            numerator = (mu_eta_hat * T_k - D)**2
            denominator = T_k**2 + sigma_B_tilde * T_k
            sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
            
            term_log = np.sum(np.log(T_k**2 + sigma_B_tilde * T_k))
            nll = 0.5 * term_log + 0.5 * N * np.log(sigma_eta_sq)
            return nll

        res = minimize_scalar(neg_log_likelihood, bounds=(1e-6, 100), method='bounded')
        sigma_B_tilde_hat = res.x
        
        term1 = np.sum(D / (T_k + sigma_B_tilde_hat))
        term2 = np.sum(T_k / (T_k + sigma_B_tilde_hat))
        self.mu_eta = term1 / term2
        
        numerator = (self.mu_eta * T_k - D)**2
        denominator = T_k**2 + sigma_B_tilde_hat * T_k
        sigma_eta_sq = (1/N) * np.sum(numerator / denominator)
        self.sigma_eta = np.sqrt(sigma_eta_sq)
        self.sigma_B = np.sqrt(sigma_eta_sq * sigma_B_tilde_hat)
        
        print(f"State Params: mu={self.mu_eta:.4f}, sig_eta={self.sigma_eta:.4f}, sig_B={self.sigma_B:.4f}")

    def estimate_measurement_params(self, train_df, sensors_to_use):
        self.sensor_list = sensors_to_use
        
        smoothed_data = {}
        units = train_df['unit_nr'].unique()
        N = len(units)
        
        # 1. Individual Sensor Estimation
        for sensor in sensors_to_use:
            unit_smoothed = []
            for unit in units:
                u_data = train_df[train_df['unit_nr'] == unit]
                y = u_data[sensor].values
                # Check for NaNs in input
                if np.isnan(y).any():
                    y = np.nan_to_num(y)
                    
                t = u_data['time_cycles'].values
                y_smooth = lowess(y, t, frac=Config.SMOOTHING_FRAC, return_sorted=False)
                unit_smoothed.append((y_smooth, t))
            smoothed_data[sensor] = unit_smoothed

            def objective_c(c):
                if c <= 0: return 1e9
                loss = 0
                sum_num_a = 0
                sum_num_b = 0
                denom = 1**c - 0**c 
                if denom == 0: return 1e9

                for y_smooth, t in unit_smoothed:
                    y_last = y_smooth[-1]
                    y_first = y_smooth[0]
                    sum_num_a += (y_last - y_first)
                    sum_num_b += (y_first * 1**c - y_last * 0**c)
                
                a_hat = (1/N) * sum_num_a / denom
                b_hat = (1/N) * sum_num_b / denom
                if abs(a_hat) < 1e-6: return 1e9 
                
                for y_smooth, t in unit_smoothed:
                    val = (y_smooth - b_hat) / a_hat
                    val = np.clip(val, 1e-9, 1.0)
                    x_approx = val**(1/c)
                    term = (x_approx - self.mu_eta * t)**2
                    loss += np.mean(term)
                return loss

            res = minimize_scalar(objective_c, bounds=(0.1, 5.0), method='bounded')
            c_hat = res.x
            
            sum_num_a = 0
            sum_num_b = 0
            for y_smooth, _ in unit_smoothed:
                sum_num_a += (y_smooth[-1] - y_smooth[0])
                sum_num_b += (y_smooth[0]) 
            
            a_hat = (1/N) * sum_num_a
            b_hat = (1/N) * sum_num_b
            
            self.sensor_params[sensor] = {'a': a_hat, 'b': b_hat, 'c': c_hat}

        # 2. Covariance Estimation
        residuals_list = []
        for unit_idx, unit in enumerate(units):
            u_data = train_df[train_df['unit_nr'] == unit]
            K_n = len(u_data)
            unit_resids = np.zeros((K_n, len(sensors_to_use)))
            
            for i, sensor in enumerate(sensors_to_use):
                y_raw = u_data[sensor].values
                y_smooth = smoothed_data[sensor][unit_idx][0]
                unit_resids[:, i] = y_raw - y_smooth
            
            residuals_list.append(unit_resids)
            
        all_residuals = np.vstack(residuals_list)
        
        # === [FIX]: Handle NaNs in residuals ===
        if np.isnan(all_residuals).any():
            print("Warning: NaNs found in residuals. Replacing with 0.")
            all_residuals = np.nan_to_num(all_residuals)
            
        self.Cov_matrix = np.cov(all_residuals, rowvar=False)
        
        # Ensure matrix validity
        if self.Cov_matrix.ndim == 0:
            self.Cov_matrix = self.Cov_matrix.reshape(1, 1)
        if np.isnan(self.Cov_matrix).any():
             print("Warning: NaNs in Covariance Matrix. Replacing with Identity.")
             self.Cov_matrix = np.eye(len(sensors_to_use)) * 0.01