# simulation_data.py
import numpy as np
import pandas as pd
from config import Config

class SimulationGenerator:
    def __init__(self, n_units=100, n_sensors=10, n_informative=3):
        self.n_units = n_units
        self.n_sensors = n_sensors
        self.n_informative = n_informative
        
        self.dt = 1.0
        self.threshold = 1.0
        
        # Measurement Params (Ground Truth)
        # a ~ N(3, 1), b ~ N(3, 1), c ~ U(1, 3)
        self.true_params = {
            'a': np.random.normal(3, 1, n_sensors),
            'b': np.random.normal(3, 1, n_sensors),
            'c': np.random.uniform(1, 3, n_sensors)
        }
        
        # Noise Covariance
        self.cov_matrix = self._generate_correlation_matrix() * 0.02
        
    def _generate_correlation_matrix(self):
        A = np.random.rand(self.n_sensors, self.n_sensors)
        Cov = np.dot(A, A.transpose())
        d = np.sqrt(np.diag(Cov))
        Corr = Cov / np.outer(d, d)
        return Corr

    def _generate_trajectory_vals(self, mode, n_steps, eta_base=None, alpha_base=None):
        """
        Helper to generate a single degradation trajectory x(t).
        Used to create independent random trajectories.
        """
        sigma_B_sq = 6e-4
        vals = []
        x = 0
        t = 0
        
        # Assign parameters from same distribution as ground truth 
        if mode == 'linear':
            # Use provided eta or sample new one
            eta = eta_base if eta_base else np.random.normal(0.05, 1e-3)
            for _ in range(n_steps):
                vals.append(x)
                omega = np.random.normal(0, np.sqrt(sigma_B_sq * self.dt))
                x = x + eta * self.dt + omega
                x = max(0.0, x) # Non-negative constraint
                t += self.dt
                
        elif mode == 'nonlinear':
            alpha = alpha_base if alpha_base else np.random.normal(0.003, np.sqrt(4e-7))
            beta = 2
            for _ in range(n_steps):
                vals.append(x)
                omega = np.random.normal(0, np.sqrt(sigma_B_sq * self.dt))
                drift = alpha * beta * (t**(beta-1)) * self.dt
                x = x + drift + omega
                x = max(0.0, x)
                t += self.dt
                
        return np.array(vals)

    def generate_dataset(self, mode='linear'):
        """
        Generates degradation data adhering to Section 4.1.
        Includes correlated trajectories for non-informative sensors.
        """
        data_list = []
        lifetimes = []
        
        # Process Noise for ground truth length determination
        sigma_B_sq = 6e-4
        
        for unit in range(1, self.n_units + 1):
            t = 0
            x = 0
            time_steps = []
            state_vals = []
            
            # 1. Generate Ground Truth State (x_true)
            # We first generate the full path until failure to determine length
            if mode == 'linear':
                eta = np.random.normal(0.05, 1e-3)
                while x < self.threshold:
                    time_steps.append(t)
                    state_vals.append(x)
                    omega = np.random.normal(0, np.sqrt(sigma_B_sq * self.dt))
                    x = x + eta * self.dt + omega
                    x = max(0.0, x)
                    t += self.dt
            elif mode == 'nonlinear':
                alpha = np.random.normal(0.003, np.sqrt(4e-7))
                beta = 2
                while x < self.threshold:
                    time_steps.append(t)
                    state_vals.append(x)
                    omega = np.random.normal(0, np.sqrt(sigma_B_sq * self.dt))
                    drift = alpha * beta * (t**(beta-1)) * self.dt
                    x = x + drift + omega
                    x = max(0.0, x)
                    t += self.dt
            
            n_steps = len(time_steps)
            lifetimes.append(t)
            x_true = np.array(state_vals)
            
            # 2. Generate Correlated Trajectories for Sensors 
            # Each sensor has its own trajectory x_p which is correlated with x_true.
            
            sensor_states = []
            for i in range(self.n_sensors):
                # Sample Correlation Coefficient rho [cite: 358, 362]
                if i < self.n_informative:
                    rho = np.random.uniform(0.8, 0.9)
                else:
                    rho = np.random.uniform(0.0, 0.5)
                
                # Generate an independent trajectory x_indep
                # using the same distribution parameters
                x_indep = self._generate_trajectory_vals(mode, n_steps)
                
                # Combine to create correlated trajectory x_p
                # Formula: x_p = rho * x_true + sqrt(1-rho^2) * x_indep
                # This ensures x_p looks like a degradation curve but has specific correlation.
                x_p = rho * x_true + np.sqrt(1 - rho**2) * x_indep
                x_p = np.maximum(0.0, x_p) # Safety clip
                sensor_states.append(x_p)
                
            # 3. Sensor Mapping (Measurement Function)
            # Noise Matrix (Correlated measurement noise)
            noise = np.random.multivariate_normal(
                np.zeros(self.n_sensors), self.cov_matrix, size=n_steps
            )
            
            unit_df = pd.DataFrame()
            unit_df['unit_nr'] = [unit] * n_steps
            unit_df['time_cycles'] = time_steps
            unit_df['state'] = x_true # Ground truth
            
            for i in range(self.n_sensors):
                a = self.true_params['a'][i]
                b = self.true_params['b'][i]
                c = self.true_params['c'][i]
                
                # Apply Measurement Function to the SENSOR-SPECIFIC state 
                # y = a * x_p^c + b + v
                x_p = sensor_states[i]
                signal = a * (x_p ** c) + b + noise[:, i]
                
                unit_df[f's_{i+1}'] = signal
                
            data_list.append(unit_df)
            
        full_df = pd.concat(data_list, ignore_index=True)
        return full_df, np.array(lifetimes)

    def get_ground_truth_params(self):
        return self.true_params