# simulation_data.py (Restructured Version)
import numpy as np
import pandas as pd
from config import Config

class SimulationGenerator:
    def __init__(self, n_units=100, n_sensors=10, n_informative=3):
        """
        Initialize simulation generator following Section 4.1 of the paper.
        
        Args:
            n_units: Number of units to generate (default: 100)
            n_sensors: Total number of sensors (default: 10)
            n_informative: Number of informative sensors (default: 3)
        """
        self.n_units = n_units
        self.n_sensors = n_sensors
        self.n_informative = n_informative
        
        self.dt = 1.0  # Time interval
        self.threshold = 1.0  # Failure threshold D = 1
        
        # Ground truth measurement parameters (Section 4.1, Step 2)
        # a ~ N(3, 1), b ~ N(3, 1), c ~ U(1, 3)
        np.random.seed(42)  # For reproducibility of true parameters
        self.true_params = {
            'a': np.random.normal(3, 1, n_sensors),
            'b': np.random.normal(3, 1, n_sensors),
            'c': np.random.uniform(1, 3, n_sensors)
        }
        
        # Measurement noise covariance matrix (Section 4.1, Step 2)
        # Noise ~ N(0, 0.02), with correlation structure
        self.cov_matrix = self._generate_correlation_matrix() * 0.02
        
        np.random.seed(None)  # Reset seed for data generation
        
    def _generate_correlation_matrix(self):
        """
        Generate a valid correlation matrix R.
        Following the method in [29]: Numpacharoen & Atsawarungruangkit (2012)
        """
        # Generate random correlation matrix
        A = np.random.rand(self.n_sensors, self.n_sensors)
        Cov = np.dot(A, A.T)  # Ensure positive semi-definite
        
        # Normalize to get correlation matrix
        d = np.sqrt(np.diag(Cov))
        Corr = Cov / np.outer(d, d)
        
        return Corr
    
    def _generate_correlated_unit_data(self, unit_id, mode):
        """
        Generate complete degradation data for one unit with all correlated sensors.
        
        This implements Section 4.1, Step 1 correctly:
        "Generate 10 correlated degradation trajectories for each unit, whose 
        parameters η, α and ω_{k-1} follow the same normal distribution but have 
        different correlations with the actual degradation trajectory."
        
        Key insight: All trajectories share the SAME drift parameters (η or α),
        but have CORRELATED Brownian motion noise terms.
        
        Args:
            unit_id: Unit identifier
            mode: 'linear' or 'nonlinear'
            
        Returns:
            time_steps: Array of time points
            x_true: Ground truth state trajectory
            sensor_states: List of sensor-specific state trajectories
            lifetime: Total lifetime (failure time)
        """
        sigma_B_sq = 6e-4  # Variance of Brownian motion (Section 4.1)
        
        # Initialize storage
        time_steps = []
        x_true_vals = []
        x_sensor_vals = [[] for _ in range(self.n_sensors)]
        
        # Initial states: x(0) = 0 (Eq 6)
        x_true = 0.0
        x_sensors = [0.0] * self.n_sensors
        t = 0
        
        # Sample shared drift parameters (Section 4.1)
        if mode == 'linear':
            # η ~ N(0.05, 1e-6) for dataset 1
            eta = np.random.normal(0.05, 1e-3)  # Using 1e-3 for numerical stability
            drift_params = {'eta': eta}
        elif mode == 'nonlinear':
            # α ~ N(0.003, 4e-7), β = 2 for dataset 2 (Eq 24)
            alpha = np.random.normal(0.003, np.sqrt(4e-7))
            beta = 2
            drift_params = {'alpha': alpha, 'beta': beta}
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Sample correlation coefficients for each sensor (Section 4.1, Step 1)
        # Informative sensors: ρ ~ U(0.8, 0.9)
        # Non-informative sensors: ρ ~ U(0.0, 0.5)
        rhos = []
        for i in range(self.n_sensors):
            if i < self.n_informative:
                rhos.append(np.random.uniform(0.8, 0.9))
            else:
                rhos.append(np.random.uniform(0.0, 0.5))
        
        # Generate trajectories until failure (x_true crosses threshold)
        while x_true < self.threshold:
            # Record current states
            time_steps.append(t)
            x_true_vals.append(x_true)
            for i in range(self.n_sensors):
                x_sensor_vals[i].append(x_sensors[i])
            
            # === CRITICAL: Generate CORRELATED Brownian increments ===
            # Use Cholesky decomposition for correlation
            # z0: for ground truth
            # z1, z2, ..., zN: for N sensors
            # All are independent N(0,1), then we correlate them
            
            z = np.random.standard_normal(self.n_sensors + 1)
            
            # Ground truth Brownian increment
            dW_true = z[0] * np.sqrt(sigma_B_sq * self.dt)
            
            # Calculate drift term (shared by all trajectories)
            if mode == 'linear':
                drift = eta * self.dt
            else:  # nonlinear
                drift = alpha * beta * (t**(beta-1)) * self.dt
            
            # Update ground truth state (Eq 6 or Eq 24)
            x_true = x_true + drift + dW_true
            x_true = max(0.0, x_true)  # Physical constraint: non-negative
            
            # Update each sensor trajectory with CORRELATED noise
            for i in range(self.n_sensors):
                rho = rhos[i]
                
                # Correlated Brownian increment (key formula!)
                # dW_sensor = ρ * dW_true + sqrt(1-ρ²) * dW_independent
                # This ensures Corr(dW_true, dW_sensor) = ρ
                dW_sensor = (rho * z[0] + np.sqrt(1 - rho**2) * z[i+1]) * \
                            np.sqrt(sigma_B_sq * self.dt)
                
                # Sensor trajectory shares SAME drift, CORRELATED noise
                x_sensors[i] = x_sensors[i] + drift + dW_sensor
                x_sensors[i] = max(0.0, x_sensors[i])  # Physical constraint
            
            # Advance time
            t += self.dt
        
        # Convert to numpy arrays
        x_true_array = np.array(x_true_vals)
        sensor_states_list = [np.array(xs) for xs in x_sensor_vals]
        
        return time_steps, x_true_array, sensor_states_list, t
    
    def generate_dataset(self, mode='linear'):
        """
        Generate complete degradation dataset following Section 4.1.
        
        Process:
        1. Generate correlated state trajectories (x_true and x_sensor for each sensor)
        2. Apply measurement function (Eq 7): y = a * x^c + b + v
        3. Add correlated measurement noise
        
        Args:
            mode: 'linear' (Dataset 1) or 'nonlinear' (Dataset 2)
            
        Returns:
            full_df: DataFrame with columns [unit_nr, time_cycles, state, s_1, ..., s_N]
            lifetimes: Array of lifetime for each unit
        """
        print(f"Generating {mode} degradation dataset...")
        
        data_list = []
        lifetimes = []
        
        for unit in range(1, self.n_units + 1):
            if unit % 20 == 0:
                print(f"  Generated {unit}/{self.n_units} units...")
            
            # Step 1: Generate all correlated trajectories at once
            time_steps, x_true, sensor_states, lifetime = \
                self._generate_correlated_unit_data(unit, mode)
            
            n_steps = len(time_steps)
            lifetimes.append(lifetime)
            
            # Step 2: Generate measurement noise (Section 4.1, Step 2)
            # Noise vector V_k ~ N(0, Σ), where Σ = 0.02 * R
            noise = np.random.multivariate_normal(
                mean=np.zeros(self.n_sensors), 
                cov=self.cov_matrix, 
                size=n_steps
            )
            
            # Step 3: Build DataFrame with measurements
            unit_df = pd.DataFrame()
            unit_df['unit_nr'] = [unit] * n_steps
            unit_df['time_cycles'] = time_steps
            unit_df['state'] = x_true  # Ground truth (for visualization only)
            
            # Apply measurement function to each sensor (Eq 7)
            for i in range(self.n_sensors):
                a = self.true_params['a'][i]
                b = self.true_params['b'][i]
                c = self.true_params['c'][i]
                
                # y_p = a_p * φ(x_sensor, c_p) + b_p + v_p
                # where φ(x, c) = x^c (polynomial function)
                x_sensor = sensor_states[i]
                
                # Apply measurement function
                signal = a * (x_sensor ** c) + b + noise[:, i]
                
                unit_df[f's_{i+1}'] = signal
            
            data_list.append(unit_df)
        
        # Concatenate all units
        full_df = pd.concat(data_list, ignore_index=True)
        
        print(f"\n[Normalization] Normalizing sensor signals...")
        sensors = [f's_{i+1}' for i in range(self.n_sensors)]
        
        for sensor in sensors:
            original_mean = full_df[sensor].mean()
            original_std = full_df[sensor].std()
            
            if original_std > 1e-6:
                # Z-score normalization
                full_df[sensor] = (full_df[sensor] - original_mean) / original_std
            else:
                print(f"  Warning: {sensor} has zero variance, skipping normalization")
        
        # Verify normalization
        sensor_means = full_df[sensors].mean().values
        sensor_stds = full_df[sensors].std().values
        print(f"  Post-normalization mean range: [{sensor_means.min():.3f}, {sensor_means.max():.3f}]")
        print(f"  Post-normalization std range: [{sensor_stds.min():.3f}, {sensor_stds.max():.3f}]")
        
        print(f"Dataset generation complete!")
        print(f"  Total units: {self.n_units}")
        print(f"  Average lifetime: {np.mean(lifetimes):.2f} cycles")
        print(f"  Lifetime range: [{np.min(lifetimes):.2f}, {np.max(lifetimes):.2f}]")
        
        return full_df, np.array(lifetimes)
    
    def get_ground_truth_params(self):
        """
        Return ground truth measurement parameters for validation.
        Used in Figure 5 to compare estimated vs. true values.
        """
        return self.true_params
    
    def validate_correlation_structure(self, df, unit_id=1):
        """
        Validation helper: Check if generated data has correct correlation structure.
        
        NOTE: This is for debugging only. The correlation between sensor MEASUREMENTS
        will be different from state correlation due to the nonlinear transformation
        and added measurement noise.
        """
        print(f"\n=== Correlation Validation for Unit {unit_id} ===")
        
        unit_data = df[df['unit_nr'] == unit_id]
        x_true = unit_data['state'].values
        
        print(f"Ground truth state range: [{x_true.min():.3f}, {x_true.max():.3f}]")
        print("\nSensor State Correlations (approximate):")
        
        for i in range(self.n_sensors):
            sensor_col = f's_{i+1}'
            if sensor_col not in unit_data.columns:
                continue
            
            y_sensor = unit_data[sensor_col].values
            
            # Approximate correlation (measurements, not states)
            corr_measurement = np.corrcoef(x_true, y_sensor)[0, 1]
            
            expected_type = "Informative" if i < self.n_informative else "Non-informative"
            expected_range = "(0.8-0.9)" if i < self.n_informative else "(0.0-0.5)"
            
            print(f"  Sensor {i+1:2d} ({expected_type:15s}): "
                  f"corr = {corr_measurement:6.3f} [expected state corr: {expected_range}]")
        
        print("\nNote: Measurement correlation ≠ State correlation due to")
        print("      nonlinear transformation φ(x) = x^c and measurement noise.")
        print("=" * 60)


# ============================================================================
# Validation Functions (Optional - for testing)
# ============================================================================

def test_correlation_generation():
    """
    Test if the correlated trajectory generation is working correctly.
    This verifies the mathematical correctness of the correlation structure.
    """
    print("\n" + "="*70)
    print("CORRELATION STRUCTURE VALIDATION TEST")
    print("="*70)
    
    # Generate small dataset for testing
    gen = SimulationGenerator(n_units=5, n_sensors=4, n_informative=2)
    
    # Test linear mode
    print("\n[Test 1] Linear Degradation Mode")
    df_linear, lifetimes_linear = gen.generate_dataset(mode='linear')
    gen.validate_correlation_structure(df_linear, unit_id=1)
    
    # Test nonlinear mode
    print("\n[Test 2] Nonlinear Degradation Mode")
    df_nonlinear, lifetimes_nonlinear = gen.generate_dataset(mode='nonlinear')
    gen.validate_correlation_structure(df_nonlinear, unit_id=1)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def visualize_trajectories(df, mode_name, n_units_to_plot=10):
    """
    Visualization helper to check generated trajectories.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    units = df['unit_nr'].unique()[:n_units_to_plot]
    
    # Plot ground truth states
    for u in units:
        u_data = df[df['unit_nr'] == u]
        axes[0].plot(u_data['time_cycles'], u_data['state'], alpha=0.6)
    axes[0].axhline(1.0, color='r', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Time (cycles)')
    axes[0].set_ylabel('State')
    axes[0].set_title(f'(a) Ground Truth States ({mode_name})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot informative sensor
    for u in units:
        u_data = df[df['unit_nr'] == u]
        axes[1].plot(u_data['time_cycles'], u_data['s_1'], alpha=0.6)
    axes[1].set_xlabel('Time (cycles)')
    axes[1].set_ylabel('Sensor Value')
    axes[1].set_title('(b) Informative Sensor (s_1)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot non-informative sensor
    for u in units:
        u_data = df[df['unit_nr'] == u]
        axes[2].plot(u_data['time_cycles'], u_data['s_4'], alpha=0.6)
    axes[2].set_xlabel('Time (cycles)')
    axes[2].set_ylabel('Sensor Value')
    axes[2].set_title('(c) Non-informative Sensor (s_4)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/validation_{mode_name}_trajectories.png', dpi=150)
    print(f"Saved visualization: validation_{mode_name}_trajectories.png")
    plt.close()


# ============================================================================
# Main (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Simulation Data Generator - Standalone Test")
    print("="*70)
    
    # Run correlation validation
    test_correlation_generation()
    
    # Generate full datasets and visualize
    print("\n" + "="*70)
    print("GENERATING FULL DATASETS")
    print("="*70)
    
    gen = SimulationGenerator(n_units=100, n_sensors=10, n_informative=3)
    
    # Linear dataset
    df_linear, lifetimes_linear = gen.generate_dataset(mode='linear')
    print(f"\nLinear dataset shape: {df_linear.shape}")
    visualize_trajectories(df_linear, "linear", n_units_to_plot=20)
    
    # Nonlinear dataset
    df_nonlinear, lifetimes_nonlinear = gen.generate_dataset(mode='nonlinear')
    print(f"Nonlinear dataset shape: {df_nonlinear.shape}")
    visualize_trajectories(df_nonlinear, "nonlinear", n_units_to_plot=20)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)