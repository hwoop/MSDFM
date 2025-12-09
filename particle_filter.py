import numpy as np
from scipy.stats import multivariate_normal, invgauss
from config import Config

class ParticleFilter:
    """
    Particle Filter for joint estimation of degradation rate and system state.
    Implements the algorithm in Table 1 with fuzzy resampling [27].
    """
    
    def __init__(self, params, sensors, initial_data=None):
        """
        Initialize particle filter.
        
        Args:
            params: MSDFM_Parameters object
            sensors: List of sensor names to use
            initial_data: IMPORTANT - First measurement for proper initialization
        """
        self.params = params
        self.sensors = sensors
        self.Ns = Config.NUM_PARTICLES
        
        # Particles: [eta, x]
        # eta: degradation rate (random variable)
        # x: system state (starts at 0)
        self.particles = np.zeros((self.Ns, 2))
        
        # Initialize degradation rate particles from estimated distribution
        # Following Eq 19 and Table 1: "sampled from N(mu_eta, sigma_eta^2)"
        # Use wider distribution for robustness
        self.particles[:, 0] = np.random.normal(
            params.mu_eta, 
            params.sigma_eta * 2.0,  # Wider initial distribution
            self.Ns
        )
        
        # Initial state x_0 = 0 (definition of virtual state, Eq 6)
        self.particles[:, 1] = 0.0
        
        # Initialize weights uniformly
        self.weights = np.ones(self.Ns) / self.Ns
        
        # ================================================================
        # CRITICAL FIX: Initialize particles using first measurement
        # ================================================================
        if initial_data is not None:
            # Infer initial state from first measurement
            x_init_estimates = self._infer_initial_state(initial_data)
            
            # Add diversity around inferred states
            self.particles[:, 1] = x_init_estimates + np.random.normal(
                0, 0.1, self.Ns  # Small perturbation
            )
            self.particles[:, 1] = np.maximum(0, self.particles[:, 1])
            
            # Perform initial update with relaxed covariance
            self.update(initial_data, inflation_factor=10.0)
            self.fuzzy_resampling()
    
    def _infer_initial_state(self, measurement_vector):
        """
        Infer initial state distribution from first measurement.
        Uses inverse measurement function.
        
        Args:
            measurement_vector: First measurement [y_1, ..., y_P]
            
        Returns:
            x_estimates: Array of inferred x values for each particle
        """
        P = len(self.sensors)
        x_estimates = []
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            y_meas = measurement_vector[i]
            
            # Inverse function: x = ((y - b) / a)^(1/c)
            try:
                val = (y_meas - p['b']) / p['a']
                if val > 0:
                    x_est = val ** (1 / p['c'])
                    x_estimates.append(x_est)
            except:
                pass
        
        if len(x_estimates) == 0:
            # Fallback: use small positive values
            return np.random.uniform(0, 0.1, self.Ns)
        
        # Use median of estimates and add particle diversity
        x_median = np.median(x_estimates)
        x_median = np.clip(x_median, 0, 0.5)  # Sanity check
        
        # Generate particles around median
        return np.random.normal(x_median, 0.1, self.Ns)
    
    def predict(self):
        """
        State transition step (Eq 19).
        
        Implements: x_k = x_{k-1} + eta_{k-1} * dt + omega
        where omega ~ N(0, sigma_B^2 * dt)
        """
        eta = self.particles[:, 0]
        x_prev = self.particles[:, 1]
        dt = Config.DT
        
        # State transition noise (Brownian motion increment)
        omega = np.random.normal(0, self.params.sigma_B * np.sqrt(dt), self.Ns)
        
        # Update state
        x_new = x_prev + eta * dt + omega
        
        # Physical constraint: x >= 0 (though rarely violated)
        x_new = np.maximum(x_new, 0.0)
        
        self.particles[:, 1] = x_new
    
    def update(self, measurement_vector, inflation_factor=1.0):
        """
        Weight update using multivariate likelihood (Eq 20).
        
        Args:
            measurement_vector: Array of sensor measurements [y_1, ..., y_P]
            inflation_factor: Covariance inflation (>1 makes it more forgiving)
        """
        P = len(self.sensors)
        x_particles = self.particles[:, 1]
        
        # Construct predicted measurements for each particle
        Y_pred = np.zeros((self.Ns, P))
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            
            # Measurement function: y = a * φ(x) + b
            # where φ(x) = x^c (polynomial function)
            x_clipped = np.clip(x_particles, 0, None)  # Ensure non-negative
            phi_x = x_clipped ** p['c']
            Y_pred[:, i] = p['a'] * phi_x + p['b']
        
        # Observed measurements
        Y_obs = measurement_vector
        
        # Residuals
        residuals = Y_obs - Y_pred  # Shape: (Ns, P)
        
        # Get covariance matrix for selected sensors
        try:
            full_sensors = self.params.sensor_list
            current_indices = [full_sensors.index(s) for s in self.sensors]
            cov = self.params.Cov_matrix[np.ix_(current_indices, current_indices)]
        except (AttributeError, ValueError, IndexError):
            # Fallback if indexing fails
            cov = self.params.Cov_matrix
        
        # Ensure 2D
        if np.ndim(cov) == 0: 
            cov = np.array([[cov]])
        
        # ================================================================
        # CRITICAL FIX: Inflate covariance for robustness
        # ================================================================
        cov = cov * inflation_factor + np.eye(P) * 1e-4
        
        # Calculate likelihoods for all particles
        # p(y_k | x_k^i) from Eq 20
        try:
            likelihoods = multivariate_normal.pdf(
                residuals, 
                mean=np.zeros(P), 
                cov=cov, 
                allow_singular=True
            )
        except Exception as e:
            print(f"Warning: Likelihood calculation failed: {e}")
            # Fallback: uniform weights
            likelihoods = np.ones(self.Ns)
        
        # ================================================================
        # CRITICAL FIX: Prevent complete weight collapse
        # ================================================================
        # Add small uniform component (defensive mixture)
        uniform_weight = 1e-10
        likelihoods = likelihoods + uniform_weight
        
        # Update weights (Eq 20)
        self.weights *= likelihoods
        
        # Prevent numerical underflow
        self.weights += 1e-300
        
        # Normalize
        max_weight = np.max(self.weights)
        if max_weight > 0:
            self.weights /= np.sum(self.weights)
        else:
            # Reset if all weights collapsed
            print("Warning: Complete weight collapse, resetting to uniform")
            self.weights = np.ones(self.Ns) / self.Ns
        
        # Additional check: if N_eff too low after update, add noise
        N_eff = self.effective_sample_size()
        if N_eff < self.Ns * 0.1:  # Less than 10%
            print(f"Warning: Low N_eff={N_eff:.1f} after update, adding noise to particles")
            # Add small random noise to state particles
            self.particles[:, 1] += np.random.normal(0, 0.05, self.Ns)
            self.particles[:, 1] = np.maximum(0, self.particles[:, 1])
    
    def effective_sample_size(self):
        """
        Calculate effective sample size: N_eff = 1 / sum(w_i^2)
        
        This metric indicates particle degeneracy:
        - N_eff ≈ Ns: Good diversity
        - N_eff << Ns: Severe degeneracy (few particles have most weight)
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def fuzzy_resampling(self):
        """
        Fuzzy resampling algorithm from Reference [27].
        
        Process (Section 3.3.1, Table 1):
        1. Systematic resampling to select particles
        2. Add "fuzzing" noise to duplicated particles to maintain diversity
        
        The fuzzing noise prevents particle collapse and improves long-term
        tracking performance.
        """
        Ns = self.Ns
        weights = self.weights
        
        # ====================================================================
        # Step 1: Systematic Resampling
        # ====================================================================
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure exact sum
        
        # Systematic sampling
        u = np.random.random()
        positions = (u + np.arange(Ns)) / Ns
        indexes = np.searchsorted(cumulative_sum, positions)
        
        # Resample particles
        new_particles = self.particles[indexes].copy()
        
        # ====================================================================
        # Step 2: Fuzzing (Add noise to duplicated particles)
        # ====================================================================
        # Identify which particles were selected multiple times
        unique, counts = np.unique(indexes, return_counts=True)
        
        # Calculate fuzzing noise std based on particle variance
        # Following [27]: "noise std = sqrt(variance / Np)"
        var_eta = np.var(self.particles[:, 0])
        
        # Fuzzing noise standard deviation
        # Note: Paper uses variance BEFORE resampling
        sigma_noise = np.sqrt(var_eta / Ns) if var_eta > 0 else 1e-6
        
        # Apply fuzzing to duplicated particles
        for idx, count in zip(unique, counts):
            if count > 1:  # Particle was selected multiple times
                # Find all copies
                mask = (indexes == idx)
                
                # Add Gaussian noise to eta (degradation rate parameter)
                # Only the RATE is fuzzed, not the state x
                noise = np.random.normal(0, sigma_noise, count)
                new_particles[mask, 0] += noise
        
        # Update particles
        self.particles = new_particles
        
        # Reset weights to uniform (standard after resampling)
        self.weights = np.ones(Ns) / Ns
    
    def estimate_rul(self, return_distribution=False):
        """
        RUL prediction following Eq 21.
        
        Args:
            return_distribution: If True, returns (mean, std) of RUL distribution
                               If False, returns only mean RUL
        
        Returns:
            If return_distribution=False: 
                rul_mean (scalar)
            If return_distribution=True:
                (rul_mean, rul_std) tuple
        
        The RUL distribution follows an Inverse Gaussian distribution (Eq 21):
        f(l | x_k, eta_k) = (D - x_k) / sqrt(2π sigma_B^2 l^3) * 
                            exp(-(l * eta_k - (D - x_k))^2 / (2 * sigma_B^2 * l))
        """
        # Use median for robustness (stated in Section 3.3.1)
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        
        D = Config.FAILURE_THRESHOLD
        
        # Check boundary conditions
        if x_hat >= D:
            # Already failed
            return (0.0, 0.0) if return_distribution else 0.0
        
        if eta_hat <= 1e-9:
            # Degradation rate too small, predict very long life
            return (1000.0, 500.0) if return_distribution else 1000.0
        
        # Mean RUL from Inverse Gaussian distribution
        # E[L] = (D - x_hat) / eta_hat
        mean_rul = (D - x_hat) / eta_hat
        
        if not return_distribution:
            return mean_rul
        
        # ====================================================================
        # Variance of RUL (from Inverse Gaussian properties)
        # ====================================================================
        # For Inverse Gaussian IG(μ, λ):
        # - μ = (D - x_hat) / eta_hat
        # - λ = (D - x_hat)^2 / sigma_B^2
        # 
        # Var[L] = μ^3 / λ = (D - x_hat)^3 / (eta_hat^2 * sigma_B^2)
        
        numerator = (D - x_hat) ** 3
        denominator = (eta_hat ** 2) * (self.params.sigma_B ** 2)
        
        if denominator > 0:
            var_rul = numerator / denominator
            std_rul = np.sqrt(max(0, var_rul))
        else:
            std_rul = mean_rul * 0.5  # Fallback: 50% CV
        
        return mean_rul, std_rul
    
    def get_state_estimate(self):
        """
        Get current state estimate (median of particles).
        
        Returns:
            (eta_hat, x_hat): Estimated degradation rate and state
        """
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        return eta_hat, x_hat
    
    def diagnose(self):
        """
        Diagnostic information for debugging.
        """
        N_eff = self.effective_sample_size()
        eta_mean = np.mean(self.particles[:, 0])
        eta_std = np.std(self.particles[:, 0])
        x_mean = np.mean(self.particles[:, 1])
        x_std = np.std(self.particles[:, 1])
        
        print(f"[PF Diagnostics]")
        print(f"  N_eff: {N_eff:.1f} / {self.Ns} ({100*N_eff/self.Ns:.1f}%)")
        print(f"  eta: {eta_mean:.6f} ± {eta_std:.6f}")
        print(f"  x:   {x_mean:.6f} ± {x_std:.6f}")
        
        if N_eff < self.Ns / 3:
            print(f"  ⚠️  Warning: Particle degeneracy detected!")