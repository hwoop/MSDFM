# particle_filter.py (Strict Adherence Version)
import numpy as np
from scipy.stats import multivariate_normal
from config import Config

class ParticleFilter:
    def __init__(self, params, sensors, initial_data=None):
        self.params = params
        self.sensors = sensors
        self.Ns = Config.NUM_PARTICLES
        
        # Particles: [eta, x]
        self.particles = np.zeros((self.Ns, 2))
        
        # === [STRICT FIX]: Follow Eq (19) text explicitly  ===
        # "sampled from the normal distribution N(mu_eta, sigma_eta^2)"
        # Do NOT use variance inflation (x2.0). Use exact estimated parameters.
        self.particles[:, 0] = np.random.normal(params.mu_eta, params.sigma_eta, self.Ns)
        
        self.particles[:, 1] = 0.0 # Initial state x=0
        
        self.weights = np.ones(self.Ns) / self.Ns
        
    def predict(self):
        """
        State Transition (Eq 19)[cite: 283].
        x_k = x_{k-1} + eta_{k-1} * dt + omega
        """
        eta = self.particles[:, 0]
        x_prev = self.particles[:, 1]
        dt = Config.DT
        
        # State transition noise omega ~ N(0, sigma_B^2 * dt) [cite: 214]
        omega = np.random.normal(0, self.params.sigma_B * np.sqrt(dt), self.Ns)
        
        # Update state x
        x_new = x_prev + eta * dt + omega
        self.particles[:, 1] = x_new
        
    def update(self, measurement_vector):
        """
        Weight Update using Multivariate Likelihood (Eq 20) [cite: 286-288].
        """
        P = len(self.sensors)
        x_particles = self.particles[:, 1]
        
        # Construct predicted Y
        Y_pred = np.zeros((self.Ns, P))
        
        for i, sensor in enumerate(self.sensors):
            p = self.params.sensor_params[sensor]
            # Clip x to [0, inf) is implicitly required for x^c if c is float
            # But strict math allows complex, yet physical degradation is real.
            # We keep clip(0) as physical constraint mentioned in paper definition (x increases from 0 to 1).
            x_clipped = np.clip(x_particles, 0, None) 
            phi_x = x_clipped ** p['c']
            Y_pred[:, i] = p['a'] * phi_x + p['b']
            
        Y_obs = measurement_vector
        residuals = Y_obs - Y_pred 
        
        # Slice Covariance Matrix
        try:
            full_sensors = self.params.sensor_list
            current_indices = [full_sensors.index(s) for s in self.sensors]
            cov = self.params.Cov_matrix[np.ix_(current_indices, current_indices)]
        except (AttributeError, ValueError):
            cov = self.params.Cov_matrix

        if np.ndim(cov) == 0: cov = np.array([[cov]])
        
        # Numerical Stability for Matrix Inversion
        # Adding tiny jitter is a standard numerical implementation detail for cholesky decomposition reliability
        # strictly not in model but required for float64 computation.
        cov = cov + np.eye(P) * 1e-6
        
        likelihoods = multivariate_normal.pdf(residuals, mean=np.zeros(P), cov=cov, allow_singular=True)
        
        self.weights *= likelihoods
        self.weights += 1e-300 
        self.weights /= np.sum(self.weights)
        
    def fuzzy_resampling(self):
        """
        Fuzzy Resampling Algorithm [cite: 792, 1047-1050].
        Strictly follows Reference [27].
        """
        Ns = self.Ns
        weights = self.weights
        
        # 1. Systematic Resampling
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0 
        indexes = np.searchsorted(cumulative_sum, np.random.random(Ns))
        
        new_particles = self.particles[indexes].copy()
        
        # 2. Fuzzing 
        unique, counts = np.unique(indexes, return_counts=True)
        
        # Calculate Variance of eta BEFORE resampling (sigma_{a_{k-1}}^2)
        var_eta = np.var(self.particles[:, 0])
        
        # Noise std = sqrt(variance / Np)
        # No arbitrary lower bound.
        sigma_noise = np.sqrt(var_eta / Ns)
        
        for idx, count in zip(unique, counts):
            if count > 1: # "fuzzy coefficient equals 1 if sampled more than once" 
                mask = (indexes == idx)
                noise = np.random.normal(0, sigma_noise, count)
                # Apply noise to eta (parameter particle)
                new_particles[mask, 0] += noise
        
        self.particles = new_particles
        self.weights = np.ones(Ns) / Ns

    def estimate_rul(self):
        """
        RUL Prediction (Eq 21 approximation).
        The paper derives analytical PDF (Eq 21). 
        However, usually Mean/Median of particles is used for point estimate in experiments.
        Section 3.3.1 says: "Median... are employed as estimation results"[cite: 296].
        """
        eta_hat = np.median(self.particles[:, 0])
        x_hat = np.median(self.particles[:, 1])
        D = Config.FAILURE_THRESHOLD
        
        if x_hat >= D:
            return 0.0
        if eta_hat <= 1e-6: # Prevent division by zero
            return 1000.0 
            
        rul = (D - x_hat) / eta_hat
        return rul