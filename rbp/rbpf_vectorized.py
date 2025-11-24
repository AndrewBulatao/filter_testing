# rbpf_with_your_kf_vectorized_multidim.py
# This version of the RBPF is fully multidimensional, keeping vectorization and Joseph form fixes

import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
import time

#time for 500 particles, 400 samples, threshold 0.5, Nplot 400: ~4.60 seconds

start_time = time.time()  # start timer

# --- LOAD DATA ---
x_data, y_data, z_data = [], [], []
with open('../data/individual/acceleration.csv', 'r') as file:
    reader = csv.reader(file)
    # Limit to first 400 rows for faster testing
    for row in islice(reader, 400): 
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
        z_data.append(float(row[2]))

measurements = np.stack([x_data, y_data, z_data], axis=1)  # shape (N_samples, 3)

# --- KALMAN FILTER DEFAULTS ---
class KalmanDefaults:
    def __init__(self):
        self.dt = 0.018
        # state: [position; velocity], 2x1 per dimension
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])  # measurement matrix
        self.Q = np.identity(2) * 0.002  # process noise
        self.R = np.identity(1) * 0.1    # measurement noise (1D per measurement channel)
        self.P0 = np.eye(2)
        self.x0 = np.array([[0.0], [0.0]])

kf_defaults = KalmanDefaults()

# --- Helper funcs ---
def gaussian_logpdf(y, mean, cov):
    """Used to evaluate how well each particle's prediction vs actual measurement"""
    y = np.atleast_1d(y)
    d = y.shape[0]
    try:
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov += 1e-8 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    diff = y - mean
    return -0.5 * (d * np.log(2*np.pi) + logdet + diff.T.dot(cov_inv).dot(diff))

def systematic_resample(weights):
    """Final step in RBPF: ensures next iteration has good particles"""
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N
    cumulative_sum = np.cumsum(weights)
    indexes = np.zeros(N, dtype=np.int32)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def effective_sample_size(w):
    """Checks how many independent particles we have"""
    return 1.0 / np.sum(np.square(w))

# --- RAO-BLACKWELLIZED PARTICLE FILTER ---
class RaoBlackwellizedPF:
    def __init__(self, N_particles, kf_defaults, measurement_dim=3, resample_threshold=0.3):
        self.N = N_particles
        self.measurement_dim = measurement_dim
        self.kf_def = kf_defaults
        
        # Initialize particles for nonlinear latent variable b
        self.particles = np.random.normal(loc=1.0, scale=0.05, size=(self.N, measurement_dim))  # b_i for each dimension
        
        # Initialize equal log-weights
        self.log_weights = np.log(np.ones(self.N) / self.N)
        
        # --- Vectorized initialization for all KF states ---
        self.z_means = np.tile(self.kf_def.x0.T, (self.N, self.measurement_dim, 1))  # shape (N, measurement_dim, state_dim)
        self.z_covs = np.tile(self.kf_def.P0, (self.N, self.measurement_dim, 1, 1))   # shape (N, measurement_dim, state_dim, state_dim)
        
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        """Simple Gaussian random walk"""
        return b_prev + self.b_rw_sigma * np.random.randn(*b_prev.shape)

    def step(self, y):
        """One RBPF time step given vector measurement y (shape: measurement_dim)"""
        # --- Vectorized particle propagation ---
        self.particles = self.sample_b(self.particles)
        
        F = self.kf_def.A
        Q = self.kf_def.Q
        R = self.kf_def.R
        
        y = np.atleast_1d(y)
        
        # --- Vectorized Kalman Filter update per particle per measurement dimension ---
        y_preds = np.zeros((self.N, self.measurement_dim))
        S_matrices = np.zeros((self.N, self.measurement_dim, 1, 1))
        Ks = np.zeros((self.N, self.measurement_dim, 2, 1))
        
        for d in range(self.measurement_dim):
            # predict
            m_pred = self.z_means[:, d, :] @ F.T  # shape (N, state_dim)
            P_pred = F @ self.z_covs[:, d, :, :] @ F.T + Q  # shape (N, state_dim, state_dim)
            
            # observation matrix scaled by particle latent variable
            Hs = self.particles[:, d, None, None] * self.kf_def.C  # shape (N, 1, state_dim)
            y_pred = np.einsum('nij,nj->ni', Hs, m_pred)  # shape (N, 1)
            S = np.einsum('nij,njk,nlk->nil', Hs, P_pred, Hs)[:, 0, 0][:, None, None] + R  # shape (N,1,1)
            
            # --- TODO: Joseph form applied ---
            S += 1e-8 * np.eye(1)[None, :, :]
            K = np.einsum('nij,njk,nlk->nil', P_pred, Hs.transpose(0,2,1), np.linalg.inv(S))  # shape (N,2,1)
            

            y_diff = (y[d] - y_pred).reshape(-1, 1, 1)
            m_upd = m_pred[:, :, None] + np.matmul(K, y_diff)
            m_upd = m_upd.squeeze(-1)
            
            P_upd = np.einsum('nij,njk->nik', np.eye(2)[None] - np.matmul(K, Hs), P_pred)
            
            self.z_means[:, d, :] = m_upd
            self.z_covs[:, d, :, :] = P_upd
            
            y_preds[:, d] = y_pred.squeeze()
            S_matrices[:, d, :, :] = S
            Ks[:, d, :, :] = K
        
        # --- Weight update using Gaussian likelihood ---
        log_lik = np.zeros(self.N)
        for i in range(self.N):
            ll = 0
            for d in range(self.measurement_dim):
                ll += gaussian_logpdf(y[d], y_preds[i, d], S_matrices[i, d])
            log_lik[i] = ll
        
        log_w = self.log_weights + log_lik
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)
        
        # --- Resampling ---
        ess = effective_sample_size(w)
        if ess < self.resample_threshold * self.N:
            idxs = systematic_resample(w)
            self.particles = self.particles[idxs].copy()
            self.z_means = self.z_means[idxs].copy()
            self.z_covs = self.z_covs[idxs].copy()
            self.log_weights = np.log(np.ones(self.N)/self.N)

    def estimate(self):
        w = np.exp(self.log_weights - np.max(self.log_weights))
        w /= np.sum(w)
        b_est = np.sum(w[:, None] * self.particles, axis=0)
        z_est = np.sum(w[:, None, None] * self.z_means, axis=0)
        return b_est, z_est, w

# --- Run RBPF ---
N_particles = 350
measurement_dim = 3
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, measurement_dim, resample_threshold=0.3)

z_estimates = []
b_estimates = []
P_traces = []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    best_idx = np.argmax(w)
    # trace of best particle and first dimension (x-axis)
    P_traces.append(np.trace(rbpf.z_covs[best_idx, 0]))

z_estimates = np.array(z_estimates)
acc_est = z_estimates[:, 0]

# --- PLOT RESULTS ---
Nplot = 400
plt.figure(figsize=(12,10))

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")

plt.subplot(3,1,1)
plt.plot(measurements[:Nplot, 0], label='Original X (measurement)')
plt.plot(acc_est[:Nplot], label='RBPF estimated accel')
plt.legend()
plt.ylabel('Value')
plt.title(f'RBPF: measurement vs estimated acceleration (first {Nplot} samples)')

plt.subplot(3,1,2)
plt.plot(np.array(b_estimates)[:Nplot, 0])
plt.ylabel('Estimated measurement scale b')
plt.title('Estimated latent scale factor b (particles, first dimension)')

plt.subplot(3,1,3)
plt.plot(P_traces[:Nplot])
plt.ylabel('Trace P (best particle)')
plt.title('Trace of Kalman error covariance with threshold of 0.5')
plt.xlabel('Sample')
plt.show()
