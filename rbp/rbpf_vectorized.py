# rbpf_with_kf_vectorized_multidim_fixed.py
# Multidimensional RBPF with Joseph form, vectorized Kalman update, and fixed log-likelihood calculation

import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
import time

start_time = time.time()  # start timer

# --- LOAD DATA ---
x_data, y_data, z_data = [], [], []
with open('../data/individual/acceleration.csv', 'r') as file:
    reader = csv.reader(file)
    for row in islice(reader, 400): 
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
        z_data.append(float(row[2]))

measurements = np.stack([x_data, y_data, z_data], axis=1)  # shape (N_samples, 3)

# --- KALMAN FILTER DEFAULTS ---
class KalmanDefaults:
    def __init__(self):
        self.dt = 0.018
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.identity(2) * 0.002
        self.R = np.identity(1) * 0.1
        self.P0 = np.eye(2)
        self.x0 = np.array([[0.0], [0.0]])

kf_defaults = KalmanDefaults()

# --- HELPER FUNCTIONS ---
def gaussian_logpdf(y, mean, cov):
    """Multivariate Gaussian log-probability (scalar)"""
    y = np.atleast_1d(y).flatten()
    mean = np.atleast_1d(mean).flatten()
    d = y.shape[0]
    try:
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov += 1e-8 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    diff = y - mean
    val = -0.5 * (d * np.log(2*np.pi) + logdet + diff.T.dot(cov_inv).dot(diff))
    return float(val)  # TODO: ensure scalar output to avoid TypeError

def systematic_resample(weights):
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
    return 1.0 / np.sum(np.square(w))

# --- RAO-BLACKWELLIZED PARTICLE FILTER ---
class RaoBlackwellizedPF:
    def __init__(self, N_particles, kf_defaults, measurement_dim=3, resample_threshold=0.3):
        self.N = N_particles
        self.measurement_dim = measurement_dim
        self.kf_def = kf_defaults
        
        # --- Nonlinear latent variable b for each particle & dimension ---
        self.particles = np.random.normal(loc=1.0, scale=0.05, size=(self.N, measurement_dim))  # b_i
        self.log_weights = np.log(np.ones(self.N) / self.N)
        
        # --- Vectorized Kalman filter initialization ---
        self.z_means = np.tile(self.kf_def.x0.T, (self.N, self.measurement_dim, 1))
        self.z_covs = np.tile(self.kf_def.P0, (self.N, self.measurement_dim, 1, 1))
        
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        return b_prev + self.b_rw_sigma * np.random.randn(*b_prev.shape)

    def step(self, y):
        self.particles = self.sample_b(self.particles)
        F, Q, R = self.kf_def.A, self.kf_def.Q, self.kf_def.R
        y = np.atleast_1d(y)

        y_preds = np.zeros((self.N, self.measurement_dim))
        S_matrices = np.zeros((self.N, self.measurement_dim, 1, 1))
        Ks = np.zeros((self.N, self.measurement_dim, 2, 1))

        for d in range(self.measurement_dim):
            # --- PREDICT ---
            m_pred = self.z_means[:, d, :] @ F.T
            P_pred = F @ self.z_covs[:, d, :, :] @ F.T + Q

            # --- OBSERVATION ---
            Hs = self.particles[:, d, None, None] * self.kf_def.C
            y_pred = np.einsum('nij,nj->ni', Hs, m_pred)
            S = np.einsum('nij,njk,nlk->nil', Hs, P_pred, Hs)[:, 0, 0][:, None, None] + R
            S += 1e-8 * np.eye(1)[None, :, :]

            # --- JOSEPH FORM ---
            K = np.einsum('nij,njk,nlk->nil', P_pred, Hs.transpose(0,2,1), np.linalg.inv(S))
            y_diff = (y[d] - y_pred).reshape(-1,1,1)
            m_upd = m_pred[:, :, None] + np.matmul(K, y_diff)
            m_upd = m_upd.squeeze(-1)
            P_upd = np.einsum('nij,njk->nik', np.eye(2)[None] - np.matmul(K, Hs), P_pred)
            self.z_means[:, d, :] = m_upd
            self.z_covs[:, d, :, :] = P_upd

            y_preds[:, d] = y_pred.squeeze()
            S_matrices[:, d, :, :] = S
            Ks[:, d, :, :] = K

        # --- LOG-LIKELIHOOD ---
        log_lik = np.zeros(self.N)
        for i in range(self.N):
            # TODO: fully 2D Gaussian likelihood with proper scalar output
            cov2d = np.diag([R[0,0]]*self.measurement_dim)
            log_lik[i] = gaussian_logpdf(y, y_preds[i], cov2d)

        log_w = self.log_weights + log_lik
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)

        # --- RESAMPLE ---
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

# --- RUN RBPF ---
N_particles = 350
measurement_dim = 3
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, measurement_dim, resample_threshold=0.3)

z_estimates, b_estimates, P_traces = [], [], []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    best_idx = np.argmax(w)
    P_traces.append(np.trace(rbpf.z_covs[best_idx, 0]))

z_estimates = np.array(z_estimates)
acc_est = z_estimates[:, 0]

# --- PLOT RESULTS ---
Nplot = 400
plt.figure(figsize=(12,10))
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")

plt.subplot(3,1,1)
plt.plot(measurements[:Nplot, 0], label='Original X')
plt.plot(acc_est[:Nplot], label='RBPF estimated X')
plt.legend()
plt.ylabel('Accel')
plt.title('RBPF: measurement vs estimate (X)')

plt.subplot(3,1,2)
plt.plot(np.array(b_estimates)[:Nplot, 0])
plt.ylabel('Latent scale b')
plt.title('Estimated latent scale factor b (X)')

plt.subplot(3,1,3)
plt.plot(P_traces[:Nplot])
plt.ylabel('Trace P')
plt.title('Trace of Kalman error covariance (best particle)')
plt.xlabel('Sample')
plt.show()
