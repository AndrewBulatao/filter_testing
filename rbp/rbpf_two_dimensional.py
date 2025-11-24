# rbpf_with_your_kf_vectorized_2d.py
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
import time

start_time = time.time()  # start timer

# --- LOAD DATA ---
x_data, y_data = [], []
with open('../data/individual/acceleration.csv', 'r') as file:
    reader = csv.reader(file)
    for row in islice(reader, 400):  # first 400 rows for testing
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

measurements = np.stack([x_data, y_data], axis=1)  # shape (num_samples, 2) #TODO: make measurements 2D

# --- KALMAN FILTER DEFAULTS ---
class KalmanDefaults:
    def __init__(self):
        self.dt = 0.018
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])  # state transition
        self.C = np.array([[1.0, 0.0]])  # observation matrix
        self.Q = np.identity(2) * 0.002
        self.R = np.identity(1) * 0.1
        self.P0 = np.eye(2)
        self.x0 = np.array([[0.0], [0.0]])

kf_defaults = KalmanDefaults()

# --- Helper functions ---
def gaussian_logpdf(y, mean, cov):
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
    def __init__(self, N_particles, kf_defaults, resample_threshold=0.5, measurement_dim=2):
        self.N = N_particles
        self.kf_def = kf_defaults
        self.measurement_dim = measurement_dim  # TODO: handle multi-dimensional measurements
        self.particles = np.random.normal(loc=1.0, scale=0.05, size=(self.N, measurement_dim))  # b_i for each dim
        
        # Initialize equal weights
        self.log_weights = np.log(np.ones(self.N) / self.N)
        
        # Vectorized initialization for all KF states
        self.z_means = np.tile(self.kf_def.x0.T, (self.N, measurement_dim, 1))  # shape (N, dim, 2)
        self.z_covs = np.tile(self.kf_def.P0, (self.N, measurement_dim, 1, 1))  # shape (N, dim, 2, 2)
        
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        return b_prev + self.b_rw_sigma * np.random.randn(*b_prev.shape)

    def step(self, y):
        """One RBPF step for multi-dimensional measurement"""
        y = np.array(y)
        self.particles = self.sample_b(self.particles)  # propagate particles
        
        F = self.kf_def.A
        Q = self.kf_def.Q
        R = self.kf_def.R

        # Loop over measurement dimensions
        for d in range(self.measurement_dim):
            m_pred = self.z_means[:, d, :] @ F.T  # (N, 2)
            P_pred = F @ self.z_covs[:, d, :, :] @ F.T + Q  # (N,2,2)

            Hs = self.particles[:, d, None, None] * self.kf_def.C  # (N,1,2)
            y_pred = np.einsum('nij,nj->ni', Hs, m_pred)  # (N,1)
            S = np.einsum('nij,njk,nlk->nil', Hs, P_pred, Hs)[:, 0, 0][:, None, None] + R  # (N,1,1)

            # TODO: Joseph form for stability
            S += 1e-8 * np.eye(1)[None, :, :]
            K = np.einsum('nij,njk,nlk->nil', P_pred, Hs.transpose(0, 2, 1), np.linalg.inv(S))

            y_diff = (y[d] - y_pred).reshape(-1,1,1)
            m_upd = m_pred[:, :, None] + np.matmul(K, y_diff)
            m_upd = m_upd.squeeze(-1)

            P_upd = np.einsum('nij,njk->nik', np.eye(2)[None] - np.matmul(K, Hs), P_pred)

            self.z_means[:, d, :] = m_upd
            self.z_covs[:, d, :, :] = P_upd

        # --- Weight update ---
        log_lik = np.zeros(self.N)
        for i in range(self.N):
            y_preds = np.array([np.einsum('ij,j->i', self.kf_def.C * self.particles[i,d], self.z_means[i,d,:]) for d in range(self.measurement_dim)])
            log_lik[i] = gaussian_logpdf(y, y_preds, np.array([[R[0,0],0],[0,R[0,0]]]))  # TODO: simple 2D covariance

        log_w = self.log_weights + log_lik
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)

        # --- Resample if needed ---
        ess = effective_sample_size(w)
        if ess < self.resample_threshold * self.N:
            idxs = systematic_resample(w)
            self.particles = self.particles[idxs].copy()
            self.z_means = self.z_means[idxs].copy()
            self.z_covs = self.z_covs[idxs].copy()
            self.log_weights = np.log(np.ones(self.N) / self.N)

    def estimate(self):
        w = np.exp(self.log_weights - np.max(self.log_weights))
        w /= np.sum(w)
        b_est = np.sum(w[:, None] * self.particles, axis=0)
        z_est = np.sum(w[:, None, None] * self.z_means, axis=0)  # (dim, state)
        return b_est, z_est, w

# --- Run RBPF ---
N_particles = 500
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, resample_threshold=0.5, measurement_dim=2)

z_estimates = []
b_estimates = []
P_traces = []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    best_idx = np.argmax(w)
    P_traces.append(np.trace(rbpf.z_covs[best_idx,0,:,:]))  # just trace of x for simplicity

z_estimates = np.array(z_estimates)
acc_est = z_estimates[:,0,:]  # x and y state estimates

# --- Plot results ---
plt.figure(figsize=(12,10))
Nplot = 400
end_time = time.time()
print(f"Total processing time (including file read): {end_time - start_time:.2f} seconds")

plt.subplot(3,1,1)
plt.plot(measurements[:Nplot,0], label='X measurement')
plt.plot(measurements[:Nplot,1], label='Y measurement')
plt.plot(acc_est[:Nplot,0], label='RBPF X estimate')
plt.plot(acc_est[:Nplot,1], label='RBPF Y estimate')
plt.legend()
plt.title('2D RBPF: Measurements vs Estimates')
plt.ylabel('Acceleration')

plt.subplot(3,1,2)
plt.plot([b[0] for b in b_estimates][:Nplot], label='b X')
plt.plot([b[1] for b in b_estimates][:Nplot], label='b Y')
plt.legend()
plt.title('Estimated latent scale factor b')

plt.subplot(3,1,3)
plt.plot(P_traces[:Nplot])
plt.title('Trace of error covariance for best-weight particle')
plt.xlabel('Sample')
plt.show()
