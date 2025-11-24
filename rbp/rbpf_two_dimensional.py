# rbpf_with_your_kf_vectorized_2d_fixed.py
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
import time

start_time = time.time()

# --- LOAD DATA ---
x_data, y_data = [], []
with open('../data/individual/acceleration.csv', 'r') as file:
    reader = csv.reader(file)
    for row in islice(reader, 400):
        try:
            x, y = float(row[0]), float(row[1])
            x_data.append(x)
            y_data.append(y)
        except ValueError:
            continue

measurements = np.stack([x_data, y_data], axis=1)  # shape (N_samples, 2)

# --- KALMAN FILTER DEFAULTS ---
class KalmanDefaults:
    def __init__(self):
        self.dt = 0.018
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * 0.002
        self.R = np.eye(1) * 0.1
        self.P0 = np.eye(2)
        self.x0 = np.array([[0.0], [0.0]])

kf_defaults = KalmanDefaults()

# --- Helper functions ---
def gaussian_logpdf(y, mean, cov):
    y = np.atleast_1d(y)
    d = y.shape[0]
    cov = np.atleast_2d(cov)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov += 1e-8 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
    diff = y - mean
    return -0.5 * (d*np.log(2*np.pi) + logdet + diff.T.dot(cov_inv).dot(diff))

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
    def __init__(self, N_particles, kf_defaults, measurement_dim=2, resample_threshold=0.5):
        self.N = N_particles
        self.kf_def = kf_defaults
        self.measurement_dim = measurement_dim
        self.particles = np.random.normal(1.0, 0.05, (self.N, measurement_dim))
        self.log_weights = np.log(np.ones(self.N)/self.N)
        self.z_means = np.tile(self.kf_def.x0.T, (self.N, measurement_dim, 1))
        self.z_covs = np.tile(self.kf_def.P0, (self.N, measurement_dim, 1, 1))
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        return b_prev + self.b_rw_sigma * np.random.randn(*b_prev.shape)

    def step(self, y):
        y = np.array(y)
        self.particles = self.sample_b(self.particles)
        F, Q, R = self.kf_def.A, self.kf_def.Q, self.kf_def.R

        y_preds = np.zeros((self.N, self.measurement_dim))

        for d in range(self.measurement_dim):
            m_pred = self.z_means[:, d, :] @ F.T
            P_pred = F @ self.z_covs[:, d, :, :] @ F.T + Q
            Hs = self.particles[:, d, None, None] * self.kf_def.C
            y_pred = np.einsum('nij,nj->ni', Hs, m_pred)
            S = np.einsum('nij,njk,nlk->nil', Hs, P_pred, Hs)[:, 0, 0][:, None, None] + R
            S += 1e-8 * np.eye(1)[None, :, :]
            K = np.einsum('nij,njk,nlk->nil', P_pred, Hs.transpose(0, 2, 1), np.linalg.inv(S))
            y_diff = (y[d] - y_pred).reshape(-1, 1, 1)
            m_upd = m_pred[:, :, None] + np.matmul(K, y_diff)
            self.z_means[:, d, :] = m_upd.squeeze(-1)
            P_upd = np.einsum('nij,njk->nik', np.eye(2)[None] - np.matmul(K, Hs), P_pred)
            self.z_covs[:, d, :, :] = P_upd
            y_preds[:, d] = y_pred.squeeze()

        # Weight update with simple 2D covariance
        log_lik = np.zeros(self.N)
        cov_2d = np.array([[R[0,0], 0], [0, R[0,0]]])
        for i in range(self.N):
            log_lik[i] = gaussian_logpdf(y, y_preds[i], cov_2d)

        log_w = self.log_weights + log_lik
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)

        # Resampling
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
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, measurement_dim=2)

z_estimates = []
b_estimates = []
P_traces = []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    best_idx = np.argmax(w)
    P_traces.append(np.trace(rbpf.z_covs[best_idx,0,:,:]))

z_estimates = np.array(z_estimates)
acc_est = z_estimates[:, :, 0]  # x and y position estimates

# --- Plot ---
plt.figure(figsize=(12,10))
Nplot = 400
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} sec")

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

# # --- Separate Plots for X and Y ---
# plt.figure(figsize=(14,8))
# Nplot = 400
# end_time = time.time()
# print(f"Total processing time: {end_time - start_time:.2f} sec")

# # X-axis plot
# plt.subplot(2,1,1)
# plt.plot(measurements[:Nplot,0], label='X measurement')
# plt.plot(acc_est[:Nplot,0], label='RBPF X estimate')
# plt.legend()
# plt.title('X-axis: Measurement vs RBPF Prediction')
# plt.ylabel('Acceleration')

# # Y-axis plot
# plt.subplot(2,1,2)
# plt.plot(measurements[:Nplot,1], label='Y measurement')
# plt.plot(acc_est[:Nplot,1], label='RBPF Y estimate')
# plt.legend()
# plt.title('Y-axis: Measurement vs RBPF Prediction')
# plt.ylabel('Acceleration')
# plt.xlabel('Sample')
# plt.show()
