# rbpf_with_your_kf_vectorized.py
import numpy as np
import csv
import matplotlib.pyplot as plt

# --- LOAD DATA ---
x_data, y_data, z_data = [], [], []
with open('../data/individual/gravity.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
        z_data.append(float(row[2]))

# vars we're working with:
measurements = np.array(x_data)  # shape (T,)

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
    def __init__(self, N_particles, kf_defaults, resample_threshold=0.5):
        # N: number of particles
        self.N = N_particles
        self.kf_def = kf_defaults
        self.particles = np.random.normal(loc=1.0, scale=0.05, size=self.N)  # b_i
        
        # Initialize equal weights
        self.log_weights = np.log(np.ones(self.N) / self.N)
        
        # --- Vectorized initialization for all KF states ---
        self.z_means = np.tile(self.kf_def.x0.T, (self.N, 1))  # shape (N, 2)
        self.z_covs = np.tile(self.kf_def.P0, (self.N, 1, 1))  # shape (N, 2, 2)
        
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        """Simple Gaussian random walk"""
        return b_prev + self.b_rw_sigma * np.random.randn(*b_prev.shape)

    def step(self, y):
        """One RBPF time step given scalar measurement y"""
        # --- Vectorized particle propagation ---
        self.particles = self.sample_b(self.particles)
        
        # --- KF matrices (shared across particles except H) ---
        F = self.kf_def.A
        Q = self.kf_def.Q
        R = self.kf_def.R

        # --- Vectorized Predict step ---
        m_pred = (self.z_means @ F.T)  # shape (N, 2)
        P_pred = F @ self.z_covs @ F.T + Q  # shape (N, 2, 2)

        # --- Vectorized Update step ---
        Hs = self.particles[:, None, None] * self.kf_def.C  # shape (N, 1, 2)
        y_preds = np.einsum('nij,nj->ni', Hs, m_pred)  # shape (N, 1)
        S = np.einsum('nij,njk,nlk->nil', Hs, P_pred, Hs)[:, 0, 0][:, None, None] + R  # (N, 1, 1)

        # Regularize for stability
        S += 1e-8 * np.eye(1)[None, :, :]
        K = np.einsum('nij,njk,nlk->nil', P_pred, Hs.transpose(0, 2, 1), np.linalg.inv(S))  # (N, 2, 1)
        
        y_diff = (y - y_preds).reshape(-1, 1, 1)  # (N, 1, 1)
        m_upd = m_pred[:, :, None] + np.matmul(K, y_diff)
        m_upd = m_upd.squeeze(-1)
        
        P_upd = np.einsum('nij,njk->nik', np.eye(2)[None] - np.matmul(K, Hs), P_pred)
        
        self.z_means = m_upd
        self.z_covs = P_upd

        # --- Weight update using Gaussian likelihood ---
        log_lik = np.zeros(self.N)
        for i in range(self.N):
            log_lik[i] = gaussian_logpdf(y, y_preds[i], S[i])
        log_w = self.log_weights + log_lik

        # normalize log weights -> weights
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)

        # --- Resampling step ---
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
        b_est = np.sum(w * self.particles)
        z_est = np.sum(self.z_means * w[:, None], axis=0)
        return b_est, z_est, w

# --- Run RBPF on measurements ---
N_particles = 10
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, resample_threshold=0.5)

z_estimates = []
b_estimates = []
P_traces = []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    best_idx = np.argmax(w)
    P_traces.append(np.trace(rbpf.z_covs[best_idx]))

z_estimates = np.array(z_estimates)
acc_est = z_estimates[:, 0]

# ------------------ Plot results ------------------
Nplot = 10 * 50  # first 10 seconds assuming 50 Hz
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(measurements[:Nplot], label='Original X (measurement)')
plt.plot(acc_est[:Nplot], label='RBPF estimated accel')
plt.legend()
plt.ylabel('Value')
plt.title('RBPF: measurement vs estimated acceleration (first {} samples)'.format(Nplot))

plt.subplot(3,1,2)
plt.plot(b_estimates[:Nplot])
plt.ylabel('Estimated measurement scale b')
plt.title('Estimated latent scale factor b (particles)')

plt.subplot(3,1,3)
plt.plot(P_traces[:Nplot])
plt.ylabel('Trace P (best particle)')
plt.title('Trace of error covariance (best-weight particle)')

plt.xlabel('Sample')
plt.show()
