# rbpf_with_your_kf.py
import numpy as np
import csv
import matplotlib.pyplot as plt
from itertools import islice
import time

# time for 350 particles, 500 samples, Nplot 10 * 100: 19.77 seconds
start_time = time.time()  

# --- LOAD DATA ---
x_data, y_data, z_data = [], [], []
with open('../data/individual/gravity.csv', 'r') as file:
    reader = csv.reader(file)
    # Limit to first 500 rows for faster testing
    for row in islice(reader, 500): 
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
        z_data.append(float(row[2]))
# vars we're working with:
measurements = np.array(x_data)  # shape (T,)

# --- KALMAN FILTER ---

# The parameters for the linear state-space model
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

# Kalman funcs: predict and update
def kf_predict(m_prev, P_prev, F, Q):
    m_pred = F.dot(m_prev)
    P_pred = F.dot(P_prev).dot(F.T) + Q
    return m_pred, P_pred

def kf_update(m_pred, P_pred, y, H, R):
    # y and H may be shaped (1,) and (1,2)
    S = H.dot(P_pred).dot(H.T) + R
    # regularize S if numeric issues
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S += 1e-8 * np.eye(S.shape[0])
        S_inv = np.linalg.inv(S)
    K = P_pred.dot(H.T).dot(S_inv)
    y = np.atleast_1d(y)
    y_pred = H.dot(m_pred)
    m_upd = m_pred + K.dot((y - y_pred).reshape(-1,1))
    P_upd = (np.eye(P_pred.shape[0]) - K.dot(H)).dot(P_pred)
    return m_upd, P_upd, y_pred.reshape(-1), S

# Used to evaluate how well each particle's prediction vs actual measurement
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

# The final step in RBPF: makes sure next iteration has good particles
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

# Checks how many independent particles we have
def effective_sample_size(w):
    return 1.0 / np.sum(np.square(w))

# --- RAO-BLACKWELLIZED PARTICLE FILTER ---
class RaoBlackwellizedPF:
    def __init__(self, N_particles, kf_defaults, resample_threshold=0.5):
        # N: numb of particles
        # kf_defaults: Holds the defaul KF matrices
        self.N = N_particles
        self.kf_def = kf_defaults
        self.particles = np.random.normal(loc=1.0, scale=0.05, size=self.N)  # b_i
        
        # Giving each particle equal weights
        self.log_weights = np.log(np.ones(self.N) / self.N)
        # per-particle KF means (2x1) and covariances (2x2)
        self.z_means = [self.kf_def.x0.copy() for _ in range(self.N)]
        self.z_covs  = [self.kf_def.P0.copy() for _ in range(self.N)]
        # process for b: random walk sigma
        self.b_rw_sigma = 0.01
        self.resample_threshold = resample_threshold

    def sample_b(self, b_prev):
        # simple Gaussian random walk
        return b_prev + self.b_rw_sigma * np.random.randn()

    def step(self, y):
        """
        One RBPF time step given scalar measurement y (float or 1-d array)
        """
        log_w = np.zeros(self.N)
        for i in range(self.N):
            # propagate b
            b_prev = self.particles[i]
            b_curr = self.sample_b(b_prev)
            self.particles[i] = b_curr

            # Prepare KF matrices for this particle
            # F: state transition matrix
            # Q: process noise covariance
            # H: measurement matrix
            # R: measurement noise covariance
            F = self.kf_def.A
            Q = self.kf_def.Q
            H = b_curr * self.kf_def.C
            R = self.kf_def.R

            # KF predict step
            m_prev = self.z_means[i]
            P_prev = self.z_covs[i]
            m_pred, P_pred = kf_predict(m_prev, P_prev, F, Q)

            # KF update step
            m_upd, P_upd, y_pred, S = kf_update(m_pred, P_pred, y, H, R)

            # Save the particle's new best guess of z
            self.z_means[i] = m_upd
            self.z_covs[i] = P_upd

            
            log_lik = gaussian_logpdf(np.atleast_1d(y), y_pred, S)
            log_w[i] = self.log_weights[i] + log_lik

        # normalize log weights -> weights
        max_logw = np.max(log_w)
        w = np.exp(log_w - max_logw)
        w /= np.sum(w)
        self.log_weights = np.log(w + 1e-300)

        # resample if ESS low
        ess = effective_sample_size(w)
        if ess < self.resample_threshold * self.N:
            idxs = systematic_resample(w)
            self.particles = self.particles[idxs].copy()
            self.z_means  = [self.z_means[j].copy() for j in idxs]
            self.z_covs   = [self.z_covs[j].copy() for j in idxs]
            self.log_weights = np.log(np.ones(self.N) / self.N)

    def estimate(self):
        w = np.exp(self.log_weights - np.max(self.log_weights))
        w /= np.sum(w)
        # estimate b
        b_est = np.sum(w * self.particles)
        # estimate linear state z (weighted average of means)
        z_stack = np.hstack([m.reshape(-1,1) for m in self.z_means]).T  # N x 2
        z_est = w.dot(z_stack)  # 2-vector
        return b_est, z_est, w

# ------------------ Run RBPF on measurements ------------------
N_particles = 350
rbpf = RaoBlackwellizedPF(N_particles, kf_defaults, resample_threshold=0.5)

z_estimates = []
b_estimates = []
P_traces = []

for t, y in enumerate(measurements):
    rbpf.step(y)
    b_est, z_est, w = rbpf.estimate()
    z_estimates.append(z_est.copy())
    b_estimates.append(b_est)
    # track trace of covariance of best particle for debugging
    best_idx = np.argmax(w)
    P_traces.append(np.trace(rbpf.z_covs[best_idx]))

z_estimates = np.array(z_estimates)  # T x 2

# Extract acceleration estimates
acc_est = z_estimates[:, 0]

# ------------------ Plot results (similar to your original plotting) ------------------
Nplot = 10 * 100  # first 10 seconds assuming 50 Hz
plt.figure(figsize=(12,10))


end_time = time.time()  # stop timer after processing
print(f"\nTotal processing time (including file read): {end_time - start_time:.2f} seconds\n")

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

