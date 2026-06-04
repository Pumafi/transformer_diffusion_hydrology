import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class GPWrapper():
    """
    A wrapper to use sklearn GP with our testing functions.
    """
    def __init__(self, kernels, restarts=3):
        self.kernels = kernels
        self.restarts = restarts

    @ignore_warnings(category=ConvergenceWarning)
    def impute(self, observed_data, observed_mask, n_samples=100, timestamps=None):
            B, L, K = observed_data.shape

            batch_samples = np.zeros((B, n_samples, L, K), dtype=np.float32)

            # move to CPU / numpy for sklearn
            observed_mask = observed_mask.cpu().numpy().astype(int)
            observed_data_np = observed_data.cpu().numpy()
            real_data_np = observed_data.cpu().numpy()

            if isinstance(timestamps, torch.Tensor):
                timepoints_np = timestamps.cpu().numpy()
            else:
                timepoints_np = np.array(timestamps)

            for b in range(B):
                if timepoints_np.ndim == 1:
                    X_all = timepoints_np.reshape(-1, 1)
                elif timepoints_np.ndim == 2:
                    X_all = timepoints_np[b].reshape(-1, 1)
                else:
                    X_all = np.arange(L).reshape(-1, 1)

                for k in range(K):
                    train_idx = np.where(observed_mask[b, :, k] == 1)[0]

                    if train_idx.size == 0:
                        # Fallback: predict zeros (or mean of training if available)
                        mean_pred = np.zeros(L, dtype=np.float32)
                        std_pred = np.ones(L, dtype=np.float32) * 1e-6
                        samples_k = np.tile(mean_pred.reshape(1, -1), (n_samples, 1))
                    else:
                        X_train = X_all[train_idx].reshape(-1, 1)  # (n_train, 1)
                        y_train = observed_data_np[b, train_idx, k].ravel()

                        # Fit GP
                        try:
                            gp = GaussianProcessRegressor(kernel=self.kernels, n_restarts_optimizer=self.restarts, normalize_y=True)
                            gp.fit(X_train, y_train)
                            try:
                                y_mean, y_cov = gp.predict(X_all, return_cov=True)
                                jitter = 1e-8
                                success = False
                                max_jitter_iters = 5

                                for jitter_iter in range(max_jitter_iters):
                                    try:
                                        if jitter_iter > 0:
                                            y_cov_j = y_cov + np.eye(L) * jitter
                                        else:
                                            y_cov_j = y_cov
                                        samples_raw = np.random.multivariate_normal(mean=y_mean, cov=y_cov_j, size=n_samples)
                                        success = True
                                        break
                                    except (np.linalg.LinAlgError, ValueError):
                                        jitter *= 10

                                if not success:
                                    y_mean2, y_std = gp.predict(X_all, return_std=True)
                                    y_std = np.clip(y_std, a_min=1e-8, a_max=None)
                                    samples_raw = np.random.normal(loc=y_mean2.reshape(1, -1), scale=y_std.reshape(1, -1), size=(n_samples, L))
                                samples_k = samples_raw.astype(np.float32)  # (nsample, L)

                            except Exception as e_cov:
                                # Fallback to approx using return_std
                                y_mean2, y_std = gp.predict(X_all, return_std=True)
                                y_std = np.clip(y_std, a_min=1e-8, a_max=None)
                                samples_k = np.random.normal(loc=y_mean2.reshape(1, -1), scale=y_std.reshape(1, -1), size=(n_samples, L)).astype(np.float32)

                        except Exception as e_fit:
                            y_all = real_data_np[b, :, k].copy()
                            mask_train = observed_mask[b, :, k].astype(bool)
                            if mask_train.sum() == 0:
                                mean_pred = np.zeros(L, dtype=np.float32)
                            else:
                                # fill na by linear interpolation
                                xp = np.where(mask_train)[0]
                                fp = y_all[mask_train]
                                # if only single point, fill with constant
                                if xp.size == 1:
                                    mean_pred = np.ones(L, dtype=np.float32) * fp[0]
                                else:
                                    mean_pred = np.interp(np.arange(L), xp, fp).astype(np.float32)

                            std_pred = np.ones(L, dtype=np.float32) * 1e-3
                            samples_k = np.tile(mean_pred.reshape(1, -1), (n_samples, 1)) + np.random.normal(scale=std_pred.reshape(1, -1), size=(n_samples, L)).astype(np.float32)
                            
                    batch_samples[b, :, :, k] = samples_k
            output = torch.from_numpy(batch_samples)
            output = output.permute(0, 1, 3, 2)
            return output