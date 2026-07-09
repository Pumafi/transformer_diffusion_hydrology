import g2s
import torch
import numpy as np
import pandas as pd


class MPSWrapper:

    def __init__(self, training_data, parameters_name, indices_step1,
                 covariates=None, covariate_names=None,
                 bool_include_covariates=True):

        self.train_data = training_data
        self.parameters_name = parameters_name
        self.indices_step1 = indices_step1
        self.bool_include_covariates = bool_include_covariates

        if bool_include_covariates and (covariates is None or covariate_names is None):
            raise ValueError("Covariates must be provided if bool_include_covariates is True")

        elif bool_include_covariates and covariates is not None and covariate_names is not None:
            covariates_step1, covariates_step2 = self.handle_covariates(
                np.expand_dims(covariates, axis=0)
            )

            self.training_covariates_step1 = covariates_step1
            self.training_covariates_step2 = covariates_step2
            self.covariate_names = covariate_names

        else:
            self.training_covariates_step1 = None
            self.training_covariates_step2 = None
            self.covariate_names = None

        self.nb_stations = len(indices_step1)
        self.nb_neighboors = 12

    def handle_covariates(self, covariates):

        covariates_step1 = covariates[:, :, [32, 38, 68]]

        sum_32_38 = np.expand_dims(
            covariates_step1[:, :, 0] + covariates_step1[:, :, 1],
            axis=-1
        )

        covariates_step1 = np.squeeze(np.concatenate((
            sum_32_38,
            np.expand_dims(covariates_step1[:, :, 2], axis=-1)
        ), axis=-1))

        covariates_step2 = np.expand_dims(
            np.squeeze(covariates[:, :, 68]),
            axis=-1
        )

        return covariates_step1, covariates_step2

    def impute(self, observed_data, observed_mask,
               covariates=None, n_samples=100, timestamps=None):

        B, L, K = observed_data.shape

        observed_mask = observed_mask.cpu().numpy().astype(int)

        observed_data_np = observed_data.cpu().numpy()
        observed_data_np[observed_mask == 0] = np.nan

        covariates = covariates.cpu().numpy() if covariates is not None else None

        if covariates is not None:
            test_covariates_step1, test_covariates_step2 = \
                self.handle_covariates(covariates)

        output = np.full((B, L, K, n_samples), 0.0, dtype=np.float32)

        for i in range(B):

            # =========================
            # STEP 1
            # =========================

            if not self.bool_include_covariates:

                M_data_s1 = self.train_data[:, self.indices_step1]
                M_cond_s1 = observed_data_np[i][:, self.indices_step1]

            else:

                M_data_s1 = np.hstack((
                    self.training_covariates_step1,
                    self.train_data[:, self.indices_step1].squeeze()
                ))

                M_cond_s1 = np.hstack((
                    test_covariates_step1[i],
                    observed_data_np[i][:, self.indices_step1].squeeze()
                ))

            M_TI_step1 = M_data_s1

            M_Simul_step1 = np.zeros(
                (*M_cond_s1.shape, n_samples),
                dtype=np.float32
            )

            V_Data_Type = np.zeros(M_TI_step1.shape[1])

            kernel_step1 = np.ones(
                (self.nb_neighboors, len(V_Data_Type))
            )
            
            if self.bool_include_covariates:

                n_covariates = self.training_covariates_step1.shape[1]
                kernel_step1[:, :n_covariates] *= 1 / n_covariates

            # 1. Pre-define the base arguments to avoid duplicating code
            g2s_base_args = [
                '-a', 'qs',
                '-ti', M_TI_step1,
                '-di', M_cond_s1,
                '-dt', V_Data_Type,
                '-k', 1.2,
                '-n', self.nb_neighboors,
                '-j', 0.5,
                '-W_CUDA', 0,
                '-silent'
            ]
            
            if self.bool_include_covariates:
                g2s_base_args.extend(['-ki', kernel_step1])
            
            # 2. Initialize tracking variables
            my_realiz = 0
            submitted_ids = []
            
            # --- PHASE 1: Initial Queue Fill ---
            # Fire off the first batch of submissions up to n_samples
            for _ in range(n_samples):
                job_id = g2s(*g2s_base_args, '-submitOnly')
                submitted_ids.append(job_id)
            
            # --- PHASE 2: Asynchronous Wait & Replenish Loop ---
            # Process jobs as they finish, and submit replacements if any fail validation
            while my_realiz < n_samples:
                # Get the oldest submitted job ID
                current_id = submitted_ids.pop(0)
                
                # Download the data (blocking step, but minimized overhead due to queueing)
                simulation, *_ = g2s('-silent', '-waitAndDownload', current_id)
                
                # Validate the simulation results
                if np.sum(simulation) != 0:
                    simulation = np.nan_to_num(simulation, nan=0.0)
                    M_Simul_step1[:, :, my_realiz] = simulation
                    my_realiz += 1
                else:
                    # Rejection handling: Submit a brand new job to keep the pipeline moving
                    print(f"Simulation rejected. Replenishing queue...")
                    new_id = g2s(*g2s_base_args, '-submitOnly')
                    submitted_ids.append(new_id)

            # =========================
            # STEP 2
            # =========================

            M_Simul_step2_total = np.zeros((L, K, n_samples))
            threshold_frequency = 0.6

            V_stationNames = np.array([
                row.split("_")[0]
                for row in self.parameters_name.squeeze()
            ])

            for my_station, water_idx in enumerate(self.indices_step1):

                station_name = np.array(
                    self.parameters_name[water_idx].split("_")[0]
                )

                inds_var_ini = np.where(
                    V_stationNames == station_name
                )[0]

                data_subset = self.train_data[:, inds_var_ini]

                V_freq_nan = (
                    np.sum(np.isnan(data_subset), axis=0)
                    / len(self.train_data)
                )

                inds_var = inds_var_ini[
                    V_freq_nan < threshold_frequency
                ]

                inds_var_no_wq = inds_var[
                    inds_var != water_idx
                ]

                n_cols_station = len(inds_var_no_wq) + 1

                if self.bool_include_covariates:
                    n_cols_station += \
                        self.training_covariates_step2.shape[1]

                M_Simul_step2 = np.zeros(
                    (L, n_cols_station, n_samples),
                    dtype=np.float32
                )

                my_realiz = 0
                max_tries = 100
                cur_tries = 0
                
                # Track active asynchronous jobs: list of tuples (job_id, realiz_index_it_was_built_for)
                active_jobs = []
                
                # Helper function to construct arguments for a specific realization index
                def submit_g2s_job(r_idx):
                    simul_water_col = M_Simul_step1[:, my_station, r_idx][:, np.newaxis]
                    obs_data_cols = observed_data_np[i][:, inds_var_no_wq]
                    train_water_col = self.train_data[:, [water_idx]]
                    train_data_cols = self.train_data[:, inds_var_no_wq]
                
                    if not self.bool_include_covariates:
                        M_TI_step2 = np.hstack((train_water_col, train_data_cols))
                        M_cond_s2 = np.hstack((simul_water_col, obs_data_cols))
                    else:
                        cov_train = self.training_covariates_step2
                        cov_test = test_covariates_step2[i]
                        M_TI_step2 = np.hstack((cov_train, train_water_col, train_data_cols))
                        M_cond_s2 = np.hstack((cov_test, simul_water_col, obs_data_cols))
                
                    V_Data_Type_step2 = np.zeros(M_TI_step2.shape[1])
                    kernel_step2 = np.ones((self.nb_neighboors, M_TI_step2.shape[1]))
                
                    job_id = g2s(
                        '-a', 'qs',
                        '-ti', M_TI_step2,
                        '-di', M_cond_s2,
                        '-dt', V_Data_Type_step2,
                        '-k', 1.2,
                        '-ki', kernel_step2,
                        '-n', self.nb_neighboors,
                        '-j', 0.5,
                        '-W_CUDA', 0,
                        '-silent',
                        '-submitOnly'
                    )
                    return job_id
                
                # --- PHASE 1: Populate Initial Queue ---
                # Prime the server with jobs for all initial targets
                for r in range(n_samples):
                    cur_tries += 1
                    if cur_tries > max_tries:
                        raise Exception("Max tries exceeded during initial submission queue setup.")
                    
                    id_sub = submit_g2s_job(r)
                    active_jobs.append((id_sub, r))
                
                # --- PHASE 2: Process & Replenish ---
                while my_realiz < n_samples:
                    # Safely guard the processing loop
                    if cur_tries > max_tries and not active_jobs:
                        raise Exception("This always raises on the second element, ie when B = 1 and my_station = 0")
                
                    # Get the next finishing job from our pipeline
                    current_id, assigned_realiz = active_jobs.pop(0)
                    
                    # Download payload
                    simulation, *_ = g2s('-silent', '-waitAndDownload', current_id)
                
                    if np.sum(simulation) != 0:
                        # Success! Save the data into our consecutive array slots
                        simulation = np.nan_to_num(simulation, nan=0.0)
                        M_Simul_step2[:, 0, my_realiz] = simulation[:, 0]
                        M_Simul_step2[:, 1:, my_realiz] = simulation[:, 1:]
                        
                        my_realiz += 1
                    else:
                        # Rejected! We need to retry this slot.
                        cur_tries += 1
                        if cur_tries > max_tries:
                            # If we run out of attempts, clear queue to break out cleanly
                            active_jobs.clear()
                            raise Exception("This always raises on the second element, ie when B = 1 and my_station = 0")
                        
                        # Re-submit a fresh attempt using the same realization index dependency
                        print(f"Simulation rejected for target slot {my_realiz}. Retrying...")
                        new_id = submit_g2s_job(assigned_realiz)
                        active_jobs.append((new_id, assigned_realiz))
                col_indices = np.concatenate(([water_idx], inds_var))

                M_Simul_step2_total[:, col_indices, :] = \
                    M_Simul_step2[:, :len(col_indices), :]

            output[i] = M_Simul_step2_total

        observed_data_np[observed_mask == 0] = 0.0

        output = torch.from_numpy(output)
        output = torch.nan_to_num(output, nan=0.0)
        output = output.permute(0, 3, 2, 1)

        return output