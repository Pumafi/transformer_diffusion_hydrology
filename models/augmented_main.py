import numpy as np
import torch
import torch.nn as nn
from .augmented_architecture import Augmented_Arch_Model
import random
import math

class TimeEmbedding(nn.Module):
    def __init__(self, d_model=16, learnable_cycles=None, device="cpu",
                 min_alpha=12.0, max_alpha=24.0*14):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        if learnable_cycles is not None:
            # initialize near given cycles by inverting sigmoid scaling
            init = []
            for c in learnable_cycles:
                val = (c - min_alpha) / (max_alpha - min_alpha)
                val = math.log(val / (1-val))  # inverse sigmoid
                init.append(val)
            self.raw_cycles = torch.tensor(init, dtype=torch.float32)
        else:
            self.raw_cycles = None

    def get_cycles(self):
        if self.raw_cycles is None:
            return None
        return self.min_alpha + (self.max_alpha - self.min_alpha) * torch.sigmoid(self.raw_cycles)

    def forward(self, pos):
        batch_size, seq_len = pos.shape
        pe = torch.zeros(pos.shape[0], pos.shape[1], self.d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, self.d_model, 2).to(self.device) / self.d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        # Learnable cycles
        cycles = self.get_cycles()
        if cycles is not None:
            feats = []
            for alpha in cycles:
                feats.append(torch.sin(2 * math.pi * position / alpha))
                feats.append(torch.cos(2 * math.pi * position / alpha))
            feats = torch.cat(feats, dim=-1)
            pe = torch.cat([pe, feats], dim=-1)

        return pe



class Augmented_CSDI_base(nn.Module):
    """
    Augmented Model. Based on CSDI, this is our custom architecture
    WITH covariates.

    Use impute() to generate new simulations
    """
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.number_of_covariates = config["data"]["number_of_covariates"]

        self.cycles_alphas = [x * 24. / config["model"]["step_size_in_hours"] for x in config["model"]["learnable_cyle_in_days"]]
        self.time_embedding = TimeEmbedding(d_model=config["model"]["timeemb"],
                                            learnable_cycles=self.cycles_alphas,
                                            min_alpha=self.cycles_alphas[0] // 2,
                                            max_alpha=self.cycles_alphas[-1] * 2,
                                            device="cuda")
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + len(self.cycles_alphas) * 2 #+ self.emb_feature_dim + self.number_of_covariates
        if self.is_unconditional == False:
            self.emb_total_dim += 2  # for conditional mask

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["nb_covariates"] = self.number_of_covariates

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = Augmented_Arch_Model(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            coin = random.uniform(0, 1)
            if coin < 0.5:
                sample_ratio = np.random.rand()  # missing ratio
            else:
                sample_ratio = 1.

            
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1 # mask a nummber of observed values. So Most observec values stay at 1, but some are put to 0
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_data, observed_tp, cond_mask, feature_id=None):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        time_embed = time_embed.permute(0, 3, 2, 1)

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        feature_embed = feature_embed.permute(0, 3, 2, 1)

        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        
        observed_data = observed_data.unsqueeze(1) * side_mask  # (B,1,K,L)

        side_info = torch.cat([observed_data, side_mask, time_embed, feature_embed], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, covariates, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, covariates, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, covariates, is_train, set_t=-1
    ):
        # observed mask is only here to compute the loss
        B, K, L = observed_data.shape

        # Noising ==> x_t
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        # First Call
        total_initial_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) # (B,3,K,L)
        initial_predicted = self.diffmodel(total_initial_input, side_info, covariates, t)  
        
        target_mask = observed_mask - cond_mask # only compute on values known
        residual_prior = (initial_predicted * target_mask) - (noise * target_mask)
        num_eval_prior = target_mask.sum()
        loss = (residual_prior ** 2).sum() / (num_eval_prior if num_eval_prior > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def __impute(self, observed_data, cond_mask, covariates, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        covariates = covariates.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, K, -1)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):

                noise = torch.randn_like(observed_data)
                noisy_obs = (self.alpha_hat[t] ** 0.5) * observed_data + self.beta[t] ** 0.5 * noise
                current_sample = ((1 - cond_mask) * current_sample) + (cond_mask * noisy_obs)

                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, covariates, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples
    
    def impute(self, observed_data, cond_mask, covariates, n_samples, timestamps=None):
        with torch.no_grad():
            observed_tp = timestamps if timestamps is not None else np.tile(np.arange(observed_data.shape[1]), (observed_data.shape[0], 1))
            observed_data = observed_data.permute(0, 2, 1)
            cond_mask = cond_mask.permute(0, 2, 1)
            
            observed_tp = torch.tensor(observed_tp, dtype=torch.float32).to(self.device)
            side_info = self.get_side_info(observed_data, observed_tp, cond_mask)
            samples = self.__impute(observed_data, cond_mask, covariates, side_info, n_samples)
            samples = samples * (1 - cond_mask[:, None, :, :]) + observed_data[:, None, :, :] * cond_mask[:, None, :, :]
        return samples


    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            covariates,
        ) = self.process_data(batch)
        if is_train == 0:
            return self.impute(observed_data, observed_mask, 1, observed_tp)
        else:
            coin = random.uniform(0, 1)
            if coin < 0.5:
                cond_mask = observed_mask * gt_mask
            else:
                cond_mask = self.get_randmask(observed_mask)
            


        side_info = self.get_side_info(observed_data, observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, covariates, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            covariates,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_data, observed_tp, cond_mask)

            samples = self.__impute(observed_data, cond_mask, covariates, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class Augmented_CSDI_WaterQual(Augmented_CSDI_base):
    def __init__(self, config, device, target_dim=12):
        super(Augmented_CSDI_WaterQual, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # Called before forward
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        covariates = batch["covariates"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        K = observed_data.shape[1]
        covariates = covariates.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, K, -1)
        #for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            covariates,
        )
