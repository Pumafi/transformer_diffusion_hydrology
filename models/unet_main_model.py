import numpy as np
import torch
import torch.nn as nn
from diff_unet_model import UNet1D
import random
import math


class Unetdiff_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = UNet1D(target_dim, 2*target_dim)

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
        elif config_diff["schedule"] == "cosine":
            s = config_diff.get("cosine_s", 0.008)
            T = self.num_steps

            t = np.linspace(0, T, T + 1)
            f = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
            alpha_hat = f / f[0]

            self.beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
            self.beta = np.clip(self.beta, 1e-5, 0.999)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            coin = random.uniform(0, 1)
            if coin < 1.:
                sample_ratio = np.random.rand()
            else:
                # generation
                sample_ratio = 1.

            
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1 # mask a nummber of observed values. So Most observec values stay at 1, but some are put to 0
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, is_train, set_t=-1
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

        cond_obs = (cond_mask * observed_data)
        #noisy_target = ((1 - cond_mask) * noisy_data)

        total_input = (noisy_data, cond_obs, cond_mask, t.to(self.device))
        predicted = self.diffmodel(total_input)  # (B,K,L)
        
        target_mask = observed_mask - cond_mask # only compute on values known
        residual_prior = (noise - predicted) * target_mask
        num_eval_prior = target_mask.sum()
        loss = (residual_prior ** 2).sum() / (num_eval_prior if num_eval_prior > 0 else 1)

        return loss


    def __impute(self, observed_data, cond_mask, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            B = observed_data.shape[0]

            for t in range(self.num_steps - 1, -1, -1):
                noise = torch.randn_like(observed_data)
                noisy_obs = (self.alpha_hat[t] ** 0.5) * observed_data + self.beta[t] ** 0.5 * noise
                current_sample = ((1 - cond_mask) * current_sample) + (cond_mask * noisy_obs)

                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

                cond_obs = cond_mask * observed_data

                eps_hat = self.diffmodel((
                    current_sample,
                    cond_obs,
                    cond_mask,
                    t_tensor
                ))
                x0_hat = (current_sample - ((1.0 - self.alpha_hat[t]) ** 0.5) * eps_hat) / (self.alpha_hat[t] ** 0.5)

                if t > 0:
                    current_sample = (
                        (self.alpha_hat[t - 1] ** 0.5) * x0_hat +
                        ((1.0 - self.alpha_hat[t - 1]) ** 0.5) * eps_hat
                    )
                else:
                    current_sample = x0_hat

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples
    
    def impute(self, observed_data, cond_mask, n_samples, timestamps=None):
        observed_data = observed_data.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        
        samples = self.__impute(observed_data, cond_mask, n_samples)
        samples = samples * (1 - cond_mask[:, None, :, :]) + observed_data[:, None, :, :] * cond_mask[:, None, :, :]

        return samples


    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            gt_mask,
        ) = self.process_data(batch)
        if is_train == 0:
            return self.impute(observed_data, observed_mask, 1)
        else:
            # During training
            coin = random.uniform(0, 1)
            if coin < 0.5:
                # Forecasting mask
                cond_mask = observed_mask * gt_mask
            else:
                # Missing at Random mask or Full Generation mask
                cond_mask = self.get_randmask(observed_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            gt_mask,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            samples = self.__impute(observed_data, cond_mask, n_samples)

        return samples, observed_data, target_mask, observed_mask


class Unet_WaterQual(Unetdiff_base):
    def __init__(self, config, device, target_dim=12):
        super(Unet_WaterQual, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # Called before forward
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        # observed_tp = batch["timepoints"].to(self.device).float() Unused with Unet
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            gt_mask,
        )
