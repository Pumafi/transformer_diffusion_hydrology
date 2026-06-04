import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    lr=1e-6,
    history={"train loss":[], "val loss": []}
):
    if "train loss" not in history:
        history["train loss"] = []
    if "val loss" not in history:
        history["val loss"] = []

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=lr)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    # Handle lr decay
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10

    # Epochs
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        # One iteration
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break
            
            # Internally the dataloader already uses a random smapler, so no need to on_epoch_end... shuffle and so on
            lr_scheduler.step()
        history["train loss"].append(avg_loss)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            history["val loss"].append(avg_loss_valid)

    if foldername != "":
        torch.save(model.state_dict(), output_path)

    return history


def original_train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    history={"train loss":[], "val loss": []}
):
    if "train loss" not in history:
        history["train loss"] = []
    if "val loss" not in history:
        history["val loss"] = []

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate_forecasting(model, test_loader, prediction_horizon=12, nsample=100, scaler=1, mean_scaler=0, foldername="."):

    with torch.no_grad():
        try:
            model.eval()
            device = model.device
        except:
            device = "cpu"
        mse_total = 0
        mae_total = 0
        smape_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                observed_mask = test_batch['observed_mask']
                # set later half to 0
                real_data = test_batch['observed_data']
                observed_data_horizon = observed_mask.shape[1] - prediction_horizon
                observed_mask_forecast = observed_mask.clone()
                observed_mask_forecast[:, observed_data_horizon:] = 0
                
                observed_data = test_batch['observed_data']
                observed_data = observed_data * observed_mask_forecast

                observed_mask_forecast = observed_mask_forecast.to(device).float()
                observed_data = observed_data.to(device).float()

                if 'covariates' in test_batch:
                    covariates_data = test_batch['covariates']
                    covariates_data = covariates_data.to(device).float()
                    output = model.impute(observed_data, observed_mask_forecast, covariates_data, n_samples=nsample, timestamps=test_batch['timepoints']).cpu()
                else:
                    output = model.impute(observed_data, observed_mask_forecast, n_samples=nsample, timestamps=test_batch['timepoints']).cpu()
                    
                output = output.permute(0, 1, 3, 2)

                real_data = torch.nan_to_num(real_data, nan=0.0)
                samples = output
                c_target = real_data
                eval_points = observed_mask * (1 - observed_mask_forecast.cpu())

                observed_points = observed_mask
                observed_time = test_batch['timepoints']

                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                y_true = c_target * scaler
                y_pred = samples_median.values * scaler

                smape_current = (
                    2.0 * torch.abs(y_pred - y_true)
                    / (torch.abs(y_pred) + torch.abs(y_true) + 1e-6)
                ) * eval_points

                smape_total += smape_current.sum().item()
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item() if eval_points.sum().item() > 0 else 1e-12

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "smape_total": 100 * smape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample)  + "forecasting" + str(prediction_horizon) + ".pk", "wb"
            ) as f:
                # Save results
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

    return {
        "rmse": np.sqrt(mse_total / evalpoints_total),
        "mae": mae_total / evalpoints_total,
        "smape": 100 * smape_total / evalpoints_total,
        "CRPS": CRPS,
        "CRPS_sum": CRPS_sum
    }

def evaluate_imputation(model, test_loader, seed, nsample=100, scaler=1, mean_scaler=0, ratio_missing=0.25, foldername="."):

    with torch.no_grad():
        try:
            model.eval()
            device = model.device
        except:
            device = "cpu"
        mse_total = 0
        mae_total = 0
        smape_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                observed_mask = test_batch['observed_mask']

                real_data = test_batch['observed_data']

                ratio_to_keep = 1. - ratio_missing
                timesteps_to_keep = int(ratio_to_keep * observed_mask.shape[1])
                
                observed_mask_imputation = torch.zeros_like(observed_mask)

                for features_idx in range(observed_mask.shape[-1]):
                    rng = np.random.RandomState(features_idx + seed)
                    observation_indices = rng.choice(np.arange(observed_mask.shape[1]), size=timesteps_to_keep, replace=False)
                    observed_mask_imputation[:, observation_indices, features_idx] = observed_mask[:, observation_indices, features_idx]

                observed_data = test_batch['observed_data']
                observed_data = observed_data * observed_mask_imputation

                observed_mask_imputation = observed_mask_imputation.to(device).float()
                observed_data = observed_data.to(device).float()
                
                if 'covariates' in test_batch:
                    covariates_data = test_batch['covariates']
                    covariates_data = covariates_data.to(device).float()
                    output = model.impute(observed_data, observed_mask_imputation, covariates_data, n_samples=nsample, timestamps=test_batch['timepoints']).cpu()
                else:
                    output = model.impute(observed_data, observed_mask_imputation, n_samples=nsample, timestamps=test_batch['timepoints']).cpu()
                    
                output = output.permute(0, 1, 3, 2)

                real_data = torch.nan_to_num(real_data, nan=0.0)

                samples = output
                c_target = real_data
                eval_points = observed_mask * (1 - observed_mask_imputation.cpu())

                observed_points = observed_mask
                observed_time = test_batch['timepoints']

                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                y_true = c_target * scaler
                y_pred = samples_median.values * scaler

                smape_current = (
                    2.0 * torch.abs(y_pred - y_true)
                    / (torch.abs(y_pred) + torch.abs(y_true) + 1e-6)
                ) * eval_points

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                smape_total += smape_current.sum().item()
                evalpoints_total += eval_points.sum().item() if eval_points.sum().item() > 0 else 1e-12

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "smape_total": 100 * smape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + "imputation" + str(ratio_missing) + ".pk", "wb"
            ) as f:
                # Save results
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
    return {
        "rmse": np.sqrt(mse_total / evalpoints_total),
        "mae": mae_total / evalpoints_total,
        "smape": 100 * smape_total / evalpoints_total,
        "CRPS": CRPS,
        "CRPS_sum": CRPS_sum
    }

def find_gaps(serie_training_mask):
    """
    Return the indices of the gaps in values based on the training mask.
    """

    padded_mask = np.pad(np.logical_not(serie_training_mask).astype(int), (1,1), mode='constant', constant_values=0)

    gaps_start = np.where(np.diff(padded_mask) == 1)[0]
    gaps_end = np.where(np.diff(padded_mask) == -1)[0]

    gaps = np.array(list(zip(gaps_start, gaps_end)))

    return gaps

def plot_single_imputation_results(generated_test, mask, num_features=3, station_name=None, parameter_names=None):
    """
    Trace les graphiques comparant les observations et les prédictions (gaps imputés).
    
    Parameters:
    -----------
    generated_test : Tensor/Array
        Les données générées ou complétées par le modèle.
    mask : Tensor/Array
        Le masque binaire indiquant la présence de données (1 pour observé, 0 pour manquant).
        Fonction utilitaire pour trouver les indices de début et fin des segments (gaps).
    num_features : int, optional
        Nombre de séries temporelles/colonnes à tracer (par défaut 3).
    station_name : str, optional
        Nom de la station météo/mesure pour le titre.
    parameter_names : list of str, optional
        Liste des noms des variables pour les titres de chaque sous-graphique.
    """
    # Nettoyage des dimensions inutiles (ex: batch size de 1)
    gen_squeezed = np.array(generated_test).squeeze()
    mask_squeezed = np.array(mask).squeeze()

    for i in range(num_features):
        fig, ax = plt.subplots(figsize=(7, 3))
        
        # --- 1. AJOUT DES OBSERVATIONS (Où le masque original vaut 1) ---
        # On inverse le masque pour trouver les segments de vraies observations
        obs_gaps = find_gaps(1 - mask_squeezed[:, i])
        first_obs = True
        
        for gap in obs_gaps:
            # Vérification des limites d'indexation
            if gap[0] >= 0 and gap[1] <= len(gen_squeezed[:, i]):
                x_axis = np.linspace(gap[0], gap[1] - 1, num=gap[1] - gap[0])
                y_axis = gen_squeezed[gap[0]:gap[1], i]
                
                # On met le label uniquement sur le premier segment pour éviter les doublons en légende
                label = 'Observations' if first_obs else ""
                ax.plot(x_axis, y_axis, color='blue', label=label)
                first_obs = False

        # --- 2. AJOUT DES PRÉDICTIONS / IMPUTATIONS (Où le masque original vaut 0) ---
        pred_gaps = find_gaps(mask_squeezed[:, i])
        first_pred = True
        
        for gap in pred_gaps:
            if gap[0] >= 0 and gap[1] <= len(gen_squeezed[:, i]):
                x_axis = np.linspace(gap[0], gap[1] - 1, num=gap[1] - gap[0])
                y_axis = gen_squeezed[gap[0]:gap[1], i]
                
                label = 'Predictions' if first_pred else ""
                ax.plot(x_axis, y_axis, color='red', label=label)
                first_pred = False
        
        # --- 3. HABILLAGE DU GRAPHIQUE ---
        # Gestion dynamique du titre si les infos sont fournies
        if station_name and parameter_names and i < len(parameter_names):
            ax.set_title(f"{station_name} - {parameter_names[i]}")
        elif parameter_names and i < len(parameter_names):
            ax.set_title(parameter_names[i])
            
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right') # Ajout de la légende (sans doublons)
        
        plt.tight_layout()
        plt.show()

def multivariate_visualize(dataind, generated, real_data, observed_mask, observed_mask_forecast):
    """
    Preprocesses the model outputs and generates the multi-variate visualization grid.
    """
    # --- 1. Vectorized Data Preprocessing ---
    # Permute and move to CPU once
    generated_test = generated.permute(0, 1, 3, 2).cpu()
    
    all_target_np = real_data.cpu()
    all_given_np = observed_mask_forecast.cpu()
    all_evalpoint_np = observed_mask.cpu() - all_given_np
    
    K = generated_test.shape[-1]  # Features / Channels
    L = generated_test.shape[-2]  # Time length

    # Vectorized Quantile Calculation: Pass the entire list of quantiles at once
    q_tensor = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    # Shape of quantiles_imp will be [5, batch, time, features]
    quantiles_imp = torch.quantile(generated_test, q_tensor, dim=1)
    
    # Broadcast the imputation formula across all quantiles at once
    # (1 - all_given_np) broadcasts perfectly over the quantile dimension
    quantiles_imp = quantiles_imp * (1 - all_given_np) + all_target_np * all_given_np

    # --- 2. Plotting Logic ---
    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=9, ncols=9, figsize=(30.0, 20.0))
    fig.delaxes(axes[-1][-1])  # Remove the 81st subplot to keep a 9x9 layout clear

    time_steps = np.arange(0, L)

    for k in range(K):
        row, col = k // 9, k % 9
        ax = axes[row][col]

        # Extract target slices for the specific sample and feature
        target_slice = all_target_np[dataind, :, k]
        eval_slice = all_evalpoint_np[dataind, :, k]
        given_slice = all_given_np[dataind, :, k]

        # Use clean boolean masking instead of creating and filtering heavy DataFrames
        eval_mask = eval_slice != 0
        given_mask = given_slice != 0

        # Plot CSDI Median (Index 2 corresponds to 0.50 quantile) and Confidence Bounds
        ax.plot(time_steps, quantiles_imp[2, dataind, :, k], color='g', label='CSDI')
        ax.fill_between(time_steps, quantiles_imp[0, dataind, :, k], quantiles_imp[4, dataind, :, k], color='g', alpha=0.3)

        # Plot actual observations (blue dots for evaluation points, red x's for given points)
        ax.plot(time_steps[eval_mask], target_slice[eval_mask], color='b', marker='o', linestyle='None')
        ax.plot(time_steps[given_mask], target_slice[given_mask], color='r', marker='x', linestyle='None')

        # Clean axis labels
        if col == 0:
            ax.set_ylabel('value')
        if row == 8: 
            ax.set_xlabel('time')

    plt.tight_layout()
    return fig, axes