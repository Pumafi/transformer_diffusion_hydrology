import torch
import numpy as np
import pandas as pd
import darts
from darts import TimeSeries
import logging
import warnings

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=".*LeafSpec.*deprecated.*",
    category=DeprecationWarning,
)

class DartsWrapper():
    def __init__(self, darts_model, is_probabilistic=False):
         self.is_probabilistic=is_probabilistic
         self.darts_model = darts_model

    def impute(self, observed_data, observed_mask, n_samples=100, timestamps=None):
          if not self.is_probabilistic:
               n_samples = 1

          B, L, K = observed_data.shape

          # move to CPU / numpy for darts
          observed_mask = observed_mask.cpu().numpy().astype(int)
          observed_data_np = observed_data.cpu().numpy()
          timestamps_np = timestamps.cpu().numpy()

          output=[]
          for i in range(B):
               ts_darts = darts.TimeSeries.from_times_and_values(pd.RangeIndex(timestamps_np[i, 0], timestamps_np[i, -1]+1), observed_data_np[i])

               mask_time = observed_mask[0].all(axis=-1)
               split_idx = np.where(mask_time == 0)[0][0]

               observed_ts, missing_ts = ts_darts.split_after(split_idx - 1)

               backtest_en = self.darts_model.historical_forecasts(
                    series=ts_darts,
                    start=missing_ts.start_time(),
                    num_samples=n_samples,
                    forecast_horizon=len(missing_ts),
                    stride=1,
                    retrain=False,
                    verbose=False,
                    last_points_only=False,
               )
               final_forecast_ts = backtest_en[-1]
               

               observed_ts = observed_ts.values()
               observed_ts = np.repeat(np.expand_dims(observed_ts, axis=-1), n_samples, axis=-1)
               
               final_forecast_ts = final_forecast_ts.all_values()
               filled_np = np.concatenate([observed_ts, final_forecast_ts], axis=0)

               output.append(filled_np)
          output = np.stack(output, axis=0)

          output = torch.from_numpy(output)
          if len(output.shape) == 3:
               output = output.unsqueeze(-1)
          output = output.permute(0, 3, 2, 1)
          return output