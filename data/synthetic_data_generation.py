import numpy as np
import matplotlib.pyplot as plt


def generate_multivariate_series(n=1440, n_stations=3, seed=0):
    np.random.seed(seed)
    t = np.arange(n)

    # =====================================================
    # GLOBAL SHOCK DRIVER (shared spike timing)
    # =====================================================
    global_spike = np.zeros(n)
    burst = np.zeros(n)

    burst_prob = 0.01
    burst_decay = 0.93
    spike_prob_normal = 0.01
    spike_prob_burst = 0.08
    spike_scale = 0.2

    for i in range(1, n):

        # Temporary volatility bursts
        if burst[i-1] < 0.1 and np.random.rand() < burst_prob:
            burst[i] = 1
        else:
            burst[i] = burst[i-1] * burst_decay

        spike_prob = spike_prob_burst if burst[i] > 0.2 else spike_prob_normal

        if np.random.rand() < spike_prob:
            global_spike[i] = np.random.exponential(spike_scale)

    # =====================================================
    # TRIPLE STRUCTURE PER STATION
    # =====================================================
    dyn_vars = []
    seasonal_vars = []
    independent_seasonal_vars = []

    for k in range(n_stations):

        # ---------------------------
        # Mean level
        # ---------------------------
        pair_mean = 0.25 + 0.07 * k

        # ---------------------------
        # Latent AR(1)
        # ---------------------------
        phi = 0.95 - 0.02*k
        latent = np.zeros(n)
        latent[0] = pair_mean

        for i in range(1, n):
            noise = np.random.normal(0, 0.01)
            latent[i] = pair_mean + phi*(latent[i-1]-pair_mean) + noise
            latent[i] += global_spike[i]
            latent[i] = max(latent[i], 0.2)

        # ---------------------------
        # Correlated seasonal component
        # ---------------------------
        p1 = 720 + np.random.randint(-10, 10)
        p2 = 24 + np.random.randint(-5, 5)

        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)

        season = (
            0.07*np.sin(2*np.pi*t/p1 + phase1) +
            0.03*np.sin(2*np.pi*t/p2 + phase2)
        )

        # ---------------------------
        # Dynamic variable
        # ---------------------------
        dyn = latent + np.random.normal(0, 0.01, n)

        # ---------------------------
        # Seasonal correlated variable
        # ---------------------------
        seasonal = (
            0.6*latent +
            season +
            np.random.normal(0, 0.01, n)
        )

        # ---------------------------
        # Independent seasonal variable
        # Completely independent of latent & spikes
        # ---------------------------
        p1_i = 600 + np.random.randint(-20, 20)
        p2_i = 48 + np.random.randint(-10, 10)

        phase1_i = np.random.uniform(0, 2*np.pi)
        phase2_i = np.random.uniform(0, 2*np.pi)

        independent_season = (
            0.15 +
            0.08*np.sin(2*np.pi*t/p1_i + phase1_i) +
            0.04*np.sin(2*np.pi*t/p2_i + phase2_i) +
            np.random.normal(0, 0.01, n)
        )

        dyn_vars.append(dyn)
        seasonal_vars.append(seasonal)
        independent_seasonal_vars.append(independent_season)

    # =====================================================
    # PROGRESSIVE DROUGHT REGIME
    # (Mutually Exclusive, Spike Priority, No Instant Drop)
    # =====================================================
    regime = np.zeros(n)
    phi_normal = 0.94
    mean_normal = 0.35

    phi_drought = 0.85        # stronger mean reversion
    mean_drought = 0.02       # near-zero target

    regime[0] = mean_normal

    state = 0                 # 0 = normal, 1 = drought
    min_drought = 60
    drought_clock = 0

    p_enter = 0.01
    p_exit = 0.03

    for i in range(1, n):

        spike_now = global_spike[i] > 0

        # =================================================
        # SPIKE PRIORITY (forces normal regime)
        # =================================================
        if spike_now:
            state = 0
            drought_clock = 0

        # =================================================
        # NORMAL REGIME
        # =================================================
        if state == 0:

            # Possibly enter drought (no spike)
            if (not spike_now) and np.random.rand() < p_enter:
                state = 1
                drought_clock = 0

            noise = np.random.normal(0, 0.01)

            regime[i] = (
                mean_normal
                + phi_normal * (regime[i-1] - mean_normal)
                + noise
                + global_spike[i] * 0.2
            )

            regime[i] = max(regime[i], 0)

        # =================================================
        # DROUGHT REGIME (PROGRESSIVE DECLINE)
        # =================================================
        else:

            drought_clock += 1
            noise = np.random.normal(0, 0.005)  # lower volatility

            regime[i] = (
                mean_drought
                + phi_drought * (regime[i-1] - mean_drought)
            )

            regime[i] = max(regime[i], 0)

            # Exit after minimum duration
            if drought_clock > min_drought and np.random.rand() < p_exit:
                state = 0


    # =====================================================
    # STACK OUTPUT
    # [D1..Dk, S1..Sk, Z1..Zk, R]
    # =====================================================
    X = np.vstack(
        dyn_vars +
        seasonal_vars +
        independent_seasonal_vars +
        [regime]
    ).T

    return X
