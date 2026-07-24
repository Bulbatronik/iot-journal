import numpy as np

def aramis_metric(
    tau_true,
    tau_pred,
    T,
    a1=10.0,
    a2=13.0,
    k_false=None,
    k_missed=None,
):
    """
    Compute the Aramis metric A (flattened formulation).

    Parameters
    ----------
    tau_true : array-like, shape (N,)
        Ground-truth onset times (NaN if no abnormal condition).
    _pred : array-like, shape (N,)
        Predicted onset times (NaN if no detection).
    T : float
        Mission time.
    a1 : float
        Shape parameter for delayed detections (default 10).
    a2 : float
        Shape parameter for anticipated detections (default 13).
    k_false : float, optional
        Penalty for false alarms. Default = 7 * T.
    k_missed : float, optional
        Penalty for missed alarms. Default = 10 * T.

    Returns
    -------
    A : float
        Aramis performance metric in [0, 1].
    """

    tau_true = np.asarray(tau_true, dtype=float)
    tau_pred = np.asarray(tau_pred, dtype=float)

    assert tau_true.shape == tau_pred.shape

    N = tau_true.size

    # Default penalties (must be > T to ensure max penalty of 1.0)
    if k_false is None:
        k_false = 7.0 * T
    if k_missed is None:
        k_missed = 10.0 * T

    # --- Step 1: compute error D_i -----------------------------------------
    D = np.zeros(N)

    true_nan = np.isnan(tau_true)
    pred_nan = np.isnan(tau_pred)

    # normal case: both defined
    mask_normal = ~true_nan & ~pred_nan
    D[mask_normal] = tau_true[mask_normal] - tau_pred[mask_normal]

    # false alarms
    mask_false = true_nan & ~pred_nan
    D[mask_false] = k_false

    # missed alarms
    mask_missed = ~true_nan & pred_nan
    D[mask_missed] = -k_missed

    # correct non-detections: D = 0 (already set)

    # --- Step 2: compute normalization constants ---------------------------
    # b1 normalizes delayed detections (-T) to 1.0
    b1 = 1.0 / (1.0 - np.exp(-T / a1))
    # b2 normalizes anticipated detections (+T) to 1.0
    b2 = 1.0 / (1.0 - np.exp(-T / a2))

    # --- Step 3: compute u(D) ----------------------------------------------
    u = np.zeros(N)

    # delayed detections (-T <= D < 0)
    mask_delayed = (D < 0) & (D >= -T)
    u[mask_delayed] = (1.0 - np.exp(D[mask_delayed] / a1)) * b1

    # anticipated detections (0 =< D <= T)
    mask_early = (D >= 0) & (D <= T)
    u[mask_early] = (1.0 - np.exp(-D[mask_early] / a2)) * b2

    # very large errors (false/missed or extreme delay)
    mask_large = (D > T) | (D < -T)
    u[mask_large] = 1.0

    # D == 0 → u = 0 (already set)

    # --- Step 4: average ---------------------------------------------------
    A = np.mean(u)
    return A
