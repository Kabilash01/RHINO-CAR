def map_visibility_to_prt(condition: str) -> float:
    """
    Maps visibility conditions to standard Perception-Reaction Time (Tpr)
    based on empirical driving behavior.
    """
    condition = condition.lower()
    if condition == "sunny":
        return 0.85
    elif condition == "foggy":
        return 2.08
    elif condition == "rainy":
        return 1.6
    else:
        return 1.5  # default conservative fallback


def estimate_prediction_horizon(tpr: float, vlv: float) -> int:
    """
    Computes the Prediction Horizon (Tph) based on:
    - Tpr: Perception-Reaction Time (in seconds)
    - VLV: Speed of Leading Vehicle (in ft/s)

    Uses separate polynomial models for congested and free-flow scenarios
    from ViCoWS research.
    """
    if vlv >= 30:  # ft/s → free-flow condition
        tph = (
            0.932 * tpr**3
            - 4.6822 * tpr**2
            + 10.481 * tpr
            + 13.16
        )
    else:  # congested condition
        tph = (
            -0.0207 * tpr**3
            + 0.3642 * tpr**2
            + 0.2078 * tpr
            + 0.6447
        )
    return max(1, round(tph))
