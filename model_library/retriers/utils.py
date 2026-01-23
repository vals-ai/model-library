import random


def jitter(wait: float) -> float:
    """
    Increase or decrease the wait time by up to 20%.
    """
    jitter_fraction = 0.2
    min_wait = wait * (1 - jitter_fraction)
    max_wait = wait * (1 + jitter_fraction)
    return random.uniform(min_wait, max_wait)
