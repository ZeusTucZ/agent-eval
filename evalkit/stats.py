from dataclasses import dataclass
from math import comb
from random import Random

def mcnemar_exact(improved: int, regressed: int) -> float:
    """Two-sided p-value: could this split of disagreements be luck?

    Only cases where the two versions disagree carry information. If the versions were
    equally good, each disagreement would be a fair coin flip, so this is the chance of
    a split at least this lopsided from `improved + regressed` flips.
    """
    flips = improved + regressed
    if flips == 0:
        return 1.0

    smaller_side = min(improved, regressed)
    tail = sum(comb(flips, k) for k in range(smaller_side + 1)) / 2 ** flips
    return min(1.0, 2 * tail)

@dataclass
class Interval:
    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

def bootstrap_mean(values: list[float], resamples: int = 10_000, seed: int = 0, confidence: float = 0.95) -> Interval:
    """Confidence interval for a mean, found by resampling the values with replacement.

    `values` must hold one number per independent unit. For paired experiments that means
    one per case, not one per trial: attempts at the same question are closely related, so
    counting them separately would make the interval falsely narrow.
    """
    if not values:
        return Interval(0.0, 0.0, 0.0)

    random = Random(seed)
    size = len(values)
    means = sorted(sum(random.choices(values, k=size)) / size for _ in range(resamples))

    tail = (1 - confidence) / 2
    return Interval(
        mean=sum(values) / size,
        low=means[int(tail * resamples)],
        high=means[min(resamples - 1, int((1 - tail) * resamples))],
    )
