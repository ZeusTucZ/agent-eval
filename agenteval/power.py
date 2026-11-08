"""agenteval power: how many cases you need before a comparison can catch a given drop.

"Inconclusive" from agenteval compare doesn't mean "no change": with too few cases a real
drop looks like chance. This simulates comparisons the way compare and the gate make them
(each case's majority over its attempts, McNemar's exact test on cases that flipped) and
reports how often a drop of each size is caught. No model is called.

The simulation treats every case as equally hard and every attempt as independent, like
the mock agent's faults. Real regressions are often consistent (the same cases always
fail), which is easier to catch, so treat these rates as a cautious lower bound.
"""
import argparse
from random import Random

from agenteval.stats import mcnemar_exact

def _case_passes(random: Random, pass_rate: float, attempts: int) -> bool | None:
    """True or False for a majority of attempts passing or failing; None for a tie."""
    passed = sum(random.random() < pass_rate for _ in range(attempts))
    if passed * 2 == attempts:
        return None
    return passed * 2 > attempts

def detection_rate(cases: int, attempts: int, pass_rate: float, drop: float,
                   simulations: int = 500, alpha: float = 0.05, seed: int = 0) -> float:
    """Share of simulated comparisons where a drop of `drop` (0.10 = 10 points) is detected."""
    random = Random(seed)
    worse = max(0.0, pass_rate - drop)
    detected = 0
    for _ in range(simulations):
        improved = regressed = 0
        for _ in range(cases):
            before, after = _case_passes(random, pass_rate, attempts), _case_passes(random, worse, attempts)
            if before is None or after is None or before == after:
                continue
            if after:
                improved += 1
            else:
                regressed += 1
        if regressed > improved and mcnemar_exact(improved, regressed) < alpha:
            detected += 1
    return detected / simulations

DEFAULT_CASES = (25, 55, 100, 200, 300, 500, 1000)
DEFAULT_DROPS = (0.05, 0.10, 0.15, 0.25)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval power", description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=int, nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--pass-rate", type=float, default=0.85, help="your agent's current pass rate, 0 to 1")
    parser.add_argument("--drop", type=float, nargs="+", default=list(DEFAULT_DROPS),
                        help="drops to test, as fractions: 0.10 is 10 points")
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--target", type=float, default=0.8, help="detection rate to aim for")
    args = parser.parse_args(argv)

    print(f"How often a drop is caught (pass rate {args.pass_rate:.0%}, {args.attempts} attempts, "
          f"{args.simulations} simulations each):\n")
    header = "  Cases  " + "".join(f"{drop * 100:>4.0f}-pt drop " for drop in args.drop)
    print(header)
    enough: dict[float, int] = {}
    for cases in args.cases:
        rates = [detection_rate(cases, args.attempts, args.pass_rate, drop, args.simulations) for drop in args.drop]
        print(f"  {cases:>5}  " + "".join(f"{rate:>11.0%}  " for rate in rates))
        for drop, rate in zip(args.drop, rates, strict=True):
            if rate >= args.target:
                enough.setdefault(drop, cases)

    print()
    for drop in args.drop:
        if drop in enough:
            print(f"  To catch a {drop * 100:.0f}-point drop {args.target:.0%} of the time: about {enough[drop]} cases.")
        else:
            print(f"  A {drop * 100:.0f}-point drop isn't caught {args.target:.0%} of the time with up to "
                  f"{max(args.cases)} cases.")
    if args.attempts % 2 == 0:
        print("\n  Tip: an odd number of attempts avoids ties, which compare can't count as a change.")
