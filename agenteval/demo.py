"""Example experiments for an empty database, so a fresh dashboard or a public demo has
something to show: runs of the fault-injecting mock agent that differ in one setting each,
which makes the compare page meaningful. No API key or network.

Used by the API server when DEMO_SEED=1 and its database is empty, and by
`python -m agenteval.demo DB` directly.
"""
import asyncio
import sys
from dataclasses import asdict

from agenteval import storage
from agenteval.cost import PRICES
from agenteval.execution import RunPlan, aexecute
from agenteval.mock_agent import FaultConfig, MockAgent
from agenteval.models import AgentConfig
from agenteval.sample_data import arithmetic_cases

# name -> the mock's faults. Each differs from the baseline in one way, as a real change would.
RUNS = {
    "baseline": FaultConfig(tool_error_rate=0.05, hallucination_rate=0.12, rate_limit_rate=0.03, seed=1),
    "fewer-hallucinations": FaultConfig(tool_error_rate=0.05, hallucination_rate=0.02, rate_limit_rate=0.03, seed=2),
    "flaky-tools": FaultConfig(tool_error_rate=0.25, hallucination_rate=0.12, rate_limit_rate=0.03, seed=3),
}

def seed(conn, cases: int = 40, attempts: int = 3) -> list[int]:
    """Add the example runs; returns their experiment ids."""
    ids = []
    dataset = arithmetic_cases(cases)
    for name, faults in RUNS.items():
        faults = FaultConfig(**{**asdict(faults), "latency_ms": (5, 40)})  # quick, but not instant
        plan = RunPlan(
            agent=MockAgent(faults), config=AgentConfig(model="mock-llm", prompt="Answer arithmetic questions.",
                                                        params=asdict(faults)),
            dataset_name=f"arithmetic-{cases}", cases=dataset, attempts=attempts, concurrency=20,
            timeout_s=5, prices=dict(PRICES),
        )
        ids.append(asyncio.run(aexecute(plan, conn, name=name, progress=False)).experiment_id)
    return ids

def seed_if_empty(path: str) -> list[int]:
    conn = storage.connect(path)
    try:
        return [] if storage.list_experiment_ids(conn) else seed(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    print(f"Added experiments {seed(storage.connect(sys.argv[1] if len(sys.argv) > 1 else 'agenteval.db'))}")
