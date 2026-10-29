from contextlib import contextmanager
from time import perf_counter_ns

from evalkit.models import Span

class Tracer:
    def __init__(self, trial_id: int):
        self.trial_id = trial_id
        self.spans: list[Span] = []

    @contextmanager
    def span(self, kind: str, name: str):
        start = perf_counter_ns()

        span = Span(
            trial_id = self.trial_id,
            kind=kind,
            name=name,
            start=start,
            end=start,
            status="running",
            input_tokens=0,
            output_tokens=0,
        )

        try:
            yield span
        # BaseException, not Exception: a timeout cancels the step with
        # asyncio.CancelledError, which is not an Exception subclass.
        except BaseException as error:
            span.status = "error"
            span.error_type = type(error).__name__
            raise
        else:
            span.status = "success"
        finally:
            span.end = perf_counter_ns()
            self.spans.append(span)
