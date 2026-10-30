import re
from contextlib import contextmanager
from time import perf_counter_ns

from evalkit.models import Span

MAX_RECORDED_TEXT = 1000  # per field: enough to read what happened; API errors can be much longer

def clip(text: str) -> str | None:
    # Provider errors can name your account, and results may be kept where others can read them.
    redacted = re.sub(r"\borg_[A-Za-z0-9]+", "org_REDACTED", text)
    return redacted[:MAX_RECORDED_TEXT] or None

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
            # Code inside the span may already have set a more precise label, e.g. "rate_limit".
            span.error_type = span.error_type or type(error).__name__
            span.error_message = clip(str(error))
            raise
        else:
            span.status = "success"
        finally:
            span.end = perf_counter_ns()
            self.spans.append(span)
