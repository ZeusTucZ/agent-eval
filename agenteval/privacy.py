"""Keep personal data out of what a run stores.

Traces hold every question, answer, tool argument and tool result, and they end up in a
database file, a CI artifact or a dashboard. With --redact, common kinds of personal data
are replaced before anything is saved (scores are computed first, from the real text).
"""
import re

from agenteval.models import Trial

# Order matters: card numbers before phone numbers, which would otherwise match parts of them.
PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}\b")),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d[\d ()-]{7,}\d(?![\w.])")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("SECRET", re.compile(r"\b(?:sk|gsk|pk|rk|xox[abp])[-_][A-Za-z0-9_-]{16,}\b")),
]

def redact(text: str | None) -> str | None:
    if not text:
        return text
    for label, pattern in PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text

def redact_trial(trial: Trial) -> Trial:
    """A copy of the trial with personal data replaced in its answer and every step."""
    spans = [
        span.model_copy(update={
            "input": redact(span.input),
            "output": redact(span.output),
            "error_message": redact(span.error_message),
        })
        for span in trial.spans
    ]
    return trial.model_copy(update={"output": redact(trial.output), "spans": spans})
