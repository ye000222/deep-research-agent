"""Generic generation I/O policy, extracted unchanged from Research Worker.

HTTPX values are per-phase inactivity limits, not an overall request deadline.
Transport retry/fallback policy belongs to each caller and is not set here.
"""

import httpx


def model_generation_timeout() -> httpx.Timeout:
    return httpx.Timeout(90.0, connect=10.0)
