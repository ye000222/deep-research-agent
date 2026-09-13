"""Test-wide pytest configuration."""

import asyncio
import sys
from contextlib import suppress

if sys.platform == "win32":
    # psycopg's async pool relies on a selector-based event loop, but Windows
    # defaults to ProactorEventLoop (IOCP). Tests that touch Postgres via asyncio
    # (async checkout, checkpoints, budget ledger) fail on Windows for that reason
    # unless the selector policy is installed before any loop is created. This
    # matches the policy already applied in app.main / cli.setup_checkpoints, so
    # async integration tests behave the same on Windows as on Linux/CI.
    with suppress(AttributeError):
        # Python 3.14+ removed WindowsSelectorEventLoopPolicy and made the
        # selector policy the only default, so there is nothing left to change.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())