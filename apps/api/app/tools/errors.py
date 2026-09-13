"""Normalized tool-layer failures safe for orchestration and public events."""


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details or {}
