"""Typed application failures shared by storage and provider boundaries."""


class QuoteMemError(ValueError):
    """Report a stable reason code without including sensitive payload content.

    Attributes:
        code: Machine-readable failure category.
        origin_frame: Optional content-free coordinate (file name, line,
            function) of the innermost frame a wrapper deliberately dropped
            the traceback of, so a retained failure reason can still name it.
    """

    origin_frame: tuple[str, int, str] | None

    def __init__(self, code: str, message: str) -> None:
        """Create a failure with an English, payload-free explanation."""
        self.code = code
        self.origin_frame = None
        super().__init__(f"{code}: {message}")


class RecoveryInProgress(QuoteMemError):
    """``delivery_busy`` while an owner's automatic resumption is starting.

    The code stays ``delivery_busy`` for every internal check. Only the HTTP
    edge reads the class: it asks for the longest honored Retry-After, because
    an attempt sent before the resumed owner holds its fresh allowance only
    spends one of the caller's bounded attempts on another immediate refusal.
    """

    def __init__(self, message: str) -> None:
        """Create the recovery refusal with an English, payload-free explanation."""
        super().__init__("delivery_busy", message)
