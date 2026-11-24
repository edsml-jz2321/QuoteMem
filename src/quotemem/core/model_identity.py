"""Response identities admitted for the supported gateway model aliases."""

import re

DEEPSEEK_FLASH_MODEL = "deepseek-v4.1-flash"
DEEPSEEK_FLASH_RESOLVED_MODEL = "deepseek-v4-1-flash-261017"
"""The alias's current dated snapshot, the identity a synthesized reply reports."""
DEEPSEEK_FLASH_FALLBACK_MODEL = "deepseek-v4.1-flash-1017"
"""The same dated snapshot under the alias that a fallback route of a gateway may report."""
_DEEPSEEK_FLASH_SNAPSHOT = re.compile(r"deepseek-v4-1-flash-[0-9]{6}")


def admits_inference_model(provider: str, requested_model: str, resolved_model: object) -> bool:
    """Whether a response's model is the requested model's own identity.

    Args:
        provider: The frozen inference transport identity.
        requested_model: The model name actually sent in the HTTP request.
        resolved_model: The model name the response reports.

    Returns:
        For the DeepSeek alias, whether the response names one of that alias's own dated
        snapshots (``deepseek-v4-1-flash-YYMMDD``) or exactly the fallback tier's name for
        that snapshot (``DEEPSEEK_FLASH_FALLBACK_MODEL``); the gateway upgrades the snapshot
        behind the alias, so pinning one would turn each upgrade into an outage. Any
        other model line, vendor or malformed id is refused. Every other requested
        model admits only itself.
    """
    if not isinstance(resolved_model, str) or not resolved_model:
        return False
    if provider == "openai" and requested_model == DEEPSEEK_FLASH_MODEL:
        return resolved_model == DEEPSEEK_FLASH_FALLBACK_MODEL or (
            _DEEPSEEK_FLASH_SNAPSHOT.fullmatch(resolved_model) is not None
        )
    return resolved_model == requested_model


def current_inference_model(provider: str, requested_model: str) -> str:
    """The response model a synthesized reply to this request reports.

    Args:
        provider: The frozen inference transport identity.
        requested_model: The model name sent in the request.

    Returns:
        The alias's current snapshot for the DeepSeek alias, else the requested model.
        Admission never compares with it; see ``admits_inference_model``.
    """
    if provider == "openai" and requested_model == DEEPSEEK_FLASH_MODEL:
        return DEEPSEEK_FLASH_RESOLVED_MODEL
    return requested_model
