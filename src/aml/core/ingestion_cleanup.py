"""Closed input selection for the opened privacy-erase producers (D.7).

Split-secret cleanup is the one product producer; source expiry and consent
withdrawal are selected only by test fixtures until a producer is authorized
(INV P13, Open item O10). These value helpers grant no maintenance authority.
The sole Writer validates the selected complete plan, admission, and current
privacy fences separately.
"""

from pydantic import TypeAdapter

from aml.core.admission import privacy_erase_target_locator
from aml.core.errors import AMLError
from aml.core.identity import ContentBindingDigest
from aml.core.privacy_persistence import PrivacyEraseInputV1
from aml.core.run_inputs import (
    ConsentWithdrawalRunInputV1,
    SourceExpiryRunInputV1,
    SplitSecretChildRunInputV1,
    SplitSecretPreAddRunInputV1,
    parse_run_input,
)
from aml.ingestion.inventory import BindContent
from aml.store.metadata import StrictModel

CleanupInput = PrivacyEraseInputV1
_INPUT: TypeAdapter[CleanupInput] = TypeAdapter(CleanupInput)
CleanupRunInput = (
    SplitSecretPreAddRunInputV1
    | SplitSecretChildRunInputV1
    | SourceExpiryRunInputV1
    | ConsentWithdrawalRunInputV1
)
RootRunInput = SplitSecretPreAddRunInputV1 | SourceExpiryRunInputV1 | ConsentWithdrawalRunInputV1
"""The independent erase roots; only an ingestion child has a parent Run."""
RUN_INPUT_OF_TRIGGER: dict[str, type[CleanupRunInput]] = {
    "pre_add": SplitSecretPreAddRunInputV1,
    "ingestion_child": SplitSecretChildRunInputV1,
    "clock": SourceExpiryRunInputV1,
    "consent_epoch": ConsentWithdrawalRunInputV1,
}
"""The run input that attaches each trigger's complete plan."""


def validate_cleanup_input(value: object) -> CleanupInput:
    """Validate and isolate a complete closed erase input without reading storage.

    Args:
        value: A strict input instance or its complete decoded mapping.

    Returns:
        An isolated privacy-erase input with no inferred fields.

    Raises:
        ValidationError: If the role, trigger, required fields, or bounds are invalid.
    """
    return _INPUT.validate_python(
        value.model_dump(mode="json") if isinstance(value, StrictModel) else value,
    )


def cleanup_target(value: CleanupInput, bind_content: BindContent) -> ContentBindingDigest:
    """Select the existing purpose-bound target without changing the input protocol.

    Args:
        value: Complete strict erase input, revalidated before deriving its locator.
        bind_content: The tenant Keyring's existing canonical HMAC callback.

    Returns:
        The privacy-erase-target HMAC.

    Raises:
        ValidationError: If a supplied or mutated input is invalid.
    """
    return privacy_erase_target_locator(validate_cleanup_input(value), bind_content)


def validate_cleanup_run_input(value: object) -> CleanupRunInput:
    """Select only opened erase inputs through the authoritative Run dispatcher.

    Args:
        value: A strict Run input or its complete decoded mapping.

    Returns:
        An isolated split-secret, source-expiry or consent-withdrawal input.

    Raises:
        AMLError: If the input is malformed, unsupported, or outside the closed erase roles.
    """
    selected = parse_run_input(
        value.model_dump(mode="json") if isinstance(value, StrictModel) else value,
    )
    if not isinstance(selected, CleanupRunInput):
        raise AMLError("unsupported_variant", "Run input is outside privacy erase")
    return selected
