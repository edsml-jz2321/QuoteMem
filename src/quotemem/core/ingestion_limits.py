"""One bounded Add producer inventory authority.

Each new Evidence part fits the minimum 1,000-word semantic policy without
splitting its pending source. The tokenizer counts at most one word per Unicode
codepoint and ASCII punctuation reaches one word per byte. UTF-8 boundary
alignment can waste at most three bytes in each nonfinal part; every message is
nonempty. Therefore B bytes over M messages produce at most
floor((B - M) / (part_bytes - 3)) + M parts. Archived larger rows remain subject
to their recorded source inventory.
"""

MAX_ADD_MESSAGES = 256
MAX_ADD_UTF8_BYTES = 1_048_576
MAX_EVIDENCE_PART_BYTES = 1_000
MAX_ADD_SOURCE_PARTS = (MAX_ADD_UTF8_BYTES - MAX_ADD_MESSAGES) // (
    MAX_EVIDENCE_PART_BYTES - 3
) + MAX_ADD_MESSAGES
MAX_ADD_PARTS_PER_MESSAGE = (MAX_ADD_UTF8_BYTES - 1) // (MAX_EVIDENCE_PART_BYTES - 3) + 1
