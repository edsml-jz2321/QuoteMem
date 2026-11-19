"""The bare-name statement syntax: an entity named by a neutral kind alone (D.3 P1 #6 (b)).

"A person is named Rowan." names its entity and says nothing else about it: no owner,
relation, role, occupation or clause, and its kind is one of a closed list of neutral
kinds that hint at no fact. Forget keeps such a name readable for a surviving naming
memory (path (b), ``store/forget_admission.py``), and nomination never presumes that a
bare-name member restates another entity's forgotten fact on the strength of a name
alone (C.4.3 #7, ML2): equal names never prove identity (C.15 #5).
"""

from __future__ import annotations

import re
from typing import Final

_BY_TYPE: Final = re.compile(
    r"(?:A|An|The) ([a-z][a-z-]*(?: [a-z][a-z-]*){0,3}) (?:is|was) (?:named|called) (.+?)\.?",
)
"""English naming syntax; `bare_naming` also requires an exact neutral kind."""

NEUTRAL_KINDS: Final = frozenset({
    "person", "man", "woman", "boy", "girl", "organization", "company", "business",
    "place", "city", "town", "country", "hospital", "clinic", "pantry", "studio",
    "bakery", "application",
})
"""Census-backed neutral kinds, never relations, roles, occupations, animals or clauses."""

CHINESE_KINDS: Final = (
    "一个人", "一个男人", "一个女人", "一个男孩", "一个女孩", "一个组织", "一家公司",
    "一家企业", "一个地方", "一座城市", "一个城镇", "一个国家", "一家医院", "一家诊所",
    "一家食品救济站", "一间工作室", "一家面包店", "一个应用程序",
)
"""Exact neutral classifier-kind phrases, with no relation, modifier or role."""

_CHINESE_BY_TYPE: Final = re.compile(
    r"(" + "|".join(CHINESE_KINDS) + r")\s*(?:名叫|名为|叫)\s*(.+?)[。.]?",
)


def bare_naming(statement: str) -> tuple[str, str] | None:
    """Return the kind and the name of a bare-name statement, or None.

    A bare-name statement is exactly ``A|An|The <kind> is|was named|called <name>.``
    with one of `NEUTRAL_KINDS`, or a Chinese classifier kind of `CHINESE_KINDS`
    followed by 叫, 名叫 or 名为 and the name, with an optional final period.

    Args:
        statement (str): A statement.

    Returns:
        tuple[str, str] | None: ``(kind, name)`` as written, or None for any other
            statement, a modifier, relation, role, occupation or animal kind included.

    Examples:
        ```python
        bare_naming("A person is named Rowan.")  # ("person", "Rowan")
        bare_naming("The user's spouse is named Rowan.")  # None
        bare_naming("A dog is named Biscuit.")  # None: an animal is no neutral kind
        bare_naming("一家公司名叫小明。")  # ("一家公司", "小明")
        ```
    """
    text = statement.strip()
    match = _BY_TYPE.fullmatch(text)
    if match is not None:
        return (match.group(1), match.group(2)) if match.group(1) in NEUTRAL_KINDS else None
    match = _CHINESE_BY_TYPE.fullmatch(text)
    return None if match is None else (match.group(1), match.group(2))
