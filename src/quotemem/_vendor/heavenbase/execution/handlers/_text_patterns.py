"""Bounded-cost pattern matching for the portable LIKE/ILIKE and regex predicates.

A Python ``re`` pattern backtracks. Translating SQL ``%`` to ``.*`` and running
one regex over the stored value costs about ``O(n**k)`` on a failing value of
length ``n`` with ``k`` ``%`` wildcards, so one ordinary multi-word pattern can
stall a scan on one long row. :class:`LikePattern` matches the ``%``-separated
segments left to right instead, in ``O(len(value) * len(pattern))``.

A user regex keeps Python ``re.search`` semantics. :meth:`RegexSearch.plan`
estimates the worst-case backtracking of Python ``re`` on a text at the
configured cap. A pattern within a fixed step budget runs on Python ``re``.
Otherwise a pattern made of regular constructs runs on a lazily built DFA whose
work is linear in the text, and a pattern that needs Python ``re`` features is
reported as unsafe.
"""

from __future__ import annotations

__all__ = ["LikePattern", "RegexSearch"]

import math
import re
import threading
from itertools import combinations

from heavenbase.utils import Any, dataclass

try:  # Python 3.11+
    from re import _constants as _sre
    from re import _parser as _sre_parse
except ImportError:  # pragma: no cover - Python 3.10
    import sre_constants as _sre  # type: ignore[no-redef]
    import sre_parse as _sre_parse  # type: ignore[no-redef]


@dataclass(frozen=True)
class LikePattern:
    """Prepared SQL LIKE pattern: ``%`` matches any run, ``_`` one character.

    Every other character is literal, the pattern must match the whole value,
    and ``%``/``_`` match line breaks. Each ``%``-free segment matches exactly
    as many characters as it has, so placing each inner segment at its leftmost
    occurrence always leaves the most room for the rest, and no placement is
    ever revisited.

    Attributes:
        head: Segment before the first ``%``; the whole pattern when it has no ``%``.
        middle: Non-empty segments between ``%`` wildcards, in order.
        tail: Segment after the last ``%``, or None when the pattern has no ``%``.
        tail_width: Characters the tail matches.
    """

    head: re.Pattern[str]
    middle: tuple[re.Pattern[str], ...]
    tail: re.Pattern[str] | None
    tail_width: int = 0

    @classmethod
    def compile(cls, pattern: str, *, ignore_case: bool = False) -> LikePattern:
        """Prepare one LIKE pattern.

        Args:
            pattern: SQL LIKE pattern text.
            ignore_case: Whether letters compare case-insensitively (ILIKE).

        Returns:
            LikePattern: The prepared pattern.
        """
        flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
        parts = pattern.split("%")
        segments = [re.compile("".join("." if char == "_" else re.escape(char) for char in part), flags) for part in parts]
        if len(segments) == 1:
            return cls(segments[0], (), None)
        middle = tuple(segment for segment, part in zip(segments[1:-1], parts[1:-1]) if part)
        return cls(segments[0], middle, segments[-1], len(parts[-1]))

    def match(self, text: str) -> bool:
        """Return whether the pattern matches the whole text.

        Args:
            text: Stored value text.

        Returns:
            bool: True when the text matches.
        """
        if self.tail is None:
            return self.head.fullmatch(text) is not None
        head = self.head.match(text)
        end = len(text) - self.tail_width
        if head is None or end < head.end():
            return False
        position = head.end()
        for segment in self.middle:
            found = segment.search(text, position, end)
            if found is None:
                return False
            position = found.end()
        return self.tail.fullmatch(text, end) is not None


# ---------------------------------------------------------------- regex

_MAXREPEAT = _sre.MAXREPEAT
_REPEATS = (_sre.MAX_REPEAT, _sre.MIN_REPEAT)
_POSSESSIVE = getattr(_sre, "POSSESSIVE_REPEAT", None)
_ATOMIC = getattr(_sre, "ATOMIC_GROUP", None)
_CHAR_OPS = (_sre.LITERAL, _sre.NOT_LITERAL, _sre.ANY, _sre.IN)
_LOOKAROUNDS = (_sre.ASSERT, _sre.ASSERT_NOT)
# A regular pattern stays on Python re while its estimated worst-case steps at the text cap stay within this budget.
_RE_STEP_BUDGET = 1e9
# A pattern that needs Python re is rejected above this estimate. At the default 8,192-character cap any
# estimate that grows with the cube of the text length is far above it (8192 ** 3 is about 5.5e11).
_RE_REJECT_BUDGET = 3e9
# Python re backtracks a quantifier over one character class much faster than one over a longer sub-pattern.
_GROUP_ITERATION_WEIGHT = 8.0
_MAX_NFA_NODES = 4096
_MAX_DFA_STATES = 4096
_MAX_CACHED_MOVES = 1 << 18
_MAX_CACHED_CHARS = 1 << 16
_ENUMERABLE = 4096
# Characters the overlap tests may scan while one pattern is planned; past it, classes count as overlapping.
_OVERLAP_BUDGET = 1 << 22
_DISJOINT_CATEGORIES = {
    frozenset(pair)
    for pair in (
        (_sre.CATEGORY_DIGIT, _sre.CATEGORY_NOT_DIGIT),
        (_sre.CATEGORY_SPACE, _sre.CATEGORY_NOT_SPACE),
        (_sre.CATEGORY_WORD, _sre.CATEGORY_NOT_WORD),
        (_sre.CATEGORY_DIGIT, _sre.CATEGORY_SPACE),
        (_sre.CATEGORY_WORD, _sre.CATEGORY_SPACE),
        (_sre.CATEGORY_DIGIT, _sre.CATEGORY_NOT_WORD),
    )
}
_CATEGORY_SOURCE = {
    _sre.CATEGORY_DIGIT: r"\d",
    _sre.CATEGORY_NOT_DIGIT: r"\D",
    _sre.CATEGORY_SPACE: r"\s",
    _sre.CATEGORY_NOT_SPACE: r"\S",
    _sre.CATEGORY_WORD: r"\w",
    _sre.CATEGORY_NOT_WORD: r"\W",
}
_FEATURES = {
    _sre.GROUPREF: "a backreference",
    _sre.GROUPREF_EXISTS: "a conditional group",
    _sre.ASSERT: "a lookaround",
    _sre.ASSERT_NOT: "a lookaround",
}
if _ATOMIC is not None:  # pragma: no branch - Python 3.11+
    _FEATURES[_ATOMIC] = "an atomic group"
if _POSSESSIVE is not None:  # pragma: no branch - Python 3.11+
    _FEATURES[_POSSESSIVE] = "a possessive quantifier"

# One character class: (negated, items). Items are ("lit", code), ("range", lo, hi), ("cat", category) or ("any",).
_CharSet = tuple[bool, tuple[tuple[Any, ...], ...]]
_ANY: _CharSet = (False, (("any",),))
_NEWLINE_SET: _CharSet = (False, (("lit", 10),))
_WORD_SET: _CharSet = (False, (("cat", _sre.CATEGORY_WORD),))
_NOT_WORD_SET: _CharSet = (False, (("cat", _sre.CATEGORY_NOT_WORD),))
# Static facts about one parsed item or sequence: (can match empty, classes a match can start with, always matches).
_Info = tuple[bool, tuple[_CharSet, ...], bool]
# Estimated cost of one item or sequence: (times its end is reached, work per entry, work of an entry that fails).
_Cost = tuple[float, float, float]
_INF = float("inf")


class _NotRegular(Exception):
    """Raised while building the linear matcher for a construct it does not support."""


class _TooLarge(Exception):
    """Raised while building the linear matcher when the pattern expands past its size limit."""


class _Full(Exception):
    """Raised by the linear matcher when its state table is full; the search then runs uncached."""


def _power(base: float, exponent: float) -> float:
    if base <= 1.0:
        return 1.0
    if exponent == _INF or exponent * math.log(base) > 700.0:
        return _INF
    return float(base**exponent)


_CASED: str | None = None


def _cased() -> str:
    """Return every character whose case-insensitive match can differ from exact equality.

    A character outside this set has no lowercase, uppercase or casefold
    variant, so Python ``re`` matches it under IGNORECASE exactly as without
    the flag, against literals, ranges and categories alike.
    """
    global _CASED
    if _CASED is None:
        found: list[str] = []
        for base in range(0, 0x110000, 0x1000):
            chunk = "".join(map(chr, range(base, base + 0x1000)))
            if chunk.lower() == chunk and chunk.upper() == chunk and chunk.casefold() == chunk:
                continue
            found.extend(char for char in chunk if char.lower() != char or char.upper() != char or char.casefold() != char)
        _CASED = "".join(found)
    return _CASED


def _charset_source(charset: _CharSet) -> str:
    """Return regex source for one character class of the cost model."""
    negated, items = charset
    parts = []
    for item in items:
        if item[0] == "lit":
            parts.append("\\U%08x" % item[1])
        elif item[0] == "range":
            parts.append("\\U%08x-\\U%08x" % (item[1], item[2]))
        elif item[0] == "cat" and item[1] in _CATEGORY_SOURCE:
            parts.append(_CATEGORY_SOURCE[item[1]])
        else:
            parts.append("\\x00-\\U0010ffff")
    return "[" + ("^" if negated else "") + "".join(parts) + "]"


def _ranges(charset: _CharSet) -> list[tuple[int, int]] | None:
    """Return the sorted, merged code ranges of a class of literals and ranges, or None."""
    negated, items = charset
    if negated:
        return None
    spans = []
    for item in items:
        if item[0] == "lit":
            spans.append((item[1], item[1]))
        elif item[0] == "range":
            spans.append((item[1], item[2]))
        else:
            return None
    spans.sort()
    merged: list[tuple[int, int]] = []
    for low, high in spans:
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _intersects(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> bool:
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] < right[j][0]:
            i += 1
        elif right[j][1] < left[i][0]:
            j += 1
        else:
            return True
    return False


class _CostModel:
    """Static estimate of Python ``re`` backtracking on one parsed pattern.

    The estimate counts steps for the worst text of ``n`` characters, where
    ``n`` is the configured text cap. A choice point (a variable quantifier or
    an alternation) multiplies the work of what follows it only when something
    after it in the same atomic scope can fail. A quantifier's iteration counts
    lead to different continuation positions only when the continuation can
    start with a character the quantifier also matches. Repeating a choice
    point inside a quantifier compounds its choices per iteration. Atomic
    groups, possessive quantifiers and lookarounds are never re-entered after
    they finish, so their work adds instead of multiplying. A backreference
    compares at most its group's width, and when the character before it
    cannot belong to the group, its comparisons together read the text about
    once per capture. Character classes are compared with Python ``re``
    itself, so IGNORECASE folds exactly as the stdlib folds it.
    """

    def __init__(self, parsed: Any, text_chars: int) -> None:
        flags = int(parsed.state.flags)
        self.multiline = bool(flags & re.MULTILINE)
        self.ignore_case = bool(flags & re.IGNORECASE)
        self.dotall = bool(flags & re.DOTALL)
        self.ascii = bool(flags & re.ASCII)
        self.mixed_types = False
        self._scan_scoped_flags(parsed)
        self.n = float(max(int(text_chars), 1))
        self.groups: dict[int, _Info] = {}
        self.group_classes: dict[int, tuple[_CharSet, ...]] = {}
        self.group_widths: dict[int, float] = {}
        self.infos: dict[int, _Info] = {}
        self.overlap_memo: dict[frozenset[_CharSet], bool] = {}
        self.class_patterns: dict[_CharSet, tuple[re.Pattern[str], ...]] = {}
        self.members: dict[_CharSet, str | None] = {}
        self.cased_members: dict[_CharSet, str] = {}
        self.class_lists: dict[int, tuple[_CharSet, ...]] = {}
        self.budget = _OVERLAP_BUDGET
        self.single_path: set[int] = set()
        self.attempts: dict[int, float] = {}
        self.info(parsed)

    def _scan_scoped_flags(self, parsed: Any) -> None:
        # Conservative: a flag scoped to any group applies to the whole analysis.
        stack = [parsed]
        while stack:
            for op, av in stack.pop():
                if op is _sre.SUBPATTERN:
                    self.ignore_case |= bool(av[1] & re.IGNORECASE)
                    self.dotall |= bool(av[1] & re.DOTALL)
                    self.mixed_types |= bool((av[1] | av[2]) & (re.ASCII | re.UNICODE))
                stack.extend(_children(op, av))

    # -- character sets

    def charset(self, op: Any, av: Any) -> _CharSet | None:
        if op is _sre.LITERAL:
            return (False, (("lit", av),))
        if op is _sre.NOT_LITERAL:
            return (True, (("lit", av),))
        if op is _sre.ANY:
            return _ANY if self.dotall else (True, (("lit", 10),))
        if op is _sre.IN:
            negated = any(item_op is _sre.NEGATE for item_op, _ in av)
            items: list[tuple[Any, ...]] = []
            for item_op, item_av in av:
                if item_op is _sre.LITERAL:
                    items.append(("lit", item_av))
                elif item_op is _sre.RANGE:
                    items.append(("range", item_av[0], item_av[1]))
                elif item_op is _sre.CATEGORY:
                    items.append(("cat", item_av))
                elif item_op is not _sre.NEGATE:
                    items.append(("any",))
            return (negated, tuple(items))
        return None

    def _patterns(self, charset: _CharSet) -> tuple[re.Pattern[str], ...]:
        """Return the class compiled as Python ``re`` matches it here (both ASCII modes when the pattern mixes them)."""
        known = self.class_patterns.get(charset)
        if known is None:
            letters = "i" if self.ignore_case else ""
            modes = (self.ascii, not self.ascii) if self.mixed_types else (self.ascii,)
            source = _charset_source(charset)
            known = tuple(re.compile(f"(?{letters}{'a' if ascii_mode else 'u'})" + source) for ascii_mode in modes)
            self.class_patterns[charset] = known
        return known

    def _scan(self, charset: _CharSet, chars: str) -> bool:
        """Return whether the class matches any of the characters; True once the scan budget is spent."""
        self.budget -= len(chars)
        if self.budget < 0:
            return True
        return any(pattern.search(chars) is not None for pattern in self._patterns(charset))

    def _members(self, charset: _CharSet) -> str | None:
        """Return the characters of a class of at most ``_ENUMERABLE`` literals and range members, or None."""
        if charset in self.members:
            return self.members[charset]
        spans = _ranges(charset)
        chars = None
        if spans is not None and sum(high - low + 1 for low, high in spans) <= _ENUMERABLE:
            chars = "".join(chr(code) for low, high in spans for code in range(low, high + 1))
        self.members[charset] = chars
        return chars

    def _cased_members(self, charset: _CharSet) -> str:
        """Return the case-sensitive characters this class matches under IGNORECASE."""
        known = self.cased_members.get(charset)
        if known is None:
            known = "".join(dict.fromkeys(char for pattern in self._patterns(charset) for char in pattern.findall(_cased())))
            self.cased_members[charset] = known
        return known

    def overlap(self, left: _CharSet, right: _CharSet) -> bool:
        """Return False only when the two classes provably share no character."""
        key = frozenset((left, right))
        known = self.overlap_memo.get(key)
        if known is None:
            known = self._overlap(left, right)
            self.overlap_memo[key] = known
        return known

    def _overlap(self, left: _CharSet, right: _CharSet) -> bool:
        # A shared character either has case variants (it is then among one class's cased members under
        # IGNORECASE) or both classes match it exactly as without the flag.
        if self.ignore_case and self._scan(right, self._cased_members(left)):
            return True
        spans_left, spans_right = _ranges(left), _ranges(right)
        if spans_left is not None and spans_right is not None:
            return _intersects(spans_left, spans_right)
        for first, second in ((left, right), (right, left)):
            chars = self._members(first)
            if chars is not None:
                return self._scan(second, chars)
        if not left[0] and not right[0] and not self.mixed_types:
            categories_left = [item[1] for item in left[1] if item[0] == "cat"]
            categories_right = [item[1] for item in right[1] if item[0] == "cat"]
            if len(categories_left) == len(left[1]) and len(categories_right) == len(right[1]):
                return any(frozenset((a, b)) not in _DISJOINT_CATEGORIES for a in categories_left for b in categories_right)
        return True

    def overlaps(self, left: tuple[_CharSet, ...], right: tuple[_CharSet, ...]) -> bool:
        for a in left:
            for b in right:
                if self.overlap(a, b):
                    return True
        return False

    def classes(self, seq: Any) -> tuple[_CharSet, ...]:
        """Return every class a match of the sequence, lookarounds included, can read a character with."""
        found: list[_CharSet] = []
        for item in seq:
            found.extend(self.item_classes(item))
        return tuple(dict.fromkeys(found))

    def item_classes(self, item: Any) -> tuple[_CharSet, ...]:
        """Return every class a match of one parsed item, lookarounds included, can read a character with."""
        known = self.class_lists.get(id(item))
        if known is None:
            found: list[_CharSet] = []
            stack = [item]
            while stack:
                op, av = stack.pop()
                charset = self.charset(op, av)
                if charset is not None:
                    found.append(charset)
                elif op is _sre.GROUPREF:
                    found.extend(self.group_classes.get(av, (_ANY,)))
                else:
                    for child in _children(op, av):
                        stack.extend(child)
            known = tuple(dict.fromkeys(found))
            self.class_lists[id(item)] = known
        return known

    # -- static facts, one recursion level per nesting level

    def info(self, seq: Any) -> _Info:
        nullable = True
        first: list[_CharSet] = []
        cannot_fail = True
        for item in seq:
            op, av = item
            charset = self.charset(op, av)
            own: _Info
            if charset is not None:
                own = (False, (charset,), False)
            elif op is _sre.AT or op in _LOOKAROUNDS:
                if op in _LOOKAROUNDS:
                    self.info(av[1])
                own = (True, (), False)
            elif op in _REPEATS or op is _POSSESSIVE:
                body = self.info(av[2])
                own = (av[0] == 0 or body[0], body[1], av[0] == 0 or body[2])
            elif op is _sre.SUBPATTERN:
                own = self.info(av[3])
                if av[0] is not None:
                    self.groups[av[0]] = own
                    self.group_classes[av[0]] = self.classes(av[3])
                    self.group_widths[av[0]] = min(float(av[3].getwidth()[1]), self.n)
            elif op is _ATOMIC:
                own = self.info(av)
            elif op is _sre.BRANCH:
                branch_nullable = False
                branch_first: list[_CharSet] = []
                branch_cannot_fail = False
                for branch in av[1]:
                    branch_info = self.info(branch)
                    branch_nullable = branch_nullable or branch_info[0]
                    branch_first.extend(branch_info[1])
                    branch_cannot_fail = branch_cannot_fail or branch_info[2]
                own = (branch_nullable, tuple(dict.fromkeys(branch_first)), branch_cannot_fail)
            elif op is _sre.GROUPREF:
                group = self.groups.get(av, (True, (_ANY,), False))
                own = (group[0], group[1], False)
            elif op is _sre.GROUPREF_EXISTS:
                yes = self.info(av[1])
                no = self.info(av[2]) if av[2] is not None else (True, (), True)
                own = (yes[0] or no[0], tuple(dict.fromkeys(yes[1] + no[1])), yes[2] and no[2])
            else:
                own = (True, (_ANY,), False)
            self.infos[id(item)] = own
            if nullable:
                first.extend(own[1])
            nullable = nullable and own[0]
            cannot_fail = cannot_fail and own[2]
        result = (nullable, tuple(dict.fromkeys(first)), cannot_fail)
        self.infos[id(seq)] = result
        return result

    def _ambiguous(self, branches: Any) -> bool:
        infos = [self.infos[id(branch)] for branch in branches]
        if any(info[0] for info in infos):
            return True
        return any(self.overlaps(a[1], b[1]) for a, b in combinations(infos, 2))

    # -- cost, one recursion level per nesting level

    def attempt(self, seq: Any) -> float:
        """Estimate the work of starting a sequence up to its first consuming item, when that item fails at once."""
        known = self.attempts.get(id(seq))
        if known is None:
            known = 0.0
            for item in seq:
                known += self.attempt_item(item)
                if not self.infos[id(item)][0]:
                    break
            known = max(known, 1.0)
            self.attempts[id(seq)] = known
        return known

    def attempt_item(self, item: Any) -> float:
        """Estimate the work of trying one item where it matches nothing or fails at its first character."""
        known = self.attempts.get(id(item))
        if known is not None:
            return known
        op, av = item
        if op in _CHAR_OPS or op is _sre.AT:
            known = 1.0
        elif op is _sre.SUBPATTERN:
            known = self.attempt(av[3])
        elif op is _sre.BRANCH:
            known = 0.0
            for branch in av[1]:
                known += self.attempt(branch)
        elif op in _REPEATS or op is _POSSESSIVE:
            known = _iteration_weight(av[2]) * (1.0 + self.attempt(av[2]))
        elif op is _ATOMIC or op in _LOOKAROUNDS:
            # These run to completion wherever they are tried.
            known = self.cost(av if op is _ATOMIC else av[1], (), False, 1.0)[1]
        elif op is _sre.GROUPREF:
            known = 2.0
        elif op is _sre.GROUPREF_EXISTS:
            known = max(self.attempt(av[1]), self.attempt(av[2]) if av[2] is not None else 1.0)
        else:
            known = _INF
        self.attempts[id(item)] = known
        return known

    def cost(self, seq: Any, follow: tuple[_CharSet, ...], tail_fails: bool, lead: float) -> _Cost:
        """Estimate one entry into a sequence.

        Args:
            seq: Parsed sequence.
            follow: Classes the continuation after the sequence can start with.
            tail_fails: Whether the continuation, within the same atomic scope, can fail.
            lead: Work of trying the continuation when it fails at its first consuming item.

        Returns:
            _Cost: Times the end is reached, work per entry, and work of an entry that fails.
        """
        items = list(seq)
        follows: list[tuple[_CharSet, ...]] = [()] * len(items)
        fails = [False] * len(items)
        leads = [1.0] * len(items)
        after, failing, after_lead = follow, tail_fails, lead
        for index in range(len(items) - 1, -1, -1):
            follows[index], fails[index], leads[index] = after, failing, after_lead
            item = items[index]
            info = self.infos[id(item)]
            after = tuple(dict.fromkeys(info[1] + after)) if info[0] else info[1]
            failing = failing or not info[2]
            own_lead = self.attempt_item(item)
            after_lead = own_lead + after_lead if info[0] else own_lead
            if index + 1 < len(items) and self._boundary(items[index + 1]) and not info[0] and not self.mixed_types:
                # After an item that ends with a word character, \b lets only a non-word character follow, and the
                # reverse.
                consumed = self.item_classes(item)
                if not self.overlaps(consumed, (_NOT_WORD_SET,)):
                    follows[index] = (_NOT_WORD_SET,)
                elif not self.overlaps(consumed, (_WORD_SET,)):
                    follows[index] = (_WORD_SET,)
        n = self.n
        paths, work = 1.0, 0.0
        captures: dict[int, float] = {}
        for index, (op, av) in enumerate(items):
            follow_here, fails_here, lead_here = follows[index], fails[index], leads[index]
            own: _Cost
            extra = 0.0
            if op in _CHAR_OPS or op is _sre.AT:
                own = (1.0, 1.0, 1.0)
            elif op is _sre.SUBPATTERN:
                own = self.cost(av[3], follow_here, fails_here, lead_here)
                if av[0] is not None:
                    captures[av[0]] = paths * own[0]
            elif op is _sre.BRANCH:
                branch_paths, branch_work = [], 0.0
                for branch in av[1]:
                    branch_cost = self.cost(branch, follow_here, fails_here, lead_here)
                    branch_paths.append(branch_cost[0])
                    branch_work += branch_cost[1]
                ambiguous = self._ambiguous(av[1])
                own = (sum(branch_paths) if ambiguous else max(branch_paths), branch_work, branch_work)
            elif op in _REPEATS or op is _POSSESSIVE:
                low, high, body = av
                possessive = op is _POSSESSIVE
                first = self.infos[id(body)][1]
                iterations = n + 1.0 if high == _MAXREPEAT or high > n else float(high)
                choices = max(iterations - low + 1.0, 1.0)
                inner_fails = (fails_here and not possessive) or low > 1
                body_follow = first if possessive else tuple(dict.fromkeys(first + follow_here))
                weight = _iteration_weight(body)
                body_lead = weight * self.attempt(body) + (0.0 if possessive else lead_here)
                body_paths, body_work, _ = self.cost(body, body_follow, inner_fails, body_lead)
                explore = _power(body_paths, iterations) if inner_fails else 1.0
                repeat_work = weight * (iterations * body_work * explore + choices)
                if possessive or not fails_here or id(items[index]) in self.single_path:
                    repeat_paths = 1.0
                else:
                    # Each iteration count is tried with the continuation after it.
                    repeat_work += choices * lead_here
                    varies = high != low and self.overlaps(first, follow_here)
                    repeat_paths = (choices if varies else 1.0) * explore
                fail_work = weight * (low * body_work * (_power(body_paths, low) if low > 1 else 1.0) + 1.0)
                own = (repeat_paths, repeat_work, fail_work)
            elif op is _ATOMIC or op in _LOOKAROUNDS:
                inner = self.cost(av if op is _ATOMIC else av[1], (), False, 1.0)
                own = (1.0, inner[1], inner[1])
            elif op is _sre.GROUPREF:
                # A backreference compares at most the referenced group's width.
                width = self.group_widths.get(av, n)
                if self._guarded(items, index, av):
                    # Every comparison past the first character reads text no other comparison of the same capture
                    # reads, so all of them together read at most the text once per capture.
                    own = (1.0, 1.0, 1.0)
                    extra = min(n * captures.get(av, paths), paths * width)
                else:
                    # When nothing after it can fail, the first comparison that succeeds ends the search, and a
                    # failing one stops at the first character that differs.
                    own = (1.0, width, width if fails_here else 1.0)
            elif op is _sre.GROUPREF_EXISTS:
                yes = self.cost(av[1], follow_here, fails_here, lead_here)
                no = self.cost(av[2], follow_here, fails_here, lead_here) if av[2] is not None else (1.0, 1.0, 1.0)
                own = (max(yes[0], no[0]), max(yes[1], no[1]), max(yes[2], no[2]))
            else:
                own = (_INF, _INF, _INF)
            if fails_here:
                work += paths * own[1] + extra
                paths *= own[0]
            else:
                # Nothing after this item can fail, so only its last entry can succeed and end the scope.
                work += own[1] + ((paths - 1.0) * own[2] if paths > 1.0 else 0.0) + extra
                paths = 1.0
        work = max(work, 1.0)
        return (paths if tail_fails else 1.0), work, work

    @staticmethod
    def _boundary(item: Any) -> bool:
        return item[0] is _sre.AT and item[1] == _sre.AT_BOUNDARY

    def _guarded(self, items: list[Any], index: int, group: int) -> bool:
        """Return whether the text a backreference compares can never start inside an earlier comparison.

        That holds when the character just before the comparison cannot belong
        to the group: the previous consuming item is a class disjoint from the
        group's classes, or a ``\\b`` sits between it and the backreference and
        the group consumes only word or only non-word characters.
        """
        consumed = self.group_classes.get(group)
        if consumed is None:
            return False
        position = index - 1
        boundary = False
        while position >= 0 and items[position][0] is _sre.AT:
            boundary = boundary or self._boundary(items[position])
            position -= 1
        if boundary and not self.mixed_types and (not self.overlaps(consumed, (_NOT_WORD_SET,)) or not self.overlaps(consumed, (_WORD_SET,))):
            return True
        if position < 0:
            return False
        before = self.charset(*items[position])
        return before is not None and not self.overlaps((before,), consumed)

    def search_steps(self, parsed: Any) -> float:
        """Estimate the worst-case steps of ``re.search`` on a text at the cap."""
        work = self.cost(parsed, (), False, 1.0)[1]
        lead = parsed[0] if len(parsed) else None
        anchored = lead is not None and lead[0] is _sre.AT and (lead[1] == _sre.AT_BEGINNING_STRING or (lead[1] == _sre.AT_BEGINNING and not self.multiline))
        if anchored:
            steps = work + self.n
        elif lead is not None and lead[0] is _sre.AT and lead[1] == _sre.AT_BEGINNING and not self.overlaps(self.classes(parsed), (_NEWLINE_SET,)):
            # A multiline ^ starts the match only after a line break, and a pattern that cannot consume one stays
            # within that line, so the starts together cost no more than one start on the whole text.
            steps = work + (self.n + 1.0) * self.attempt(parsed)
        else:
            steps = (self.n + 1.0) * work
            lead_repeat = self._lead_repeat(parsed)
            if lead_repeat is not None:
                # The repeat's runs after different starts are disjoint, so its iteration counts over all starts sum
                # to at most the text length: every start but the costliest one counts it as a single path.
                self.single_path.add(id(lead_repeat))
                single = self.cost(parsed, (), False, 1.0)[1]
                self.single_path.clear()
                steps = min(steps, (self.n + 1.0) * single + work)
        return _INF if math.isnan(steps) else steps

    def _lead_repeat(self, parsed: Any) -> Any:
        """Return the repeat over one class that directly follows a leading class disjoint from it, or None."""
        if len(parsed) < 2:
            return None
        before = self.charset(*parsed[0])
        item = parsed[1]
        while item[0] is _sre.SUBPATTERN and len(item[1][3]):
            item = item[1][3][0]
        if before is None or item[0] not in _REPEATS:
            return None
        body = item[1][2]
        inner = self.charset(*body[0]) if len(body) == 1 else None
        return item if inner is not None and not self.overlap(before, inner) else None


# -- linear-time matcher

_CHAR, _SPLIT, _ASSERT, _MATCH = range(4)
_A_BEGIN, _A_BEGIN_LINE, _A_END, _A_END_LINE, _A_END_STRING, _A_BOUND, _A_NOT_BOUND, _A_BOUND_ASCII, _A_NOT_BOUND_ASCII = range(9)
# Context bits of the character before a position (or of the character after it).
_START, _NEWLINE, _WORD, _WORD_ASCII = 1, 2, 4, 8
_WORD_RE = re.compile(r"\w")
_WORD_ASCII_RE = re.compile(r"\w", re.ASCII)


def _iteration_weight(body: Any) -> float:
    """Return the relative cost of one iteration of a quantifier over ``body`` in Python re."""
    return 1.0 if len(body) == 1 and body[0][0] in _CHAR_OPS else _GROUP_ITERATION_WEIGHT


def _scoped_flags(flags: int, add: int, delete: int) -> int:
    """Return the flags inside ``(?add-delete:...)``, as the stdlib compiler combines them."""
    if add & (re.ASCII | re.UNICODE):
        flags &= ~(re.ASCII | re.UNICODE)
    return (flags | add) & ~delete


def _children(op: Any, av: Any) -> list[Any]:
    """Return the nested sequences of one parsed item."""
    if op in _REPEATS or op is _POSSESSIVE:
        return [av[2]]
    if op is _sre.SUBPATTERN:
        return [av[3]]
    if op is _ATOMIC:
        return [av]
    if op in _LOOKAROUNDS:
        return [av[1]]
    if op is _sre.BRANCH:
        return list(av[1])
    if op is _sre.GROUPREF_EXISTS:
        return [av[1]] + ([av[2]] if av[2] is not None else [])
    return []


def _class_source(op: Any, av: Any) -> str:
    """Return regex source matching exactly the characters one parsed item matches."""
    if op is _sre.LITERAL:
        return "\\U%08x" % av
    if op is _sre.NOT_LITERAL:
        return "[^\\U%08x]" % av
    if op is _sre.ANY:
        return "."
    parts = []
    negated = ""
    for item_op, item_av in av:
        if item_op is _sre.NEGATE:
            negated = "^"
        elif item_op is _sre.LITERAL:
            parts.append("\\U%08x" % item_av)
        elif item_op is _sre.RANGE:
            parts.append("\\U%08x-\\U%08x" % item_av)
        elif item_op is _sre.CATEGORY and item_av in _CATEGORY_SOURCE:
            parts.append(_CATEGORY_SOURCE[item_av])
        else:
            raise _NotRegular(f"the class item {item_op}")
    return "[" + negated + "".join(parts) + "]"


class _Nfa:
    """Thompson automaton over one parsed pattern, built right to left."""

    def __init__(self) -> None:
        self.kind: list[int] = [_MATCH]
        self.out: list[tuple[int, ...]] = [()]
        self.arg: list[int] = [0]
        self.sets: list[re.Pattern[str]] = []
        self.set_index: dict[str, int] = {}
        self.assertions: set[int] = set()

    def node(self, kind: int, out: tuple[int, ...], arg: int = 0) -> int:
        if len(self.kind) >= _MAX_NFA_NODES:
            raise _TooLarge()
        self.kind.append(kind)
        self.out.append(out)
        self.arg.append(arg)
        return len(self.kind) - 1

    def charset(self, op: Any, av: Any, flags: int) -> int:
        letters = ("i" if flags & re.IGNORECASE else "") + ("s" if flags & re.DOTALL else "") + ("a" if flags & re.ASCII else "")
        source = (f"(?{letters})" if letters else "") + _class_source(op, av)
        index = self.set_index.get(source)
        if index is None:
            index = len(self.sets)
            self.sets.append(re.compile(source))
            self.set_index[source] = index
        return index

    def assertion(self, code: Any, flags: int) -> int:
        if code == _sre.AT_BEGINNING:
            kind = _A_BEGIN_LINE if flags & re.MULTILINE else _A_BEGIN
        elif code == _sre.AT_BEGINNING_STRING:
            kind = _A_BEGIN
        elif code == _sre.AT_END:
            kind = _A_END_LINE if flags & re.MULTILINE else _A_END
        elif code == _sre.AT_END_STRING:
            kind = _A_END_STRING
        elif code == _sre.AT_BOUNDARY:
            kind = _A_BOUND_ASCII if flags & re.ASCII else _A_BOUND
        elif code == _sre.AT_NON_BOUNDARY:
            kind = _A_NOT_BOUND_ASCII if flags & re.ASCII else _A_NOT_BOUND
        else:
            raise _NotRegular(f"the anchor {code}")
        self.assertions.add(kind)
        return kind

    def build(self, seq: Any, flags: int, cont: int) -> int:
        """Return the entry node of a sequence whose matches continue at ``cont``."""
        for op, av in reversed(list(seq)):
            if op in _CHAR_OPS:
                cont = self.node(_CHAR, (cont,), self.charset(op, av, flags))
            elif op is _sre.AT:
                cont = self.node(_ASSERT, (cont,), self.assertion(av, flags))
            elif op is _sre.SUBPATTERN:
                scoped = _scoped_flags(flags, av[1], av[2])
                if flags & re.ASCII and not scoped & re.ASCII:
                    # CPython's search prefilter tests a leading class of such a group with ASCII \w, \d and \s.
                    raise _NotRegular("a (?u:...) group inside an ASCII pattern")
                cont = self.build(av[3], scoped, cont)
            elif op is _sre.BRANCH:
                entries = []
                for branch in av[1]:
                    entries.append(self.build(branch, flags, cont))
                cont = self.node(_SPLIT, tuple(entries))
            elif op in _REPEATS:
                low, high, body = av
                if low > _MAX_NFA_NODES or (high != _MAXREPEAT and high - low > _MAX_NFA_NODES):
                    raise _TooLarge()
                if high == _MAXREPEAT:
                    loop = self.node(_SPLIT, ())
                    self.out[loop] = (self.build(body, flags, loop), cont)
                    tail = loop
                else:
                    tail = cont
                    for _ in range(high - low):
                        tail = self.node(_SPLIT, (self.build(body, flags, tail), cont))
                for _ in range(low):
                    tail = self.build(body, flags, tail)
                cont = tail
            else:
                raise _NotRegular(_FEATURES.get(op, f"the {op} construct"))
        return cont


class _Dfa:
    """One generation of the lazily built DFA: its states and cached transitions."""

    def __init__(self) -> None:
        self.states: list[tuple[frozenset[int], int]] = []
        self.state_index: dict[tuple[frozenset[int], int], int] = {}
        self.rows: list[dict[str, int]] = []
        self.moves: dict[tuple[int, int, bool], int] = {}
        self.ends: dict[tuple[int, bool], bool] = {}
        self.cached = 0


class _LinearRegex:
    """Answer ``re.search(...) is not None`` for a regular pattern in time linear in the text.

    The automaton is simulated as a lazily built DFA: a state is the set of
    automaton nodes that can continue a match, plus the context bits of the
    previous character that anchors and word boundaries need. Per character the
    search does one cached table lookup, or one pass over the automaton when the
    transition is new, so its work is at most proportional to the text length
    times the expanded pattern size. When a text needs more than
    ``_MAX_DFA_STATES`` states, the rest of that text is simulated without the
    table and the next text starts a fresh one.
    """

    def __init__(self, nfa: _Nfa, start: int) -> None:
        self._nfa = nfa
        self._kind = tuple(nfa.kind)
        self._out = tuple(nfa.out)
        self._arg = tuple(nfa.arg)
        self._sets = tuple(nfa.sets)
        used = nfa.assertions
        self._needs_last = _A_END in used
        mask = 0
        if used & {_A_BEGIN_LINE, _A_END, _A_END_LINE}:
            mask |= _NEWLINE
        if used & {_A_BOUND, _A_NOT_BOUND}:
            mask |= _WORD
        if used & {_A_BOUND_ASCII, _A_NOT_BOUND_ASCII}:
            mask |= _WORD_ASCII
        self._mask = mask
        self._start_node = start
        self._lock = threading.Lock()
        self._chars: dict[str, int] = {}
        self._classes: list[tuple[frozenset[int], int]] = []
        self._class_index: dict[tuple[frozenset[int], int], int] = {}
        self._dfa = self._fresh()

    def __reduce__(self) -> tuple[Any, ...]:
        # Copies and pickles rebuild the automaton with empty caches and a fresh lock.
        return (_LinearRegex, (self._nfa, self._start_node))

    def _fresh(self) -> _Dfa:
        """Return an empty DFA generation whose state 0 is the start state."""
        dfa = _Dfa()
        self._intern(dfa, frozenset((self._start_node,)), _START)
        return dfa

    def _intern(self, dfa: _Dfa, nodes: frozenset[int], prev: int) -> int:
        key = (nodes, prev)
        state = dfa.state_index.get(key)
        if state is None:
            with self._lock:
                state = dfa.state_index.get(key)
                if state is None:
                    if len(dfa.states) >= _MAX_DFA_STATES:
                        raise _Full()
                    state = len(dfa.states)
                    dfa.states.append(key)
                    dfa.rows.append({})
                    dfa.state_index[key] = state
        return state

    def _classify(self, char: str) -> int:
        known = self._chars.get(char)
        if known is not None:
            return known
        members = frozenset(index for index, pattern in enumerate(self._sets) if pattern.match(char))
        bits = (_NEWLINE if char == "\n" else 0) | (_WORD if _WORD_RE.match(char) else 0) | (_WORD_ASCII if _WORD_ASCII_RE.match(char) else 0)
        key = (members, bits & self._mask)
        with self._lock:
            index = self._class_index.get(key)
            if index is None:
                index = len(self._classes)
                self._classes.append(key)
                self._class_index[key] = index
        if len(self._chars) < _MAX_CACHED_CHARS:
            self._chars[char] = index
        return index

    def _holds(self, kind: int, prev: int, after: int | None, last: bool, empty: bool) -> bool:
        if kind == _A_BEGIN:
            return bool(prev & _START)
        if kind == _A_BEGIN_LINE:
            return bool(prev & (_START | _NEWLINE))
        if kind == _A_END_STRING:
            return after is None
        if kind == _A_END:
            return after is None or (last and bool(after & _NEWLINE))
        if kind == _A_END_LINE:
            return after is None or bool(after & _NEWLINE)
        if empty:
            # Python re: \b and \B never match inside an empty string.
            return False
        bit = _WORD if kind in (_A_BOUND, _A_NOT_BOUND) else _WORD_ASCII
        boundary = bool(prev & bit) != (after is not None and bool(after & bit))
        return boundary if kind in (_A_BOUND, _A_BOUND_ASCII) else not boundary

    def _closure(self, nodes: frozenset[int], prev: int, after: int | None, last: bool, empty: bool) -> list[int] | None:
        """Return the character nodes reachable without consuming, or None when a match ends here."""
        kinds, outs = self._kind, self._out
        stack = list(nodes)
        seen = set(nodes)
        chars: list[int] = []
        while stack:
            node = stack.pop()
            kind = kinds[node]
            if kind == _MATCH:
                return None
            if kind == _CHAR:
                chars.append(node)
                continue
            if kind == _ASSERT and not self._holds(self._arg[node], prev, after, last, empty):
                continue
            for target in outs[node]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return chars

    def _step(self, nodes: frozenset[int], prev: int, klass: int, last: bool) -> tuple[frozenset[int], int] | None:
        members, bits = self._classes[klass]
        chars = self._closure(nodes, prev, bits, last, False)
        if chars is None:
            return None
        targets = {self._start_node}
        for node in chars:
            if self._arg[node] in members:
                targets.add(self._out[node][0])
        return frozenset(targets), bits

    def _move(self, dfa: _Dfa, state: int, klass: int, last: bool) -> int:
        key = (state, klass, last)
        target = dfa.moves.get(key)
        if target is None:
            nodes, prev = dfa.states[state]
            step = self._step(nodes, prev, klass, last)
            target = -1 if step is None else self._intern(dfa, *step)
            if len(dfa.moves) < _MAX_CACHED_MOVES:
                dfa.moves[key] = target
        return target

    def _finish(self, dfa: _Dfa, state: int, empty: bool) -> bool:
        key = (state, empty)
        found = dfa.ends.get(key)
        if found is None:
            nodes, prev = dfa.states[state]
            found = self._closure(nodes, prev, None, False, empty) is None
            dfa.ends[key] = found
        return found

    def search(self, text: str) -> bool:
        """Return whether the pattern matches anywhere in the text.

        Args:
            text: Text to search.

        Returns:
            bool: True when ``re.search`` would find a match.
        """
        dfa = self._dfa
        if not text:
            return self._finish(dfa, 0, True)
        try:
            return self._search(dfa, text)
        except _Full:
            # Later texts start a fresh table instead of simulating every text that needs a new state.
            fresh = self._fresh()
            with self._lock:
                if self._dfa is dfa:
                    self._dfa = fresh
            return self._simulate(text)

    def _search(self, dfa: _Dfa, text: str) -> bool:
        rows = dfa.rows
        state = 0
        body = text[:-1] if self._needs_last else text
        for char in body:
            row = rows[state]
            target = row.get(char)
            if target is None:
                target = self._move(dfa, state, self._classify(char), False)
                if dfa.cached < _MAX_CACHED_MOVES:
                    dfa.cached += 1
                    row[char] = target
            if target < 0:
                return True
            state = target
        if self._needs_last:
            state = self._move(dfa, state, self._classify(text[-1]), True)
            if state < 0:
                return True
        return self._finish(dfa, state, False)

    def _simulate(self, text: str) -> bool:
        nodes, prev = frozenset((self._start_node,)), _START
        last_index = len(text) - 1
        for index, char in enumerate(text):
            step = self._step(nodes, prev, self._classify(char), index == last_index)
            if step is None:
                return True
            nodes, prev = step[0], step[1] & self._mask
        return self._closure(nodes, prev, None, False, False) is None


@dataclass(frozen=True)
class RegexSearch:
    """Prepared regex filter with Python ``re.search`` semantics.

    Attributes:
        pattern: The compiled Python pattern.
        linear: Linear-time matcher used instead of Python ``re``, or None.
    """

    pattern: re.Pattern[str]
    linear: _LinearRegex | None = None

    @property
    def engine(self) -> str:
        """str: ``"re"`` or ``"linear"``, the matcher this pattern runs on."""
        return "re" if self.linear is None else "linear"

    def search(self, text: str) -> bool:
        """Return whether the pattern matches anywhere in the text.

        Args:
            text: Text to search.

        Returns:
            bool: The same answer as ``pattern.search(text) is not None``.
        """
        if self.linear is None:
            return self.pattern.search(text) is not None
        return self.linear.search(text)

    @classmethod
    def plan(cls, pattern: re.Pattern[str], *, max_text_chars: int) -> tuple[RegexSearch, str | None]:
        """Choose the matcher for one compiled pattern.

        A pattern stays on Python ``re`` when the static estimate of its
        worst-case backtracking on a text of ``max_text_chars`` characters is
        within a step budget. Otherwise a pattern made only of regular
        constructs runs on the linear-time matcher, which gives the same
        answers. A pattern that needs Python ``re`` (backreferences,
        lookarounds, conditionals, atomic groups, possessive quantifiers) or is
        too large for the linear matcher stays on Python ``re`` up to a larger
        budget, and gets a reason above it.

        Args:
            pattern: Pattern compiled from source text with no flags argument.
            max_text_chars: Longest text the filter accepts.

        Returns:
            tuple[RegexSearch, str | None]: The matcher, and why the pattern is
            unsafe on Python ``re`` when no linear matcher could replace it
            (the matcher is then plain Python ``re``).
        """
        fallback = cls(pattern)
        steps = 0.0
        try:
            parsed = _sre_parse.parse(pattern.pattern)
            steps = _CostModel(parsed, max_text_chars).search_steps(parsed)
            if steps <= _RE_STEP_BUDGET:
                return fallback, None
            nfa = _Nfa()
            start = nfa.build(parsed, int(parsed.state.flags), 0)
        except RecursionError:
            return fallback, "it is nested too deeply to check"
        except (_NotRegular, _TooLarge) as exc:
            if steps <= _RE_REJECT_BUDGET:
                return fallback, None
            why = (
                f"uses {exc}, which the linear-time matcher does not support"
                if isinstance(exc, _NotRegular)
                else f"expands past the linear-time matcher's {_MAX_NFA_NODES} nodes"
            )
            return (
                fallback,
                f"its estimated worst-case backtracking in Python re on texts of up to {max_text_chars} characters exceeds {_RE_REJECT_BUDGET:.0e} steps, and it {why}",
            )
        return cls(pattern, _LinearRegex(nfa, start)), None
