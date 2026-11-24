"""Column-summary folds for the aggregate stage.

Each ``*_summary`` function takes one group's non-null values and returns a
small payload describing that column. ``column_summary`` is the registered
reducer over them: the engine feeds it every row of the column and only the
summary travels back — no paging, no sample, no rows in the caller's process.

They live here rather than in a caller because summarizing a column is a query
capability, and because a named reducer needs no permission to run: routing
these folds through ``code_reduce`` would have made every column statistic
depend on a switch that exists to refuse *caller-authored* code.

The engine drops nulls before invoking a fold, so no function here can see or
count them; a null count comes from a row count minus a ``count`` of the field.
"""

from __future__ import annotations

__all__ = [
    "STAT_LIST_LIMIT",
    "STAT_VALUE_CHARS",
    "bool_summary",
    "column_summary",
    "elided_value",
    "histogram",
    "json_summary",
    "numeric_summary",
    "parse_time",
    "quantiles",
    "set_top",
    "stat_key_row",
    "stat_kind",
    "stat_value_row",
    "text_summary",
    "time_summary",
    "vector_summary",
]

from heavenbase.utils.typing import Any, Dict, List

STAT_VALUE_CHARS = 120
# A fold now sees every row, so a distribution is as long as the column is
# varied. These lists stay generous — the defect this replaced cut them to eight
# over a sample — but a payload cannot be unbounded, so the cut is declared.
STAT_LIST_LIMIT = 200


def stat_kind(dtype_id: str) -> str:
    """Map a dtype identifier onto a column-statistics kind.

    Args:
        dtype_id (str): Logical type identifier, for example ``"integer"``.

    Returns:
        str: One of ``vector``/``time``/``numeric``/``bool``/``json``/``list``/``text``.
    """
    name = (dtype_id or "").lower()
    if name.startswith("vector"):
        return "vector"
    if name in ("timestamp", "date", "datetime", "time"):
        return "time"
    if name in ("integer", "int", "float", "double", "number"):
        return "numeric"
    if name in ("boolean", "bool"):
        return "bool"
    if name == "json":
        return "json"
    if name.startswith("array") or name == "list":
        return "list"
    return "text"


def quantiles(values: List[float]) -> Dict[str, float]:
    """Return p25/p50/p75 (inclusive method) over every value.

    Args:
        values (List[float]): Numeric values, already filtered of nulls.

    Returns:
        Dict[str, float]: ``{"p25", "p50", "p75"}``. Fewer than two values give
        the single value for all three, since there is no spread to report.
    """
    import statistics

    if len(values) < 2:
        only = round(values[0], 4) if values else 0.0
        return {"p25": only, "p50": only, "p75": only}
    p25, p50, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return {"p25": round(p25, 4), "p50": round(p50, 4), "p75": round(p75, 4)}


def histogram(values: List[float], bins: int = 12) -> List[Dict[str, Any]]:
    """Bucket numeric values into equal-width bins (unit-wide for small integer ranges).

    Args:
        values (List[float]): Numeric values, already filtered of nulls.
        bins (int): Maximum bucket count. Defaults to ``12``.

    Returns:
        List[Dict[str, Any]]: ``{"lo", "hi", "count"}`` rows whose counts sum to
        ``len(values)``.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [{"lo": low, "hi": high, "count": len(values)}]
    integral = all(float(value).is_integer() for value in values)
    if integral and (high - low) < bins:
        edges = [low + step for step in range(int(high - low) + 2)]
    else:
        width = (high - low) / bins
        edges = [low + width * index for index in range(bins + 1)]
    counts = [0] * (len(edges) - 1)
    span = edges[-1] - edges[0]
    for value in values:
        index = min(int((value - edges[0]) / span * (len(edges) - 1)), len(counts) - 1)
        counts[index] += 1
    return [{"lo": round(edges[i], 4), "hi": round(edges[i + 1], 4), "count": counts[i]} for i in range(len(counts))]


def parse_time(value: Any) -> tuple:
    """Parse one timestamp-ish value.

    Args:
        value (Any): Stored cell value.

    Returns:
        tuple: ``(epoch_seconds | None, format_label)``.
    """
    import datetime as dt

    if isinstance(value, bool):
        return None, "unknown"
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e11:
            return number / 1000.0, "epoch millis"
        if number > 1e8:
            return number, "epoch seconds"
        return None, "unknown"
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None, "unknown"
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp(), "ISO date" if len(text) == 10 else "ISO 8601"
    return None, "unknown"


def elided_value(value: Any) -> Dict[str, Any]:
    """Return one display-safe value cell, marking any elision it had to make.

    Quoting a prefix as if it were the value is the same defect as quoting a
    slice as if it were the whole list, so every cut carries ``value_truncated``
    and the true ``value_len`` beside it.

    Args:
        value (Any): Value to publish.

    Returns:
        Dict[str, Any]: ``{"value"}`` plus ``value_truncated``/``value_len`` when cut.
    """
    text = str(value)
    cell: Dict[str, Any] = {"value": text[:STAT_VALUE_CHARS]}
    if len(text) > STAT_VALUE_CHARS:
        cell["value_truncated"] = True
        cell["value_len"] = len(text)
    return cell


def stat_value_row(value: Any, count: int) -> Dict[str, Any]:
    """Return one element/count entry, marking a value elided for display.

    Args:
        value (Any): Counted value.
        count (int): Occurrences over the whole column.

    Returns:
        Dict[str, Any]: ``{"value", "count"}`` plus any elision markers.
    """
    return {**elided_value(value), "count": count}


def stat_key_row(key: Any, count: int) -> Dict[str, Any]:
    """Return one key/count entry, marking a key elided for display.

    Args:
        key (Any): Json object key.
        count (int): Rows carrying the key.

    Returns:
        Dict[str, Any]: ``{"key", "count"}`` plus ``key_elided``/``key_len`` when cut.
    """
    text = str(key)
    row: Dict[str, Any] = {"key": text[:STAT_VALUE_CHARS], "count": count}
    if len(text) > STAT_VALUE_CHARS:
        row["key_elided"] = True
        row["key_len"] = len(text)
    return row


def set_top(stats: Dict[str, Any], key: str, counter: Any, limit: int) -> None:
    """Write one most-common list into ``stats``, declaring every cut it makes.

    Counters key on whole values so counts and ``distinct`` stay true. Ties break
    on the value name so the published list does not depend on row order. The
    companion ``<key>_truncated`` flag appears only when entries were dropped.

    Args:
        stats (Dict[str, Any]): Statistics payload being assembled.
        key (str): Payload key receiving the list.
        counter (Any): Counter over whole values.
        limit (int): Maximum entries to publish.

    Returns:
        None.
    """
    # most_common leaves ties in insertion order, so both the order AND which
    # tied entries survive the cut would follow row order; break on the name.
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    stats[key] = [stat_value_row(value, count) for value, count in ranked[:limit]]
    if len(counter) > limit:
        stats[f"{key}_truncated"] = True


def _set_ranked(summary: Dict[str, Any], key: str, counter: Any, row: Any) -> None:
    """Write one whole-column distribution, ranked and bounded, declaring any cut.

    Ties break on the name so the list does not follow row order, and a cut sets
    ``<key>_truncated`` — the counts themselves stay exact over every row, which
    is the property the cut must not be mistaken for.

    Args:
        summary (Dict[str, Any]): Payload being assembled.
        key (str): Payload key receiving the list.
        counter (Any): Counter over the whole column.
        row (Any): Row builder, ``stat_value_row`` or ``stat_key_row``.

    Returns:
        None.
    """
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    summary[key] = [row(name, count) for name, count in ranked[:STAT_LIST_LIMIT]]
    if len(ranked) > STAT_LIST_LIMIT:
        summary[f"{key}_truncated"] = True


def numeric_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a numeric column over every one of its non-null values.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``distinct`` plus ``min``/``max``/``mean``/``percentiles``/
        ``histogram`` when any value was numeric, and ``top`` for a low-cardinality
        column. Booleans are excluded: folding ``True`` as ``1`` would report a
        mean over values that were never numbers.
    """
    from collections import Counter

    numbers = [float(value) for value in values if not isinstance(value, bool) and isinstance(value, (int, float))]
    counter: Counter = Counter(str(value) for value in values if not isinstance(value, bool) and isinstance(value, (int, float)))
    summary: Dict[str, Any] = {}
    if counter:
        summary["distinct"] = len(counter)
    if numbers:
        summary["min"] = min(numbers)
        summary["max"] = max(numbers)
        summary["mean"] = round(sum(numbers) / len(numbers), 4)
        summary["percentiles"] = quantiles(numbers)
        summary["histogram"] = histogram(numbers)
        if len(counter) <= 10:
            set_top(summary, "top", counter, 5)
    return summary


def bool_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a boolean column over every one of its non-null values.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``distinct`` and a ``top`` list of ``true``/``false`` counts.
    """
    from collections import Counter

    counter: Counter = Counter(str(bool(value)).lower() for value in values)
    summary: Dict[str, Any] = {}
    if counter:
        summary["distinct"] = len(counter)
        set_top(summary, "top", counter, 2)
    return summary


def time_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a time column over every one of its non-null values.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``format``/``earliest``/``latest``/``span_days`` when any
        value parsed, plus ``unparsed`` counting the values that did not. A column
        holding more than one format reports ``mixed`` rather than one of them.
    """
    import datetime as dt

    from collections import Counter

    stamps: List[float] = []
    formats: Counter = Counter()
    unparsed = 0
    for value in values:
        parsed, label = parse_time(value)
        if parsed is None:
            unparsed += 1
        else:
            stamps.append(parsed)
            formats[label] += 1
    summary: Dict[str, Any] = {}
    if stamps:
        fmt = formats.most_common(1)[0][0] if len(formats) == 1 else "mixed"
        summary["format"] = fmt
        earliest = dt.datetime.fromtimestamp(min(stamps), tz=dt.timezone.utc)
        latest = dt.datetime.fromtimestamp(max(stamps), tz=dt.timezone.utc)
        date_only = fmt == "ISO date"
        summary["earliest"] = earliest.date().isoformat() if date_only else earliest.isoformat()
        summary["latest"] = latest.date().isoformat() if date_only else latest.isoformat()
        summary["span_days"] = round((max(stamps) - min(stamps)) / 86400, 1)
    if unparsed:
        summary["unparsed"] = unparsed
    return summary


def vector_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a vector column over every one of its non-null values.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``dims`` as a single width or a ``low–high`` range, and
        ``norm`` min/mean/max over the vectors whose components were numeric.
    """
    import math

    from collections import Counter

    dims: Counter = Counter()
    norms: List[float] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        dims[len(value)] += 1
        try:
            norms.append(math.sqrt(sum(float(item) * float(item) for item in value)))
        except (TypeError, ValueError):
            continue
    summary: Dict[str, Any] = {}
    if dims:
        low, high = min(dims), max(dims)
        summary["dims"] = str(low) if low == high else f"{low}–{high}"
    if norms:
        summary["norm"] = {"min": round(min(norms), 4), "mean": round(sum(norms) / len(norms), 4), "max": round(max(norms), 4)}
    return summary


def json_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a Json or list column over every one of its non-null values.

    Object and list payloads can share one column, so both distributions are
    computed and each is emitted only when that shape was present. Counts cover
    the whole column; published lists carry a truncation flag when their display
    bound omits entries.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``keys``/``distinct_keys``/``key_count`` for object
        payloads, ``elements``/``items`` for list payloads, and ``size`` over
        both, measured on the serialized payload. A payload that is neither is
        counted by ``scalars`` and distributed by ``scalar_values``, so a column
        holding only scalars still describes itself and a mixed one says which
        rows the other numbers leave out. Each distribution is exact over every
        row and published up to ``STAT_LIST_LIMIT`` entries, setting
        ``<key>_truncated`` when longer; ``distinct_keys`` stays the true count
        either way.
    """
    from collections import Counter

    from heavenbase.utils.serialize import dumps_json

    key_counter: Counter = Counter()
    element_counter: Counter = Counter()
    scalar_counter: Counter = Counter()
    key_counts: List[int] = []
    item_lengths: List[int] = []
    sizes: List[int] = []
    for value in values:
        if isinstance(value, dict):
            key_counts.append(len(value))
            sizes.append(len(dumps_json(value)))
            for key in value:
                key_counter[str(key)] += 1
        elif isinstance(value, (list, tuple)):
            item_lengths.append(len(value))
            sizes.append(len(dumps_json(list(value))))
            for item in value:
                if item is None or isinstance(item, (str, int, float, bool)):
                    element_counter[str(item)] += 1
        else:
            # A Json column may hold a bare scalar. It yields neither keys nor
            # elements, so it is counted and distributed on its own: dropping it
            # would let a mostly-scalar column publish ranges over the few rows
            # that had a shape, with nothing saying the rest were left out, and
            # counting it without describing it would answer nothing at all.
            scalar_counter[str(value)] += 1
    summary: Dict[str, Any] = {}
    if scalar_counter:
        summary["scalars"] = sum(scalar_counter.values())
        _set_ranked(summary, "scalar_values", scalar_counter, stat_value_row)
    if key_counter:
        _set_ranked(summary, "keys", key_counter, stat_key_row)
        summary["distinct_keys"] = len(key_counter)
        summary["key_count"] = {"min": min(key_counts), "mean": round(sum(key_counts) / len(key_counts), 1), "max": max(key_counts)}
    if item_lengths:
        summary["items"] = {"min": min(item_lengths), "mean": round(sum(item_lengths) / len(item_lengths), 1), "max": max(item_lengths)}
        if element_counter:
            # same tie-break as keys: most_common leaves ties in insertion order,
            # which makes the published list depend on row order
            _set_ranked(summary, "elements", element_counter, stat_value_row)
    if sizes:
        summary["size"] = {"mean": round(sum(sizes) / len(sizes), 1), "max": max(sizes)}
    return summary


def text_summary(values: List[Any]) -> Dict[str, Any]:
    """Summarize a text column over every one of its non-null values.

    Args:
        values (List[Any]): One group's non-null values.

    Returns:
        Dict[str, Any]: ``distinct``, ``length``, and ``length_histogram``, then
        either ``top`` when some value repeats or up to three ``examples``. The
        examples deduplicate on the whole value, so two long values sharing a
        prefix do not collapse into one.
    """
    from collections import Counter

    from heavenbase.utils.serialize import dumps_json

    counter: Counter = Counter()
    lengths: List[int] = []
    examples: List[Dict[str, Any]] = []
    seen: set = set()
    for value in values:
        text = value if isinstance(value, str) else dumps_json(value)
        lengths.append(len(text))
        counter[text] += 1
        if len(examples) < 3 and text not in seen:
            seen.add(text)
            examples.append(elided_value(text))
    summary: Dict[str, Any] = {}
    if counter:
        summary["distinct"] = len(counter)
    if lengths:
        summary["length"] = {"min": min(lengths), "mean": round(sum(lengths) / len(lengths), 1), "max": max(lengths)}
        summary["length_histogram"] = histogram([float(size) for size in lengths])
    repeated = counter.most_common(1)
    if repeated and repeated[0][1] > 1:
        set_top(summary, "top", counter, 5)
    elif examples:
        summary["examples"] = examples
    return summary


_SUMMARY_BY_KIND = {
    "numeric": numeric_summary,
    "bool": bool_summary,
    "time": time_summary,
    "vector": vector_summary,
    "json": json_summary,
    "list": json_summary,
    "text": text_summary,
}


def column_summary(values: List[Any], kind: str) -> Dict[str, Any]:
    """Summarize one column's values the way its kind calls for.

    This is the registered ``column_summary`` reducer. Being registered is the
    point: the fold is repository code with a fixed meaning, so it must not
    depend on the switch that refuses caller-authored expressions.

    Args:
        values (List[Any]): One group's non-null values, fed by the engine.
        kind (str): Statistics kind from :func:`stat_kind`.

    Returns:
        Dict[str, Any]: The kind-specific summary payload.

    Raises:
        ValueError: If ``kind`` is not one this fold summarizes. Naming a kind
            it cannot handle is a caller error, not an empty result.
    """
    fold = _SUMMARY_BY_KIND.get(kind)
    if fold is None:
        raise ValueError(f"Unsupported column_summary kind {kind!r}; expected one of {', '.join(sorted(_SUMMARY_BY_KIND))}")
    return fold(values)
