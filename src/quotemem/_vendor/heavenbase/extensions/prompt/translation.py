"""Prompt extension translation entity and service."""

from __future__ import annotations

__all__ = ["Translation"]

import re
from time import time

from heavenbase.entity import Entity, field
from heavenbase.types import Float, Json, LongText, ShortText
from heavenbase.utils import Any, Dict, List, Mapping, Optional, sha256hash

from ._workspace import prompt_config, resolve_prompt_workspace
from .name import check_prompt_name, translation_row_id

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _pattern(source: str) -> Dict[str, Any]:
    text = str(source)
    names: List[str] = []
    groups: List[str] = []
    counts: Dict[str, int] = {}
    parts: List[str] = []
    pos = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        parts.append(re.escape(text[pos : match.start()]))
        name = match.group(1)
        index = counts.get(name, 0)
        counts[name] = index + 1
        names.append(name)
        group = f"__hb_{name}_{index}"
        groups.append(group)
        parts.append(f"(?P<{group}>.+?)")
        pos = match.end()
    parts.append(re.escape(text[pos:]))
    return {"source": text, "placeholders": names, "groups": groups, "regex": "".join(parts) if names else ""}


def _pattern_match(pattern: Mapping[str, Any], text: str) -> Optional[Dict[str, str]]:
    regex = str((pattern or {}).get("regex") or "")
    if not regex:
        return None
    match = re.fullmatch(regex, str(text))
    if not match:
        return None
    captured = match.groupdict()
    names = list(pattern.get("placeholders") or [])
    groups = list(pattern.get("groups") or names)
    values: Dict[str, str] = {}
    for name, group in zip(names, groups):
        value = captured.get(str(group), captured.get(str(name), ""))
        if name in values and values[name] != value:
            return None
        values[str(name)] = value
    return values or captured


class Translation(Entity):
    """Translation row entity and prompt-bound helper.

    ``Translation`` stores source-to-target text rows as ``sys-translation`` entities. When created by ``Prompt``, the
    same object is exposed as ``prompt.tr`` for lookup, mutation, and translation dictionary binding.

    Warnings:
        Translated template rows are later formatted with runtime values. Treat shared translation workspaces as trusted
        prompt data and avoid loading translations from unreviewed sources.

    Examples:
        >>> prompt = Prompt("Hello, {name}", name="demo.hello")
        >>> prompt.tr.set("Hello, {name}", "zz", "HEY, {name}")
        '...'
        >>> prompt(name="Ada", lang="zz")
        'HEY, Ada'
    """

    identifier = "sys-translation"

    dict_name = field(ShortText).desc("Dotted translation dictionary name")
    source_key = field(LongText).desc("Source text or template key")
    lang = field(ShortText).desc("Target language")
    target_value = field(LongText).desc("Translated text")
    pattern = field(Json).default({}).desc("Parsed placeholder pattern metadata")
    status = field(ShortText).default("active").desc("Translation status")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    updated_at = field(Float).default(0.0).desc("Last update timestamp")

    def __init__(self, prompt: Any = None, ws: Any = None, dicts: Optional[List[str]] = None, **values: Any) -> None:
        """Create a translation row object or prompt-bound translation helper.

        Args:
            prompt (Any | None): Optional ``Prompt`` instance to bind lookup and mutation methods to.
            ws (Any | None): Workspace used for translation rows. Defaults to the prompt workspace or active workspace.
            dicts (List[str] | None): Ordered translation dictionary names for standalone helpers.
            **values (Any): Entity row values when constructing ``Translation`` as a row object.

        Warnings:
            Translation rows are executable-adjacent prompt data. Treat shared translation workspaces as trusted content,
            because translated templates are later formatted with runtime values.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__(**values)
        self.prompt = prompt
        self.ws = ws
        self.dicts = list(dicts or [])

    def _workspace(self) -> Any:
        if self.ws is not None:
            return self.ws
        if self.prompt is not None and self.prompt.ws is not None:
            self.ws = self.prompt.ws
            return self.ws
        self.ws = resolve_prompt_workspace()
        if self.prompt is not None:
            self.prompt.ws = self.ws
        return self.ws

    def _ensure(self, ws: Any) -> None:
        enable_extension = getattr(ws, "enable_extension", None)
        if callable(enable_extension):
            enable_extension("prompt")
            return
        if self.prompt is not None:
            ws.register(type(self.prompt))
        ws.register(type(self))

    def _dicts(self, dict_name: Optional[str] = None) -> List[str]:
        if dict_name:
            return [check_prompt_name(dict_name)]
        refs = list(self.dicts or getattr(self.prompt, "dicts", []) or [])
        if not refs and self.prompt is not None and self.prompt.name:
            refs = [self.prompt.name]
            self.prompt.dicts = refs
        return [check_prompt_name(ref) for ref in refs]

    def _default_dict(self, dict_name: Optional[str] = None) -> str:
        dicts = self._dicts(dict_name)
        if not dicts:
            raise ValueError("Translation dict name is required")
        return dicts[0]

    def _rows(self, lang: str, dict_name: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            ws = self._workspace()
        except KeyError:
            return []
        self._ensure(ws)
        dicts = self._dicts(dict_name)
        rows = ws.query(type(self)).where({"lang": str(lang)}).where({"status": "active"}).execute().rows()
        by_dict = {name: [] for name in dicts}
        for row in rows:
            if row.get("dict_name") in by_dict:
                by_dict[row.get("dict_name")].append(row)
        return [row for name in dicts for row in by_dict[name]]

    def invalidate(self) -> None:
        """Refresh translation helper state.

        Args:
            None.

        Returns:
            None: Translation lookups read current workspace rows, so no local cache is retained.
        """
        return None

    def set(self, key: str, lang: str, value: str, dict_name: Optional[str] = None) -> str:
        """Set one translation row.

        Args:
            key (str): Source text or source template.
            lang (str): Target language key.
            value (str): Translated text or translated template.
            dict_name (str | None): Optional dictionary override. Defaults to the first bound dictionary.

        Returns:
            str: Stable translation row ``object_id``.

        Warnings:
            Pattern sources with ``{placeholder}`` captures are compiled into an internal regex for lookup.
        """
        ws = self._workspace()
        self._ensure(ws)
        selected = self._default_dict(dict_name)
        row_id = translation_row_id(selected, key, lang)
        now = time()
        existing = ws.get(row_id, entity=type(self))
        ws.upsert(
            type(self),
            {
                "object_id": row_id,
                "dict_name": selected,
                "source_key": str(key),
                "lang": str(lang),
                "target_value": str(value),
                "pattern": _pattern(key),
                "status": "active",
                "created_at": float((existing or {}).get("created_at") or now),
                "updated_at": now,
            },
        )
        self.invalidate()
        return row_id

    def set_many(self, data: Mapping[str, Mapping[str, str]], dict_name: Optional[str] = None) -> List[str]:
        """Set many translation rows.

        Args:
            data (Mapping[str, Mapping[str, str]]): Mapping shaped as ``{lang: {source: target}}``.
            dict_name (str | None): Optional dictionary override. Defaults to the first bound dictionary.

        Returns:
            List[str]: Stable translation row ``object_id`` values in input iteration order.
        """
        return [self.set(source, lang, target, dict_name=dict_name) for lang, rows in data.items() for source, target in rows.items()]

    def list(self, lang: Optional[str] = None, dict_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active translation rows for this prompt or dictionary.

        Args:
            lang (str | None): Optional language filter.
            dict_name (str | None): Optional dictionary override. Defaults to all bound dictionaries.

        Returns:
            List[Dict[str, Any]]: Active translation rows sorted by dictionary, language, and source key.
        """
        ws = self._workspace()
        self._ensure(ws)
        dicts = set(self._dicts(dict_name))
        rows = ws.query(type(self)).where({"status": "active"}).execute().rows()
        if lang is not None:
            rows = [row for row in rows if row.get("lang") == str(lang)]
        rows = [row for row in rows if row.get("dict_name") in dicts]
        return sorted(rows, key=lambda row: (str(row.get("dict_name") or ""), str(row.get("lang") or ""), str(row.get("source_key") or "")))

    def get(self, key: str, lang: str, dict_name: Optional[str] = None) -> str:
        """Get a translated value or return the configured fallback.

        Args:
            key (str): Source text to translate.
            lang (str): Target language key.
            dict_name (str | None): Optional dictionary override. Defaults to bound dictionaries in order.

        Returns:
            str: Translated text. If no row matches and elicit is ``"none"``, returns ``key`` unchanged.

        Raises:
            NotImplementedError: If missing translations use the reserved ``elicit="llm"`` policy.
            ValueError: If a pattern translation row cannot format with captured placeholders.

        Examples:
            >>> prompt.tr.get("Hello", "zz")
            'HEY'
        """
        text = str(key)
        for row in self._rows(str(lang), dict_name=dict_name):
            if row.get("source_key") == text:
                return str(row.get("target_value") or "")
        for row in self._rows(str(lang), dict_name=dict_name):
            values = _pattern_match(row.get("pattern") or {}, text)
            if values is not None:
                try:
                    return str(row.get("target_value") or "").format(**values)
                except Exception as error:
                    raise ValueError(f"Translation {row.get('object_id')!r} failed to format source {text!r}: {error}") from error
        workspace = self.ws if self.ws is not None else getattr(self.prompt, "ws", None)
        config = prompt_config(workspace)
        main_lang = str(getattr(self.prompt, "main_lang", None) or config.get("heavenbase.prompt.main_lang", default="en") or "en")
        if str(lang) == main_lang:
            return text
        elicit = str(getattr(self.prompt, "elicit", None) or config.get("heavenbase.prompt.elicit", default="none") or "none")
        if elicit == "llm":
            raise NotImplementedError("Prompt translation elicit='llm' is not implemented in TAL-194")
        return text

    def delete(self, key: str, lang: str, dict_name: Optional[str] = None) -> bool:
        """Tombstone one translation row.

        Args:
            key (str): Source text or source template.
            lang (str): Target language key.
            dict_name (str | None): Optional dictionary override. Defaults to the first bound dictionary.

        Returns:
            bool: Whether a matching row was found and tombstoned.
        """
        ws = self._workspace()
        self._ensure(ws)
        selected = self._default_dict(dict_name)
        row_id = translation_row_id(selected, key, lang)
        row = ws.get(row_id, entity=type(self))
        if row is None:
            return False
        row = dict(row)
        row["object_id"] = row_id
        row["status"] = "deleted"
        row["updated_at"] = time()
        ws.set(row, entity=type(self))
        self.invalidate()
        return True

    def bind(self, dict_name: str) -> str:
        """Append a translation dictionary reference to the bound prompt.

        Args:
            dict_name (str): Dotted translation dictionary name.

        Returns:
            str: Validated dictionary name.
        """
        selected = check_prompt_name(dict_name)
        if self.prompt is not None and selected not in self.prompt.dicts:
            self.prompt.dicts.append(selected)
            self._persist_refs()
        elif self.prompt is None and selected not in self.dicts:
            self.dicts.append(selected)
        self.invalidate()
        return selected

    def unbind(self, dict_name: str) -> str:
        """Remove a translation dictionary reference from the bound prompt.

        Args:
            dict_name (str): Dotted translation dictionary name.

        Returns:
            str: Validated dictionary name.
        """
        selected = check_prompt_name(dict_name)
        if self.prompt is not None:
            self.prompt.dicts = [ref for ref in self.prompt.dicts if ref != selected]
            self._persist_refs()
        else:
            self.dicts = [ref for ref in self.dicts if ref != selected]
        self.invalidate()
        return selected

    def _persist_refs(self) -> None:
        if self.prompt is None or self.prompt.ws is None or not self.prompt.name:
            return
        ws = self._workspace()
        self._ensure(ws)
        prompt_object_id = self.prompt._row_id()
        row = ws.get(prompt_object_id, entity=type(self.prompt))
        if row is None:
            return
        row = dict(row)
        row["object_id"] = prompt_object_id
        row["dicts"] = list(self.prompt.dicts)
        manifest = dict(row.get("manifest") or {})
        manifest["dicts"] = list(self.prompt.dicts)
        row["manifest"] = manifest
        row["fingerprints"] = {"manifest": sha256hash(manifest), "capsule": sha256hash(row.get("capsule") or {})}
        row["updated_at"] = time()
        ws.set(row, entity=type(self.prompt))
