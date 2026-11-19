"""Closed time vocabularies beyond English for deterministic time resolution (B.10, ML-1).

`temporal` resolves a fact's source time expression without a model call. The
extractor copies that expression as written in its quotes (B.6.3), so a source in
Spanish, Russian or Thai yields ``ayer``, ``в 2022 году`` or ``ปีหน้า``. This module
holds the exact phrases, with which `temporal` gives such expressions the windows
their English equivalents get, and nothing more:

- `DAYS`: yesterday, today and tomorrow;
- `PERIODS`: last, this and next week, month and year;
- `WEEKDAYS` and `WEEKENDS`: relative weekdays and Saturday/Sunday windows;
- `AGO`: a count of days, hours, weeks or weekends before or after the source time;
- `CALENDAR_AGO`: a count of months or years before or after it, which only the
  ``calendar-unit-v1`` time reading anchors;
- `YEAR_FORMS`: a four-digit year inside a closed preposition or year word;
- `MONTHS`: month names and their inflected forms, for the named-month and
  full-date forms;
- `NAMED_MONTH_FORMS` and `YEAR_MONTH_FORMS`: "this, last or next <month>" and
  "<month> of this, last or next year";
- `DATE_FORMS`: a day, a month and a year.

Every phrase is matched whole against `fold_time` of the expression, never searched
inside it, so an expression with any other word stays unanchored. A word whose
reading depends on context is left out: Hindi and Urdu ``कल``/``kal`` mean both
yesterday and tomorrow, and ``परसों`` both two days before and after. A year of 2400
or more is never anchored: in Thai text it is a Buddhist-era year, and only a Thai
year form converts it (`buddhist_era`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from aml.core.statement import folded

Relation = Literal["last", "this", "next"]
Unit = Literal["week", "month", "year"]

_ARABIC_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u0640]")
_ARABIC_LETTERS = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627",
                                 "\u0649": "\u064a"})
_EDGES = " .,;:!?\u2026\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f\u060c\u061b\u061f\u0964\u0965"


def fold_time(text: str) -> str:
    """Return the form every phrase of this module is written and matched in.

    The form is `aml.core.statement.folded` (NFKC, casefold, dotted I folded) with
    curly apostrophes made straight, Arabic short vowels and tatweel removed,
    Arabic alef and final-ya variants unified, whitespace collapsed to single
    spaces, and edge punctuation and spaces stripped.

    Args:
        text (str): A time expression as the extractor copied it.

    Returns:
        str: The folded expression.
    """
    value = folded(text).replace("\u2019", "'").replace("\u02bc", "'")
    value = _ARABIC_MARKS.sub("", value).translate(_ARABIC_LETTERS)
    return " ".join(value.split()).strip(_EDGES)


def _table[T](rows: Mapping[str, T]) -> Mapping[str, T]:
    folded_rows = {fold_time(phrase): value for phrase, value in rows.items()}
    if len(folded_rows) != len(rows):
        raise ValueError("Two phrases of one time table fold to the same form")
    return MappingProxyType(folded_rows)


_YESTERDAY, _TODAY, _TOMORROW = -1, 0, 1
DAYS: Final[Mapping[str, int]] = _table({
    # English and Chinese (the earlier vocabulary).
    "yesterday": _YESTERDAY, "today": _TODAY, "tomorrow": _TOMORROW,
    "昨天": _YESTERDAY, "今天": _TODAY, "明天": _TOMORROW,
    # Chinese and Japanese: 昨日, 今日 and 明日 are written alike in both.
    "昨日": _YESTERDAY, "今日": _TODAY, "明日": _TOMORROW,
    "きのう": _YESTERDAY, "きょう": _TODAY, "あした": _TOMORROW, "あす": _TOMORROW,
    "어제": _YESTERDAY, "오늘": _TODAY, "내일": _TOMORROW,
    "ayer": _YESTERDAY, "hoy": _TODAY, "mañana": _TOMORROW,
    "ontem": _YESTERDAY, "hoje": _TODAY, "amanhã": _TOMORROW,
    "hier": _YESTERDAY, "aujourd'hui": _TODAY, "demain": _TOMORROW,
    "ieri": _YESTERDAY, "oggi": _TODAY, "domani": _TOMORROW,
    "gestern": _YESTERDAY, "heute": _TODAY, "morgen": _TOMORROW,
    "вчера": _YESTERDAY, "сегодня": _TODAY, "завтра": _TOMORROW,
    "أمس": _YESTERDAY, "الأمس": _YESTERDAY, "بالأمس": _YESTERDAY, "البارحة": _YESTERDAY,
    "اليوم": _TODAY,
    "غدا": _TOMORROW,
    "आज": _TODAY,
    "เมื่อวาน": _YESTERDAY, "เมื่อวานนี้": _YESTERDAY, "วันนี้": _TODAY, "พรุ่งนี้": _TOMORROW,
    "dün": _YESTERDAY, "bugün": _TODAY, "yarın": _TOMORROW,
    "hôm qua": _YESTERDAY, "hôm nay": _TODAY, "ngày mai": _TOMORROW,
    "kemarin": _YESTERDAY, "hari ini": _TODAY, "besok": _TOMORROW,
    "aaj": _TODAY,
})
"""Exact day words with their day offset from the source day."""


_LW: Final[tuple[Relation, Unit]] = ("last", "week")
_TW: Final[tuple[Relation, Unit]] = ("this", "week")
_NW: Final[tuple[Relation, Unit]] = ("next", "week")
_LM: Final[tuple[Relation, Unit]] = ("last", "month")
_TM: Final[tuple[Relation, Unit]] = ("this", "month")
_NM: Final[tuple[Relation, Unit]] = ("next", "month")
_LY: Final[tuple[Relation, Unit]] = ("last", "year")
_TY: Final[tuple[Relation, Unit]] = ("this", "year")
_NY: Final[tuple[Relation, Unit]] = ("next", "year")
_Period = tuple[Relation, Unit]
PERIODS: Final[Mapping[str, _Period]] = _table({
    "last week": _LW, "this week": _TW, "next week": _NW,
    "last month": _LM, "this month": _TM, "next month": _NM,
    "last year": _LY, "this year": _TY, "next year": _NY,
    # Chinese.
    "上周": _LW, "本周": _TW, "下周": _NW, "这周": _TW, "上星期": _LW, "下星期": _NW,
    "上个星期": _LW, "这个星期": _TW, "下个星期": _NW,
    "上个月": _LM, "本月": _TM, "下个月": _NM, "这个月": _TM, "上月": _LM, "下月": _NM,
    "去年": _LY, "今年": _TY, "明年": _NY,
    # Japanese (去年 and 今年 as in Chinese).
    "先週": _LW, "今週": _TW, "来週": _NW, "先月": _LM, "今月": _TM, "来月": _NM,
    "昨年": _LY, "来年": _NY,
    # Korean, with and without the particle 에.
    "지난주": _LW, "지난 주": _LW, "이번 주": _TW, "이번주": _TW, "다음 주": _NW, "다음주": _NW,
    "지난주에": _LW, "이번 주에": _TW, "다음 주에": _NW,
    "지난달": _LM, "지난 달": _LM, "이번 달": _TM, "이번달": _TM, "다음 달": _NM, "다음달": _NM,
    "지난달에": _LM, "이번 달에": _TM, "다음 달에": _NM,
    "작년": _LY, "지난해": _LY, "올해": _TY, "금년": _TY, "내년": _NY,
    "작년에": _LY, "지난해에": _LY, "올해에": _TY, "내년에": _NY,
    # Spanish.
    "la semana pasada": _LW, "semana pasada": _LW, "esta semana": _TW,
    "la semana que viene": _NW, "la próxima semana": _NW, "la semana próxima": _NW,
    "el mes pasado": _LM, "mes pasado": _LM, "este mes": _TM,
    "el mes que viene": _NM, "el próximo mes": _NM, "el mes próximo": _NM,
    "el año pasado": _LY, "año pasado": _LY, "este año": _TY,
    "el año que viene": _NY, "el próximo año": _NY, "el año próximo": _NY,
    # Portuguese.
    "a semana passada": _LW, "semana passada": _LW,
    "a próxima semana": _NW, "na próxima semana": _NW, "semana que vem": _NW,
    "o mês passado": _LM, "mês passado": _LM, "este mês": _TM,
    "o próximo mês": _NM, "no próximo mês": _NM, "mês que vem": _NM,
    "o ano passado": _LY, "ano passado": _LY, "este ano": _TY,
    "o próximo ano": _NY, "no próximo ano": _NY, "ano que vem": _NY,
    # French.
    "la semaine dernière": _LW, "semaine dernière": _LW, "cette semaine": _TW,
    "la semaine prochaine": _NW, "semaine prochaine": _NW,
    "le mois dernier": _LM, "mois dernier": _LM, "ce mois-ci": _TM, "ce mois": _TM,
    "le mois prochain": _NM, "mois prochain": _NM,
    "l'année dernière": _LY, "l'an dernier": _LY, "l'année passée": _LY, "cette année": _TY,
    "l'année prochaine": _NY, "l'an prochain": _NY,
    # Italian.
    "la settimana scorsa": _LW, "settimana scorsa": _LW, "questa settimana": _TW,
    "la settimana prossima": _NW, "la prossima settimana": _NW, "settimana prossima": _NW,
    "il mese scorso": _LM, "mese scorso": _LM, "questo mese": _TM,
    "il mese prossimo": _NM, "il prossimo mese": _NM, "mese prossimo": _NM,
    "l'anno scorso": _LY, "anno scorso": _LY, "quest'anno": _TY,
    "l'anno prossimo": _NY, "il prossimo anno": _NY, "anno prossimo": _NY,
    # German.
    "letzte woche": _LW, "vergangene woche": _LW, "diese woche": _TW, "nächste woche": _NW,
    "kommende woche": _NW,
    "letzten monat": _LM, "vergangenen monat": _LM, "diesen monat": _TM,
    "nächsten monat": _NM, "kommenden monat": _NM,
    "letztes jahr": _LY, "vergangenes jahr": _LY, "dieses jahr": _TY, "nächstes jahr": _NY,
    "kommendes jahr": _NY, "im nächsten jahr": _NY,
    # Russian.
    "на прошлой неделе": _LW, "на этой неделе": _TW, "на следующей неделе": _NW,
    "в прошлом месяце": _LM, "в этом месяце": _TM, "в следующем месяце": _NM,
    "в прошлом году": _LY, "в этом году": _TY, "в следующем году": _NY,
    # Arabic (written folded: no short vowels, bare alef).
    "الأسبوع الماضي": _LW, "في الأسبوع الماضي": _LW, "هذا الأسبوع": _TW,
    "الأسبوع المقبل": _NW, "الأسبوع القادم": _NW, "في الأسبوع المقبل": _NW,
    "الشهر الماضي": _LM, "في الشهر الماضي": _LM, "هذا الشهر": _TM,
    "الشهر المقبل": _NM, "الشهر القادم": _NM, "في الشهر المقبل": _NM,
    "العام الماضي": _LY, "في العام الماضي": _LY, "السنة الماضية": _LY,
    "هذا العام": _TY, "هذه السنة": _TY,
    "العام المقبل": _NY, "العام القادم": _NY, "السنة القادمة": _NY, "السنة المقبلة": _NY,
    "في العام المقبل": _NY,
    # Hindi, with and without the nukta, and romanized (Hinglish).
    "पिछले हफ़्ते": _LW, "पिछले हफ्ते": _LW, "पिछले सप्ताह": _LW,
    "इस हफ़्ते": _TW, "इस हफ्ते": _TW, "इस सप्ताह": _TW,
    "अगले हफ़्ते": _NW, "अगले हफ्ते": _NW, "अगले सप्ताह": _NW,
    "पिछले महीने": _LM, "इस महीने": _TM, "अगले महीने": _NM,
    "पिछले साल": _LY, "पिछले वर्ष": _LY, "इस साल": _TY, "इस वर्ष": _TY,
    "अगले साल": _NY, "अगले वर्ष": _NY,
    "pichle hafte": _LW, "is hafte": _TW, "agle hafte": _NW,
    "pichle mahine": _LM, "is mahine": _TM, "agle mahine": _NM,
    "pichle saal": _LY, "is saal": _TY, "agle saal": _NY,
    # Thai.
    "สัปดาห์ที่แล้ว": _LW, "อาทิตย์ที่แล้ว": _LW, "สัปดาห์นี้": _TW, "อาทิตย์นี้": _TW,
    "สัปดาห์หน้า": _NW, "อาทิตย์หน้า": _NW,
    "เดือนที่แล้ว": _LM, "เดือนนี้": _TM, "เดือนหน้า": _NM,
    "ปีที่แล้ว": _LY, "ปีนี้": _TY, "ปีหน้า": _NY,
    # Turkish.
    "geçen hafta": _LW, "bu hafta": _TW, "gelecek hafta": _NW, "önümüzdeki hafta": _NW,
    "haftaya": _NW,
    "geçen ay": _LM, "bu ay": _TM, "gelecek ay": _NM, "önümüzdeki ay": _NM,
    "geçen yıl": _LY, "geçen sene": _LY, "bu yıl": _TY, "bu sene": _TY,
    "gelecek yıl": _NY, "gelecek sene": _NY, "önümüzdeki yıl": _NY, "seneye": _NY,
    # Vietnamese, with and without vào.
    "tuần trước": _LW, "tuần này": _TW, "tuần sau": _NW, "tuần tới": _NW,
    "tháng trước": _LM, "tháng này": _TM, "tháng sau": _NM, "tháng tới": _NM,
    "năm ngoái": _LY, "năm nay": _TY, "năm sau": _NY, "năm tới": _NY, "sang năm": _NY,
    "vào tuần trước": _LW, "vào tuần sau": _NW, "vào tháng trước": _LM,
    "vào tháng sau": _NM, "vào năm ngoái": _LY, "vào năm sau": _NY, "vào năm tới": _NY,
    # Indonesian and Malay.
    "minggu lalu": _LW, "minggu ini": _TW, "minggu depan": _NW,
    "bulan lalu": _LM, "bulan ini": _TM, "bulan depan": _NM,
    "tahun lalu": _LY, "tahun ini": _TY, "tahun depan": _NY,
})
"""Exact phrases for last, this and next week, month and year."""

WeekdayReading = Literal[
    "previous", "current_week", "following", "previous_week", "following_week",
]


def _weekdays() -> Mapping[str, tuple[WeekdayReading, int]]:
    """Closed weekday phrases; explicit week words retain their ISO-week reading."""
    rows: dict[str, tuple[WeekdayReading, int]] = {}

    def add(names: tuple[str, ...], forms: tuple[tuple[WeekdayReading, str], ...]) -> None:
        for weekday, name in enumerate(names):
            for reading, form in forms:
                rows[form.format(name)] = reading, weekday

    english: tuple[tuple[WeekdayReading, str], ...] = (
        ("previous", "last {}"), ("current_week", "this {}"), ("following", "next {}"),
    )
    add(("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"), english)
    add(("mon", "tue", "wed", "thu", "fri", "sat", "sun"), english)
    add(("mon", "tues", "weds", "thurs", "fri", "sat", "sun"), english)
    rows.update({form.format("thur"): (reading, 3) for reading, form in english})
    explicit: tuple[tuple[WeekdayReading, str], ...] = (
        ("previous_week", "上周{}"), ("current_week", "本周{}"),
        ("current_week", "这周{}"), ("following_week", "下周{}"),
        ("previous_week", "上星期{}"), ("current_week", "本星期{}"),
        ("current_week", "这星期{}"), ("following_week", "下星期{}"),
    )
    add(("一", "二", "三", "四", "五", "六", "日"), explicit)
    rows.update({form.format("天"): (reading, 6) for reading, form in explicit})
    add(("月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"), (
        ("previous_week", "先週の{}"), ("current_week", "今週の{}"),
        ("following_week", "来週の{}"),
    ))
    add(("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"), (
        ("previous", "지난 {}"), ("current_week", "이번 {}"), ("following", "다음 {}"),
        ("previous_week", "지난주 {}"), ("previous_week", "지난 주 {}"),
        ("current_week", "이번 주 {}"), ("following_week", "다음 주 {}"),
    ))
    add(("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"), (
        ("previous", "el {} pasado"), ("previous", "{} pasado"),
        ("current_week", "este {}"), ("following", "el próximo {}"),
        ("following", "próximo {}"),
    ))
    add(("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira",
         "sábado", "domingo"), (
        ("previous_week", "{} da semana passada"), ("current_week", "{} desta semana"),
        ("following_week", "{} da próxima semana"),
    ))
    for weekday, name in enumerate(("segunda-feira", "terça-feira", "quarta-feira",
                                    "quinta-feira", "sexta-feira", "sábado", "domingo")):
        feminine = weekday < 5
        rows[f"{name} passad{'a' if feminine else 'o'}"] = "previous", weekday
        rows[f"{'esta' if feminine else 'este'} {name}"] = "current_week", weekday
        rows[f"próxim{'a' if feminine else 'o'} {name}"] = "following", weekday
    add(("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"), (
        ("previous", "{} dernier"), ("current_week", "ce {}"),
        ("following", "{} prochain"),
    ))
    for weekday, name in enumerate(("lunedì", "martedì", "mercoledì", "giovedì", "venerdì",
                                    "sabato", "domenica")):
        feminine = weekday == 6
        rows[f"{name} scors{'a' if feminine else 'o'}"] = "previous", weekday
        rows[f"{'questa' if feminine else 'questo'} {name}"] = "current_week", weekday
        rows[f"{name} prossim{'a' if feminine else 'o'}"] = "following", weekday
    add(("montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"), (
        ("previous", "letzten {}"), ("current_week", "diesen {}"),
        ("following", "nächsten {}"),
    ))
    for weekday, name in enumerate(("понедельник", "вторник", "среду", "четверг", "пятницу",
                                    "субботу", "воскресенье")):
        ending = "ую" if weekday in (2, 4, 5) else "ое" if weekday == 6 else "ый"
        rows[f"в прошл{ending} {name}"] = "previous", weekday
        next_ending = "ую" if weekday in (2, 4, 5) else "ее" if weekday == 6 else "ий"
        rows[f"в следующ{next_ending} {name}"] = "following", weekday
        rows[f"в {name} на этой неделе"] = "current_week", weekday
    add(("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"), (
        ("previous", "{} الماضي"), ("previous", "{} الماضية"),
        ("current_week", "{} هذا الأسبوع"), ("following", "{} المقبل"),
        ("following", "{} القادمة"),
    ))
    add(("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"), (
        ("previous", "पिछले {}"), ("current_week", "इस {}"), ("following", "अगले {}"),
    ))
    add(("วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์"), (
        ("previous", "{}ที่แล้ว"), ("current_week", "{}นี้"), ("following", "{}หน้า"),
    ))
    add(("pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"), (
        ("previous", "geçen {}"), ("current_week", "bu {}"), ("following", "gelecek {}"),
    ))
    add(("thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"), (
        ("previous_week", "{} tuần trước"), ("current_week", "{} tuần này"),
        ("following_week", "{} tuần sau"),
    ))
    add(("senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"), (
        ("previous", "{} lalu"), ("current_week", "{} ini"), ("following", "{} depan"),
    ))
    return _table(rows)


WEEKDAYS: Final[Mapping[str, tuple[WeekdayReading, int]]] = _weekdays()
"""Whole weekday phrases and their occurrence or explicit ISO-week reading (Monday=0)."""

WEEKENDS: Final[Mapping[str, Relation]] = _table({
    "last weekend": "last", "this weekend": "this", "next weekend": "next",
    "this past weekend": "last",
    "上周末": "last", "本周末": "this", "这周末": "this", "下周末": "next",
    "先週末": "last", "今週末": "this", "来週末": "next",
    "지난 주말": "last", "이번 주말": "this", "다음 주말": "next",
    "el fin de semana pasado": "last", "este fin de semana": "this",
    "el próximo fin de semana": "next",
    "o fim de semana passado": "last", "este fim de semana": "this",
    "o próximo fim de semana": "next",
    "le week-end dernier": "last", "ce week-end": "this", "le week-end prochain": "next",
    "lo scorso fine settimana": "last", "questo fine settimana": "this",
    "il prossimo fine settimana": "next",
    "letztes wochenende": "last", "dieses wochenende": "this", "nächstes wochenende": "next",
    "в прошлые выходные": "last", "в эти выходные": "this", "в следующие выходные": "next",
    "نهاية الأسبوع الماضي": "last", "نهاية هذا الأسبوع": "this", "نهاية الأسبوع المقبل": "next",
    "पिछले सप्ताहांत": "last", "इस सप्ताहांत": "this", "अगले सप्ताहांत": "next",
    "สุดสัปดาห์ที่แล้ว": "last", "สุดสัปดาห์นี้": "this", "สุดสัปดาห์หน้า": "next",
    "geçen hafta sonu": "last", "bu hafta sonu": "this", "gelecek hafta sonu": "next",
    "cuối tuần trước": "last", "cuối tuần này": "this", "cuối tuần sau": "next",
    "akhir pekan lalu": "last", "akhir pekan ini": "this", "akhir pekan depan": "next",
})
"""Relative weekends, always Saturday through Sunday of the named ISO week."""

COUNT_WORDS: Final[Mapping[str, int]] = MappingProxyType({
    "a": 1, "an": 1, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
})
"""Closed English counts; numeric counts remain available in every AGO form."""

_D = r"(\d{1,4})"
_D_EN = r"(\d{1,4}|" + "|".join(COUNT_WORDS) + ")"
_DAYS_EN = r"(?P<unit>hours?|days?|weeks?|weekends?)"
AGO: Final[tuple[tuple[re.Pattern[str], int], ...]] = tuple(
    (re.compile(pattern), sign) for pattern, sign in (
        (rf"{_D_EN} {_DAYS_EN} ago", -1), (rf"{_D_EN} {_DAYS_EN} later", 1),
        (rf"hace {_D} (?P<unit>días?|horas?|semanas?|fin(?:es)? de semana)", -1),
        (rf"dentro de {_D} (?P<unit>días?|horas?)", 1),
        (rf"há {_D} (?P<unit>dias?|horas?|semanas?|f(?:im|ins) de semana)", -1),
        (rf"daqui a {_D} (?P<unit>dias?|horas?)", 1),
        (rf"il y a {_D} (?P<unit>jours?|heures?|semaines?|week-?ends?)", -1),
        (rf"dans {_D} (?P<unit>jours?|heures?)", 1),
        (rf"{_D} (?P<unit>giorni|giorno|ore|ora|settimane|settimana|fin[ei] settimana) fa", -1),
        (rf"(?:tra|fra) {_D} (?P<unit>giorni|giorno|ore|ora)", 1),
        (rf"vor {_D} (?P<unit>tagen|tag|stunden|stunde|wochen?|wochenenden?)", -1),
        (rf"in {_D} (?P<unit>tagen|stunden)", 1),
        (rf"{_D} (?P<unit>день|дня|дней|час|часа|часов|неделю|недели|недель|"
         r"уикенд|уикенда|уикендов) назад", -1),
        (rf"через {_D} (?P<unit>день|дня|дней|час|часа|часов)", 1),
        (rf"منذ {_D} (?P<unit>يوم|يوما|ايام|ساعة|ساعات|اسبوع|اسابيع|عطلات نهاية الاسبوع)", -1),
        (rf"بعد {_D} (?P<unit>يوم|يوما|ايام|ساعة|ساعات)", 1),
        (rf"{_D} (?P<unit>दिन|घंटे|घंटा|सप्ताह|हफ्ते|हफ़्ते|सप्ताहांत) पहले", -1),
        (rf"{_D} (?P<unit>दिन|घंटे|घंटा) बाद", 1),
        (rf"{_D} ?(?P<unit>วัน|ชั่วโมง|สัปดาห์|สุดสัปดาห์)ที่แล้ว", -1),
        (rf"อีก ?{_D} ?(?P<unit>วัน|ชั่วโมง)", 1),
        (rf"{_D} ?(?P<unit>일|시간|주|주말) ?전", -1),
        (rf"{_D} ?(?P<unit>일|시간) ?(?:후|뒤)", 1),
        (rf"{_D}(?P<unit>日|時間|週間|週末)前", -1), (rf"{_D}(?P<unit>日|時間)後", 1),
        (rf"{_D}(?P<unit>天|小时|个小时|周|星期|个周末)前", -1),
        (rf"{_D}(?P<unit>天|小时|个小时)后", 1),
        (rf"{_D} (?P<unit>gün|saat|hafta|hafta sonu) önce", -1),
        (rf"{_D} (?P<unit>gün|saat) sonra", 1),
        (rf"{_D} (?P<unit>ngày|giờ|tuần|cuối tuần) trước", -1),
        (rf"{_D} (?P<unit>ngày|giờ) (?:sau|nữa)", 1),
        (rf"{_D} (?P<unit>hari|jam|minggu|akhir pekan) (?:yang )?lalu", -1),
        (rf"{_D} (?P<unit>hari|jam) lagi", 1),
    )
)
"""``(pattern, sign)``: a whole folded expression with a count and a unit."""

HOUR_UNITS: Final = frozenset({
    "hour", "hours", "hora", "horas", "heure", "heures", "ora", "ore", "stunde", "stunden",
    "час", "часа", "часов", "ساعة", "ساعات", "घंटे", "घंटा", "ชั่วโมง", "시간", "時間", "小时",
    "个小时", "saat", "giờ", "jam",
})
"""Units of `AGO` that count hours rather than days, weeks or weekends."""

WEEK_UNITS: Final = frozenset({
    "week", "weeks", "semana", "semanas", "semaine", "semaines", "settimana", "settimane",
    "woche", "wochen", "неделю", "недели", "недель", "اسبوع", "اسابيع", "सप्ताह", "हफ्ते",
    "हफ़्ते", "สัปดาห์", "주", "週間", "周", "星期", "hafta", "tuần", "minggu",
})
"""AGO units counting seven-day offsets: the offset ISO week (``calendar-unit-v1``)."""

WEEKEND_UNITS: Final = frozenset({
    "weekend", "weekends", "fin de semana", "fines de semana", "fim de semana", "fins de semana",
    "week-end", "week-ends", "fine settimana", "fini settimana", "wochenende", "wochenenden",
    "уикенд", "уикенда", "уикендов", "عطلات نهاية الاسبوع", "सप्ताहांत", "สุดสัปดาห์", "주말",
    "週末", "个周末", "hafta sonu", "cuối tuần", "akhir pekan",
})
"""AGO units naming the Saturday/Sunday range of an offset ISO week."""

CALENDAR_AGO: Final[tuple[tuple[re.Pattern[str], int], ...]] = tuple(
    (re.compile(pattern), sign) for pattern, sign in (
        (rf"{_D_EN} (?P<unit>months?|years?) ago", -1),
        (rf"{_D_EN} (?P<unit>months?|years?) later", 1),
        (rf"hace {_D} (?P<unit>mes|meses|año|años)", -1),
        (rf"há {_D} (?P<unit>mês|meses|ano|anos)", -1),
        (rf"il y a {_D} (?P<unit>mois|an|ans|année|années)", -1),
        (rf"{_D} (?P<unit>mese|mesi|anno|anni) fa", -1),
        (rf"vor {_D} (?P<unit>monat|monaten|jahr|jahren)", -1),
        (rf"{_D} ?(?P<unit>개월|년) ?전", -1),
        (rf"{_D}(?P<unit>か月|ヶ月|カ月|个月|年)前", -1),
    )
)
"""``(pattern, sign)``: a whole folded count of calendar months or years from the source time.

`AGO`'s units keep their reading under every time reading; these anchor only under
``calendar-unit-v1``, which reads them as the calendar month or year the count names.
"""

YEAR_UNITS: Final = frozenset({
    "year", "years", "año", "años", "ano", "anos", "an", "ans", "année", "années", "anno",
    "anni", "jahr", "jahren", "년", "年",
})
"""`CALENDAR_AGO` units counting calendar years; every other unit counts calendar months."""

_MONTH_NAMES: Final[tuple[tuple[str, ...], ...]] = (
    # English (with abbreviations), Spanish, Portuguese, French, Italian, German,
    # Turkish, Indonesian; Russian nominative, genitive and prepositional;
    # Arabic (Levant and Gulf/Egypt); Hindi; Thai; Vietnamese words.
    ("january", "jan", "enero", "janeiro", "janvier", "gennaio", "januar", "jänner", "ocak",
     "januari", "январь", "января", "январе", "كانون الثاني", "يناير", "जनवरी", "มกราคม",
     "tháng một", "tháng giêng"),
    ("february", "feb", "febrero", "fevereiro", "février", "febbraio", "februar", "şubat",
     "februari", "февраль", "февраля", "феврале", "شباط", "فبراير", "फ़रवरी", "फरवरी",
     "กุมภาพันธ์", "tháng hai"),
    ("march", "mar", "marzo", "março", "mars", "märz", "mart", "maret", "март", "марта",
     "марте", "اذار", "مارس", "मार्च", "มีนาคม", "tháng ba"),
    ("april", "apr", "abril", "avril", "aprile", "nisan", "апрель", "апреля", "апреле",
     "نيسان", "ابريل", "अप्रैल", "เมษายน", "tháng tư"),
    ("may", "mayo", "maio", "mai", "maggio", "mayıs", "mei", "май", "мая", "мае", "ايار",
     "مايو", "मई", "พฤษภาคม", "tháng năm"),
    ("june", "jun", "junio", "junho", "juin", "giugno", "juni", "haziran", "июнь", "июня",
     "июне", "حزيران", "يونيو", "जून", "มิถุนายน", "tháng sáu"),
    ("july", "jul", "julio", "julho", "juillet", "luglio", "juli", "temmuz", "июль", "июля",
     "июле", "تموز", "يوليو", "जुलाई", "กรกฎาคม", "tháng bảy"),
    ("august", "aug", "agosto", "août", "ağustos", "agustus", "август", "августа", "августе",
     "اب", "اغسطس", "अगस्त", "สิงหาคม", "tháng tám"),
    ("september", "sep", "sept", "septiembre", "setembro", "septembre", "settembre", "eylül",
     "сентябрь", "сентября", "сентябре", "ايلول", "سبتمبر", "सितंबर", "सितम्बर", "กันยายน",
     "tháng chín"),
    ("october", "oct", "octubre", "outubro", "octobre", "ottobre", "oktober", "ekim",
     "октябрь", "октября", "октябре", "تشرين الاول", "اكتوبر", "अक्टूबर", "ตุลาคม",
     "tháng mười"),
    ("november", "nov", "noviembre", "novembro", "novembre", "kasım", "ноябрь", "ноября",
     "ноябре", "تشرين الثاني", "نوفمبر", "नवंबर", "नवम्बर", "พฤศจิกายน", "tháng mười một"),
    ("december", "dec", "diciembre", "dezembro", "décembre", "dicembre", "dezember", "aralık",
     "desember", "декабрь", "декабря", "декабре", "كانون الاول", "ديسمبر", "दिसंबर", "दिसम्बर",
     "ธันวาคม", "tháng mười hai"),
)


def _months() -> Mapping[str, int]:
    found: dict[str, int] = {}
    for index, names in enumerate(_MONTH_NAMES, 1):
        for name in names:
            key = fold_time(name)
            if found.setdefault(key, index) != index:
                raise ValueError(f"Month name {name!r} names two months")
    return MappingProxyType(found)


MONTHS: Final[Mapping[str, int]] = _months()
"""Folded month names, their abbreviations and inflected forms, with the month number."""

THAI_MONTHS: Final = frozenset(name for name in MONTHS if "\u0e00" <= name[0] <= "\u0e7f")
"""The Thai month names of `MONTHS`: a year written with one may be Buddhist-era."""

ENGLISH_WORD_MONTHS: Final = frozenset({"march", "mar", "may"})
"""Month names that are also English words: they name a month only when capitalized."""

_MONTH = "(?P<month>" + "|".join(
    sorted((re.escape(name) for name in MONTHS), key=len, reverse=True)) + ")"
_NUMBER_MONTH = r"(?P<number>1[0-2]|0?[1-9])"
_CJK_MONTH = "(?P<cjk>十二|十一|十|[一二三四五六七八九])"

_NAMED_MONTHS: tuple[tuple[str, Relation], ...] = (
    (rf"este {_MONTH}", "this"), (rf"(?:el )?{_MONTH} pasado", "last"),
    (rf"el próximo {_MONTH}", "next"), (rf"(?:el )?{_MONTH} próximo", "next"),
    (rf"(?:o )?{_MONTH} passado", "last"), (rf"(?:o )?próximo {_MONTH}", "next"),
    (rf"ce {_MONTH}", "this"), (rf"(?:en )?{_MONTH} dernier", "last"),
    (rf"(?:en )?{_MONTH} prochain", "next"),
    (rf"questo {_MONTH}", "this"), (rf"(?:lo )?scorso {_MONTH}", "last"),
    (rf"{_MONTH} scorso", "last"), (rf"(?:il )?prossimo {_MONTH}", "next"),
    (rf"diesen {_MONTH}", "this"), (rf"im {_MONTH} dieses jahres", "this"),
    (rf"letzten {_MONTH}", "last"), (rf"nächsten {_MONTH}", "next"),
    (rf"в этом {_MONTH}", "this"), (rf"в прошлом {_MONTH}", "last"),
    (rf"в следующем {_MONTH}", "next"),
    (rf"इस {_MONTH}", "this"), (rf"पिछले {_MONTH}", "last"), (rf"अगले {_MONTH}", "next"),
    (rf"bu {_MONTH}(?:da|de|ta|te)?", "this"), (rf"geçen {_MONTH}(?:da|de|ta|te)?", "last"),
    (rf"gelecek {_MONTH}(?:da|de|ta|te)?", "next"),
    (rf"(?:vào )?{_MONTH} này", "this"), (rf"(?:vào )?tháng {_NUMBER_MONTH} này", "this"),
    (rf"เดือน{_MONTH}นี้", "this"), (rf"{_MONTH}นี้", "this"),
    (rf"이번 {_NUMBER_MONTH}월(?:에)?", "this"),
)
NAMED_MONTH_FORMS: Final[tuple[tuple[re.Pattern[str], Relation], ...]] = tuple(
    (re.compile(pattern), relation) for pattern, relation in _NAMED_MONTHS
)
"""``(pattern, relation)``: "this, last or next <month>" as a whole folded expression."""

_THIS_YEAR = (r"今年の?", r"올해 ?", r"금년 ?")
_LAST_YEAR = (r"去年の?", r"昨年の?", r"작년 ?", r"지난해 ?")
_NEXT_YEAR = (r"明年", r"来年の?", r"내년 ?")
_CJK_YEAR_MONTH = rf"(?:{_NUMBER_MONTH}|{_CJK_MONTH})(?:月|월)(?:に|에)?"
YEAR_MONTH_FORMS: Final[tuple[tuple[re.Pattern[str], int], ...]] = tuple(
    (re.compile(pattern), offset) for pattern, offset in (
        *((year + _CJK_YEAR_MONTH, 0) for year in _THIS_YEAR),
        *((year + _CJK_YEAR_MONTH, -1) for year in _LAST_YEAR),
        *((year + _CJK_YEAR_MONTH, 1) for year in _NEXT_YEAR),
        (rf"(?:في )?{_MONTH} (?:هذا العام|من هذا العام|هذه السنة)", 0),
        (rf"(?:في )?{_MONTH} (?:العام الماضي|من العام الماضي|السنة الماضية)", -1),
        (rf"(?:في )?{_MONTH} (?:العام المقبل|من العام المقبل|العام القادم|السنة القادمة)", 1),
        (rf"(?:vào )?(?:{_MONTH}|tháng {_NUMBER_MONTH}) năm nay", 0),
        (rf"(?:vào )?(?:{_MONTH}|tháng {_NUMBER_MONTH}) năm ngoái", -1),
        (rf"(?:vào )?(?:{_MONTH}|tháng {_NUMBER_MONTH}) (?:năm sau|năm tới)", 1),
        (rf"(?:เดือน)?{_MONTH}ปีนี้", 0), (rf"(?:เดือน)?{_MONTH}ปีที่แล้ว", -1),
        (rf"(?:เดือน)?{_MONTH}ปีหน้า", 1),
        (rf"(?:en )?{_MONTH} de este año", 0), (rf"(?:en )?{_MONTH} del año pasado", -1),
        (rf"(?:en )?{_MONTH} del año que viene", 1), (rf"(?:en )?{_MONTH} del próximo año", 1),
    )
)
"""``(pattern, year offset)``: a month of the source's year, the year before or after."""

_Y = r"(?P<year>\d{4})"
YEAR_FORMS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(pattern) for pattern in (
    rf"(?:en|em|nel|im jahr) {_Y}",
    rf"(?:в )?{_Y} (?:году|г)",
    rf"(?:في )?(?:عام|سنة) {_Y}",
    rf"{_Y} में",
    rf"(?:साल|वर्ष) {_Y}(?: में)?",
    rf"{_Y}(?:'de|'da|'te|'ta| yılında)",
    rf"(?:vào )?năm {_Y}",
    rf"{_Y}년(?:에|도)?",
    rf"在{_Y}年",
    rf"{_Y}年に",
    rf"(?:pada )?tahun {_Y}",
))
"""Whole folded expressions that name exactly one Gregorian year (named group ``year``)."""

SINCE_YEAR_FORMS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(pattern) for pattern in (
    rf"(?:desde|depuis|seit|dal) {_Y}",
    rf"с {_Y} (?:года|г)",
    rf"منذ (?:عام |سنة )?{_Y}",
    rf"{_Y} से",
    rf"{_Y}(?:'den|'dan|'ten|'tan) beri",
    rf"từ năm {_Y}",
    rf"{_Y}년부터",
    rf"{_Y}年から",
    rf"(?:从|自){_Y}年(?:起|以来)?",
    rf"{_Y}年以来",
    rf"sejak (?:tahun )?{_Y}",
))
"""Whole folded expressions that open a boundary at the start of a Gregorian year ("since")."""

THAI_YEAR_FORMS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(pattern) for pattern in (
    rf"(?:ใน|เมื่อ)?ปี ?{_Y}", rf"(?:ใน|เมื่อ)?(?:ปี )?(?P<be>พ\.ศ\.) ?{_Y}",
    rf"(?:ใน|เมื่อ)?(?:ปี )?(?P<ce>ค\.ศ\.) ?{_Y}",
))
"""Thai year forms: ``พ.ศ.`` marks a Buddhist-era year, ``ค.ศ.`` a Gregorian one, and an
unmarked year of 2400 or more is Buddhist-era (`buddhist_era`)."""

_DAY = r"(?P<day>3[01]|[12]\d|0?[1-9])"
DATE_FORMS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(pattern) for pattern in (
    rf"(?:el )?{_DAY} de {_MONTH} de(?:l)? {_Y}",
    rf"(?:le )?{_DAY}(?:er)? {_MONTH} {_Y}",
    rf"(?:am )?{_DAY}\. {_MONTH} {_Y}",
    rf"{_DAY} {_MONTH} {_Y}(?: (?:г|года))?",
    rf"(?:في )?{_DAY} {_MONTH} {_Y}",
    rf"{_Y}년 {_NUMBER_MONTH}월 {_DAY}일(?:에)?",
    rf"(?:vào )?ngày {_DAY} tháng {_NUMBER_MONTH} năm {_Y}",
    rf"(?:วันที่ )?{_DAY} {_MONTH} (?:(?P<be>พ\.ศ\.) ?)?{_Y}",
))
"""Whole folded expressions that name one day: day, month and year (named groups)."""

MONTH_YEAR_FORMS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(pattern) for pattern in (
    rf"(?:en |im |em |nel |в )?{_MONTH}(?: de(?:l)?)? {_Y}(?: (?:года|г))?",
    rf"{_MONTH} {_Y}(?:'de|'da|'te|'ta| में)?",
    rf"(?:في )?(?:شهر )?{_MONTH} (?:عام |سنة )?{_Y}",
    rf"(?:pada )?(?:bulan )?{_MONTH} {_Y}",
    rf"{_Y}년 {_NUMBER_MONTH}월(?:에)?",
    rf"(?:vào )?(?:tháng {_NUMBER_MONTH}|{_MONTH}) năm {_Y}",
    rf"(?:เดือน)?{_MONTH} (?:(?P<be>พ\.ศ\.) ?)?{_Y}",
))
"""Whole folded expressions that name one month of one year (named groups)."""

CJK_NUMERALS: Final = MappingProxyType({
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12,
})
"""Chinese and Japanese numerals of the twelve months."""

GREGORIAN_LIMIT: Final = 2400
"""The first year never read as Gregorian: from it on, a Thai year is Buddhist-era."""

BUDDHIST_ERA_OFFSET: Final = 543
"""Years between the Buddhist era and the Gregorian calendar: 2567 BE is 2024."""


def buddhist_era(year: int) -> int | None:
    """Return the Gregorian year of a year written in a Thai year form.

    A year from `GREGORIAN_LIMIT` on is a Buddhist-era year and loses
    `BUDDHIST_ERA_OFFSET`; an earlier one is already Gregorian, as Thai text
    also writes them.

    Args:
        year (int): The year as written.

    Returns:
        int | None: The Gregorian year, or None when the converted year would be
            an earlier one than any Gregorian year written in this form.
    """
    if year < GREGORIAN_LIMIT:
        return year
    converted = year - BUDDHIST_ERA_OFFSET
    return converted if converted >= 1800 else None
