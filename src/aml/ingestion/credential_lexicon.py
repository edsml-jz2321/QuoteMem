"""Closed credential labels and separators beyond English for reception (B.2 #3, ML-1).

`boundary` refuses an Add whose text states a credential: a known key format, or a
label, a separator and a value (``password: x``, ``password is x``). Other languages
state a password with their own labels and separators (``Mi contraseña del banco es
…``, ``我的银行密码是…``, ``銀行のパスワードは…``, ``Mật khẩu ngân hàng của tôi là …``),
and each of those passed reception and was stored. `FORMS` holds them for the
languages the sensitivity classifier covers (D.8.1): Spanish, Portuguese, French,
Italian, German, Russian, Arabic, Hindi and Hinglish, Thai, Korean, Japanese,
Chinese, Turkish, Vietnamese and Indonesian. English keeps `boundary`'s own list.

A form is a closed label, at most `Language.gap` qualifier words before its separator
(``contraseña del banco es``; never across a listed clause word such as ``porque``),
a closed separator and a value:

- after a colon or an equals sign, and after a Latin-script copula, any token, as
  English ``password is x`` takes; a listed question word (``Mật khẩu wifi là gì?``)
  is not a value;
- after a copula or particle of another script, a quoted value or a run of four or
  more typed characters (ASCII, full-width letters, digits) holding a letter or a
  digit, unless a number runs into a unit (``2024年``): prose there continues in its
  own script (``パスワードは変更しました``, ``密码是多少``, ``пароль — это секрет``),
  a typed secret does not;
- where nothing separates the label from the value (Turkish ``Banka şifrem
  tulip-42``; Hindi and Hinglish ``password tulip-42 hai``, closed by the copula), a
  quoted value or a token of four or more characters holding a digit or a symbol:
  a next word alone states nothing (``Şifrem değişti``, ``password galat hai``);
- for a verification code, four to ten digits after an optional separator.

Every word matches case-insensitively in its composed and decomposed forms.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from aml.core.statement import COMBINING_MARKS, folded
from aml.ingestion.word_patterns import alternation

CredentialCategory = Literal["password", "verification_code"]
Script = Literal["spaced", "compound", "attached", "unspaced"]
ValueRule = Literal["any", "typed", "marked"]


@dataclass(frozen=True, slots=True)
class Language:
    """The closed credential vocabulary of one language (module docstring).

    Attributes:
        script (Script): How words meet. ``spaced`` labels and copulas are whole
            words; a ``compound`` label may end a longer word (German
            ``Bankpasswort``); an ``attached`` copula follows its label directly
            (Korean ``비밀번호는``); ``unspaced`` text has no word boundaries.
        passwords (tuple[str, ...]): Labels of a password.
        codes (tuple[str, ...]): Labels of a verification code.
        copulas (tuple[str, ...]): Separators stating the value (``es``, ``是``, ``—``).
        value (ValueRule): What a copula, a closing or a bare label takes.
        gap (int): Most qualifier words (in unspaced text, characters) between a
            label and its separator.
        stops (tuple[str, ...]): Clause words a gap never crosses.
        questions (tuple[str, ...]): Question words that are never a value.
        closings (tuple[str, ...]): Verb-final copulas after a value (Hindi ``है``).
        suffixes (tuple[str, ...]): Copula suffixes after an apostrophe that a bare
            value may carry (Turkish ``tulip-42'dir``); any other suffix is prose.
        bare (bool): Whether a value may follow the label with nothing between.
    """

    script: Script
    passwords: tuple[str, ...]
    codes: tuple[str, ...] = ()
    copulas: tuple[str, ...] = ()
    value: ValueRule = "any"
    gap: int = 0
    stops: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    closings: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    bare: bool = False


LANGUAGES: Final[Mapping[str, Language]] = MappingProxyType({
    # Spanish, and Spanish mixed with English ("Mi bank password es …").
    "es": Language(
        "spaced",
        passwords=("contraseña", "contrasena", "clave de acceso", "clave secreta", "password"),
        codes=("código de verificación", "codigo de verificacion", "código de confirmación",
               "codigo de confirmacion", "código otp", "codigo otp"),
        copulas=("es",), gap=5,
        stops=("porque", "pero", "aunque", "si", "cuando", "mientras", "pues", "sino"),
        questions=("cuál", "cual", "qué", "que"),
    ),
    "pt": Language(
        "spaced",
        passwords=("senha", "palavra-passe", "palavra passe", "password"),
        codes=("código de verificação", "codigo de verificacao", "código de confirmação",
               "codigo de confirmacao", "código otp", "codigo otp"),
        copulas=("é",), gap=5,
        stops=("porque", "mas", "embora", "se", "quando", "enquanto", "pois"),
        questions=("qual", "que"),
    ),
    "fr": Language(
        "spaced",
        passwords=("mot de passe", "mot-de-passe", "mdp", "password"),
        codes=("code de vérification", "code de verification", "code de confirmation",
               "code otp", "code à usage unique"),
        copulas=("est", "c'est", "c’est", ", c'est", ", c’est"), gap=5,
        stops=("parce", "mais", "car", "si", "quand", "lorsque", "puisque", "sinon"),
        questions=("quoi", "quel", "quelle"),
    ),
    "it": Language(
        "spaced",
        passwords=("password", "parola d'ordine", "parola d’ordine"),
        codes=("codice di verifica", "codice di conferma", "codice otp"),
        copulas=("è", "e'"), gap=4,
        stops=("perché", "perche", "ma", "però", "se", "quando", "mentre"),
        questions=("quale", "cosa", "che"),
    ),
    "de": Language(
        "compound",
        passwords=("passwort", "kennwort"),
        codes=("bestätigungscode", "bestaetigungscode", "verifizierungscode",
               "verifikationscode", "einmalcode"),
        copulas=("ist", "lautet", "heißt", "heisst"), gap=4,
        stops=("dass", "weil", "aber", "wenn", "ob", "obwohl", "sondern", "denn"),
        questions=("was", "welches"),
    ),
    "ru": Language(
        "spaced",
        passwords=("пароль",),
        codes=("код подтверждения", "код проверки", "проверочный код", "одноразовый код"),
        copulas=("—", "–", "-", "это", "— это", "– это", "- это"), value="typed", gap=4,
        stops=("потому", "но", "если", "когда", "хотя", "а"),
    ),
    "ar": Language(
        "spaced",
        passwords=("كلمة المرور", "كلمة مرور", "كلمة السر", "كلمه المرور", "كلمه مرور",
                   "كلمه السر", "الرقم السري", "الرمز السري", "باسورد", "الباسورد"),
        codes=("رمز التحقق", "رمز التأكيد", "رمز التفعيل", "كود التحقق", "كود التفعيل"),
        copulas=("هي", "هو"), value="typed", gap=4,
        stops=("لأن", "لكن", "ولكن", "إذا", "عندما"),
    ),
    "hi": Language(
        "spaced",
        passwords=("पासवर्ड", "पास्वर्ड", "password"),
        codes=("ओटीपी", "सत्यापन कोड", "वेरिफिकेशन कोड"),
        value="typed", closings=("है", "हैं", "था", "थी"),
    ),
    # Romanized Hindi with English labels ("Mera bank password tulip-42 hai").
    "hinglish": Language(
        "spaced", passwords=("password",), value="marked", closings=("hai", "hain", "tha", "thi"),
    ),
    "th": Language(
        "unspaced",
        passwords=("รหัสผ่าน", "พาสเวิร์ด"),
        codes=("รหัสยืนยัน", "รหัส otp", "รหัสotp", "โอทีพี"),
        copulas=("คือ", "เป็น"), value="typed", gap=24,
    ),
    "ko": Language(
        "attached",
        passwords=("비밀번호", "비밀 번호", "비번", "패스워드", "암호"),
        codes=("인증번호", "인증 번호", "인증코드", "인증 코드", "확인코드", "확인 코드"),
        copulas=("는", "은", "가", "이"), value="typed",
    ),
    "ja": Language(
        "unspaced",
        passwords=("パスワード", "ﾊﾟｽﾜｰﾄﾞ", "暗証番号", "パスコード"),
        codes=("認証コード", "確認コード", "認証番号", "確認番号", "ワンタイムパスワード",
               "ワンタイムコード"),
        copulas=("は", "って"), value="typed",
    ),
    "zh": Language(
        "unspaced",
        passwords=("密码", "密碼", "口令"),
        codes=("验证码", "驗證碼", "校验码", "校驗碼", "动态码", "動態碼"),
        copulas=("是", "为", "為", "就是"), value="typed",
    ),
    "tr": Language(
        "spaced",
        passwords=("şifre", "şifrem", "şifren", "şifresi", "şifremiz", "şifreniz", "parola",
                   "parolam", "parolan", "parolası", "parolamız", "parolanız", "sifre", "sifrem",
                   "sifresi", "sifreniz"),
        codes=("doğrulama kodu", "dogrulama kodu", "doğrulama kodunuz", "onay kodu",
               "onay kodunuz"),
        value="marked", gap=2, bare=True,
        suffixes=("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür"),
    ),
    "vi": Language(
        "spaced",
        passwords=("mật khẩu", "mat khau", "password"),
        codes=("mã xác minh", "mã xác thực", "mã xác nhận", "mã otp", "ma xac minh",
               "ma xac thuc", "ma xac nhan", "ma otp"),
        copulas=("là", "la"), gap=7,
        stops=("nhưng", "nhung", "vì", "nếu", "neu", "khi", "tuy", "rằng", "rang"),
        questions=("gì", "gi", "nào", "nao", "bao"),
    ),
    "id": Language(
        "spaced",
        passwords=("kata sandi", "katasandi", "password"),
        codes=("kode verifikasi", "kode otp", "kode konfirmasi"),
        copulas=("adalah", "ialah", "yaitu", "yakni"), gap=4,
        stops=("karena", "tetapi", "tapi", "jika", "kalau", "ketika", "walaupun", "meskipun"),
        questions=("apa",),
    ),
})

_COLON: Final = r"\s*[:=\uff1a\uff1d]\s*"
_QUOTED: Final = (
    r"""["'\u201c\u2018\u300c\u300e\u00ab]"""
    r"""(?P<quoted>[^"'\u201d\u2019\u300d\u300f\u00bb\r\n]+)"""
    r"""["'\u201d\u2019\u300d\u300f\u00bb]"""
)
_BARE: Final = r"""[^\s"',;<>]"""
_TYPED: Final = r"(?:[!#-&(-+\--:=?-~\uff21-\uff3a\uff41-\uff5a]|\d)"
_TYPED_ALNUM: Final = r"(?:[0-9A-Za-z\uff21-\uff3a\uff41-\uff5a]|\d)"
_MARK: Final = r"(?:\d|[#-&(-+\-/:=@\[-`{-~])"
_UNIT: Final = (
    "[年月日号號位个個次元块塊分秒点點时時字文桁歳岁歲円回件度"
    "년월일자개번원시분초살명회]"
)
_WORD: Final = r"[^\s.!?;:,\u060c\u061b\u061f\u0964\u3001\u3002\uff01\uff0c\uff1f]+"
_CODE: Final = r"""["'\u201c\u300c]?(?P<bare>\d{4,10})(?!\d)"""


def _label(language: Language, words: Iterable[str]) -> str:
    label = alternation(words, r"\s+")
    if language.script == "spaced":
        return rf"(?<!\w){label}(?!\w)"
    if language.script == "compound":
        return rf"{label}(?!\w)"
    return label


def _gap(language: Language) -> str:
    if not language.gap:
        return ""
    if language.script == "unspaced":
        return rf"(?:[^\W\d_]|[{COMBINING_MARKS}]| ){{0,{language.gap}}}?"
    stops = rf"(?!{alternation(language.stops, r'\s+')}(?!\w))" if language.stops else ""
    return rf"(?:\s+{stops}{_WORD}){{0,{language.gap}}}?"


def _separator(language: Language) -> str:
    words = [copula for copula in language.copulas if copula[0].isalpha()]
    marks = [copula for copula in language.copulas if not copula[0].isalpha()]
    choices: list[str] = []
    if words:
        spaced = language.script in ("spaced", "compound")
        lead, tail = (r"\s+", r"(?!\w)") if spaced else (r"\s*", "")
        choices.append(lead + alternation(words, r"\s+") + tail)
    if marks:
        # A mark ending in a word (Russian "— это") ends at that word's boundary.
        choices.append(r"\s*" + alternation(marks, r"\s*") + r"(?!(?<=[^\W\d_])\w)")
    return "(?:" + "|".join(choices) + r")\s*(?:[:\uff1a]\s*)?"


def _value(language: Language, rule: ValueRule) -> str:
    if rule == "any":
        questions = (
            rf"(?!{alternation(language.questions, r'\s+')}(?!\w))" if language.questions else ""
        )
        return rf"{questions}(?:{_QUOTED}|(?P<bare>{_BARE}+))"
    if rule == "typed":
        return (
            rf"(?:{_QUOTED}|(?P<bare>(?={_TYPED}*?{_TYPED_ALNUM}){_TYPED}{{4,}}+)"
            rf"(?!(?<=\d){_UNIT}))"
        )
    suffix = (
        rf"(?!'(?!{alternation(language.suffixes, '')}(?!\w)))" if language.suffixes else ""
    )
    return rf"(?:{_QUOTED}|(?P<bare>(?={_BARE}*?{_MARK}){_BARE}{{4,}}+){suffix})"


def _forms() -> tuple[tuple[CredentialCategory, re.Pattern[str]], ...]:
    forms: list[tuple[CredentialCategory, str]] = []
    for language in LANGUAGES.values():
        label, gap = _label(language, language.passwords), _gap(language)
        forms.append(("password", label + gap + _COLON + _value(language, "any")))
        if language.copulas:
            forms.append((
                "password", label + gap + _separator(language) + _value(language, language.value),
            ))
        if language.closings:
            forms.append((
                "password",
                label + r"\s+" + _value(language, language.value) + r"\s*"
                + alternation(language.closings, r"\s+") + r"(?!\w)",
            ))
        if language.bare:
            forms.append(("password", label + r"\s+" + _value(language, language.value)))
        if language.codes:
            # A qualifier gap needs a separator after it; without one the code follows
            # its label directly ("código de verificación 482913").
            separator = _separator(language) + "|" if language.copulas else ""
            forms.append((
                "verification_code",
                _label(language, language.codes)
                + rf"(?:{_gap(language)}(?:{separator}{_COLON})|\s*)" + _CODE,
            ))
    return tuple((category, re.compile(pattern, re.IGNORECASE)) for category, pattern in forms)


FORMS: Final[tuple[tuple[CredentialCategory, re.Pattern[str]], ...]] = _forms()
"""Each language's colon, copula, closing, bare and code forms, with their category.

A match's ``quoted`` or ``bare`` group is its value."""


def _fold(text: str) -> str:
    # The regex case rule also equates the Turkish dotless i with i; casefolding does not.
    return folded(text).replace("ı", "i")


_LABELS: Final = re.compile("|".join(sorted(
    {
        r"\s+".join(re.escape(part) for part in _fold(word).split())
        for language in LANGUAGES.values() for word in (*language.passwords, *language.codes)
    },
    key=lambda label: (-len(label), label),
)))


def has_label(text: str) -> bool:
    """Whether the folded text holds a label of `LANGUAGES`; without one no form matches.

    `aml.core.statement.folded` composes, case-folds and unifies compatibility forms
    (and the dotless i is read as i), so a label that a form finds in any normal form
    or case is found here too, for a fraction of the cost of the forms themselves.

    Args:
        text (str): Text to scan, possibly holding lone surrogates.

    Returns:
        bool: Whether `FORMS` can match the text.
    """
    return _LABELS.search(_fold(text)) is not None
