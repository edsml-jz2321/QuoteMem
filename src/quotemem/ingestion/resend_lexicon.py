"""Closed short-assertion signals for the empty-reply resend.

`extraction_stage.needs_fresh_call` gives a reply without facts its one fresh call when
a fresh, readable, non-assistant stretch names someone in a declarative sentence:
English ``My cousin Inez is a glassblower.`` (`_NAMED_POSSESSIVE`). Other
languages say it with their own possessives and nouns (``Mi prima Inez es …``,
``Kuzenim Inez …``, ``我的表妹伊内兹是…``), which that English shape never
matches, so a short empty reply on such a source stood. `names_someone` finds them:

- a first-person possessive with a closed noun for a relative, partner, friend,
  colleague, boss, neighbour or pet, in the language's order (``mi hermana``,
  ``em gái tôi``, ``kardeşim``, ``أختي``, ``พี่สาวของฉัน``, ``妹の``); a possessive
  that English never writes (Portuguese ``minha``, German ``meine``, Russian ``мой``
  in every case) takes any noun, as English ``my`` does;
- followed by a name: in a cased script a capitalized word, as English requires; in a
  script without case any word, since no spelling marks a name there.

Korean also permits a listed noun without its possessive when a Hangul word with
a subject/topic particle (``은``, ``는``, ``이``, ``가``) follows it (``남편 로완이``).
Vietnamese also puts a capitalized name of one to four words between the noun and
``của tôi`` or ``của mình`` (``hamster Mochi của tôi``).

A sentence ending in a question mark never qualifies. Like the English shape, a phrase
only buys the one fresh call; it never establishes a fact or an entity.

``asserts_personal_fact`` adds closed subject/predicate/content shapes for
short personal assertions without a named relative: the closed first-person
subjects and predicates below. Both signals share the existing one-call allowance
with the declaration, length and attribution triggers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final

from quotemem.ingestion.word_patterns import alternation


@dataclass(frozen=True, slots=True)
class _Phrases:
    """One language's possessive phrases; ``cased`` asks a capitalized name after them."""

    pattern: re.Pattern[str]
    cased: bool


def _before(possessives: Iterable[str], nouns: Iterable[str]) -> str:
    """A possessive word before its noun, both whole words (``mi hermana``)."""
    return (rf"(?<!\w){alternation(possessives, r'\s+')}\s+"
            rf"{alternation(nouns, r'\s+')}(?!\w)")


def _after(nouns: Iterable[str], possessives: Iterable[str]) -> str:
    """A noun before its possessive word, both whole words (``em gái tôi``)."""
    return (rf"(?<!\w){alternation(nouns, r'\s+')}\s+"
            rf"{alternation(possessives, r'\s+')}(?!\w)")


def _before_any(possessives: Iterable[str]) -> str:
    """A possessive word English never writes before any one word (``мой брат``)."""
    return rf"(?<!\w){alternation(possessives, r'\s+')}\s+[^\W\d_][\w'\u2019-]*"


def _whole(words: Iterable[str]) -> str:
    """Possessed nouns written as one word (``kardeşim``, ``أختي``)."""
    return rf"(?<!\w){alternation(words, r'\s+')}(?!\w)"


_ES_NOUNS: Final = (
    "hermana", "hermano", "madre", "padre", "mamá", "papá", "hijo", "hija", "esposa",
    "esposo", "marido", "mujer", "novia", "novio", "pareja", "amigo", "amiga", "prima",
    "primo", "tía", "tío", "abuela", "abuelo", "sobrina", "sobrino", "jefe", "jefa",
    "compañero", "compañera", "colega", "vecino", "vecina", "perro", "perra", "gato", "gata",
    "hámster", "hamster", "conejo", "loro", "partner",
)
_FR_NOUNS: Final = (
    "sœur", "soeur", "frère", "mère", "père", "fils", "fille", "femme", "mari", "épouse",
    "époux", "copine", "copain", "compagne", "compagnon", "partenaire", "ami", "amie",
    "cousine", "cousin", "tante", "oncle", "grand-mère", "grand-père", "patron", "patronne",
    "collègue", "voisin", "voisine", "chien", "chienne", "chat", "chatte",
    "hamster", "lapin", "perroquet",
)
_IT_NOUNS: Final = (
    "sorella", "fratello", "madre", "padre", "mamma", "papà", "figlio", "figlia", "moglie",
    "marito", "ragazza", "ragazzo", "compagna", "compagno", "partner", "amico", "amica",
    "cugina", "cugino", "zia", "zio", "nonna", "nonno", "capo", "collega", "vicino",
    "vicina", "cane", "gatto", "gatta",
    "criceto", "coniglio", "pappagallo",
)
_TR_NOUNS: Final = (
    "kardeşim", "kız kardeşim", "erkek kardeşim", "ablam", "ağabeyim", "abim", "annem",
    "babam", "oğlum", "kızım", "eşim", "karım", "kocam", "sevgilim", "arkadaşım",
    "kuzenim", "teyzem", "halam", "amcam", "dayım", "anneannem", "babaannem", "dedem",
    "patronum", "meslektaşım", "komşum", "köpeğim", "kedim",
    "partnerim", "hamsterım", "tavşanım", "papağanım",
)
_VI_NOUNS: Final = (
    "chị gái", "em gái", "anh trai", "em trai", "chị", "em", "anh", "mẹ", "bố", "ba",
    "cha", "con trai", "con gái", "vợ", "chồng", "bạn gái", "bạn trai", "bạn", "chị họ",
    "anh họ", "em họ", "cô", "dì", "chú", "bác", "bà", "ông", "sếp", "đồng nghiệp",
    "hàng xóm", "chó", "mèo",
    "người yêu", "hamster", "thỏ", "vẹt",
)
_ID_NOUNS: Final = (
    "kakak", "adik", "kakak perempuan", "adik perempuan", "kakak laki-laki",
    "adik laki-laki", "ibu", "ayah", "mama", "papa", "anak", "putra", "putri", "istri",
    "suami", "pacar", "teman", "sepupu", "bibi", "tante", "paman", "om", "nenek", "kakek",
    "bos", "atasan", "rekan kerja", "tetangga", "anjing", "kucing",
    "pasangan", "hamster", "kelinci", "burung",
)
_HINGLISH_NOUNS: Final = (
    "behen", "bahan", "bhai", "maa", "mummy", "papa", "beta", "beti", "biwi", "patni",
    "pati", "wife", "husband", "girlfriend", "boyfriend", "dost", "friend", "cousin",
    "chacha", "chachi", "mausi", "bua", "mama", "mami", "nani", "nana", "dadi", "dada",
    "boss", "padosi", "sister", "brother",
    "partner", "hamster",
)
_HI_NOUNS: Final = (
    "बहन", "भाई", "माँ", "मां", "माता", "पिता", "पापा", "मम्मी", "बेटा", "बेटी", "पत्नी", "पति",
    "बीवी", "गर्लफ्रेंड", "बॉयफ्रेंड", "पार्टनर", "दोस्त", "मित्र", "मौसी", "बुआ", "चाचा",
    "चाची", "मामा", "मामी", "दादी", "दादा", "नानी", "नाना", "बॉस", "सहकर्मी", "पड़ोसी",
    "पड़ोसन", "कुत्ता", "बिल्ली",
    "हैम्स्टर", "खरगोश", "तोता",
)
_AR_NOUNS: Final = (
    "أختي", "أخي", "أمي", "والدتي", "أبي", "والدي", "ابني", "ابنتي", "بنتي", "زوجتي",
    "زوجي", "صديقتي", "صديقي", "خطيبتي", "خطيبي", "شريكي", "شريكتي", "عمتي", "عمي",
    "خالتي", "خالي", "جدتي", "جدي", "مديري", "مديرتي", "زميلي", "زميلتي", "جاري", "جارتي",
    "كلبي", "قطتي", "قطي",
    "أرنبي", "ببغائي",
)
_TH_NOUNS: Final = (
    "พี่สาว", "น้องสาว", "พี่ชาย", "น้องชาย", "แม่", "พ่อ", "ลูกชาย", "ลูกสาว", "ภรรยา",
    "สามี", "แฟน", "เพื่อน", "ลูกพี่ลูกน้อง", "ป้า", "น้า", "ลุง", "ยาย", "ย่า", "ปู่",
    "หัวหน้า", "เจ้านาย", "เพื่อนร่วมงาน", "เพื่อนบ้าน", "หมา", "สุนัข", "แมว",
    "แฮมสเตอร์", "กระต่าย", "นกแก้ว",
)
_KO_NOUNS: Final = (
    "여동생", "남동생", "동생", "누나", "언니", "오빠", "형", "엄마", "어머니", "아빠",
    "아버지", "아들", "딸", "아내", "남편", "와이프", "여자친구", "남자친구", "여친",
    "남친", "파트너", "친구", "사촌", "이모", "고모", "삼촌", "할머니", "할아버지", "상사",
    "동료", "이웃", "강아지", "고양이",
    "햄스터", "토끼", "앵무새",
)
_JA_NOUNS: Final = (
    "妹", "姉", "兄", "弟", "母", "父", "母親", "父親", "息子", "娘", "妻", "夫", "彼女",
    "彼氏", "パートナー", "友達", "友人", "いとこ", "叔母", "伯母", "叔父", "伯父", "祖母",
    "祖父", "上司", "同僚", "隣人", "犬", "猫",
    "ハムスター", "ウサギ", "うさぎ", "インコ", "鳥",
)
_ZH_NOUNS: Final = (
    "妹妹", "姐姐", "哥哥", "弟弟", "妈妈", "媽媽", "母亲", "母親", "爸爸", "父亲", "父親",
    "儿子", "兒子", "女儿", "女兒", "妻子", "老婆", "丈夫", "老公", "女朋友", "男朋友",
    "女友", "男友", "伴侣", "伴侶", "朋友", "表姐", "表妹", "表哥", "表弟", "堂姐", "堂妹",
    "堂哥", "堂弟", "阿姨", "叔叔", "舅舅", "姑姑", "奶奶", "爷爷", "爺爺", "外婆", "外公",
    "老板", "老闆", "上司", "同事", "邻居", "鄰居", "狗", "猫", "貓", "宠物", "寵物",
    "仓鼠", "倉鼠", "兔子", "鹦鹉", "鸚鵡",
)

_PHRASES: Final[tuple[_Phrases, ...]] = tuple(
    _Phrases(re.compile(pattern, re.IGNORECASE), cased) for pattern, cased in (
        (_before(("mi", "mis", "nuestro", "nuestra"), _ES_NOUNS), True),
        (_before_any(("meu", "minha", "meus", "minhas", "nosso", "nossa")), True),
        (_before(("mon", "ma", "notre"), _FR_NOUNS), True),
        (_before(("mio", "mia", "nostro", "nostra"), _IT_NOUNS), True),
        (_before_any(("mein", "meine", "meinem", "meinen", "meiner", "meines", "unser",
                      "unsere", "unserem", "unseren", "unserer", "unseres")), True),
        (_before_any(("мой", "моя", "моё", "мое", "мои", "моего", "моему", "моим", "моём",
                      "моем", "моей", "мою", "моих", "наш", "наша", "наше", "наши", "нашего",
                      "нашему", "нашим", "нашем", "нашей", "нашу", "наших")), True),
        (_whole(_TR_NOUNS), True),
        (_after(_VI_NOUNS, ("tôi", "của tôi", "mình", "của mình")), True),
        (_after(_ID_NOUNS, ("saya", "aku", "kami")), True),
        (_whole(noun + "ku" for noun in _ID_NOUNS if " " not in noun), True),
        (_before(("meri", "mera", "mere", "hamari", "hamara"), _HINGLISH_NOUNS), True),
        (_before(("मेरी", "मेरा", "मेरे", "हमारी", "हमारा", "हमारे"), _HI_NOUNS), False),
        (_whole(_AR_NOUNS), False),
        (r"(?<!\w)ال[^\W\d_]+\s+(?:الخاص|الخاصة)\s+بي(?!\w)", False),
        (alternation(_TH_NOUNS, "") + r"(?:ของ)?\s*" + alternation(
            ("ฉัน", "ผม", "เรา", "ดิฉัน", "หนู"), ""), False),
        (r"(?<!\w)(?:제|내|우리|저의|나의)\s*" + alternation(_KO_NOUNS, ""), False),
        (r"(?:私|僕|俺|わたし|あたし|うち)の" + alternation(_JA_NOUNS, "")
         + "|" + alternation(_JA_NOUNS, "") + "(?=の)", False),
        (r"(?:我的|我们的|我|我们)" + alternation(_ZH_NOUNS, ""), False),
    )
)
_ENDS: Final = r".!?\n\u3002\uff01\uff1f\u061f\u0964"
_SENTENCE: Final = re.compile(f"[^{_ENDS}]+[{_ENDS}]?")
_QUESTION: Final = frozenset("?\uff1f\u061f")
_GAP: Final = re.compile(r"[\s,\u3001\uff0c:\uff1a]*(?:\u306e)?[\s,\u3001\uff0c]*")
_WORD: Final = re.compile(r"[^\W\d_][\w'\u2019-]*")
_VI_NAME_BEFORE_POSSESSIVE: Final = re.compile(
    rf"{_whole(_VI_NOUNS)}\s+(?P<name>{_WORD.pattern}(?:\s+{_WORD.pattern}){{0,3}})"
    rf"\s+{_whole(('của tôi', 'của mình'))}", re.IGNORECASE,
)


def _capitalized(word: str) -> bool:
    """Whether a word begins uppercase and has a later lowercase letter."""
    return word[0].isupper() and any(char.islower() for char in word[1:])
_VI_NAMED: Final = re.compile(
    _whole(_VI_NOUNS)
    + r"\s+(?P<name>(?!của\b|tôi\b|mình\b)[^\W\d_][\w'\u2019-]*"
    r"(?:\s+(?!của\b|tôi\b|mình\b)[^\W\d_][\w'\u2019-]*){0,3})"
    r"\s+(?:của\s+)?(?:tôi|mình)(?!\w)", re.IGNORECASE,
)
_KO_NAMED: Final = re.compile(
    _whole(_KO_NOUNS) + r"\s+[가-힣]{1,8}(?:은|는|이|가)\s+"
    r"(?:저|나|우리|저희)를\s+돌봐줘요(?=[.!。！]?$)",
)
_VI_NAME_INTRO: Final = re.compile(
    _after(_VI_NOUNS, ("tôi", "của tôi", "mình", "của mình")) + r"\s+tên\s+là",
    re.IGNORECASE,
)
_KO_OMITTED: Final = re.compile(
    _whole(_KO_NOUNS) + r"\s+(?P<word>[가-힣]+?)(?:은|는|이|가)(?!\w)",
)
_KO_TALK_WORDS: Final = frozenset({"이야기", "얘기", "이름", "말", "소식"})
_KO_QUESTION_END: Final = re.compile(r"(?:뭐|무엇|누구|어디|언제|왜)(?:이에요|예요)?[.!。！]?$")
_KO_NAME_INTRO: Final = re.compile(
    r"(?<!\w)(?:제|내|우리|저의|나의)\s*" + alternation((*_KO_NOUNS, "개"), "")
    + r"\s*이름은\s*(?!(?:뭐|무엇|누구)(?:이에요|예요)(?=[.!。！]?$))"
    r"[가-힣]{1,8}(?:이에요|예요)(?=[.!。！]?$)",
)


def _named(sentence: str, end: int, cased: bool) -> bool:
    """Whether a name follows a phrase ending at ``end``: capitalized when ``cased``."""
    gap = _GAP.match(sentence, end)
    word = _WORD.match(sentence, gap.end() if gap is not None else end)
    if word is None:
        return False
    text = word.group()
    return not cased or _capitalized(text)


def names_someone(text: str) -> bool:
    """Tell whether a declarative sentence names a relative or companion (module docstring).

    Args:
        text (str): One fresh, readable, non-assistant stretch of a source.

    Returns:
        bool: True when a sentence not ending in a question mark holds a listed
            possessive phrase followed by a name, or, outside double quotes,
            nonassertion words and Korean conditional markers, a Vietnamese name
            before its possessive, a Vietnamese or Korean name introduction, a
            Korean relation, Hangul name and particle before
            저를/나를/우리를/저희를 돌봐줘요, or a listed Korean relation before a
            Hangul word with 은/는/이/가 that is no talk or name noun, in a
            sentence not ending in a question word.
    """
    for match in _SENTENCE.finditer(text):
        sentence = match.group()
        if sentence[-1] in _QUESTION:
            continue
        for phrases in _PHRASES:
            for found in phrases.pattern.finditer(sentence):
                if _named(sentence, found.end(), phrases.cased):
                    return True
        if (any(c in sentence for c in '"“”') or _NONASSERTION.search(sentence)
                or _KO_NONASSERTION.search(sentence)):
            continue
        # These recorded orders put the possessive after the name (Vietnamese)
        # or leave it implicit in a named relation's care assertion (Korean).
        for found in _VI_NAME_BEFORE_POSSESSIVE.finditer(sentence):
            if all(_capitalized(word) for word in found["name"].split()):
                return True
        for found in _VI_NAMED.finditer(sentence):
            if all(word[0].isupper() and any(char.islower() for char in word[1:])
                   for word in found['name'].split()):
                return True
        if _KO_NAMED.search(sentence):
            return True
        for found in _KO_OMITTED.finditer(sentence):
            if found["word"] not in _KO_TALK_WORDS and not _KO_QUESTION_END.search(sentence):
                return True
        if (any(_named(sentence, found.end(), True)
                for found in _VI_NAME_INTRO.finditer(sentence))
                or _KO_NAME_INTRO.search(sentence)):
            return True
    return False


_FINITE: Final = (
    r"am|is|are|was|were|have|has|had|live|lives|lived|work|works|worked|"
    r"moved|relocated|married|adopted|own|owns|owned|keep|keeps|kept|"
    r"study|studies|studied|attend|attends|attended|prefer|prefers|preferred|"
    r"like|likes|liked|love|loves|loved|hate|hates|hated|enjoy|enjoys|enjoyed|"
    r"volunteer|volunteers|volunteered|started|speak|speaks|spoke|"
    r"teach|teaches|taught|commute|commutes|commuted|play|plays|played|plan|plans|planned|"
    r"watch|watches|watched|read|reads|write|writes|wrote|run|runs|ran|swim|swims|swam|"
    r"cook|cooks|cooked|eat|eats|ate|drink|drinks|drank|drive|drives|drove|"
    r"visit|visits|visited|collect|collects|collected|"
    r"buy|buys|bought|join|joins|joined|switch|switches|switched|grow|grows|grew|"
    r"brew|brews|brewed|compost|composts|composted|sew|sews|sewed|"
    r"hike|hikes|hiked|design|designs|designed|expire|expires|expired"
)
_ADVERBS: Final = (
    r"just|recently|currently|still|always|never|really|now|also|finally|usually|often"
)
"""Closed adverbs allowed between the subject and its predicate.

``now`` carries an update (``I now have a dental appointment at Maple Street Clinic``, a
measured reply that came back empty in one of two samples under the current template);
``also``, ``finally``, ``usually`` and ``often`` are as common inside a short personal
assertion. Each only lets the closed predicate and content rules apply.
"""
_PERSONAL: Final = re.compile(
    rf"(?:i|we|(?:my|our)(?:\s+[^\W\d_]+){{1,5}})\s+"
    rf"(?:(?:{_ADVERBS})\s+)?"
    rf"(?:(?:do|does|did)\s+not\s+|(?:don|doesn|didn)['\u2019]t\s+)?"
    rf"(?:(?:{_ADVERBS})\s+)?"
    rf"(?:{_FINITE})\s+(?P<content>.+)", re.IGNORECASE,
)
_VI_PERSONAL: Final = re.compile(
    r"(?:(?:tôi|chúng tôi)\s+(?:(?:đã|đang|vừa mới)\s+)?"
    r"(?:(?:không còn|không hề|không|chưa)\s+)?"
    r"(?:sống ở|chuyển đến|làm(?: việc)?|học|bị|thích|kết hôn với|định đi|"
    r"bắt đầu|lên kế hoạch|có lịch hẹn|có (?:một )?con)"
    r"|sinh nhật(?: của)? tôi\s+là)\s+(?P<content>.+)", re.IGNORECASE,
)
_VI_RESIDENCE_QUESTION: Final = re.compile(
    r"(?:tôi|chúng tôi)\s+(?:(?:đã|đang|vừa mới)\s+)?sống ở\s+đâu(?:\s+(?:vậy|nhỉ))?",
    re.IGNORECASE,
)
_KO_PERSONAL: Final = re.compile(
    r"(?:저는|나는|우리는|저희는)\s+(?P<content>.+?)\s*"
    r"(?:에\s+살아요|일해요|(?:수업을|레슨을)\s+(?:듣고|받고)\s+있어요|"
    r"알레르기가\s+있어요|에\s+걸렸어요|"
    r"(?:를|을)\s+(?:안\s+좋아해요|좋아하지\s+(?:않아요|않습니다))|"
    r"안\s+먹어요|먹지\s+않아요|"
    r"(?:와|과)\s+결혼했어요|로\s+이사했어요|에\s+갈\s+계획이에요|"
    r"시작했어요|배우고\s+있어요|준비하고\s+있어요|금식해요|뛰었어요|키워요)",
)
_KO_BIRTHDAY: Final = re.compile(r"(?:제|내|우리)\s+생일은\s+(?P<content>.+?)(?:이에요|예요)")
_KO_RESIDENCE_QUESTION: Final = re.compile(
    r"(?:저는|나는|우리는|저희는)\s+(?:(?:지금|현재)\s+)?어디에\s+살아요",
)
_LABEL: Final = re.compile(r"(?P<name>[^\W\d_][\w'-]*(?:\s+[^\W\d_][\w'-]*){0,3}):\s*")
_LEAD_IN: Final = re.compile(
    r"(?:(?:separate|another)\s+)?(?:correction|update)\s*:\s*", re.IGNORECASE,
)
"""A closed correction/update lead-in after a speaker label (``Avery: Correction: I now
have …``). Without a label the lead-in already reads as one; after one it hid the clause."""
_NONASSERTION: Final = re.compile(
    r"\b(?:if|whether|suppose|supposing|imagine|nếu|giả sử|tưởng tượng)\b", re.IGNORECASE,
)
_KO_NONASSERTION: Final = re.compile(r"만약|[가-힣]+면(?=[\s,，:：.!]|$)")
_SMALL_TALK: Final = re.compile(
    r"(?:not\s+)?(?:(?:just|very|so|really|quite)\s+)*"
    r"(?:been\s+)?"
    r"(?:fine|good|well|okay|ok|here|there|ready|sorry|glad|happy|excited|tired|"
    r"bored|busy|wondering|asking|joking|kidding|doing|feeling|talking|chatting|"
    r"speaking|thanking|having a (?:good|great|nice|lovely|wonderful) day|"
    r"a question|no idea)\b", re.IGNORECASE,
)
_FUNCTION_WORDS: Final = frozenset(
    "a an the to of for at in on with and or but as by from about not no very just "
    "really so quite it this that there here you your me my our us i we have has had "
    "am is are was were be been being do does did help thanks thank".split()
)
_VI_FUNCTION_WORDS: Final = frozenset(
    "ở tại với vào một những các của là và có đã đang tôi chúng này đó đây bạn gì cả".split(),
)
_KO_QUESTION_WORDS: Final = frozenset(
    "뭐 뭘 뭐를 무엇 무엇을 누구 어디 어디에 어디서 언제 왜 어떻게 어떤 무슨".split(),
)

# These are sentence shapes, not a language detector. Each requires an
# explicit first-person subject, a listed predicate and content; its paired
# exclusions reject questions, conditionals and transient conversational states.
# Keep them separate from the existing English, Vietnamese and Korean rules so
# adding a language cannot change those rules' decisions.
_FIRST_PERSON: Final = tuple(
    (re.compile(pattern, re.IGNORECASE), re.compile(excluded, re.IGNORECASE))
    for pattern, excluded in (
        (r"(?:我|我们|我們)(?:的[\w]{1,24})?\s*"
         r"(?:是|有|叫|住在|居住在|在|喜欢|喜歡|不喜欢|不喜歡|爱|愛|养了|養了|"
         r"学习|學習|会说|會說|搬到了|很)(?P<content>.+)",
         r"[吗嗎么麼呢嘛]|是否|如果|假如|要是|假设|假設|想象|想像|请|請|"
         r"什么|什麼|哪|谁|誰|何|多少|几|幾|怎么|怎麼|为什么|為什麼|"
         r"问题|問題|开玩笑|開玩笑|高兴|高興|开心|開心|抱歉|对不起|對不起|"
         r"疲[倦劳勞]|累|很好|忙|无聊|無聊|饿|餓|渴|这里|這裡|那里|那裡"),
        (r"(?:(?:私|僕|俺|私たち|僕たち|我々)(?:の[\w]{1,24})?(?:は|が)|"
         r"[^。！？.!?\n]{1,48}は(?:私|僕|俺|私たち|僕たち|我々)の)\s*"
         r"(?P<content>.+?)(?:に住んでいます|で働いています|を習っています|"
         r"を勉強しています|が好きです|が嫌いです|を飼っています|を持っています|"
         r"に引っ越しました|です|だ)",
         r"もし|なら|たら|れば|仮に|例えば|想像|ください|でしょうか|ですか|ますか|"
         r"どこ|だれ|誰|何|なぜ|どうして|いつ|どの|どちら|いくつ|いくら|"
         r"質問|冗談|元気|嬉しい|うれしい|疲れ|ごめん|すみません|大丈夫|ここ|そこ|忙しい|退屈"),
        (r"(?:(?:저는|나는|우리는|저희는)|"
         r"(?:제|내|저의|나의|우리)\s*[가-힣A-Za-z]{1,16}(?:은|는|이|가))\s*"
         r"(?P<content>.+?)(?:읽어요|좋아해요|싫어해요|키워요|살아요|일해요|이에요|예요)",
         r"만약|상상|질문|농담|안녕하세요|고마워|미안|피곤|행복|기뻐|여기|거기|괜찮|바빠|"
         r"(?:뭐|뭘|무엇|누구|어디|언제|왜|어떻게|어떤|무슨)"),
        (r"(?:yo|nosotros|nosotras|(?:mi|mis|nuestro|nuestra|nuestros|nuestras)"
         r"(?:\s+[^\W\d_]+){1,5}?)\s+(?:no\s+)?"
         r"(?:soy|somos|tengo|tenemos|vivo|vivimos|trabajo|trabajamos|estudio|estudiamos|"
         r"prefiero|preferimos|hablo|hablamos|tiene|tienen|vive|viven|es|son)\s+(?P<content>.+)",
         r"\b(?:si|supongamos|imagina|imaginemos|qué|quién|dónde|cuándo|por qué|cómo|cuál|cuánto|"
         r"pregunta|broma|bien|feliz|felices|cansado|cansada|listo|lista|perdón)\b"),
        (r"(?:(?:je|nous|(?:mon|ma|mes|notre|nos)(?:\s+[^\W\d_]+){1,5}?)\s+"
         r"(?:suis|sommes|vis|vivons|habite|habitons|travaille|travaillons|étudie|étudions|"
         r"préfère|préférons|parle|parlons|ai|avons|est|sont|a|ont)|"
         r"j['’](?:ai|habite|aime|étudie))\s+(?P<content>.+)",
         r"\b(?:si|supposons|imagine|imaginons|quoi|qui|où|quand|pourquoi|comment|quel|quelle|"
         r"question|blague|bien|heureux|heureuse|fatigué|fatiguée|prêt|prête|désolé|désolée)\b"),
        (r"(?:ich|wir|(?:mein|meine|meinen|meinem|meiner|meines|unser|unsere)"
         r"(?:\s+[^\W\d_]+){1,5}?)\s+"
         r"(?:bin|sind|habe|haben|wohne|wohnen|lebe|leben|arbeite|arbeiten|lerne|lernen|"
         r"mag|mögen|spreche|sprechen|ist|hat|wohnt|lebt)\s+(?P<content>.+)",
         r"\b(?:wenn|falls|angenommen|stell|stellen|was|wer|wo|wann|warum|wie|welche|welcher|welches|"
         r"Frage|Scherz|gut|glücklich|müde|bereit|froh|entschuldigung)\b"),
        (r"(?:я|мы|(?:мой|моя|моё|мое|мои|наш|наша|наше|наши)"
         r"(?:\s+[^\W\d_]+){1,5}?)\s+(?:не\s+)?"
         r"(?:живу|живём|живем|живёт|живет|живут|работаю|работаем|работает|работают|"
         r"учусь|учимся|люблю|любим|любит|любят|говорю|говорим|переехал|переехала|"
         r"переехали|имею|имеем)\s+(?P<content>.+)",
         r"\b(?:если|предположим|представь|представим|что|кто|где|куда|когда|почему|как|сколько|какой|"
         r"вопрос|шутка|хорошо|рад|рада|рады|устал|устала|готов|готова)\b"),
        (r"(?:eu|nós|(?:meu|minha|meus|minhas|nosso|nossa|nossos|nossas)"
         r"(?:\s+[^\W\d_]+){1,5}?)\s+(?:não\s+)?"
         r"(?:sou|somos|tenho|temos|moro|moramos|vivo|vivemos|trabalho|trabalhamos|"
         r"estudo|estudamos|prefiro|preferimos|falo|falamos|é|são|tem|têm)\s+(?P<content>.+)",
         r"\b(?:se|suponha|imagine|quem|onde|quando|por que|como|qual|quanto|pergunta|brincadeira|"
         r"bem|feliz|felizes|cansado|cansada|pronto|pronta|desculpa)\b"),
        (r"(?:io|noi|(?:(?:il|la|i|le)\s+)?(?:mio|mia|miei|mie|nostro|nostra|nostri|nostre)"
         r"(?:\s+[^\W\d_]+){1,5}?)\s+(?:non\s+)?"
         r"(?:sono|siamo|ho|abbiamo|vivo|viviamo|abito|abitiamo|lavoro|lavoriamo|"
         r"studio|studiamo|preferisco|preferiamo|parlo|parliamo|è|ha|hanno)\s+(?P<content>.+)",
         r"\b(?:se|supponiamo|immagina|immaginiamo|cosa|chi|dove|quando|perché|come|quale|quanto|"
         r"domanda|scherzo|bene|felice|felici|stanco|stanca|pronto|pronta|scusa)\b"),
    )
)
_ASSERTION_QUOTES: Final = frozenset('「」『』«»„‟‘’')
_QUOTED_ASSERTIONS: Final = re.compile(
    r'"[^"]*(?:"|$)|“[^”]*(?:”|$)|「[^」]*(?:」|$)|『[^』]*(?:』|$)|'
    r'«[^»]*(?:»|$)|„[^“]*(?:“|$)|‘[^’]*(?:’|$)',
)
_ASSERTION_FUNCTION_WORDS: Final = frozenset(
    "en un una el la los las de del a al à le les des du et à avec pour ce ça y "
    "in im am an mit ein eine der die das und zu von es um com em no na os as o "
    "e di con per lo gli che в на у и это из к с во не мне нам "
    "을 를 에 은 는 이 가 を に は が の".split(),
)


def _first_person_assertion(sentence: str) -> bool:
    """Match the closed first-person predicates with nontrivial content."""
    # Apostrophes inside French words are not quotation delimiters.
    if any(char in sentence for char in _ASSERTION_QUOTES - {"’"}):
        return False
    for pattern, excluded in _FIRST_PERSON:
        clause = pattern.fullmatch(sentence)
        if clause is None or excluded.search(sentence):
            continue
        words = re.findall(r"[^\W\d_]+", clause["content"].casefold())
        if any(word not in _ASSERTION_FUNCTION_WORDS for word in words):
            return True
    return False


def asserts_personal_fact(text: str) -> bool:
    """Recognize closed assertion shapes for the empty-reply resend.

    A sentence starts with I/we, my/our plus one to five subject words, or the
    same name as its leading speaker label; its predicate is in ``_FINITE``, also
    negated (``I don't like``, ``I do not own``, ``my sister doesn't like``, with a
    listed verb), with a listed adverb (`_ADVERBS`) before or after the negation
    (``I don't really like``, ``I now live in``), and its complement has
    a letter word outside ``_FUNCTION_WORDS``. Contractions
    I'm/we're/I've/we've expand before matching. Questions, quoted/conditional text and the
    closed transient/small-talk complements never qualify. Vietnamese adds the
    closed ``_VI_PERSONAL`` first-person predicates and birthday subject; Korean
    adds ``_KO_PERSONAL``'s first-person predicates and ``_KO_BIRTHDAY``. These
    match complete sentences, with a letter word as Vietnamese content and a
    letter or digit as Korean content (a birthday can contain only a date).
    Vietnamese allows the closed negative prefixes không còn/không hề/không/chưa
    after its tense marker. Korean adds the closed negative preference and eating
    endings in ``_KO_PERSONAL``; an exact ``_KO_QUESTION_WORDS`` content word is
    excluded, including an interrogative written without a question mark.
    Closed residence questions are excluded even without a question mark;
    indefinite or negative location assertions keep their existing eligibility.
    The lexicon adds explicit first-person pronouns and possessives with closed predicates
    in Chinese, Japanese, Korean, Spanish, French, German, Russian, Portuguese and
    Italian; each language excludes its listed interrogative, conditional and
    conversational words. The extraction stage bounds short unquoted assertions
    with statement tokens, including individual Han, Kana and Hangul characters.
    This conservative signal buys a sample; admission and the judge still
    decide what is a fact.

    Args:
        text (str): One fresh, readable, non-assistant stretch, never joined
            across a withheld range.

    Returns:
        bool: Whether one complete clause has the closed assertion shape.
    """
    return next(assertion_spans(text), None) is not None


def assertion_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield the character range of each sentence of ``text`` with a closed assertion shape.

    The ranges are those of `asserts_personal_fact`'s sentences, so a caller can tell
    which sentence of a stretch a reply's quotes leave uncovered.

    Args:
        text (str): One fresh, readable, non-assistant stretch.

    Yields:
        tuple[int, int]: The start and end of each qualifying sentence in ``text``.
    """
    quoted = [(match.start(), match.end()) for match in _QUOTED_ASSERTIONS.finditer(text)]
    for match in _SENTENCE.finditer(text):
        in_quote = any(start < match.end() and end > match.start() for start, end in quoted)
        if _asserts(match.group().strip(), quoted=in_quote):
            yield match.start(), match.end()


def _asserts(sentence: str, *, quoted: bool = False) -> bool:
    """Tell whether one stripped sentence has a closed assertion shape."""
    if (not sentence or sentence[-1] in _QUESTION
            or any(c in sentence for c in '"“”') or _NONASSERTION.search(sentence)
            or _KO_NONASSERTION.search(sentence)):
        return False
    label = _LABEL.match(sentence)
    if label is not None:
        sentence = sentence[label.end():]
        lead_in = _LEAD_IN.match(sentence)
        if lead_in is not None:
            sentence = sentence[lead_in.end():]
        # A labelled speaker can also use their own name as the subject.
        sentence = re.sub(rf"^{re.escape(label['name'])}\s+", "I ", sentence)
    sentence = re.sub(r"(?i)^i['’]m\b", "I am", sentence)
    sentence = re.sub(r"(?i)^we['’]re\b", "We are", sentence)
    sentence = re.sub(r"(?i)^(i|we)['’]ve\b", r"\1 have", sentence)
    sentence = sentence.rstrip(".!。！")
    if (_VI_RESIDENCE_QUESTION.fullmatch(sentence)
            or _KO_RESIDENCE_QUESTION.fullmatch(sentence)):
        return False
    vietnamese = _VI_PERSONAL.fullmatch(sentence)
    if vietnamese is not None:
        words = re.findall(r"[^\W\d_]+", vietnamese['content'].casefold())
        if any(word not in _VI_FUNCTION_WORDS for word in words):
            return True
    korean = _KO_PERSONAL.fullmatch(sentence) or _KO_BIRTHDAY.fullmatch(sentence)
    if (korean is not None
            and not _KO_QUESTION_WORDS.intersection(korean['content'].split())
            and any(char.isalpha() or char.isdigit() for char in korean['content'])):
        return True
    clause = _PERSONAL.fullmatch(sentence)
    if clause is None or _SMALL_TALK.match(clause['content']):
        return not quoted and _first_person_assertion(sentence)
    words = re.findall(r"[^\W\d_]+", clause['content'].casefold())
    return any(word not in _FUNCTION_WORDS for word in words)
