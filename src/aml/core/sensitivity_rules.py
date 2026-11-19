"""The deterministic sensitivity classifier and protection rules of D.8.1 and D.8.2.

``classify`` finds the sensitive topics a text mentions with the copied term
patterns of ``ingestion/grounding.py:161-191`` and the deterministic floors of
``:286-293``. It never establishes identity, diagnosis, trust or permission;
it only raises protection. Reception uses only the base term patterns; admission
includes the floors and applies the classifier twice (B.7.3):

- the classes of a memory are the union over its statement, its located quotes
  and the model's valid finding categories, and they set its security level;
- the mechanical floor refuses a statement that names a protected class its
  located source text does not (``unsupported_sensitive_inference``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Final, Literal, get_args

from aml.core.semantic import SecurityLevel
from aml.core.sensitivity import SensitivityClass
from aml.core.statement import canonical_entity_type

Profile = Literal["product", "bench"]
"""The trusted deployment profile; model output never selects it."""

SENSITIVITY_CLASSES: Final[tuple[SensitivityClass, ...]] = get_args(SensitivityClass)
PROTECTED_CLASSES: Final = frozenset[SensitivityClass]({
    "health", "religion", "political_affiliation", "sexual_orientation", "gender_identity",
})
"""Classes the mechanical floor requires the sources to state (B.7.3)."""

_PATTERNS: Final[tuple[tuple[SensitivityClass, str, Literal["base", "floor"]], ...]] = (
    ("health",
     r"\b(?:diagnos\w*|cancer|diabetes|medical|hiv|aids|therapy|therapist|depression|autism)\b"
     "|健康|医疗|抑郁|糖尿病", "base"),
    ("religion",
     r"\b(?:religion|religious|christian|muslim|jewish|hindu|buddhist|church|mosque"
     r"|synagogue)\b|宗教|基督|佛教|穆斯林", "base"),
    ("political_affiliation",
     r"\b(?:political|democrat|republican|socialist|communist|voted for|voting for)\b"
     "|政治|党派", "base"),
    ("sexual_orientation",
     r"\b(?:gay|lesbian|bisexual|asexual|homosexual|sexual orientation|lgbtq?)\b"
     "|性取向|同性恋|双性恋", "base"),
    ("gender_identity",
     r"\b(?:transgender|nonbinary|gender identity)\b|性别认同", "base"),
    ("financial",
     r"\b(?:bank account|credit card|social security number|tax identifier|salary)\b"
     "|银行账号|信用卡|工资", "base"),
    ("credentials",
     r"\b(?:password|api[ _-]?key|session[ _-]?token|verification code|private key)\b"
     "|密码|密钥|验证码", "base"),
    # The deterministic floors for expressions models have omitted.
    ("health", r"\b(?:mental\s+health|asthma)\b|哮喘|心理健康", "floor"),
    ("gender_identity",
     r"\b(?:trans\s+(?:woman|women|man|men|person|people)|non[- ]binary)\b"
     "|跨性别|非二元性别", "floor"),
)
_TERMS: Final[tuple[tuple[SensitivityClass, re.Pattern[str], Literal["base", "floor"]], ...]] = (
    tuple((category, re.compile(pattern, re.IGNORECASE), rule)
          for category, pattern, rule in _PATTERNS)
)


def _words(*alternatives: str) -> str:
    """Whole words of space-delimited languages; the forms are written in full."""
    return r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)"


def _runs(*alternatives: str) -> str:
    """Terms of scripts that attach particles or write no spaces: found anywhere."""
    return "|".join(alternatives)


_MULTILINGUAL_PATTERNS: Final[
    tuple[tuple[SensitivityClass, str, Literal["base", "floor"]], ...]
] = (
    # The same topics as the English and Chinese terms above, in the other languages
    # whose sources extraction states in English statements (ML-1). Latin, Cyrillic and
    # Vietnamese forms are written in full so that no English word gains a class.
    ("health", "|".join((
        _words(
            r"cáncer", r"cancro", r"câncer", r"depresión", r"depressão", r"dépression",
            r"depressione", r"terapia", r"terapeuta", r"thérapie", r"thérapeute",
            r"therapie", r"therapeut(?:in|en|innen)?", r"autismo", r"autisme", r"autismus",
            r"diabète", r"diabete", r"diagnóstico", r"diagnosticad[oa]", r"diagnostiqu[ée]e?",
            r"krebs(?:erkrankung)?", r"tumore", r"sida", r"vih",
            r"рак(?:а|ом|е|у)?", r"диабет\w*", r"депресси\w*", r"терапи\w*",
            r"(?:психо)?терапевт\w*", r"аутизм\w*", r"вич", r"спид\w*", r"диагноз\w*",
            r"kanser\w*", r"diyabet\w*", r"depresyon\w*", r"terapist\w*", r"otizm\w*",
            r"teşhis\w*", r"ung thư", r"tiểu đường", r"trầm cảm", r"trị liệu", r"tự kỷ",
            r"chẩn đoán", r"kanker", r"depresi", r"terapis", r"autisme",
        ),
        _runs(
            r"سرطان", r"السكري", r"اكتئاب", r"معالج نفسي", r"معالجة نفسية", r"علاج نفسي",
            r"توحد", r"الإيدز", r"تشخيص", r"कैंसर", r"मधुमेह", r"डायबिटीज", r"अवसाद",
            r"डिप्रेशन", r"थेरेपी", r"थेरेपिस्ट", r"ऑटिज्म", r"ऑटिज़्म", r"एड्स", r"एचआईवी",
            r"निदान", r"มะเร็ง", r"เบาหวาน", r"ซึมเศร้า", r"บำบัด", r"ออทิสติก", r"เอชไอวี",
            r"เอดส์", r"วินิจฉัย", r"암(?:에|이|을|은|으로| 진단| 환자| 투병)", r"당뇨",
            r"우울증", r"치료사", r"심리 치료", r"상담 치료", r"자폐", r"에이즈", r"진단",
            r"がん(?:患者|治療|診断|検診|です|でした|になり|と診断|を患)",
            r"ガン(?:患者|治療|です)", r"癌", r"糖尿病", r"うつ病", r"鬱病", r"セラピー",
            r"セラピスト",
            r"心理療法", r"自閉症", r"エイズ", r"診断", r"治疗师", r"心理治疗", r"心理咨询",
            r"自闭症", r"艾滋", r"诊断",
        ),
    )), "base"),
    ("religion", "|".join((
        _words(
            r"religión", r"religios[oa]s?", r"cristian[oa]s?", r"musulm[aá]n(?:a|es|as)?",
            r"jud[ií][oa]s?", r"hindú", r"budistas?", r"iglesia", r"mezquita", r"sinagoga",
            r"religião", r"cristã", r"cristão", r"muçulman[oa]s?", r"judeu", r"judia",
            r"igreja", r"mesquita", r"religieux", r"religieuse", r"chrétien(?:ne)?s?",
            r"musulman(?:e|es|s)?", r"juif", r"juive", r"juifs", r"hindoue?s?",
            r"bouddhistes?", r"église", r"mosquée", r"religiös\w*", r"christ(?:in|en)",
            r"christlich\w*", r"muslim(?:e|in|innen|isch\w*)", r"juden", r"jüd\w*",
            r"buddhist(?:in|en|isch\w*)", r"kirche", r"moschee", r"synagoge", r"religione",
            r"cristian[oa]", r"musulman[oa]", r"ebre[oa]", r"induist[ae]", r"buddist[ae]",
            r"chiesa", r"moschea", r"религи\w*", r"христиан\w*", r"мусульман\w*",
            r"еврей\w*", r"иуде\w*", r"индуис\w*", r"буддис\w*", r"церк\w*", r"мечет\w*",
            r"синагог\w*", r"dini", r"dindar\w*", r"hristiyan\w*", r"müslüman\w*",
            r"yahudi\w*", r"budist\w*", r"kilise\w*", r"cami(?:ye|de|ler|i|ler\w*)?",
            r"sinagog\w*", r"tôn giáo", r"công giáo", r"kitô giáo", r"cơ đốc", r"hồi giáo",
            r"đạo hồi", r"do thái", r"phật giáo", r"đạo phật", r"nhà thờ", r"agama",
            r"kristen", r"gereja", r"masjid",
        ),
        _runs(
            r"ديانة", r"متدين", r"مسيحي", r"مسلم", r"يهودي", r"هندوسي", r"بوذي", r"كنيسة",
            r"مسجد", r"كنيس", r"धर्म", r"धार्मिक", r"ईसाई", r"मुसलमान", r"मुस्लिम", r"यहूदी",
            r"हिंदू", r"हिन्दू", r"बौद्ध", r"चर्च", r"गिरजाघर", r"मस्जिद", r"ศาสนา", r"คริสต์",
            r"มุสลิม", r"อิสลาม", r"ฮินดู", r"พุทธ", r"โบสถ์", r"มัสยิด", r"종교", r"기독교",
            r"그리스도교", r"크리스천", r"천주교", r"무슬림", r"이슬람", r"유대교", r"유대인",
            r"힌두교", r"불교", r"교회", r"성당", r"모스크", r"宗教", r"キリスト教",
            r"クリスチャン", r"イスラム", r"ムスリム", r"ユダヤ", r"ヒンドゥー", r"仏教", r"教会",
            r"モスク", r"シナゴーグ", r"伊斯兰", r"犹太", r"印度教", r"教堂", r"清真寺", r"天主教",
        ),
    )), "base"),
    ("political_affiliation", "|".join((
        _words(
            r"polític[oa]s?", r"dem[oó]cratas?", r"republican[oa]s?", r"socialistas?",
            r"comunistas?", r"vot[eéó] por", r"politique", r"démocrates?", r"républicaine?s?",
            r"socialistes?", r"communistes?", r"voté pour", r"politisch\w*",
            r"demokrat(?:in|en)?", r"republikaner\w*", r"sozialist(?:in|en|isch\w*)?",
            r"kommunist(?:in|en|isch\w*)?", r"votei em", r"politic[oa]", r"democratic[oa]",
            r"repubblican[oa]", r"votato per", r"политическ\w*", r"демократ(?:ка|ы|ов)?",
            r"республиканц\w*", r"республиканец", r"социалист\w*", r"коммунист\w*",
            r"голосовал\w* за", r"siyasi", r"cumhuriyetçi\w*", r"sosyalist\w*", r"komünist\w*",
            r"chính trị", r"đảng dân chủ", r"đảng cộng hòa", r"cộng sản", r"bỏ phiếu cho",
        ),
        _runs(
            r"سياسي", r"اشتراكي", r"شيوعي", r"الحزب الديمقراطي", r"الحزب الجمهوري",
            r"राजनीतिक", r"डेमोक्रेट", r"रिपब्लिकन", r"समाजवादी", r"साम्यवादी", r"कम्युनिस्ट",
            r"การเมือง", r"เดโมแครต", r"รีพับลิกัน", r"สังคมนิยม", r"คอมมิวนิสต์", r"정치",
            r"민주당", r"공화당", r"사회주의", r"공산주의", r"政治", r"民主党", r"共和党",
            r"社会主義", r"共産主義", r"共産党", r"社会主义", r"共产主义", r"共产党",
        ),
    )), "base"),
    ("sexual_orientation", "|".join((
        _words(
            r"lesbianas?", r"bisexuales", r"asexuales", r"homosexuales", r"orientación sexual",
            r"lesbiennes?", r"bisexuel(?:le)?s?", r"asexuel(?:le)?s?", r"homosexuel(?:le)?s?",
            r"orientation sexuelle", r"schwul\w*", r"lesbisch\w*", r"bisexuell\w*",
            r"asexuell\w*", r"homosexuell\w*", r"sexuelle orientierung", r"lésbica",
            r"bissexual", r"assexual", r"homossexual", r"orientação sexual", r"lesbica",
            r"bisessuale", r"asessuale", r"omosessuale", r"orientamento sessuale", r"гей\w*",
            r"лесбиян\w*", r"бисексуал\w*", r"асексуал\w*", r"гомосексуал\w*",
            r"сексуальн\w* ориентаци\w*", r"лгбт\w*", r"gey", r"eşcinsel\w*", r"lezbiyen\w*",
            r"biseksüel\w*", r"aseksüel\w*", r"cinsel yönelim\w*", r"đồng tính", r"song tính",
            r"vô tính", r"xu hướng tính dục", r"biseksual", r"homoseksual",
            r"orientasi seksual",
        ),
        _runs(
            r"مثلي", r"سحاقية", r"مزدوج الميول", r"لاجنسي", r"التوجه الجنسي",
            r"الميول الجنسية", r"समलैंगिक", r"लेस्बियन", r"उभयलिंगी", r"अलैंगिक",
            r"यौन अभिविन्यास", r"เกย์", r"เลสเบี้ยน", r"ไบเซ็กชวล", r"รักร่วมเพศ", r"เพศวิถี",
            r"รักเพศเดียวกัน", r"게이(?!트|머|밍|지)", r"레즈비언", r"양성애", r"무성애", r"동성애",
            r"성적 지향", r"ゲイ(?!ン|ト)", r"レズビアン", r"バイセクシュアル", r"アセクシュアル",
            r"同性愛", r"性的指向", r"男同", r"女同", r"无性恋",
        ),
    )), "base"),
    ("gender_identity", "|".join((
        _words(
            r"transg[eé]neros?", r"identidad de g[eé]nero", r"transgenres?",
            r"identité de genre", r"geschlechtsidentität", r"transgêner[oa]s?",
            r"identidade de gênero", r"identità di genere", r"трансгендер\w*",
            r"гендерн\w* идентичност\w*", r"trans birey\w*", r"transseksüel\w*",
            r"cinsiyet kimliği\w*", r"chuyển giới", r"bản dạng giới", r"identitas gender",
        ),
        _runs(
            r"متحول جنسيا", r"متحولة جنسيا", r"عابر جنسيا", r"عابرة جنسيا",
            r"الهوية الجندرية", r"الهوية الجنسية", r"ट्रांसजेंडर", r"लैंगिक पहचान",
            r"คนข้ามเพศ", r"ทรานส์เจนเดอร์", r"อัตลักษณ์ทางเพศ", r"트랜스젠더", r"성전환",
            r"성 정체성", r"トランスジェンダー", r"性自認", r"性同一性",
        ),
    )), "base"),
    ("financial", "|".join((
        _words(
            r"cuenta bancaria", r"tarjeta de crédito", r"número de (?:la )?seguridad social",
            r"salario", r"sueldo", r"compte bancaire", r"carte de crédit", r"carte bancaire",
            r"numéro de sécurité sociale", r"salaire", r"bankkonto\w*", r"kreditkarte\w*",
            r"sozialversicherungsnummer", r"steuernummer", r"gehalt", r"conta bancária",
            r"cartão de crédito", r"salário", r"conto bancario", r"carta di credito",
            r"codice fiscale", r"stipendio", r"банковск\w* сч[её]т\w*", r"кредитн\w* карт\w*",
            r"зарплат\w*", r"banka hesab\w*", r"kredi kart\w*", r"maaş\w*",
            r"tài khoản ngân hàng", r"thẻ tín dụng", r"tiền lương", r"mức lương",
            r"rekening bank", r"kartu kredit", r"gaji",
        ),
        _runs(
            r"حساب بنكي", r"حساب مصرفي", r"بطاقة ائتمان", r"بطاقة الائتمان", r"راتب",
            r"बैंक खाता", r"बैंक खाते", r"क्रेडिट कार्ड", r"वेतन", r"तनख्वाह", r"บัญชีธนาคาร",
            r"บัตรเครดิต", r"เงินเดือน", r"은행 계좌", r"계좌번호", r"신용카드", r"월급", r"급여",
            r"연봉", r"銀行口座", r"口座番号", r"クレジットカード", r"給料", r"給与", r"年収",
            r"银行卡", r"薪水", r"薪资", r"年薪",
        ),
    )), "base"),
    ("credentials", "|".join((
        _words(
            r"contraseñas?", r"clave privada", r"código de verificación", r"mot de passe",
            r"clé privée", r"code de vérification", r"passwort\w*", r"kennwort\w*",
            r"privater schlüssel", r"bestätigungscode\w*", r"verifizierungscode\w*",
            r"senhas?", r"chave privada", r"código de verificação", r"chiave privata",
            r"codice di verifica", r"парол\w*", r"закрыт\w* ключ\w*",
            r"код\w* подтверждени\w*", r"şifre\w*", r"parola(?:m|sı|nız|n)",
            r"doğrulama kodu\w*", r"özel anahtar\w*", r"mật khẩu", r"mã xác minh",
            r"mã xác nhận", r"khóa riêng", r"kata sandi", r"kode verifikasi",
        ),
        _runs(
            r"كلمة المرور", r"كلمة السر", r"رمز التحقق", r"المفتاح الخاص", r"पासवर्ड",
            r"सत्यापन कोड", r"ओटीपी", r"निजी कुंजी", r"รหัสผ่าน", r"รหัสยืนยัน", r"คีย์ส่วนตัว",
            r"비밀번호", r"패스워드", r"인증번호", r"인증 코드", r"개인 키", r"パスワード",
            r"暗証番号", r"認証コード", r"確認コード", r"秘密鍵", r"口令",
        ),
    )), "base"),
    ("health", "|".join((
        _words(
            r"salud mental", r"asma", r"saúde mental", r"santé mentale", r"asthme",
            r"psychische gesundheit", r"salute mentale", r"психическ\w* здоровь\w*",
            r"астм\w*", r"ruh sağlığı", r"astım\w*", r"sức khỏe tâm thần", r"hen suyễn",
            r"kesehatan mental",
        ),
        _runs(
            r"الصحة النفسية", r"الصحة العقلية", r"الربو", r"मानसिक स्वास्थ्य", r"अस्थमा",
            r"สุขภาพจิต", r"หอบหืด", r"정신 건강", r"천식", r"メンタルヘルス", r"心の健康", r"喘息",
        ),
    )), "floor"),
    ("gender_identity", "|".join((
        _words(
            r"no binari[oa]s?", r"mujer trans", r"hombre trans", r"non[- ]binaires?",
            r"femme trans", r"homme trans", r"nicht-?binär\w*", r"non-binär\w*",
            r"trans(?:frau|mann)", r"non binari[oa]", r"não binári[oa]", r"небинарн\w*",
            r"ikili olmayan", r"phi nhị nguyên",
        ),
        _runs(r"غير ثنائي", r"नॉन-बाइनरी", r"นอนไบนารี", r"논바이너리", r"ノンバイナリー"),
    )), "floor"),
)
_MULTILINGUAL_TERMS: Final[
    tuple[tuple[SensitivityClass, re.Pattern[str], Literal["base", "floor"]], ...]
] = tuple(
    (category, re.compile(unicodedata.normalize("NFKC", pattern), re.IGNORECASE), rule)
    for category, pattern, rule in _MULTILINGUAL_PATTERNS
)
"""The other languages' terms, matched in the NFKC form of a text (ML-1)."""


def classify(
    texts: Iterable[str], *, include_floors: bool = True,
) -> frozenset[SensitivityClass]:
    """Return the sensitive topics the texts mention (D.8.1).

    The English and Chinese terms read the text as written; the other languages'
    terms (ML-1) read its NFKC form. Extraction states facts of sources in those
    languages in English statements, so without their terms the mechanical floor
    would refuse every protected fact of such a source.

    Args:
        texts (Iterable[str]): Statements or exact source text; each is read on
            its own, so no match spans two texts.
        include_floors (bool): Include the two deterministic floors for statement
            admission. Reception passes ``False`` to preserve Evidence security.

    Returns:
        frozenset[SensitivityClass]: The topics found; empty when none.
    """
    found: set[SensitivityClass] = set()
    for text in texts:
        found.update(
            category for category, pattern, rule in _TERMS
            if (include_floors or rule == "base") and pattern.search(text)
        )
        normal = unicodedata.normalize("NFKC", text)
        found.update(
            category for category, pattern, rule in _MULTILINGUAL_TERMS
            if (include_floors or rule == "base") and category not in found
            and pattern.search(normal)
        )
    return frozenset(found)


def stored_entity_type(text: str) -> str:
    """Return the type an entity is stored with (D.8.4).

    A type that names a sensitive topic would disclose it through every catalog
    and Entity SDK read of the entity, so it is stored as ``entity``; any other
    type keeps its canonical form.

    Args:
        text (str): A type as an extraction reply declares it or the Entity SDK
            supplies it.

    Returns:
        str: ``entity`` when the classifier finds a sensitive class in the
            canonical type, else the canonical type (``canonical_entity_type``).

    Raises:
        ValueError: If the canonical type is outside 1..64 characters.
    """
    stored = canonical_entity_type(text)
    return "entity" if classify([stored]) else stored


def security_level(profile: Profile, classes: Iterable[str], *, forced: bool) -> SecurityLevel:
    """Return the security level of a new memory (B.7.3).

    Args:
        profile (Profile): ``product`` always stores ``restricted``.
        classes (Iterable[str]): The memory's sensitivity classes.
        forced (bool): An invalid finding or a bench secret forced protection.

    Returns:
        SecurityLevel: ``restricted``, or ``confidential`` for a bench memory
            with no class and nothing forced.
    """
    if profile == "product" or forced or any(True for _ in classes):
        return "restricted"
    return "confidential"
