"""General category words for the named items a memory's keywords name (Search lexical document).

A list question names a category ("Which cities has Kai visited?", "What
instruments does Blair play?") while a memory names only the member ("Kai went
to Paris yesterday.", keywords ``Paris``, ``trip``). Search's lexical and
trigger routes match words, so such a member shares nothing with the question
but the person's name. `category_terms` maps each keyword through a small,
fixed lexicon of common categories (cities, countries and their continent, US
states, continents, musical instruments, sports, pets and animals, languages
and dishes) to the category words a question would use, in both singular and
plural form. Search adds them to the memory's lexical document and keyword
forms (`aml.core.recall_corpora.MemoryDocument`); the stored memory, its
statement and its keywords are unchanged, so every retained workspace gains
them on its next Search.

The lexicon is deterministic and has no provider. A proper-name category
(place) matches a whole keyword only, so ``Minnesota Wolves`` is no animal; a
common-noun category also matches a keyword's last word, so ``rescue dog`` is a
pet and ``chicken pot pie`` a dish. Names that are also common given names or
words (``Jordan``, ``Charlotte``, ``Nice``) are left out.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

CATEGORY_LEXICON_VERSION: Final = "category-terms-v1"
"""Names the lexicon below; a change to it changes Search's lexical documents."""


def _words(*groups: str) -> tuple[str, ...]:
    return tuple(word for group in groups for word in group.split())


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace(".", " ").split())


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_CITY: Final = _words("city cities")
_COUNTRY: Final = _words("country countries")
_STATE: Final = _words("state states")
_CONTINENT: Final = _words("continent continents")
_REGIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "europe": _words("europe european"),
    "asia": _words("asia asian"),
    "africa": _words("africa african"),
    "north america": _words("america american"),
    "south america": _words("america american latin"),
    "oceania": _words("oceania"),
})
_INSTRUMENT: Final = _words("instrument instruments musical")
_SPORT: Final = _words("sport sports")
_PET: Final = _words("pet pets animal animals")
_ANIMAL: Final = _words("animal animals")
_LANGUAGE: Final = _words("language languages")
_DISH: Final = _words("food foods dish dishes")

_CITIES: Final = """
new york|new york city|nyc|los angeles|chicago|houston|phoenix|philadelphia|san antonio|san diego|
dallas|austin|san jose|san francisco|seattle|denver|boston|detroit|nashville|portland|las vegas|
memphis|louisville|baltimore|milwaukee|albuquerque|tucson|fresno|sacramento|atlanta|miami|
orlando|tampa|new orleans|cleveland|pittsburgh|cincinnati|minneapolis|st. louis|kansas city|
salt lake city|honolulu|anchorage|raleigh|indianapolis|columbus|oklahoma city|omaha|
buffalo|charleston|santa fe|santa barbara|san juan|berkeley|palo alto|boulder|
toronto|vancouver|montreal|ottawa|calgary|edmonton|quebec city|winnipeg|halifax|
london|paris|rome|berlin|madrid|barcelona|lisbon|porto|amsterdam|brussels|vienna|prague|
budapest|warsaw|krakow|munich|hamburg|frankfurt|cologne|zurich|geneva|milan|venice|
naples|athens|istanbul|dublin|edinburgh|glasgow|manchester|liverpool|oxford|cambridge|
copenhagen|stockholm|oslo|helsinki|reykjavik|moscow|saint petersburg|st petersburg|kyiv|
seville|valencia|lyon|marseille|bordeaux|salzburg|bruges|dubrovnik|
tokyo|kyoto|osaka|seoul|busan|beijing|shanghai|hong kong|shenzhen|guangzhou|taipei|singapore|
bangkok|hanoi|ho chi minh city|kuala lumpur|jakarta|bali|manila|mumbai|delhi|new delhi|
bangalore|kolkata|chennai|dubai|abu dhabi|doha|tel aviv|jerusalem|cairo|marrakech|casablanca|
cape town|johannesburg|nairobi|lagos|sydney|melbourne|brisbane|perth|auckland|wellington|
mexico city|cancun|guadalajara|havana|rio de janeiro|rio|sao paulo|buenos aires|lima|bogota|
santiago|quito|cusco|montevideo|medellin
"""

_COUNTRIES: Final[Mapping[str, str]] = MappingProxyType({
    "europe": """
england|scotland|wales|ireland|northern ireland|united kingdom|uk|great britain|britain|france|
germany|italy|spain|portugal|netherlands|holland|belgium|luxembourg|switzerland|austria|
denmark|sweden|norway|finland|iceland|poland|czech republic|czechia|slovakia|hungary|romania|
bulgaria|greece|croatia|serbia|slovenia|bosnia|montenegro|albania|north macedonia|estonia|
latvia|lithuania|ukraine|belarus|moldova|russia|malta|cyprus|monaco|andorra|liechtenstein|
san marino|vatican
""",
    "asia": """
china|japan|south korea|korea|north korea|taiwan|mongolia|india|pakistan|bangladesh|sri lanka|
nepal|bhutan|maldives|thailand|vietnam|cambodia|laos|myanmar|malaysia|indonesia|philippines|
brunei|east timor|afghanistan|iran|iraq|syria|lebanon|israel|palestine|saudi arabia|
united arab emirates|uae|qatar|kuwait|bahrain|oman|yemen|turkey|turkiye|armenia|azerbaijan|
kazakhstan|uzbekistan|turkmenistan|kyrgyzstan|tajikistan
""",
    "africa": """
egypt|morocco|algeria|tunisia|libya|sudan|south sudan|ethiopia|eritrea|somalia|kenya|uganda|
tanzania|rwanda|burundi|nigeria|ghana|senegal|mali|niger|cameroon|ivory coast|liberia|
sierra leone|guinea|congo|angola|zambia|zimbabwe|mozambique|malawi|botswana|namibia|
south africa|madagascar|mauritius|seychelles|lesotho|eswatini
""",
    "north america": """
united states|united states of america|usa|america|canada|mexico|greenland|cuba|jamaica|
haiti|dominican republic|puerto rico|bahamas|barbados|trinidad|guatemala|belize|honduras|
el salvador|nicaragua|costa rica|panama
""",
    "south america": """
brazil|argentina|chile|peru|colombia|venezuela|ecuador|bolivia|paraguay|uruguay|guyana|
suriname
""",
    "oceania": "australia|new zealand|fiji|samoa|tonga|papua new guinea|vanuatu",
})

_STATES: Final = """
alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|hawaii|
idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|
minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|
north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|
south dakota|tennessee|texas|utah|vermont|virginia|west virginia|wisconsin|wyoming
"""

_CONTINENTS: Final = "europe|asia|africa|north america|south america|australia|antarctica|oceania"

_INSTRUMENTS: Final = """
piano|guitar|bass guitar|violin|viola|cello|double bass|harp|flute|piccolo|clarinet|oboe|
bassoon|saxophone|sax|trumpet|trombone|tuba|french horn|horn|drums|drum|drum kit|percussion|
xylophone|marimba|ukulele|banjo|mandolin|harmonica|accordion|synthesizer|
bagpipes|sitar|tabla|erhu|guzheng
"""

_SPORTS: Final = """
basketball|football|soccer|baseball|softball|tennis|table tennis|ping pong|badminton|golf|
hockey|ice hockey|volleyball|rugby|cricket|lacrosse|swimming|surfing|skiing|snowboarding|
skateboarding|skating|ice skating|cycling|biking|mountain biking|running|marathon|
boxing|wrestling|martial arts|karate|judo|taekwondo|kickboxing|fencing|archery|rowing|
kayaking|canoeing|sailing|diving|scuba diving|climbing|rock climbing|bouldering|gymnastics|
triathlon|squash|handball|water polo|horse riding|equestrian|bowling
"""

_PETS: Final = """
dog|dogs|puppy|pup|cat|kitten|kitty|turtle|tortoise|hamster|gerbil|guinea pig|rabbit|bunny|
parrot|parakeet|budgie|canary|goldfish|fish|ferret|hedgehog|lizard|gecko|iguana|snake|
chinchilla|horse|pony
"""

_ANIMALS: Final = """
bird|birds|eagle|owl|duck|cow|pig|goat|sheep|deer|fox|wolf|bear|lion|tiger|elephant|
giraffe|zebra|monkey|dolphin|whale|shark|penguin|frog|butterfly|bee|squirrel|raccoon|
cockroach|spider|seal|otter|koala|kangaroo|panda|camel
"""

_LANGUAGES: Final = """
english|spanish|french|german|italian|portuguese|dutch|russian|chinese|mandarin|cantonese|
japanese|korean|arabic|hindi|bengali|urdu|turkish|greek|hebrew|polish|swedish|norwegian|
danish|finnish|vietnamese|thai|indonesian|tagalog|swahili|latin|sign language|asl
"""

_DISHES: Final = """
pizza|pasta|spaghetti|lasagna|sushi|ramen|curry|tacos|taco|burrito|burger|hamburger|sandwich|
salad|soup|stew|steak|pie|pot pie|cake|cupcake|cupcakes|muffin|muffins|cookies|cookie|brownies|
pancakes|waffles|bread|bagel|croissant|dumplings|noodles|fried rice|paella|risotto|omelette|
chili|barbecue|bbq|roast chicken|roasted chicken|fried chicken|meatballs|lasagne|
enchiladas|quesadilla|nachos|pho|kimchi|tofu|ice cream|smoothie|chocolate|cheesecake|
pudding|granola|oatmeal|casserole|kebab|falafel|hummus|tamales|empanadas|gumbo
"""


def _entries(*sources: tuple[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    table: dict[str, list[str]] = {}
    for names, words in sources:
        for name in names.replace("\n", "").split("|"):
            key = _fold(name)
            if key:
                found = table.setdefault(key, [])
                found.extend(word for word in words if word not in found)
    return {key: tuple(words) for key, words in table.items()}


_NAMED: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(_entries(
    (_CITIES, _CITY),
    *((names, (*_COUNTRY, *_REGIONS[region])) for region, names in _COUNTRIES.items()),
    (_STATES, _STATE),
    (_CONTINENTS, _CONTINENT),
))
"""Proper-name categories: the whole folded keyword names the item."""

_COMMON: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(_entries(
    (_INSTRUMENTS, _INSTRUMENT),
    (_SPORTS, _SPORT),
    (_PETS, _PET),
    (_ANIMALS, _ANIMAL),
    (_LANGUAGES, _LANGUAGE),
    (_DISHES, _DISH),
))
"""Common-noun categories: the whole keyword or its last word names the item."""


def category_terms(keywords: Iterable[str]) -> tuple[str, ...]:
    """Return the category words of the named items among a memory's keywords.

    Args:
        keywords (Iterable[str]): A memory's keywords, in stored order.

    Returns:
        tuple[str, ...]: The category words, each once, in first-found order;
            empty when no keyword names a listed item.

    Examples:
        ```python
        category_terms(["Kai", "Paris", "trip"])  # ('city', 'cities')
        category_terms(["Spain"])  # ('country', 'countries', 'europe', 'european')
        category_terms(["rescue dog"])  # ('pet', 'pets', 'animal', 'animals')
        ```
    """
    found: dict[str, None] = {}
    for keyword in keywords:
        text = _fold(keyword)
        last = text.rsplit(" ", 1)[-1]
        for words in (
            _NAMED.get(text), _COMMON.get(text), _COMMON.get(_singular(text)),
            _COMMON.get(last), _COMMON.get(_singular(last)),
        ):
            for word in words or ():
                found.setdefault(word)
    return tuple(found)


__all__ = ["CATEGORY_LEXICON_VERSION", "category_terms"]
