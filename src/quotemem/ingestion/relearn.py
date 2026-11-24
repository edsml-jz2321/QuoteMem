"""The ``agent`` mode's test of a user sentence that states something.

``core/relearn.py`` decides when the user told a forgotten fact again; this module supplies
what counts as a statement: a sentence that is no forget request (``forget_requests``, the
``agent`` mode's forms), names no forgetting or erasing anywhere and no negated
remembering (`FORGET_CUE`: a request in a form those forms miss, "那个我每天早上喝绿茶的事你就
忘了吧", "Like I said, forget that ...", "I don't want you to remember that ..."), and is not
only questions (``no-question-support-v1``'s test). Such a sentence never lifts a forget:
refusing a real restatement that happens to name forgetting only keeps the fact forgotten.
"""

from __future__ import annotations

import re
from typing import Final

from quotemem.ingestion.extraction_admission import only_questions
from quotemem.ingestion.forget_requests import is_request

FORGET_CUE: Final = re.compile(
    r"\b(?:forget|forgot|forgetting|erase|erasing|delete|deleting|remove|removing|wipe|"
    r"purge|unlearn)\b"
    r"|\b(?:don'?t|do not|never|no longer|stop|not)\b[^.!?。！？]{0,40}?"
    r"\b(?:remember\w*|keep|store|save|record|retain)\b"
    r"|忘|删|清除|抹掉|抹去|(?:不要|别|不用|不必|无需|不想让你|不希望你)再?(?:记|存|保存|保留)",
    re.IGNORECASE)
"""A word of forgetting or erasing anywhere in a sentence, or a negated remembering or
keeping ("I don't want you to remember ...", "以后不要再记得 ..."). Asking to remember ("I am
telling you again so that you remember it", "请记住 ...") is no cue: a user re-teaching a
forgotten fact says it that way."""


def asserts(text: str) -> bool:
    """Whether a user's sentence states something: no forget request, no forget cue
    (`FORGET_CUE`), not only questions."""
    return (bool(text.strip()) and FORGET_CUE.search(text) is None
            and not is_request(text, complete=True) and not only_questions([text]))
