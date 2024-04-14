import re
from .constants import PERSIAN_CHARS, DIACRITIC_CHARS

_WHITE_SPACES = re.compile(r"\s+")


def collapse_whitespaces(text):
    return re.sub(_WHITE_SPACES, " ", text)


def basic_cleaner(text):
    text = collapse_whitespaces(text)
    return text.strip()


def persian_cleaner(text):  
    clean_text = filter(lambda characters: characters in PERSIAN_CHARS, text)
    clean_text = collapse_whitespaces("".join(list(clean_text)))
    return clean_text
    
    
def diacritics_cleaner(text):
    return text.translate(str.maketrans("", "", "".join(DIACRITIC_CHARS)))