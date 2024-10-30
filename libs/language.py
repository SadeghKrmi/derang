# definition of language specific characters, constants and diacritics
"""
character references using https://character-table.netlify.app/persian/
Saria definition using https://hmotahari.com/blog/%DB%8C-%DA%A9%D9%88%DA%86%DA%A9-%D9%87%D9%85%D8%B2%D9%87-%D9%86%DB%8C%D8%B3%D8%AA/
"""

from enum import Enum


class PersianDiacriticsEndOfWord(Enum):
    """define diacritics at the end of each word in Persian language that an alphabet can take."""
    NO_DIACRITICS = ""
    ZIR = chr(0x0650)
    # SARIA = chr(0x06C0)

    @classmethod
    def chars(cls):
        return {
            cls.ZIR,
            # cls.SARIA,
        }
    
    @classmethod
    def valid(cls):
        return {
            cls.ZIR,
            # cls.SARIA,
        }


    @classmethod
    def diacritic_to_label(cls):
        return {member.value: name for (name, member) in cls.__members__.items()}
    


# define letters, numbers and separators, reference from https://character-table.netlify.app/persian/
HARF_SEPARATOR = chr(0x200C)    # تعریف نیم فاصله
WORD_SEPARATOR = chr(0x20)
WHITESPACE = chr(0x20)

PERSIAN_LETTERS = frozenset(
    chr(x) for x in (
        0x621, 0x622, 0x623, 0x624, 0x626, 0x627, 0x628, 0x629, 0x62A, 0x62B, 0x62C, 0x62D, 0x62E, 0x62F,
        0x630, 0x631, 0x632, 0x633, 0x634, 0x635, 0x636, 0x637, 0x638, 0x639, 0x63A, 0x641, 0x642, 0x644,
        0x645, 0x646, 0x647, 0x648, 0x67E, 0x686, 0x698, 0x6A9, 0x6AF, 0x6CC, 0x06C0
    ))

PERSIAN_NUMBERS = frozenset(chr(x) for x in (list(range(0x06F0, 0x6FA))))   # اعداد نوشتاری در فارسی
PUNCTUATIONS = frozenset({".", "،", ":", "؛", "-", "؟", "!", "(", ")", "[", "]", '"', "«", "»",})

DIACRITIC_CHARS = frozenset(
    chr(x) for x in (
        0x0652, 0x0651, 0x0650, 0x064F, 0x654, 0x064B, 0x064E
    ))

DIACRITIC_CHARS_END_OF_WORD = {diac.value for diac in PersianDiacriticsEndOfWord.chars()}
DIACRITIC_LABELS_END_OF_WORD = PersianDiacriticsEndOfWord.diacritic_to_label()


PERSIAN_CHARS = {HARF_SEPARATOR, WORD_SEPARATOR, *PERSIAN_LETTERS, *PUNCTUATIONS, *DIACRITIC_CHARS, *DIACRITIC_CHARS_END_OF_WORD}
PERSIAN_CHARS_WITHOUT_DIACRITIC = {HARF_SEPARATOR, WORD_SEPARATOR, *PERSIAN_LETTERS, *PUNCTUATIONS}
VALID_PERSIAN = {*PERSIAN_LETTERS, *DIACRITIC_CHARS_END_OF_WORD}


# Some often mistakes used in the language, replace with proper chars, @todo: INVALID_SEQUENCES shall be adjusted for persian
INVALID_SEQUENCES = {
    "َّ": "َّ",
    "ِّ": "ِّ",
    "ُّ": "ُّ",
    "ًّ": "ًّ",
    "ٍّ": "ٍّ",
    "ٌّ": "ٌّ",
    " ،": "،",
    " .": ".",
}
if __name__ == '__main__':
    print(f'All valid chars in Persian: \n {VALID_PERSIAN}')
