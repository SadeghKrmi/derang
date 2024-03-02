import enum

class PersianDiacritics():
    """define all existing diacritics in Persian language that an alphabet can take."""
    
    
    NO_DIACRITICS = ""
    SOKOON = chr(0x0652)
    TASHDID = chr(0x0651)
    ZIR = chr(0x0650)
    ZEBAR = chr(0x064E)
    PISH = chr(0x064F)
    HAMZEH = chr(0x654)
    TANWEEN_ZEBAR = chr(0x064B)
    TASHDID_PLUS_ZIR = chr(0x0651) + chr(0x0650)
    TASHDID_PLUS_ZEBAR = chr(0x0651) + chr(0x064E)
    TASHDID_PLUS_PISH = chr(0x0651) + chr(0x064F)
    
    
    @classmethod
    def chars(cls):
        return {
            cls.SOKOON,
            cls.TASHDID,
            cls.ZIR,
            cls.ZEBAR,
            cls.PISH,
            cls.HAMZEH,
            cls.TANWEEN_ZEBAR,
            cls.TASHDID_PLUS_ZIR,
            cls.TASHDID_PLUS_ZEBAR,
            cls.TASHDID_PLUS_PISH
        }

HARF_SEPARATOR = chr(0x200C)
WORD_SEPARATOR = chr(0x20)

# https://character-table.netlify.app/persian/
PERSIAN_LETTERS = frozenset(
    chr(x) for x in (
        0x621, 0x622, 0x623, 0x624, 0x626, 0x627, 0x628, 0x629, 0x62A, 0x62B, 0x62C, 0x62D, 0x62E, 0x62F,
        0x630, 0x631, 0x632, 0x633, 0x634, 0x635, 0x636, 0x637, 0x638, 0x639, 0x63A, 0x641, 0x642, 0x644,
        0x645, 0x646, 0x647, 0x648, 0x67E, 0x686, 0x698, 0x6A9, 0x6AF, 0x6CC
    ))

PERSIAN_NUMBERS = frozenset(chr(x) for x in (list(range(0x06F0, 0x6FA))))
PUNCTUATIONS = frozenset({".", "،", ":", "؛", "-", "؟", "!", "(", ")", "[", "]", '"', "«", "»",})

DIACRITIC_CHARS = {diac for diac in PersianDiacritics.chars()}

# Order is critical
SENTENCE_BOUNDRY_PUNCS = [".", "؟", "!", "،", "؛"]


PERSIAN_CHARS = {HARF_SEPARATOR, WORD_SEPARATOR, *PERSIAN_LETTERS, *PUNCTUATIONS, *DIACRITIC_CHARS}
