# define the encoder class for Persian lang
import dataclasses
from itertools import chain
from logger import logger
from collections import OrderedDict



from language import (
    HARF_SEPARATOR, WORD_SEPARATOR, PERSIAN_LETTERS, 
    PUNCTUATIONS, DIACRITIC_CHARS_END_OF_WORD, DIACRITIC_CHARS, WHITESPACE
)

PAD = "_"
NUM = "#"
NO_DIACRITICS = ""
INPUT_TOKENS = [PAD, HARF_SEPARATOR, WORD_SEPARATOR, *DIACRITIC_CHARS, *sorted(chain(PUNCTUATIONS, PERSIAN_LETTERS))]
TARGET_TOKENS = [NO_DIACRITICS, *sorted(DIACRITIC_CHARS_END_OF_WORD)]

@dataclasses.dataclass
class TokenConfig:
    pad: str
    num: str
    white: str
    input_tokens: list[str]
    target_tokens: list[str]

    def __post_init__(self):
        self.input_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.input_tokens))
        self.target_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.target_tokens))


    @classmethod
    def default(cls):
        return cls(
            pad = PAD,
            num = NUM,
            white = WHITESPACE,
            input_tokens = INPUT_TOKENS,
            target_tokens = TARGET_TOKENS
        )

class TextEncoder:
    """
    Clean text, process, generate input and outputs
    """
    def __init__(self, token_config: TokenConfig = None):
        self.token_config = TokenConfig.default() if token_config is None else token_config

        self.input_symbols: list[str] = self.token_config.input_tokens
        self.target_symbols: list[str] = self.token_config.target_tokens

        self.input_symbol_to_id: OrderedDict[list, int] = dict(self.token_config.input_id_map)
        self.input_id_to_symbol: OrderedDict[list, int] = OrderedDict(
            sorted((id, char) for char,id in self.input_symbol_to_id.items())
        )

        self.target_symbol_to_id: OrderedDict[list, int] = dict(self.token_config.target_id_map)
        self.target_id_to_symbol: OrderedDict[list, int] = OrderedDict(
            sorted((id, char) for char,id in self.target_symbol_to_id.items())
        )

        self.pad = self.token_config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]

        self.meta_input_token_ids = {
            self.input_pad_id,
        }

        self.whitespace = self.token_config.white
        self.input_whitespace_id = self.input_symbol_to_id[self.whitespace]

    def input_to_sequence(self, text: str) -> list[int]:
        seq = [self.input_symbol_to_id[c] for c in text]
        return seq
    
    def target_to_sequence(self, diacritics: str) -> list[int]:
        seq = [self.target_symbol_to_id[c] for c in diacritics]
        return seq
    
    def sequence_to_text_input(self, sequence: list[int]):
        return [self.input_id_to_symbol[sid] for sid in sequence if (sid not in self.meta_input_token_ids)]
    
    def sequence_to_text_target(self, sequence: list[int]):
        return [self.target_id_to_symbol[sid] for sid in sequence]
    
    def input_to_word_boundary(self, text: str)-> list[int]:
        bound = [1 if char==self.whitespace else 0 for char in text]
        # shift by one to the left because 1 indicates the last char of a word
        bound = bound[1:] + [1]
        return bound

    def combine_text_and_diacritics(self, input_ids: list[int], output_ids: list[str]):
        """
        combile the sequnce of input and generated sequence for Kasreh at the end of words
        Args:
            input_ids: a list of integers representing the input text
            output_ids: a list of integers representing the Kasreh at the end of words
            
        Returns:
            text: the text after merging the input text and Kasreh at the end of words
        """
        return "".join(letter + diacritic for (letter, diacritic) in zip (self.sequence_to_text_input(input_ids), self.sequence_to_text_target(output_ids)))


    def extractor(self, text):
        """
            extract_kasreh_from_end_of_word_in_text
            This function extract diacritics from text, and return the original text, list of chars and list of related diacritics
            text: ایرانِ زیبا
            chars_list_in_text: ['ا', 'ی', 'ر', 'ا', 'ن', ' ', 'ز', 'ی', 'ب', 'ا']
            diacritics_list_in_text: ['', '', '', '', 'ِ', '', '', '', '']
        """

        if len(text.strip()) == 0:
            return text, [" "] * len(text), [""] * len(text)
        
        chars_list_in_text = []
        diacritics_in_end_of_word_text = []
        for i, char in enumerate(text):
            if char in DIACRITIC_CHARS_END_OF_WORD:
                if (i == len(text) - 1) or (text[i+1] == self.whitespace):
                    diacritics_in_end_of_word_text.pop()
                    diacritics_in_end_of_word_text.append(char)
            else:
                chars_list_in_text.append(char)
                diacritics_in_end_of_word_text.append("")

        ntext = "".join(c for c in chars_list_in_text)  # new text
        sentece_vector = [self.input_symbol_to_id[c] for c in chars_list_in_text if c in self.input_symbol_to_id]
        diacritics_vector = [self.target_symbol_to_id[c] for c in diacritics_in_end_of_word_text if c in self.target_symbol_to_id]
        boundary_vect = self.input_to_word_boundary(ntext)
        return sentece_vector, boundary_vect, diacritics_vector



# run directly the text_encoder.py to check input_tokens, target_tokens...
if __name__ == '__main__':
    text_encoder = TextEncoder()
    # logger.info(f"list of input tokens or input symbols \n{text_encoder.input_symbols}")
    # logger.info(f"list of target tokens or target symbols \n{text_encoder.target_symbols}")
    # logger.info(f"mapping of input symbols to ids \n{text_encoder.input_symbol_to_id}")
    # logger.info(f"mapping of target symbols to ids \n{text_encoder.target_symbol_to_id}")
    
    # txt = 'ساعتِ چهارِ بعد از ظهرِ و عبدالله که معمولاً در این ساعت بسیار خسته است می‌خواهد به خانه‌اش برگردد'
    txt = 'ساعتِ چهارِ بعداز ظهر'

    # txt = txt.replace('ِ', '')


    text, chars_list_in_text, diacritics_in_text = text_encoder.extractor(txt)
    print(chars_list_in_text)
    print(diacritics_in_text)


    sentence = [37, 39,  2, 55, 29, 50, 47,  2, 29, 52, 56, 39, 29, 45, 55, 39,  2, 41,
        29, 30, 50,  2, 30, 54,  2, 42, 62, 55, 61,  2, 59, 42, 62, 42,  1, 54,
        29,  2, 56, 62, 39, 29, 54, 53, 62,  2, 30, 51, 53, 37,  2, 55,  2, 41,
        49, 62, 37,  2, 30, 54,  2, 32, 53,  2, 37, 29, 42, 32,  2, 59, 54,  2,
        32, 29,  2, 53, 55, 59,  2, 56, 29, 54, 29, 62, 42,  2, 52, 62,  1, 39,
        41, 62, 37,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    
    boundary = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0]
    

    diacritics = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0]


    rtxt = text_encoder.combine_text_and_diacritics(sentence, diacritics)
    print(rtxt)