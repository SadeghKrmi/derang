import dataclasses
from typing import Any
from itertools import chain
from derang.logger import _LOGGER
from collections import OrderedDict
from functools import  cached_property
from derang.text_cleaner import persian_cleaner
from derang.constants import (
    HARF_SEPARATOR,
    WORD_SEPARATOR,
    PERSIAN_LETTERS,
    PUNCTUATIONS,
    DIACRITIC_LABELS,
    ALL_VALID_DIACRITIC_CHARS
)

PAD = "_"
INPUT_TOKENS = [
    PAD,
    HARF_SEPARATOR,
    WORD_SEPARATOR,
    *sorted(chain(PUNCTUATIONS, PERSIAN_LETTERS))
]

TARGET_TOKENS = [PAD, *sorted(ALL_VALID_DIACRITIC_CHARS)]


@dataclasses.dataclass
class TockenConfig:
    pad: str
    input_tokens: list[str]
    target_tokens: list[str]
    
    def __post_init__(self):
        self.input_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.input_tokens))
        self.target_id_map: OrderedDict[str, int] = OrderedDict((char, idx) for idx, char in enumerate(self.target_tokens))
    
    @classmethod
    def default(cls):
        return cls(
            pad = PAD,
            input_tokens = INPUT_TOKENS,
            target_tokens = TARGET_TOKENS
        )
        

class TextEncoder:
    """Clean text, preprocess input, convert and generate output."""
    def __init__(self, config: TockenConfig = None):
        print('******************** TEXT ENCODER **********************************')
        self.config = TockenConfig.default() if config is None else config
        self.input_symbols: list[str] = self.config.input_tokens
        self.target_symbols: list[str] = self.config.target_tokens
        
        self.input_symbol_to_id: OrderedDict[str, int] = dict(self.config.input_id_map)
        self.input_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char,id in self.input_symbol_to_id.items())
        )
        
        self.target_symbol_to_id: OrderedDict[str, int] = self.config.target_id_map
        self.target_id_to_symbol: OrderedDict[int, str] = OrderedDict(
            sorted((id, char) for char, id in self.target_symbol_to_id.items())
        )
        
        self.pad = self.config.pad
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]
        
        self.meta_input_token_ids = {
            self.input_pad_id,
        }

        self.meta_target_token_ids = {
            self.target_pad_id,
        }
        
    def input_to_sequence(self, text: str) -> list[int]:
        seq = [self.input_symbol_to_id[c] for c in text]
        return seq
    
    
    def target_to_sequence(self, diacritics: str) -> list[int]:
        seq = [self.target_symbol_to_id[c] for c in diacritics]
        return seq
    
    def sequence_to_text_input(self, sequence: list[int]):
        return [self.input_id_to_symbol[sid] for sid in sequence if (sid not in self.meta_input_token_ids)]
        
    def sequence_to_text_target(self, sequence: list[int]):
        return [self.target_id_to_symbol[sid] for sid in sequence if (sid not in self.meta_target_token_ids)]
    
    def clean(self, text):
        clean_text = persian_cleaner(text)
        return clean_text
    
    def combine_text_and_diacritics(self, input_ids: list[int], output_ids: list[int]):
        """
        combile the sequnce of input and generated sequence for diacritics
        Args:
            input_ids: a list of integers representing the input text
            output_ids: a list of integers representing the diacritics
            
        Returns:
            text: the text after merging the input text and diacritics
        """
        
        return "".join(letter + diacritic for (letter, diacritic) in zip (self.sequence_to_text_input(input_ids), self.sequence_to_text_target(output_ids)))
    
    @cached_property
    def target_id_to_label(self):
        ret = {}
        for (idx, symbol) in self.target_id_to_symbol.items():
            ret[idx] = DIACRITIC_LABELS.get(symbol, symbol)
        return OrderedDict(sorted(ret.items()))
    
    def dump_tokens(self) -> dict[Any, Any]:
        data = {
            "pad": self.config.pad,
            "input_id_map": dict(self.input_symbol_to_id),
            "target_id_map": dict(self.target_symbol_to_id)
        }
        return {"text_encoder": data}
        
        
# run directly the text_encoder.py to check input_tokens, target_tokens...
if __name__ == '__main__':
    text_encoder = TextEncoder()
    _LOGGER.info(f"list of input tokens or input symbols \n{text_encoder.input_symbols}")
    _LOGGER.info(f"list of target tokens or target symbols \n{text_encoder.target_symbols}")
    _LOGGER.info(f"mapping of input symbols to ids \n{text_encoder.input_symbol_to_id}")
    _LOGGER.info(f"mapping of target symbols to ids \n{text_encoder.target_symbol_to_id}")
    
