from derang.constants import DIACRITIC_CHARS
import  random
from pathlib import Path
def has_diacritics_chars(line):
    if any(c in line for c in DIACRITIC_CHARS):
        return line
    return

def take_out_samples(lines, n) -> (list, list):
    sample = random.sample(list(lines), n)
    return (list(set(lines).difference(sample)), list(sample))
    
def save_lines(filename, lines):
    Path(filename).write_text("\n".join(lines), encoding="utf-8", newline="\n")