import re
import os
import torch
import  random
from derang.logger import _LOGGER
from typing import Optional, Any
from pathlib import Path
from derang.constants import (
    DIACRITIC_CHARS, 
    PERSIAN_LETTERS, 
    ALL_VALID_DIACRITIC_CHARS, 
    VALID_PERSIAN
)

from more_itertools import last

def has_diacritics_chars(line):
    if any(c in line for c in DIACRITIC_CHARS):
        return line
    return

def take_out_samples(lines, n) -> (list, list):
    sample = random.sample(list(lines), n)
    return (list(set(lines).difference(sample)), list(sample))
    
def save_lines(filename, lines):
    Path(filename).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def sequence_mask(length, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def categorical_accuracy(preds, y, tag_pad_idx, device="cuda"):
    """
    Returns accuracy per batch, if 8 out of 10 is correct, you get 0.8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = torch.nonzero((y != tag_pad_idx))
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)



def extract_stack(stack, correct_reversed: bool = True):
    """
    Given stack, we extract its content to string, and check whether this string is
    available at all_possible_haraqat list: if not we raise an error. When correct_reversed
    is set, we also check the reversed order of the string, if it was not already correct.
    
    copied from https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/util.py
    """
    char_haraqat = []
    while len(stack) != 0:
        char_haraqat.append(stack.pop())
    full_haraqah = "".join(char_haraqat)
    reversed_full_haraqah = "".join(reversed(char_haraqat))
    if full_haraqah in ALL_VALID_DIACRITIC_CHARS:
        out = full_haraqah
    elif reversed_full_haraqah in ALL_VALID_DIACRITIC_CHARS and correct_reversed:
        out = reversed_full_haraqah
    else:
        raise ValueError(
            f"""The chart has the following haraqat which are not found in
        all possible haraqat: {'|'.join([ALL_VALID_DIACRITIC_CHARS[diacritic]
                                         for diacritic in full_haraqah ])}"""
        )
    return out


def extract_haraqat(text: str, correct_reversed: bool = True):
    """
    Args:
    text (str): text to be diacritized
    Returns:
    text: the text as came
    text_list: all text that are not haraqat
    haraqat_list: all haraqat_list

    copied from https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/util.py
    for a text of "ای همُّ و غَم من"
    
    
    text: ای همُّ و غَم من
    text_list: ['ا', 'ی', ' ', 'ه', 'م', ' ', 'و', ' ', 'غ', 'م', ' ', 'م', 'ن']
    haraqat_list: ['', '', '', '', 'ُّ', '', '', '', 'َ', '', '', '', '']
    """
    print('-----------------------------------------------------------------------')
    print(text)
    if len(text.strip()) == 0:
        return text, [" "] * len(text), [""] * len(text)
    
    stack = []
    haraqat_list = []
    txt_list = []
    for char in text:
        # if chart is a diacritic, then extract the stack and empty it
        if char not in DIACRITIC_CHARS:
            stack_content = extract_stack(stack, correct_reversed=correct_reversed)
            haraqat_list.append(stack_content)
            txt_list.append(char)
            stack = []
        else:
            stack.append(char)
    if len(haraqat_list) > 0:
        del haraqat_list[0]
    haraqat_list.append(extract_stack(stack))

    return text, txt_list, haraqat_list


def format_as_table(*cols: tuple[str, list[str]]) -> str:
    """Taken from lightening"""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]
    
    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width
    return summary



def format_error_rates_as_table(error_rates):
    metrics, values = [e[0] for e in error_rates], [e[1] for e in error_rates]
    cols = [
        ("".ljust(10), ["   DER", "   WER"]),
        ("With CE".ljust(10), [error_rates["DER"], error_rates["WER"]]),
        ("Without CE".ljust(10), [error_rates["DER*"], error_rates["WER*"]]),
    ]
    return format_as_table(*cols)


def calculate_error_rates(original_file_path: str, target_file_path: str) -> dict[str, float]:
    """
    Calculates der/wer error rates from paths
    """
    assert os.path.isfile(original_file_path)
    assert os.path.isfile(target_file_path)

    _wer = calculate_wer_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=True
    )

    _wer_without_case_ending = calculate_wer_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=False
    )

    _der = calculate_der_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=True
    )

    _der_without_case_ending = calculate_der_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=False
    )

    return {
        "DER": _der,
        "WER": _wer,
        "DER*": _der_without_case_ending,
        "WER*": _wer_without_case_ending,
    }



def has_persian_letters(text: str):
    for char in VALID_PERSIAN:
        if char in text:
            return True

    return False


def calculate_rate(equal: int, not_equal: int) -> float:
    """
    Given an Equal and  not equal values, this function calculate the error rate
    Args:
    equal: the number of matching values when comparing two files, this can be in
    a word level or diacritic level
    not_equal: the number of un_matching
    Returns:
    The error rate
    """
    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


def get_case_ending_indices_from_un_diacritized_txt(text):
    text = text + [" "]
    indices = []
    for i in range(len(text)):
        if text[i] not in  PERSIAN_LETTERS and text[i - 1] in PERSIAN_LETTERS:
            indices.append(i - 1)
    return indices


def combine_txt_and_haraqat(txt_list, haraqat_list):
    """
    Rejoins text with its corresponding haraqat
    Args:
    txt_list: The text that does not contain any haraqat
    haraqat_list: The haraqat that are corresponding to the text list
    """

    assert len(txt_list) == len(haraqat_list)
    out = []
    for i, char in enumerate(txt_list):
        out.append(char)
        out.append(haraqat_list[i])
    return "".join(out)


def get_word_without_case_ending(word: str):
    _, text, haraqat = extract_haraqat(word)
    indices = get_case_ending_indices_from_un_diacritized_txt(text)

    if len(indices) == 0:
        return -1

    idx = indices[-1]
    text = text[:idx] + text[idx + 1 :]
    haraqat = haraqat[:idx] + haraqat[idx + 1 :]
    output = combine_txt_and_haraqat(text, haraqat)
    return output


def calculate_der(original_content: str, predicted_content: str, case_ending: bool = True) -> float:
    """Given the original text and the predicted text,
    this function calculate the DER
    between these two files.
    Args
        original_content (str): the original text that contains the correct
        diacritization.
        predicted_content (str): the predicted text
        case_ending (str): whether to include the last character of each word darning
        the calculation
    Returns:
        DER: the  diacritic error rate (DER)
    """
    _, original_text, original_haraqat = extract_haraqat(original_content)
    _, predicted_text, predicted_haraqat = extract_haraqat(predicted_content)

    SKIP_CASE_ENDING_VALUE = -1
    if not case_ending:
        indices = get_case_ending_indices_from_un_diacritized_txt(original_text)
        for i in indices:
            original_haraqat[i] = SKIP_CASE_ENDING_VALUE

    equal = 0
    not_equal = 0
    for i, (original_char, predicted_chart) in enumerate(zip(original_haraqat, predicted_haraqat)):
        if not case_ending:
            if original_char == SKIP_CASE_ENDING_VALUE:
                continue
        if original_char == predicted_chart:
            equal += 1
        else:
            not_equal += 1

    return calculate_rate(equal, not_equal)


def calculate_der_from_path(original_path: str, predicted_path: str, case_ending: bool = True) -> float:
    """Given the original_ path and the predicted_path, this function read the content
    of both files and call calculate_der function.
    Args:
        original_path (str): the path to the original file
        predicted_path (str): the path to the generated file
        case_ending (bool): whether to calculate the last character of each word or not
    Return:
     DER: the diacritic error rate between the two files
    """
    with open(original_path, encoding="utf8") as file:
        original_content = file.read()

    with open(predicted_path, encoding="utf8") as file:
        predicted_content = file.read()

    return calculate_der(original_content, predicted_content, case_ending)



def calculate_wer(
    original_content, predicted_content, case_ending=True, include_non_arabic=False
):
    """
    Calculate Word Error Rate (WER) from two text content
    Args
        original_content (str): the original text that contains the correct
        diacritization.
        predicted_content (str): the predicted text
        case_ending (str): whether to include the last character of each word darning
        the calculation
        include_non_arabic (bool): any space separated word other than Arabic,
        such as punctuations
    Returns:
        WER : The  word error rate (WER)

    """

    original = original_content.split()
    prediction = predicted_content.split()

    # If the whole word is a diacritic, then skip it since it my cause error in the WER caclulation
    #by shifting all remaining words.
    prediction = [
        word for word in prediction if word not in ALL_VALID_DIACRITIC_CHARS.keys()
    ]
    original = [word for word in original if word not in ALL_VALID_DIACRITIC_CHARS.keys()]

    assert len(prediction) == len(original)

    equal = 0
    not_equal = 0

    for _, (original_word, predicted_word) in enumerate(zip(original, prediction)):
        if not include_non_arabic:

            if not has_persian_letters(original_word) and not has_persian_letters(
                predicted_word
            ):
                continue

        if not case_ending:
            # When not using case_ending, exclude the last char of each word from
            # calculation
            original_word = get_word_without_case_ending(original_word)
            predicted_word = get_word_without_case_ending(predicted_word)

        if original_word == predicted_word:
            equal += 1
        else:
            not_equal += 1

    return calculate_rate(equal, not_equal)


def calculate_wer_from_path(
    original_path: str,
    predicted_path: str,
    case_ending: bool = True,
    include_non_arabic: bool = False,
) -> float:
    """
    Given the input path and the out_path, this function read the content
    of both files and call calculate_der function.
    Args:
        original_path: the path to the original file
        predicted_path: the path to the predicted file
        case_ending: whether to calculate the last character of each word or not
    Return:
     DER: the diacritic error rate between the two files
    """
    with open(original_path, encoding="utf8") as file:
        original_content = file.read()

    with open(predicted_path, encoding="utf8") as file:
        predicted_content = file.read()

    return calculate_wer(
        original_content,
        predicted_content,
        case_ending=case_ending,
        include_non_arabic=include_non_arabic,
    )
    
    

CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>[0-9]+)-step=(?P<step>[0-9]+)")


def find_last_checkpoint(logs_root_directory):
    checkpoints_dir = Path(logs_root_directory)
    available_checkpoints = {
        file: CHECKPOINT_RE.match(file.stem)
        for file in (
            item for item in checkpoints_dir.rglob("*.ckpt")
            if item.is_file()
        )
    }
    available_checkpoints = {
        filename: (int(match.groupdict()["epoch"]), int(match.groupdict()["step"]))
        for (filename, match) in available_checkpoints.items()
        if match is not None
    }
    available_checkpoints = sorted(available_checkpoints.items(), key=lambda item: item[1])
    checkpoint = last(available_checkpoints, default=None)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoints were found")
    filename, (epoch, step) = checkpoint
    return os.fspath(filename.absolute()), epoch, step
