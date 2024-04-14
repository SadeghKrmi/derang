import argparse
from os import cpu_count, fspath
from derang.logger import _LOGGER
from derang.config import Config
import derang.utils as utils
from pathlib import Path
from functools import reduce, partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random

def process_corpus_arg_parser():
    _LOGGER.info("Running the argument parser in corpus prcessor...")
    
    parser = argparse.ArgumentParser(
        description = "argument parser for derang, preprocessing and generating the training, valiation and test sets."
    )
    
    parser.add_argument(
        "corpus", nargs="+", type=str,  
        help=
        """corpus text file containing multiple lines, each line sentences in Persian with proper diacritics.
            python derang/process_corpus.py --config config/alef/alef.json dataset/sentenceses1.txt dataset/sentenceses2.txt
        """
    )
    
    parser.add_argument(
        "--config", type=str, required=True, help="Model configiguration path"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./datasets", help="output directory for processed corpus."
    )
    
    parser.add_argument(
        "--workers", type=int, default=0, help="number of process for corpus processing"
    )
    
    parser.add_argument(
        "--chunk_size", type=int, default=0, help="the size of batches sent to each process"
    )
    
    parser.add_argument(
        "--validate", action="store_true", help="validate sentences to make sure includes valid diactritics."
    )
    
    parser.add_argument(
        "--n_val", type=int, default=0, help="number of sentences in validation set, for example 50."
    )
    
    parser.add_argument(
        "--n_test", type=int, default=0, help="number of sentences in test set, for example 100."
    )

    return parser


def process_corpus_data(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers, chunksize = (args.workers or (cpu_count() * 4), args.chunk_size or 720)
    _LOGGER.info(f"Reading the text from corpus ... ")
    corpus_paths = set(Path(c).resolve() for c in args.corpus)
    _LOGGER.info(" and ".join(fspath(p) for p in corpus_paths))
    text = "\n".join(corpus.read_text(encoding="utf-8".strip()) for corpus in corpus_paths)
    lines = text.splitlines()
    max_chars = args.max_chars
    _LOGGER.info(f"Maximom length of each sentence allowed: {max_chars}")
    
    # longer lines (len(line)>max_chars) are being ignored at this version.
    valid_lines = set(l for l in lines if len(l) <= max_chars)
    invalid_lines = set(lines).difference(valid_lines)

    if invalid_lines:
        _LOGGER.info(f"lines with length > {max_chars} are ignored. lines being ignored: \n" + "\n".join(ln for ln in invalid_lines))
    else:
        _LOGGER.info(f"no lines being ignored due to lenght more than {max_chars}")
    lines = valid_lines
    
    if args.validate:
        total_lines = len(lines)
        _LOGGER.info("Ignoring the sentences without valid diacritics or no diacritic ... ")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(utils.has_diacritics_chars, lines, chunksize=chunksize)
            processed_lines = [ln for ln in tqdm(iterator, total=total_lines)]            
        lines = list(filter(None, processed_lines))
        _LOGGER.info(f"number of lines without diacritics is {total_lines - len(lines)} ")
    _LOGGER.info(f"number of lines after preprocess is {len(lines)} ")

    _LOGGER.info(f"generating the training, validation and test set.")
    random.shuffle(list(lines))
    n_lines = len(lines)
    
    n_val = args.n_val or round(n_lines * 0.01)
    lines, val_lines = utils.take_out_samples(lines, n_val)

    n_test = args.n_test or round(n_lines * 0.02)
    lines, test_lines = utils.take_out_samples(lines, n_test)
    
    utils.save_lines(output_dir.joinpath("train.txt"), lines)
    utils.save_lines(output_dir.joinpath("val.txt"), val_lines)
    utils.save_lines(output_dir.joinpath("test.txt"), test_lines)
    return
    
def main():
    args = process_corpus_arg_parser().parse_args()
    config = Config(args.config)
    
    # reserve 2 empty poses for the SOS and EOS tokens
    args.max_chars = config.config["max_len"] - 2
    process_corpus_data(args)


if __name__ == '__main__':
    main()