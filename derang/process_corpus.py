import argparse
from os import cpu_count, fspath
from logger import _LOGGER
from config import Config
from pathlib import Path
from functools import reduce, partial
from concurrent.futures import ProcessPoolExecutor
from constants import SENTENCE_BOUNDRY_PUNCS

def process_corpus_arg_parser():
    _LOGGER.info("Running the argument parser in corpus prcessor...")
    
    parser = argparse.ArgumentParser(
        description = "argument parser for derang, training, valiation and test sets."
    )
    
    parser.add_argument(
        "corpus", nargs="+", type=str,  
        help=
        """corpus text file containing multiple lines, each line sentences in Persian with proper diacritics.
            python derang/process_corpus.py --config config/alef/alef.json dataset/sentenceses1.txt dataset/sentenceses2.txt
        """
    )
    
    parser.add_argument(
        "--config", type=str, required=True, help="Model config"
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

    _LOGGER.info(f"lines with length > {max_chars} are ignored. lines being ignored: \n" + "\n".join(ln for ln in invalid_lines))
    
    
  

def main():
    arguments = process_corpus_arg_parser().parse_args()
    config = Config(arguments.config)
    
    # reserve 2 empty poses for the SOS and EOS tokens
    arguments.max_chars = config.config["max_len"] - 2
    process_corpus_data(arguments)


if __name__ == '__main__':
    main()