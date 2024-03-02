from os import cpu_count
import argparse
from logger import _LOGGER
from config import Config
from pathlib import Path


def process_corpus_arg_parser():
    _LOGGER.info("Running the argument parser in corpus prcessor...")
    
    parser = argparse.ArgumentParser(
        description = "argument parser for derang, training, valiation and test sets."
    )
    
    parser.add_argument(
        "corpus", action="append", type=str,  
        help=
        """corpus text file containing multiple lines, each line sentences in Persian with proper diacritics.
            python derang/process_corpus.py --config config/alef/alef.json dataset/sentences1.txt dataset/sentences2.txt
        """
    )
    
    parser.add_argument(
        "--config", type=str, required=True, help="Model config"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="~/datasets", help="output directory for processed corpus."
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
    corp_paths = set(Path(c).resolve() for c in args.corpus)
    print(corp_paths)
    pass


def main():
    arguments = process_corpus_arg_parser().parse_args()
    config = Config(arguments.config)
    process_corpus_data(arguments)


if __name__ == '__main__':
    main()