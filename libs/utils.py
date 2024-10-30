import argparse

def arguments():
    parser = argparse.ArgumentParser(
        description = """
            corpus text file containing multiple lines, each line sentences in Persian with proper diacritics.
        """
    )

    parser.add_argument(
        "--config", type=str, help="json configuration containing the config inluding, directory, etc."
    )

    return parser

