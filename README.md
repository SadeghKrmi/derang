## derang
derang (گرفته شده از دِرنگ‌نما), diacritics creation for Persian/Farsi language

### Preprocessing the text corpus
```bash
python -m derang.process_corpus --config config/alef/alef.json

python -m derang.process_corpus --config config/alef/alef.json datasets/sentences1.txt --output-dir datasets/processed-data

python -m derang.process_corpus --config config/alef/alef.json datasets/sentences1.txt datasets/sentences2.txt --output-dir datasets/processed-data

python -m derang.process_corpus --config config/alef/alef.json datasets/sentences1.txt datasets/sentences2.txt --output-dir datasets/processed-data --n_val 1 --n_test 1
```


### text encoder
run the following command to check list of input/target symbols, and how is the input/target symbol to id mapping.
```bash
python -m derang.text_encoder
```


### training
```bash
python -m derang.alef.train --config config/alef/alef.json
```
