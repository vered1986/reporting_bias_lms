## Data:

Assuming you have a tokenized version of Wikipedia, e.g. at `~/corpora/wiki/en_corpus_tokenized`:

Create directory `texts` and run `grep_colors.sh` to obtain sentences for each color. 

Then, run `create_dataset.py`, and finally, split to train and dev/test:

```bash
shuf data/dataset.jsonl > temp.jsonl
head -10000 temp.jsonl > data/dev.jsonl
tail -1169590 temp.jsonl > data/train.jsonl
rm temp.jsonl
```

## Model:

### Train:

For example, fine-tuning BERT base:

```
python  -m colors.src.fine_tune \
        --output_dir output/bert_ft \
        --model_name_or_path bert-base-uncased \
        --train_data_file data/train.jsonl \
        --eval_data_file data/dev.jsonl \
        --do_eval \
        --per_gpu_eval_batch_size 8 \
        --device 0 
```

### Evaluate:

For example, evaluating a pre-trained model:

```
python  -m colors.src.evaluate \
        --model_name_or_path roberta-large \
        --eval_data_file data/dev.jsonl \
        --device 0 
```
