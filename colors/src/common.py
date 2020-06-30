import os
import json
import torch
import pickle
import logging

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class JsonTextDataset(Dataset):
    """
    Loads a masked sentence with its label.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename = f"{args.model_name_or_path}_cached_{block_size}_{filename}"
        cached_features_file = os.path.join(directory, filename)

        # Load tensors from cache
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples, self.labels, self.masked_indices = pickle.load(handle)

        # Load from file and save tensors to cache
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f_in:
                examples = [json.loads(line.strip()) for line in f_in]

            examples, labels = zip(*[(ex["sentence"].replace("[MASK]", tokenizer.mask_token), ex["color"])
                                     for ex in examples])
            labels = [tokenizer.encode(f"the color is {label}", add_special_tokens=False)[-1] for label in labels]

            examples = tokenizer.batch_encode_plus(
                examples, add_special_tokens=True, max_length=min(args.max_length, block_size))

            self.examples, self.labels, self.masked_indices = zip(*[
                (example, label, example.index(tokenizer.mask_token_id))
                for example, label in zip(examples["input_ids"], labels)
                if tokenizer.mask_token_id in example])

            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump([self.examples, self.labels, self.masked_indices], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), \
               torch.tensor(self.labels[i], dtype=torch.long).unsqueeze(0),\
               torch.tensor(self.masked_indices[i], dtype=torch.long).unsqueeze(0)
