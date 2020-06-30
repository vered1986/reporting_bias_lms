import tqdm
import scipy
import torch
import logging
import argparse

import numpy as np

from colors.src.common import JsonTextDataset
from sklearn.metrics import accuracy_score

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)

from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead


def main():
    """
    Evaluate a LM on masked prediction of colors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, help="Pretrained LM")
    ap.add_argument("--eval_data_file", default=None, type=str, help="Evaluation data file")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or number for GPU device")
    ap.add_argument("--max_length", type=int, default=50, help="")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")

    logger.info(f"Loading the model from {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    logger.info(f"Loading eval file from {args.eval_data_file}") 
    dataset = JsonTextDataset(tokenizer, args, file_path=args.eval_data_file)

    colors = [color for color in [line.strip() for line in open("color_list.txt")] if color != ""]
    colors_ids = [tokenizer.encode(f"the color is {color}", add_special_tokens=False)[-1] for color in colors]
    label2index = {label_id: index for index, label_id in enumerate(colors_ids)}

    logger.info(f"Evaluating")
    predictions, label_ranks, labels = [], [], []

    for input_ids, label, masked_index in tqdm.tqdm(dataset):
        probs, ranks = get_substitute_probabilities(model, input_ids, masked_index.item(), colors_ids, device)
        label = label2index[label.detach().item()]
        predictions.append(np.argmax(probs))
        labels.append(label)
        label_ranks.append(ranks[label])

    labels = np.array(labels)
    accuracy = accuracy_score(labels, np.array(predictions))
    average_rank = np.mean(np.array(label_ranks))
    logger.info(f"Accuracy: {accuracy:.3f}, Average Rank: {average_rank:.3f}")


def get_substitute_probabilities(model, input_ids, masked_index, word_ids, device):
    """
    Predict the masked index and return: 1) the rank of each word in word_ids; and
    2) the normalized distribution among the words in word_ids.
    :param model: the language model
    :param input_ids: the IDs of the words in the sentence
    :param masked_index: index of the mask token
    :param word_ids: the IDs of the target words
    :param device: GPU / CPU device
    :return: 1) the rank of each word in word_ids; and
    2) the normalized distribution among the words in word_ids.
    """
    input_ids = input_ids.unsqueeze(0).long().to(device)
    segments_tensors = torch.zeros_like(input_ids, device=device, dtype=input_ids.dtype)

    model.eval()

    with torch.no_grad():
        predictions = model(input_ids, token_type_ids=segments_tensors)[0]

    predictions = predictions[0, masked_index]

    # Compute the rank of each word
    sorted_pred = {word_id: rank + 1
        for rank, (word_id, _) in enumerate(
            sorted(enumerate(predictions.detach().cpu().numpy().tolist()), key=lambda item: item[1], reverse=True))}
    ranks = [sorted_pred[word_id] for word_id in word_ids]

    # Compute the normalized probability of the words
    probs = scipy.special.softmax([predictions[w].detach().cpu().item() for w in word_ids])

    return probs, ranks


if __name__ == '__main__':
    main()
