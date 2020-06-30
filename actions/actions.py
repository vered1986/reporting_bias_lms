import os
import torch
import spacy
import logging
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelWithLMHead

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


def main():
    """
    Evaluate a LM on masked prediction of colors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, help="Pretrained LM")
    ap.add_argument("--eval_data_file", default=None, type=str, help="Evaluation data file")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or number for GPU device")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")

    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    roberta_model = AutoModelWithLMHead.from_pretrained("roberta-large")
    roberta_model.to(device)

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    bert_model = AutoModelWithLMHead.from_pretrained("bert-large-uncased")
    bert_model.to(device)

    # Actual frequencies
    freq = dict([
        # Average life expectancy: 78.54 years in the US
        # Estimate per event: from the web
        ("thinking", 1433355000),  # 50,000 per day * 78.54 * 365
        ("breathing", 660489984),  # 23,040 per day
        ("blinking", 344005200),  # 12,000 per day
        ("talking", 286671000),  # 10,000 per day
        ("eating", 86001.3),  # 3 times per day
        ("sleeping", 28667.1),  # 1 time per day
        ("working", 20420.4),  # 5 times a week
        ("exercising", 8168.16),  # 2-3 times a week
        ("travelling", 154.9),  # a few times a year
        ("getting married", 1.66),  # 0-3 times per life
        ("getting divorced", 1.1),  # 0-2 times per life
        ("being born", 1.1),
        ("being named", 1.1),
        ("dying", 1.1),

        # Some of these can happen multiple times, so these are lower bounds, but even so, the chances are
        # much smaller than 1
        ("being injured", 0.1263),  # https://www.cdc.gov/nchs/fastats/injury.htm
        ("being murdered", 1.0 / 229),  # https://www.businessinsider.com/us-gun-death-murder-risk-statistics-2018-3
        ("being killed", 1.0 / 229 + 1.0 / 28),  # Including murder or accident
        ("being arrested", 0.031526),
        # 3,152.6 arrests per 100,000
        # https://ucr.fbi.gov/crime-in-the-u.s/2018/crime-in-the-u.s.-2018/topic-pages/persons-arrested
        ("being adopted", 7 / 328.2),  # Around 7 million out of 328.2 https://adoptionnetwork.com/adoption-statistics
        ("being raped", 0.183 * 0.508 + 0.014 * 0.492),  # https://www.nsvrc.org/statistics
        ("being abandoned", 0.000175),
        # https://www.encyclopedia.com/social-sciences-and-law/law/law/abandonment (7000 each year, out of 4M births)
        ("being abused", 0.5)  # https://ncadv.org/statistics
    ])

    freq = dict([(key, {"true": val}) for key, val in freq.items()])

    variations = {"thinking": ["thinking", "thinks", "think", "thought"],
                  "breathing": ["breathing", "breathe", "exhale", "inhale"],
                  "blinking": ["blinking", "blink", "blinks", "blinked"],
                  "talking": ["talking", "talk", "talked", "say", "said", "saying", "converse", "conversed",
                              "conversing"],
                  "eating": ["eat", "eating", "ate", "dine", "dining", "dined"],
                  "sleeping": ["sleeping", "sleep", "sleeps", "slept"],
                  "working": ["working", "work", "worked", "employeed"],
                  "exercising": ["exercising", "exercise", "exercised"],
                  "travelling": ["travelling", "traveling", "travel", "travelled", "traveled"],
                  "getting married": ["married"],
                  "getting divorced": ["divorced"],
                  "being born": ["born"],
                  "being named": ["named", "called"],
                  "dying": ["died", "die", "dies", "dying"],
                  "being injured": ["injured"],
                  "being arrested": ["arrested"],
                  "being murdered": ["murdered", "killed"],
                  "being killed": ["killed"],
                  "being raped": ["raped"],
                  "being abused": ["abused", "molested", "assaulted", "beat", "bullied", "oppressed", "tortured"],
                  "being shot": ["shot"],
                  "being adopted": ["adopted"],
                  "being abandoned": ["abandoned"]
                  }

    # Estimate probabilities with each knowledge source
    with open("events.jsonl", "w") as f_out:
        event_freq = {}
        entity = "person"

        # Google Ngrams: search for "person is <verb>"
        items = get_following_words(f"{entity} is", k=1500)
        items = filter_and_sort(nlp, items)

        # Fall back to "person <verb>"
        if len(items) == 0:
            items = get_following_words(entity, k=1500)
            items = filter_and_sort(nlp, items)

        event_freq["gngrams"] = items[:500]

        # BERT
        items = get_top_k_predictions(
            bert_model, bert_tokenizer, f"The {entity} is [MASK].", mask="[MASK]", k=1500)
        items = filter_and_sort(nlp, items)
        event_freq["bert"] = items[:500]

        # RoBERTa
        items = get_top_k_predictions(
            roberta_model, roberta_tokenizer, f"The {entity} is <mask>.", mask="<mask>", k=1500)
        items = filter_and_sort(nlp, items)
        event_freq["roberta"] = items[:500]

    for resource in ["gngrams", "bert", "roberta"]:
        for key in freq.keys():
            count = np.sum(
                [dict(event_freq[resource]).get(variation, 0) for variation in variations[key]])
            freq[key][resource] = count

    # Scale it
    scaled = freq.copy()

    for resource in ["roberta", "bert", "gngrams", "true"]:
        for label in scaled.keys():
            all_sum = np.sum([freq[l][resource] for l in freq.keys()])
            scaled[label][resource] = scaled[label][resource] * 1.0 / all_sum

    scaled = OrderedDict(sorted(scaled.items(), key=lambda x: x[1]["true"], reverse=True))
    draw_event_frequencies(scaled, "A person is ____")
    print_pred(bert_model, bert_tokenizer, roberta_model, roberta_tokenizer)


def get_top_k_predictions(bert_model, bert_tokenizer, text, mask="[MASK]", k=10):
    """
    Find the best word to replace the mask, using BERT or RoBERTa.
    """
    tokenized_text = bert_tokenizer.tokenize(text)
    masked_index = [i for i, token in enumerate(tokenized_text) if token == mask][0]
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).long().to(bert_model.device)

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(tokens_tensor)
        predictions = outputs[0]

    probs = predictions[0, masked_index].cpu().numpy()
    top_k_probs = np.argpartition(probs, -k)[-k:]
    top_k_probs = top_k_probs[np.argsort(probs[top_k_probs])[::-1]]
    tokens = [t.replace("Ġ", "") for t in bert_tokenizer.convert_ids_to_tokens(top_k_probs)]
    top_k_probs = list(zip(tokens, [probs[i] for i in top_k_probs]))
    return top_k_probs


def get_following_words(phrase, google_ngram_dir='~/corpora/google_ngrams'):
    """
    Searches for occurrences that start with `phrase` in Google Ngrams.
    Assumes a Google Ngram corpus under ~/corpora.
    """
    google_ngram_dir = os.path.expanduser(google_ngram_dir)
    phrase = phrase.lower()
    phrase_words = phrase.split()
    n = len(phrase_words) + 1
    prefix = phrase[:2]
    results = []
    curr_ngram_file = f'{google_ngram_dir}/googlebooks-eng-all-{n}gram-20120701-{prefix}_filtered'

    # No file for this prefix
    if not os.path.exists(curr_ngram_file):
        logging.warning(f'file {curr_ngram_file} does not exist')
        return results

    # The Google ngrams file is tab separated, containing: ngram and count.
    with open(curr_ngram_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            ngram, count = line.lower().strip().split('\t')
            words = ngram.split()

            if n == 3:
                if (words[0] > phrase_words[0] and not phrase_words[0] in words[0]) or \
                        (words[0] == phrase_words[0] and phrase_words[1] not in words[1] and \
                         words[1] > phrase_words[1]):
                    break
                elif words[0] == phrase_words[0] and words[1] == phrase_words[1]:
                    results.append((words[-1], float(count)))
                else:
                    continue

            elif n == 2:
                if words[0] > phrase_words[0] and not phrase_words[0] in words[0]:
                    break
                elif words[0] == phrase_words[0]:
                    results.append((words[-1], float(count)))
                else:
                    continue

    return results


def filter_and_sort(nlp, items):
    """
    Only keep non-auxiliary verbs, sort by descending score
    """
    docs = [nlp(f"The person {item}.") for item, count in items]
    verbs = [(item, count)
             for (item, count), doc in zip(items, docs)
             if "##" not in item and
             item not in {"be", "can", "could", "dare", "do", "have", "may", "might",
                          "must", "need", "ought", "shall", "should", "will", "would"} and
             [t.pos_ for t in doc][-2] == "VERB"]

    # Normalize and sort
    all_sum = np.sum([count for _, count in verbs])
    curr_items = list(sorted(
        [(word, count * 1.0 / all_sum) for word, count in verbs],
        key=lambda x: x[1], reverse=True))
    return curr_items


def draw_event_frequencies(freq, title):
    """
    Draw the event frequencies
    """
    font = {'size': 16}

    sns.set(color_codes=True)
    sns.set_context("notebook")
    sns.set_style("dark")
    plt.rc("text", usetex=False)
    sns.despine(left=True)

    resources = ["roberta", "bert", "gngrams", "true"]
    labels = list(freq.keys())
    items = {label: [freq[label][resource] for resource in resources] for label in labels}

    x = np.arange(len(labels))
    y = np.arange(len(resources))
    plt.figure(figsize=(24, 6))
    w = 0.5

    colors = ["purple", "green", "orange", "blue"]

    df = pd.DataFrame({
        'X': [i for i in range(len(labels)) for j in range(len(resources))],
        'Y': [j for i in range(len(labels)) for j in range(len(resources))],
        'colors': [colors[j] for i in range(len(labels)) for j in range(len(resources))],
        "bubble_size": np.array([items[label][i] for label in labels for i in range(len(resources))]) * 7500})

    plt.scatter('X', 'Y', s='bubble_size', c='colors', alpha=0.2, data=df)

    resource_names = ["RoBERTa", "BERT", "Google Ngrams", "Actual"]

    plt.title(title, fontsize=22)
    plt.xlim(-1, len(labels) - 0.5)
    plt.ylim(-.5, len(resources) - 0.25)
    plt.yticks(y, resource_names, fontsize=16)
    plt.xticks(x + w / 2, [label.replace(" ", "\n") for label in labels], rotation='vertical', fontsize=16)
    plt.savefig("a_person_is.png", format="png", bbox_inches="tight")
    plt.show()


def print_pred(bert_model, bert_tokenizer, roberta_model, roberta_tokenizer):
    """
    Print the LaTex table with the predictions
    """
    print("""\\begin{tabular}{c c c c c c}""")
    print("""\\toprule""")
    print("""& \\textbf{BERT} & \\textbf{RoBERTa} & & \\textbf{BERT} & \\textbf{RoBERTa} \\\\""")
    print("\\midrule")

    templates = ["People are [MASK] every day.", "All people are [MASK].",
                 "Most people are [MASK].", "Some people are [MASK]."]
    results = {template: {} for template in templates}

    for template in templates:
        # BERT
        items = get_top_k_predictions(bert_model, bert_tokenizer, template, mask="[MASK]", k=500)
        items = filter_and_sort(nlp, items)
        results[template]["bert"] = [(e, str(c)) for e, c in items][:10]

        # RoBERTa
        items = get_top_k_predictions(
            roberta_model, roberta_tokenizer, template.replace("[MASK]", "<mask>"), mask="<mask>", k=500)
        items = filter_and_sort(nlp, items)
        results[template]["roberta"] = [(e, str(c)) for e, c in items][:10]

    for t1, t2 in [templates[:2], templates[2:]]:
        t1_text = "\\multirow{10}{*}{" + t1.replace("[MASK]", "\\underline{~~~~~}") + "} "
        t2_text = "\\multirow{10}{*}{" + t2.replace("[MASK]", "\\underline{~~~~~}") + "} "

        for (b1, bc1), (r1, rc1), (b2, bc2), (r2, rc2) in zip(
                results[t1]["bert"], results[t1]["roberta"], results[t2]["bert"], results[t2]["roberta"]):
            r1, r2 = map(str.lower, [r1, r2])
            bc1, rc1, bc2, rc2 = map(float, [bc1, rc1, bc2, rc2])
            print(
                f"{t1_text} & {b1} ({bc1:.3f}) & {r1} ({rc1:.3f}) & {t2_text} & {b2} ({bc2:.3f}) & {r2} ({rc2:.3f}) \\\\")
            t1_text = t2_text = ""

        print("\\midrule")

    print("""\\bottomrule""")
    print("""\end{tabular}""")


if __name__ == '__main__':
    main()