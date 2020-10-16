import os
import json
import torch
import spacy
import scipy
import logging
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelWithLMHead

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

AUX_VERBS = {"be", "can", "could", "dare", "do", "have", "may", "might",
             "must", "need", "ought", "shall", "should", "will", "would"}


def main():
    """
    Evaluate a LM on masked prediction of colors.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--google_ngram_dir", type=str, required=True,
                    help="directory to Google Ngram files. "
                         "See https://github.com/vered1986/PythonUtils/tree/master/corpora/google_ngrams")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or number for GPU device")
    ap.add_argument("--k", default=5000, type=int, help="How many items to predict")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")

    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    roberta_model = AutoModelWithLMHead.from_pretrained("roberta-large")
    roberta_model.to(device)

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    bert_model = AutoModelWithLMHead.from_pretrained("bert-large-uncased")
    bert_model.to(device)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2-xl")
    gpt2_model.to(device)

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
    event_freq = {}
    entity = "person"

    # Google Ngrams: search for "person is <verb>"
    items = get_following_words(f"{entity} is", google_ngram_dir=args.google_ngram_dir, k=args.k)
    items = filter_and_sort(nlp, items)

    # Fall back to "person <verb>"
    if len(items) == 0:
        items = get_following_words(entity, google_ngram_dir=args.google_ngram_dir, k=args.k)
        items = filter_and_sort(nlp, items)

    event_freq["gngrams"] = items

    # BERT
    items = get_top_k_predictions_masked_lm(
        bert_model, bert_tokenizer,
        ["The person is [MASK].", "The person [MASK].",
         "People are [MASK].", "All people [MASK]."],
        mask="[MASK]", k=args.k)
    items = filter_and_sort(nlp, items)
    event_freq["bert"] = items

    # RoBERTa
    items = get_top_k_predictions_masked_lm(
        roberta_model, roberta_tokenizer,
        ["The person is <mask>.", "The person <mask>.",
         "People are <mask>.", "All people <mask>."],
        mask="<mask>", k=args.k)
    items = filter_and_sort(nlp, items)
    event_freq["roberta"] = items

    # GPT2
    items = get_top_k_predictions_lm(
        gpt2_model, gpt2_tokenizer,
        ["The person is ", "The person ", "People are ", "All people "],
        k=args.k)
    items = filter_and_sort(nlp, items)
    event_freq["gpt2"] = items

    for resource in ["gngrams", "bert", "roberta"]:
        for key in freq.keys():
            count = np.sum(
                [dict(event_freq[resource]).get(variation, 0) for variation in variations[key]])
            freq[key][resource] = count

    # Scale it
    scaled = {label: {} for label in freq.keys()}

    for resource in ["roberta", "bert", "gngrams", "true"]:
        all_sum = np.sum([freq[l][resource] for l in freq.keys()])
        for label in freq.keys():
            scaled[label][resource] = freq[label][resource] / all_sum

    scaled = OrderedDict(sorted(scaled.items(), key=lambda x: freq[x[0]]["true"], reverse=True))

    # Save to file
    with open("actions.jsonl", "w") as f_out:
        f_out.write(json.dumps(scaled) + "\n")

    draw_event_frequencies(slice_odict(scaled, 0, 10), "A person is ____")
    draw_event_frequencies(slice_odict(scaled, 10, 20), legend=True)
    print(kl_divergence(scaled))

    print_pred(bert_model,
               bert_tokenizer,
               roberta_model,
               roberta_tokenizer,
               gpt2_model,
               gpt2_tokenizer)


def get_top_k_predictions_masked_lm(bert_model, bert_tokenizer, texts, mask="[MASK]", k=10):
    """
    Find the best word to replace the mask, using BERT or RoBERTa.
    """
    mask_id = bert_tokenizer.encode(mask, add_special_tokens=False)[0]
    indexed_tokens = bert_tokenizer.batch_encode_plus(
        texts, pad_to_max_length=True, max_length=10)['input_ids']
    masked_indices = [[i for i, token in enumerate(instance)
                       if token == mask_id][0] for instance in indexed_tokens]
    tokens_tensor = torch.tensor(indexed_tokens).long().to(bert_model.device)

    bert_model.eval()

    with torch.no_grad():
        predictions = bert_model(tokens_tensor)[0]

    masked_token_probs = [
        predictions[i, masked_index, :].cpu().numpy()
        for i, masked_index in enumerate(masked_indices)]

    probs = np.max(masked_token_probs, axis=0)
    probs = scipy.special.softmax(probs, axis=-1)
    top_k_probs = np.argpartition(probs, -k)[-k:]
    top_k_probs = top_k_probs[np.argsort(probs[top_k_probs])[::-1]]
    tokens = [t.replace("Ġ", "") for t in bert_tokenizer.convert_ids_to_tokens(top_k_probs)]
    top_k_probs = list(zip(tokens, [probs[i] for i in top_k_probs]))
    return top_k_probs


def get_top_k_predictions_lm(model, tokenizer, texts, k=10):
    """
    Find the best word to continue the sentence.
    """
    next_token_logits = []
    for text in texts:
        context_tokens = tokenizer.encode(text)
        input_ids = torch.tensor(context_tokens, device=model.device).unsqueeze(0)

        model.eval()

        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits.append(outputs[0][:, -1, :].cpu().numpy())

    probs = np.max(np.vstack(next_token_logits), axis=0)
    probs = scipy.special.softmax(probs, axis=-1)
    top_k_probs = np.argpartition(probs, -k)[-k:]
    top_k_probs = top_k_probs[np.argsort(probs[top_k_probs])[::-1]]
    tokens = [t.replace("Ġ", "").lower() for t in tokenizer.convert_ids_to_tokens(top_k_probs)]
    top_k_probs = list(zip(tokens, [probs[i] for i in top_k_probs]))
    return top_k_probs


def get_following_words(phrase, google_ngram_dir, k=-1):
    """
    Searches for occurrences that start with `phrase` in Google Ngrams.
    :google_ngram_dir: a Google Ngram directory. See
    `https://github.com/vered1986/PythonUtils/tree/master/corpora/google_ngrams`
    """
    phrase = phrase.lower()
    phrase_words = phrase.split()
    n = 5
    prefix = phrase[:2]
    results = []
    curr_ngram_file = os.path.join(google_ngram_dir, f"googlebooks-eng-all-{n}gram-20120701-{prefix}_filtered")

    # No file for this prefix
    if not os.path.exists(curr_ngram_file):
        logging.warning(f'file {curr_ngram_file} does not exist')
        return results

    # The Google ngrams file is tab separated, containing: ngram and count.
    with open(curr_ngram_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            ngram, count = line.lower().strip().split('\t')
            words = ngram.split()

            if len(phrase_words) > 1:
                if (words[0] > phrase_words[0] and not phrase_words[0] in words[0]) or \
                        (words[0] == phrase_words[0] and phrase_words[1] not in words[1] and
                         words[1] > phrase_words[1]):
                    break
                elif words[0] == phrase_words[0] and words[1] == phrase_words[1]:
                    results.append((words[-1], float(count)))
                else:
                    continue

            else:
                if words[0] > phrase_words[0] and not phrase_words[0] in words[0]:
                    break
                elif words[0] == phrase_words[0]:
                    results.append((words[-1], float(count)))
                else:
                    continue

    # Limit the number of results
    if k > 0:
        results = results[:k+1]

    return results


def filter_and_sort(nlp, items):
    """
    Only keep non-auxiliary verbs, sort by descending score
    """
    docs = [nlp(f"The person {item.lower()}.") for item, count in items]
    verbs = [(item, count)
             for (item, count), doc in zip(items, docs)
             if "##" not in item and
             item not in AUX_VERBS and [t.pos_ for t in doc][-2] == "VERB"]

    # Normalize and sort
    all_sum = np.sum([count for _, count in verbs])
    curr_items = list(sorted(
        [(word, count * 1.0 / all_sum) for word, count in verbs],
        key=lambda x: x[1], reverse=True))
    return curr_items


def draw_event_frequencies(freq, title=None, legend=False):
    """
    Draw the event frequencies
    """
    sns.set_style("white")
    resources = ["true", "gngrams", "bert", "roberta", "gpt2"]
    labels = list(freq.keys())
    items = {label: [freq[label][resource] for resource in resources] for label in labels}

    x = np.arange(len(labels))
    y = np.arange(len(resources))
    plt.figure(figsize=(int(len(freq) * 2), 2))
    plt.box(on=None)
    w = 0.5

    resource_names = ["Actual", "Google Ngrams", "BERT", "RoBERTa", "GPT2"]
    colors = sns.color_palette("colorblind", len(resources))
    styles = ["-", ":", "--", "--", "--"]

    df = pd.DataFrame({
        'X': [i for i in range(len(labels)) for j in range(len(resources))],
        'Y': [0 for i in range(len(labels)) for j in range(len(resources))],
        "bubble_size": np.array([items[label][i] for label in labels for i in range(len(resources))]) * 15000,
        "linestyle": [styles[j] for i in range(len(labels)) for j in range(len(resources))]
    })

    plt.scatter('X', 'Y', s='bubble_size', facecolors='none',
                edgecolors=colors, data=df,
                linestyle=df['linestyle'], linewidth=2)

    if title is not None:
        plt.title(title, fontsize=22)

    plt.xlim(-0.5, len(labels) - 0.5)
    plt.ylim(-.1, .1)
    plt.yticks([])
    plt.xticks(x - .2 + w / 2, [label.replace(" ", "\n") for label in labels], fontsize=16)

    if legend:
        handles = [Line2D(
            [0], [0], color=colors[i], linewidth=2,
            label=resource_names[i], markersize=18, linestyle=styles[i])
            for i in range(len(resource_names))]
        plt.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.22, -0.65), ncol=len(resources))

    plt.show()


def print_pred(bert_model,
               bert_tokenizer,
               roberta_model,
               roberta_tokenizer,
               gpt2_model,
               gpt2_tokenizer):
    """
    Print the LaTex table with the predictions
    """
    print("""\\begin{tabular}{llllllllllll}""")
    print("""\\toprule""")
    print("""& \\textbf{BERT} & \\textbf{RoBERTa} & \\textbf{GPT-2} & & 
             & \\textbf{BERT} & \\textbf{RoBERTa} & \\textbf{GPT-2} \\\\""")
    print("\\midrule")

    templates = ["The person [MASK].",
                 "The person is [MASK]."]
    results = {template: {} for template in templates}

    for template in templates:
        # BERT
        items = get_top_k_predictions_masked_lm(
            bert_model, bert_tokenizer, [template], mask="[MASK]", k=500)
        items = filter_and_sort(nlp, items)
        results[template]["bert"] = [(e, str(c)) for e, c in items][:10]

        # RoBERTa
        items = get_top_k_predictions_masked_lm(
            roberta_model, roberta_tokenizer,
            [template.replace("[MASK]", "<mask>")], mask="<mask>", k=500)
        items = filter_and_sort(nlp, items)
        results[template]["roberta"] = [(e, str(c)) for e, c in items][:10]

        # GPT2
        items = get_top_k_predictions_lm(
            gpt2_model, gpt2_tokenizer, templates, k=500)
        items = filter_and_sort(nlp, items)
        results[template]["gpt2"] = \
            [(e, str(c)) for e, c in items if e != "ċċ"][:10]

    template_displays = [
        "\\multirow{10}{*}{\\specialcellleft{" +
        "\\\\".join(t.replace("[MASK]", "\\underline{~~~~~}").split()) + "}} "
        for t in templates]

    for i in range(10):
        for j, template in enumerate(templates):
            print(f"{template_displays[j] if i == 0 else ''} & ", end="")
            print(f"{results[template]['bert'][i][0]} ({100 * float(results[template]['bert'][i][1]):.1f}) & ", end="")
            print(f"{results[template]['roberta'][i][0]} ({100 * float(results[template]['roberta'][i][1]):.1f}) & ",
                  end="")
            print(f"{results[template]['gpt2'][i][0]} ({100 * float(results[template]['gpt2'][i][1]):.1f}) & ", end="")

        print("\\\\")
    print("""\\bottomrule""")
    print("""\\end{tabular}""")


def slice_odict(odict, start=None, end=None):
    return OrderedDict([
        (k,v) for (k,v) in odict.items()
        if k in list(odict.keys())[start:end]
    ])


def kl_divergence(frequencies):
    """
    Computes the KL divergence of each distribution with
    the actual distribution
    """
    epsilon = 0.0000001
    resources = ["true", "gngrams", "bert", "roberta"]
    by_resource = {resource:
        np.array([
            max(epsilon, frequencies[action][resource])
            for action in frequencies.keys()])
        for resource in resources}

    for resource in resources[1:]:
        print(resource, scipy.stats.entropy(by_resource['true'], by_resource[resource]))


if __name__ == '__main__':
    main()