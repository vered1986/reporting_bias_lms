import tqdm
import json
import torch
import random
import argparse

from orderedset import OrderedSet
from collections import defaultdict

from outcomes.src.common import init_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", type=str, help="cpu or number for GPU device")
    ap.add_argument("--copa_dir", default="data/copa", type=str, help="COPA data directory")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")

    with open(f"{args.copa_dir}/dev.jsonl") as f_in:
        events = [json.loads(line.strip())["premise"] for line in f_in]

    out = defaultdict(lambda: defaultdict(list))
    lms = [(lm, *init_model(lm, device)) for lm in ["openai-gpt", "gpt2", "gpt2-xl"]]

    for event in tqdm.tqdm(random.sample(events, 20)):
        for lm, model, tokenizer in lms:
            prefix = f"{event} As a result,"

            preds_topk = generate(
                tokenizer, model, prefix, device, num_return_sequences=10, max_length=10, k=10)

            preds_topp = generate(
                tokenizer, model, prefix, device, num_return_sequences=10, max_length=10, p=0.9)

            preds_beam = generate(
                tokenizer, model, prefix, device, num_return_sequences=5, max_length=10, beams=5)

            out[event][f"{lm}_preds_top10"] = preds_topk
            out[event][f"{lm}_preds_top0.9"] = preds_topp
            out[event][f"{lm}_preds_beam5"] = preds_beam

    print_latex_table(out)


def generate(tokenizer, model, prompt, device, num_return_sequences=1, max_length=10, beams=0, p=0, k=0):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    context_tokens = tokenizer.encode(prompt)
    max_length = max_length + len(context_tokens)
    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)
    eos_token_id = tokenizer.encode(".", add_special_tokens=False)[-1]

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=beams == 0,
        max_length=max_length,
        #         temperature=temperature,
        top_p=p if p > 0 else None,
        top_k=k if k > 0 else None,
        eos_token_id=eos_token_id,
        num_beams=beams if beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=3,
        num_return_sequences=num_return_sequences
    )

    preds = [tokenizer.decode(output, skip_special_tokens=True)[len(prompt):].strip() for output in outputs]
    print(preds)
    preds = list(OrderedSet([pred.split(".")[0].strip() for pred in preds]))
    preds = [t for t in preds if len(t) > 0]

    return preds


def print_latex_table(out):
    """
    Print the example generated outcomes
    """
    examples = [(event, fields)
                for event, fields in out.items()
                if len(fields) > 0 and
                all([len(v) > 0 for v in fields.values()])]

    print("""\\begin{tabular}{lllll}""")
    print("""\\toprule""")
    print("""\\textbf{Event} & \\textbf{LM} & \\textbf{Sampling} & \\textbf{Outcome} \\\\""")
    print("\\midrule")

    lms = ["openai-gpt", "gpt2", "gpt2-xl"]

    for event, fields in examples:
        print("\\multirow{9}{*}{\\textbf{" + event + "}} ")

        by_lm = {lm: {k.replace(f"{lm}_preds_", ""): v[0] for k, v in fields.items() if lm in k} for lm in lms}

        for i, lm in enumerate(lms):
            for j, sampling in enumerate(["top10", "top0.9", "beam5"]):
                first_col = "\\multirow{3}{*}{" + lm + "} " if j == 0 else ""
                print(f"& {first_col} & {sampling} & {by_lm[lm][sampling]} \\\\")

                if sampling == "beam5":
                    print("\\midrule")

            if i == 2:
                print("\\midrule")

        print("\\midrule")

    print("""\\bottomrule""")
    print("""\end{tabular}""")


if __name__ == '__main__':
    main()
