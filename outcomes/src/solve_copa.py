import json
import tqdm
import torch
import logging
import argparse

from sklearn.metrics import accuracy_score

from outcomes.src.common import init_model
from outcomes.src.zero_shot_copa import ZeroShotCOPA
from outcomes.src.zero_shot_copa_de import ZeroShotCOPAWithDE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODALS = {'will', 'shall'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", type=str, help="cpu or number for GPU device")
    ap.add_argument("--copa_dir", default="data/copa", type=str, help="COPA data directory")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if args.device != "cpu" else torch.device("cpu")

    lms = ["openai-gpt", "gpt2", "gpt2-xl", "xlnet-base-cased", "xlnet-large-cased"]
    results = {lm: {"with": None, "without": None} for lm in lms}
    results["majority"] = {"without": 0.55, "with": 0.55}

    for lm in lms:
        torch.cuda.empty_cache()
        model, tokenizer = init_model(lm, device)
        data = f"{args.copa_dir}/dev.jsonl"
        acc_zs = solve(data, ZeroShotCOPA(model, tokenizer, device), f"../data/copa/out_dev_zs_{lm}.jsonl")
        acc_zsde = solve(data, ZeroShotCOPAWithDE(model, tokenizer, device), f"../data/copa/out_dev_zsde_{lm}.jsonl")
        results[lm] = {"with": acc_zsde, "without": acc_zs}

    print_pred(results)


def print_pred(results):
    """
    Print the LaTex table with the results
    """
    lm_display = {"majority": "Majority",
                  "openai-gpt": "GPT",
                  "gpt2": "GPT2-S",
                  "gpt2-xl": "GPT2-XL",
                  "xlnet-base-cased": "XLNet-S",
                  "xlnet-large-cased": "XLNet-L"}
    print("""\\begin{tabular}{lll}""")
    print("""\\toprule""")
    print("""\\textbf{LM} & \\textbf{Without} & \\textbf{With} \\\\""")
    print("\\midrule")

    for lm, curr_results in results.items():
        print(f"{lm_display[lm]} & {results[lm]['without']:.2f} & {results[lm]['with']:.2f} \\\\""")

    print("""\\bottomrule""")
    print("""\\end{tabular}""")


def solve(dataset_file, solver, out_file):
    gold = []
    predictions = []

    # Predict instances
    with open(out_file, "w") as f_out:
        with open(dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                gold.append(fields["label"])
                prediction = solver.predict(fields)
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")

    # Don't report accuracy if we don't have the labels
    accuracy = None
    if None not in gold:
        accuracy = accuracy_score(gold, predictions)

    return accuracy


if __name__ == '__main__':
    main()
