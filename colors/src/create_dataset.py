import os
import re
import json
import logging

logger = logging.getLogger(__name__)


def main():
    """
    Creates the dataset
    """
    color_list = [color for color in [line.strip() for line in open("color_list.txt")] if color != ""]

    os.path.mkdirs("data")
    with open("data/dataset.jsonl", "w", encoding="utf-8") as f_out:
        for color in color_list:
            if color == "":
                continue

            with open(f"texts/{color}.txt", 'r', encoding="utf-8") as f_in:
                for line in f_in:
                    line = re.sub(rf"\b{color}\b", "[MASK]", line)
                    if len([w for w in line.strip().split() if w == "[MASK]"]) == 1:
                        f_out.write(json.dumps(
                            {"color": color, "sentence": line.strip()}) + "\n")


if __name__ == '__main__':
    main()
