# Do Neural Language Models Overcome Reporting Bias?

This is the code and data for the paper:

**Do Neural Language Models Overcome Reporting Bias?**
_Vered Shwartz and Yejin Choi._ 
COLING 2020.


## 1. Actions and Events

How well do LMs estimate action frequency compared to their real-world frequency? 

Run `python -m actions.action --device=[device] --google_ngram_dir=[google_ngram_dir]` to generate the results for this experiment. 

To create a copy of Google Ngram in your machine, follow the instructions [here](https://github.com/vered1986/PythonUtils/tree/master/corpora/google_ngrams). 

## 2. Event Outcomes

Can LMs predict the more likely cause / effect of an event?

Run `python -m outcomes.src.solve_copa --device=[device] --copa_dir=outcomes/data/copa`

## 3. Properties

See [colors/src/README.md](colors/src/README.md).

#### How to cite this repository?

```
@inproceedings{shwartz2020reporting,
  title={Do Neural Language Models Overcome Reporting Bias?},
  author={Vered Shwartz and Yejin Choi},
  booktitle={COLING},
  year={2020}
}

```

