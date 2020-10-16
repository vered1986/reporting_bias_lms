import math
import torch
import logging
import numpy as np

from torch.nn import CrossEntropyLoss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


class ZeroShotCOPA:
    """
    Solves COPA by LM-scoring the answer candidates.
    """
    def __init__(self, model, tokenizer, device, is_xlnet=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.is_xlnet = is_xlnet

    def get_lm_score(self, batch):
        """
        Get the lowest cross entropy loss for each instance (list of statements) in the batch
        using the langage model
        """
        # Batch: [num_clarifications, max_length]
        with torch.no_grad():
            num_clarifications, max_length = batch.shape
            shift_labels = batch[..., 1:].contiguous().view(-1)
            lm_logits = self.model(batch)[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(num_clarifications, -1).mean(1).min().cpu().item()

        return loss

    def predict(self, fields):
        context = fields['premise']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        choices = [fields['choice1'], fields['choice2']]

        supporting_statements = [self.get_supporting_statements(
            context, choices[i], choices[1 - i], question) for i in range(2)]

        # XLNet padding trick
        # (https://github.com/huggingface/transformers/blob/0866669e751bef636fa693b704a28c1fea9a17f3/examples/text-generation/run_generation.py#L62-L71)
        if self.is_xlnet:
            if self.is_xlnet:
                supporting_statements = [
                    [" ".join((PADDING_TEXT, statement)) for statement in curr_supporting_statements]
                    for curr_supporting_statements in supporting_statements]

        tokenized = [[self.tokenizer.encode(text) for text in curr] for curr in supporting_statements]
        max_length = [max([len(text) for text in curr]) for curr in tokenized]
        tokenized = [[text + [self.pad_token_id] * (max_len - len(text)) for text in per_clar]
                     for per_clar, max_len in zip(tokenized, max_length)]

        num_choices = 2
        num_batches = int(math.ceil(len(tokenized[0]) / BATCH_SIZE))
        per_choice_score = [1000] * num_choices

        for batch_index in range(0, num_batches):
            curr_batch = [tokenized[i][batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
                          for i in range(num_choices)]

            # XLNet: predict same token, not next token
            if self.is_xlnet:
                curr_batch = [torch.cat(
                    (instance, torch.zeros(
                        (instance.shape[0], 1),
                        dtype=torch.long, device=self.device)),
                    dim=1)
                    for instance in curr_batch]

            curr_batch = [torch.tensor(per_clar).long().to(self.device) for per_clar in curr_batch]
            curr_scores = [self.get_lm_score(clars_choice) for clars_choice in curr_batch]
            per_choice_score = [min(per_choice_score[i], curr_scores[i]) for i in range(num_choices)]

        return int(np.argmin(per_choice_score))

    def get_supporting_statements(self, context, choice, question):
        """
        Gets sentences supporting the statement, e.g.
        [cause]. As a result, [effect].
        """
        support_markers = ["So", "As a result,", "As one would expect,"]

        if question == "effect":
            sent1, sent2 = context, choice
        else:
            sent1, sent2 = choice, context

        sent2 = sent2[0].lower() + sent2[1:]

        support = [" ".join((sent1, support_marker, sent2))
                   for support_marker in support_markers] + [" ".join((sent1, sent2[0].upper() + sent2[1:]))]

        return support
