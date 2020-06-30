import spacy
import logging

from pattern import en
from overrides import overrides

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODALS = {'will', 'shall'}

from outcomes.src.zero_shot_copa import ZeroShotCOPA


class ZeroShotCOPAWithDE(ZeroShotCOPA):
    """
    Gets sentences supporting the statement, e.g.
    [cause]. As a result, [effect].
    """
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.nlp = spacy.load("en_core_web_sm")

    @overrides
    def get_supporting_statements(self, context, choice, question):
        """
        Gets sentences supporting the statement, e.g.
        [cause]. As a result, [effect].
        """
        support = super().get_supporting_statements(context, choice, other_choice, question)

        disconfirmed_markers = [f"{neg} {adv}".strip()
                                for neg in ["yet,", "but", "however,"]
                                for adv in ["surprisingly", "for some reason", "somehow", ""]
                                ]

        disconfirmed_markers = [d[0].upper() + d[1:] for d in disconfirmed_markers if len(d) > 0]

        if question == "effect":
            sent1, sent2 = context, choice
        else:
            sent1, sent2 = choice, context

        disconfirmed_expectations = []
        negated = negate_last_verb(sent2)

        if negated is not None:
            sent2 = negated[0].lower() + negated[1:]
            disconfirmed_expectations = [" ".join((sent1, disconfirmed_marker, sent2))
                                         for disconfirmed_marker in disconfirmed_markers]

        return support + disconfirmed_expectations

    def negate_last_verb(self, statement):
        """
        Takes a statement and negates it
        :param statement: string statement
        :return: the negated statement
        """
        try:
            doc = [t for t in self.nlp(statement)]
            last_verb_index = [i for i, t in enumerate(doc) if t.pos_ == "VERB"][-1]
            token = doc[last_verb_index]
            verb = token.text
            morph_features = self.nlp.vocab.morphology.tag_map[token.tag_]
            if morph_features is not None:
                morph_features = dict([key.split("_")
                                       for key in morph_features.keys()
                                       if isinstance(key, str) and "_" in key]
                                      )

            new_verb = negate(token, morph_features)
            tokens = [t.text for t in doc]
            tokens[last_verb_index] = new_verb
            return " ".join(tokens)
        except:
            return None


def negate(token, morph_features):
    """
    Get a head verb and negate it
    :param token: the SpaCy token of the main verb
    :param morph_features: SpaCy `nlp.vocab.morphology.tag_map`
    :return: the negated string or None if failed to negate
    """
    verb = token.text
    tense = morph_features.get('Tense', None)
    person = morph_features.get("Person", 3)
    number = {"sing": en.SG, "plur": en.PL}.get(morph_features.get("Number", "sing"))
    auxes = [t for t in token.lefts if t.dep_ == 'aux' or t.lemma_ in {"be", "have"}]

    # ing: add not after the have/be verb or future: add "not" after the modal
    if (token.tag_ == "VBG" and len(auxes) > 0) or is_future_tense(token):
        return "not " + token.text

    # Present simple
    elif tense in {"pres", "past"}:
        aux_tense = {"pres": en.PRESENT, "past": en.PAST}[tense]

        if token.lemma_ == "be":
            return " ".join((verb, "not"))

        else:
            aux = en.conjugate("do",
                               tense=aux_tense,
                               person=person,
                               number=number,
                               parse=True)

            verb = en.conjugate(verb,
                                tense=en.INFINITIVE,
                                person=person,
                                number=number,
                                parse=True)

            return " ".join((aux, "not", verb))

    else:
        return None


def is_future_tense(token):
    """
    Is this verb with a future tense?
    :param token: SpaCy token
    :return: Is this verb with a future tense?
    """
    return token.tag_ == 'VB' and any([t.text in MODALS and t.dep_ == 'aux' for t in token.lefts])

