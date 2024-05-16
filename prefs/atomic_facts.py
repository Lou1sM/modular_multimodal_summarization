import json
import time
import numpy as np
import re
import string
import spacy
import nltk
import openai
from prefs.openai_lm import OpenAIModel
from rank_bm25 import BM25Okapi
import os
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


class AtomicFactGenerator(object):
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy NLP model...")
            print("This may take a few minutes and it's one time process...")
            os.system(
                "python -m spacy download en")
            self.nlp = spacy.load("en_core_web_sm")
        self.demo_path = 'prefs/demos/demos_complex.json'
        self.oailm = OpenAIModel()

        with open(self.demo_path, 'r') as f:
            self.demos = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demos.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.sent_cache_fp = 'prefs/sent2facts_cache.json'
        if os.path.exists(self.sent_cache_fp):
            with open(self.sent_cache_fp) as f:
                self.sent_cache = json.load(f)
            assert isinstance(self.sent_cache, dict)
        else:
            self.sent_cache = {}

    def extract_facts(self, generation, cost_estimate=None):
        assert isinstance(generation, str)
        generation = re.sub(r' (?=[A-Z][a-z]+:)', '. ', generation)
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]

        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))
            paragraph = re.sub(r'\.\.+','<ellipsis>. ',paragraph)

            curr_sentences = [r.strip()+'.' for s in paragraph.split('. ') for r in sent_tokenize(s)]
            sentences = [s.replace('<ellipsis>','...') for s in sentences]
            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(sentences)

        if cost_estimate:
           return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if any(sent.startswith(x) for x in ['Sure', 'Here are', 'Please', 'I hope']):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs

    def get_init_atomic_facts_from_sentence(self, sentences):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        k = 1
        n = 7

        atoms = {}
        for i,sentence in enumerate(sentences):
            if sentence in atoms:
                continue
            elif sentence in self.sent_cache:
                atoms[sentence] = self.sent_cache[sentence]
                continue

            top_machings = best_demos(sentence, self.bm25, list(self.demos.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(self.demos.keys())[i])
                for fact in self.demos[list(self.demos.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
                for fact in self.demos[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            if sentence.split()[0] in ('The', 'A') and len(sentence.split())==2:
                atoms[sentence] = ['<MALFORMED SENTENCE>']
            else:
                if is_first_or_second_person(sentence):
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is first person')
                    continue
                if '?' in sentence:
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is question')
                    continue
                if ':' in sentence:
                    atoms[sentence] = ['<MALFORMED SENTENCE>']
                    print(sentence, 'is script-like line') # sometimes models just repeat part of transcript
                    continue
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(sentence)
                output = self.oailm.generate(prompt, max_output_tokens=32)
                if not output.startswith('-'):
                    print(sentence)
                    if i==len(sentences)-1: # just because context size output ran out
                        atoms[sentence] = []
                    else:
                        atoms[sentence] = ['<MALFORMED SENTENCE>']
                else:
                    if '...' in sentence:
                        print(f'sencence {sentence} got through even with elipsis as {output}')
                    maybe_facts = text_to_sentences(output)
                    if maybe_facts == []:
                        breakpoint()
                    atoms[sentence] = maybe_facts
            self.sent_cache[sentence] = atoms[sentence]

        print(atoms)
        with open(self.sent_cache_fp, 'w') as f:
            json.dump(self.sent_cache, f)
        for key, value in self.demos.items():
            if key not in atoms:
                atoms[key] = value

        return atoms

def is_first_or_second_person(s):
    s = s.replace('"','')
    s = s.replace('\'', ' \'')
    found = False
    for w in ('And','But','So','Well','Because'):
        if s.startswith(w):
            body = s[len(w):]
            found = True
    if not found:
        body = s[1:]
    if 'we' not in s.lower().split() and 'I' not in s.split() and 'you' not in s.split():
        return False
    return body.lower() == body.replace('I','i')

def best_demos(query, bm25, demos_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demos_sents, k)
    return top_machings

# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0:
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.'
    else:
        sentences = []
    return sentences

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list

def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


if __name__ == "__main__":
    generator = AtomicFactGenerator("api.key", "demos", gpt3_cache_file=None)
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    print(para_breaks)
