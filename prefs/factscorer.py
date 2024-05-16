from dl_utils.misc import check_dir
import string
import json
import numpy as np
import os

from prefs.openai_lm import OpenAIModel
from nltk.corpus import names

english_names = names.words('male.txt') + names.words('female.txt')


class FactScorer(object):
    def __init__(self, model_name, cache_dir_prefix):
        self.model_name = model_name

        self.openai_key = 'prefs/api.key'
        self.cache_dir_prefix = cache_dir_prefix
        if not os.path.exists(cache_dir_prefix):
            os.makedirs(cache_dir_prefix)

        self.lm = OpenAIModel()

    def get_score(self, atomic_facts, ref_summaries_dict, summname, topic, overwrite_cache):
        decisions = []
        cache_dir = os.path.join(self.cache_dir_prefix, 'is_supported_factscore_caches')
        print(f'\nScoring facts for {summname}\n')
        cache_path = os.path.join(cache_dir,f'{summname}-{self.model_name}.json')
        had_cache = check_dir(cache_dir) and os.path.exists(cache_path)
        if (use_cache:=(had_cache and not overwrite_cache)):
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            print('no is-supported cached found at', cache_path)
            cache = {}
        for i,atom in enumerate(atomic_facts):
            atom = atom.strip()
            definition = f'Answer the question about {topic} based on the given context.\n\n'
            for k,v in ref_summaries_dict.items():
                definition += f'Title: {k}\nText: {v}\n\n'
            definition = ' '.join([x for x in definition.strip().split()][:3000])
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = f'{definition}\n\nInput: {atom.strip()} True or False?\nOutput:'

            maybe_names = [w for w in atom.rstrip('.').split() if w in english_names]
            bad_substrings = ['airs on', 'season finale', 'click', 'samaritans', '.com']
            if not all(n in definition for n in maybe_names):
                print(f'The following names are not in the summ: {[n for n in maybe_names if n not in definition]}')
                is_supported = False
            elif atom in atomic_facts[:i]: # mark repeated facts as wrong
                if atom!='<MALFORMED SENTENCE>':
                    print('penalizing repeated fact')
                is_supported = False
            elif atom =='<MALFORMED SENTENCE>':
                is_supported = False
            elif any(x in atom.lower() for x in bad_substrings):
                is_supported = False
            elif atom in cache:
                is_supported = cache[atom]
            else:
                if use_cache:
                    print('atom:', atom, 'not in cache at', cache_path)
                output = self.lm.generate(prompt, max_output_tokens=1)

                is_supported = output.lower()=='true'
                cache[atom] = bool(is_supported)

            print(atom, is_supported)
            decisions.append({"atom": atom, "is_supported": is_supported})

        if use_cache:
            with open(cache_path) as f:
                orig_cache = json.load(f)
            for k,v in orig_cache.items():
                assert cache[k] == v

        score = np.mean([d["is_supported"] for d in decisions])
        with open(cache_path, 'w') as f:
            json.dump(cache, f)

        return score, decisions
