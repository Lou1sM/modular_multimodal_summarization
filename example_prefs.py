from prefs.atomic_facts import AtomicFactGenerator
from prefs.factscorer import FactScorer

afg=AtomicFactGenerator()
example_text = 'Jim is the man who works at the shop. Bob also works at the shop. Mick has a cat.'
facts_and_sources = afg.extract_facts(example_text) # list of (sent, facts) tuples
facts = [x for item in facts_and_sources for x in item[1]]

gold_summary = 'Jim and Bob work at the shop.'
fs = FactScorer(cache_dir_prefix='.')
prefs_score, score_per_fact = fs.get_score(facts,
                                           gold_summary,
                                           summname='test-summary',
                                           )
