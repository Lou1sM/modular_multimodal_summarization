from prefs.atomic_facts import AtomicFactGenerator
from prefs.factscorer import FactScorer


example_output = 'Jim is the man who works at the shop. Bob also works at the shop. Mick has a cat.'
gold_summary = 'Jim and Bob work at the shop.'

afg=AtomicFactGenerator()
predicted_facts_and_sources = afg.extract_facts(example_output) # list of (sent, facts) tuples
predicted_facts = [x for item in predicted_facts_and_sources for x in item[1]]
gold_facts_and_sources = afg.extract_facts(gold_summary) # list of (sent, facts) tuples
gold_facts = [x for item in gold_facts_and_sources for x in item[1]]

fs = FactScorer(cache_dir_prefix='.')
fact_precision, score_per_fact = fs.get_score(predicted_facts,
                                              gold_summary,
                                              summname='test-summary',
                                              )

fact_recall, score_per_fact = fs.get_score(gold_facts,
                                           example_output,
                                           summname='test-summary',
                                           )

prefs_score = (2*fact_precision*fact_recall) / (fact_precision+fact_recall)
