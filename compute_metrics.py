import numpy as np
import re
import sys
import pandas as pd
import json
from PREFS.factscorer import FactScorer
from dl_utils.misc import check_dir
from utils import rouge_from_multiple_refs
from os.path import join
import argparse
from episode import episode_from_epname
import os
from PREFS.atomic_facts import AtomicFactGenerator
from tqdm import tqdm


def filter_facts(facts):
    """Some facts are vacuous of any real information, exclude these before scoring."""
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or (not any(x in f.lower() for x in ['someone','something','somebody','is a person','is a character', 'are people', 'are characters']) and len(f.split())>2)]
    facts = [f for f in facts if not f.lower().startswith('there is a') and not 'is in a room' in f and not 'is talking' in f and not 'are talking' in f and not 'made a statement' in f]
    facts = [f for f in facts if not 'is mentioned' in f.lower() and not 'are mentioned' in f.lower() and not 'is there' in f.lower() and not 'are there' in f.lower()]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or not f.endswith(' to')]
    facts = [f for f in facts if not 'is there' in f and not 'are there' in f]
    return facts

def get_maybe_cached_atomic_facts(maybe_cached_path, generator, nl_text=None, path=None):
    assert (nl_text is None) != (path is None)
    if nl_text is None:
        with open(path) as f:
            nl_text = f.read()
    if ':' not in nl_text and os.path.exists(maybe_cached_path):
        print('using extraction cache at', maybe_cached_path)
        with open(maybe_cached_path) as f:
            facts = f.read().split('\n')
    else:
        print('no extraction cache found at', maybe_cached_path)
        facts_and_sources, para_breaks = generator.run(nl_text)
        facts = [x for line in facts_and_sources for x in line[1]]
        with open(maybe_cached_path,'w') as f:
            f.write('\n'.join([pf for pf in facts]))

    filtered_facts = filter_facts(facts)
    print('filtering:', [x for x in facts if x not in filtered_facts])
    if ARGS.is_test:
        filtered_facts = filtered_facts[:3]
    return filtered_facts


if __name__ == '__main__':
    generator = AtomicFactGenerator()

    all_metrics = ['factscore', 'rouge', 'rev-factscore']
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',type=str,default='tmp')
    parser.add_argument('--ep',type=str,default='none')
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--allow_bs_cache',action='store_true')
    parser.add_argument('--print_results',action='store_true')
    parser.add_argument('--scorer_model', type=str)
    parser.add_argument('--expdir_prefix', type=str, default='experiments')
    parser.add_argument('--metrics', type=str, nargs='+', choices=all_metrics+['all','just-get-facts'], default=['all'])
    parser.add_argument('--n_dpoints', type=int, default=-1)
    ARGS = parser.parse_args()

    if ARGS.metrics == ['all']:
        ARGS.metrics = all_metrics

    fs = FactScorer('gpt-4-turbo-preview', cache_dir_prefix='.')


    expdir = join(ARGS.expdir_prefix, ARGS.expname)
    gendir = join(expdir, 'generations_test')

    if ARGS.metrics == ['all']:
        ARGS.metrics = all_metrics
    if 'rouge' in ARGS.metrics:
        ARGS.metrics = [m for m in ARGS.metrics if m!='rouge']+['r1','r2','rl','rlsum']
    if 'factscore' in ARGS.metrics:
        ARGS.metrics.append('n_facts')

    if ARGS.expname == 'gt-upperbound':
        info_df = pd.read_csv('dset_info.csv',index_col=0)
        epnames_val = info_df[(info_df['split']=='val') & info_df['usable']].index.tolist()
        epnames_test = info_df[(info_df['split']=='test') & info_df['usable']].index.tolist()
        all_epnames = epnames_val + epnames_test
    else:
        all_epnames = os.listdir(gendir) if ARGS.ep == 'none' else [f'{ARGS.ep}.txt']
    if ARGS.n_dpoints != -1:
        all_epnames = all_epnames[:ARGS.n_dpoints]
    full_results = {m:{} for m in ARGS.metrics}
    all_factscores = []
    all_rouges = []
    for i,fn in enumerate(pbar := tqdm(all_epnames)):
        if ARGS.expname != 'gt-upperbound':
            assert fn.endswith('.txt')
            epname = fn[:-4]
        else:
            assert not fn.endswith('.txt')
            epname = fn

        ep = episode_from_epname(epname, False)
        gt_summs = ep.summaries # this is a dict
        good_summ_sources = ('soapcentral_condensed','tvdb','tvmega_recap')

        if ARGS.expname == 'gt-upperbound':
            possible_summ_sources = [k for k in good_summ_sources
                                     if gt_summs.get(k,None) is not None]
            assert len(possible_summ_sources) > 0
            longest_summ_source = max(possible_summ_sources, key=lambda x:200 if x=='soapcentral_condensed' else len(gt_summs[x]))
            print('\n'+longest_summ_source+'\n')
            pred_summ = gt_summs.pop(longest_summ_source)
        else:
            with open(f'{gendir}/{epname}.txt') as f:
                pred_summ = f.read()


        # Now compute specified metrics
        if 'factscore' in ARGS.metrics or ARGS.metrics==['just-get-facts']:
            pbar.set_description(f'Computing factscore for {epname}')
            check_dir(cache_dir := f'gpt-extracted-facts/{ARGS.expname}')
            cache_fpath = f'{cache_dir}/{epname}'
            pred_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=pred_summ)
            if 'factscore' in ARGS.metrics:
                epname_to_use = f'{epname}-gtup' if ARGS.expname == 'gt-upperbound' else epname
                score, decisions = fs.get_score(pred_facts,
                                                ref_summaries_dict=gt_summs,
                                                summname=epname_to_use,
                                                topic='A summary of a TV show',
                                                overwrite_cache=False)
                full_results['factscore'][epname] = score
                full_results['n_facts'][epname] = len(pred_facts)

        if 'rev-factscore' in ARGS.metrics:
            pbar.set_description(f'Computing rev-factscore for {epname}')
            check_dir(cache_dir := 'gpt-extracted-facts/gt_summ_facts')
            facts_by_summ_name = {}
            gt_facts = []
            cache_fpath = f'{cache_dir}/{epname}'
            summ = ep.summaries.get('tvmega_recap','')
            gt_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=summ)
            pred_sents = pred_summ.split('. ')

            def is_malformed(sent):
                if len(sent.split())==2 and sent.strip().startswith('The'):
                    return False
                if len(sent.split())==1:
                    return False
                if any(ord(c)>=128 for c in list(sent)):
                    return False
                if ':' in sent or '?' in sent:
                    return False
                if any(x in sent.split() for x in ['I', 'you', 'we']):
                    return False
                return True

            pred_sents = ['<MALFORMED SENTENCE>' if is_malformed(ps) else ps for ps in pred_sents]
            pred_summ_to_use = '. '.join(pred_sents)
            print(pred_summ)
            print(pred_summ_to_use)
            pred_summ_dict = {'pred':pred_summ_to_use}
            if all([ps=='<MALFORMED SENTENCE>' for ps in pred_sents]):
                full_results['rev-factscore'][epname] = 0
            else:
                score, decisions = fs.get_score(gt_facts,
                                                ref_summaries_dict=pred_summ_dict,
                                                summname=f'{epname}_{ARGS.expname}',
                                                topic='A summary of a TV show',
                                                overwrite_cache=False)
                full_results['rev-factscore'][epname] = score

        if 'r1' in ARGS.metrics:
            pbar.set_description(f'Computing rouge for {epname}')

            if len([v for v in gt_summs.values() if v is not None]) < 2:
                continue

            rouge_results = rouge_from_multiple_refs(pred_summ,
                                                     gt_summs.values(),
                                                     benchmark_rl=True,
                                                     return_full=False)

            for r,s in zip(['r1','r2','rl','rlsum'], rouge_results):
                full_results[r][epname] = s

        if ARGS.is_test and i==1:
            break
        results_df_tmp = pd.DataFrame(full_results)
        if 'factscore' in ARGS.metrics:
            print('running factscore:', results_df_tmp.mean(axis=0)['factscore'])
        if 'rev-factscore' in ARGS.metrics:
            print('running rev-factscore:', results_df_tmp.mean(axis=0)['rev-factscore'])

    results_df = pd.DataFrame(full_results)
    results_df.loc['mean'] = results_df.mean(axis=0)
    results_df.loc['std'] = results_df.std(axis=0)
    assert set(results_df.columns) == set(ARGS.metrics)

    for m in ARGS.metrics:
        m_results = results_df[m].to_dict()
        check_dir(res_dir := join(expdir,'results_by_metric'))
        with open(join(res_dir, f'{m}-full-results.json'), 'w') as f:
            json.dump(m_results,f)

    results_df.to_csv(join(expdir, 'full_results.csv'))
    if ARGS.print_results:
        print(results_df)
