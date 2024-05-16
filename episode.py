import pandas as pd
import re
import json
import os
import math
from utils import rouge_from_multiple_refs
import numpy as np
from dl_utils.label_funcs import accuracy


def episode_from_epname(epname, infer_splits):
    with open(f'SummScreen/transcripts/{epname}.json') as f:
        transcript_data = json.load(f)
    tdata = transcript_data['Transcript']
    if tdata[0].count(':') > 1 and len(tdata)<150:
        print(f'Seems transcript stored as scene list instead of line list for {epname}, fixing')
        transcript_data['Transcript'] = '\n'.join(tdata).split('\n')
        with open(f'SummScreen/transcripts/{epname}.json','w') as f:
            json.dump(transcript_data,f)
    with open(f'SummScreen/summaries/{epname}.json') as f:
        summary_data = json.load(f)
    return Episode(epname, transcript_data, summary_data, infer=infer_splits)

class Episode(): # Nelly stored transcripts and summaries as separate jsons
    def __init__(self, epname, transcript_data, summary_data, infer=False, force_infer=False):
        self.transcript = [x.replace('\r\n','') for x in transcript_data['Transcript']]
        self.epname = epname
        self.summaries = summary_data
        self.summaries = {k:v for k,v in self.summaries.items() if len(v) > 0}
        self.title = transcript_data['Show Title'].lower().replace(' ','_')
        self.show_name = epname.split('.')[0]
        #if self.epname == 'gl-01-03-06':
            #assert self.transcript.count('') <= 1
            #if self.transcript.count('') == 1:
                #assert self.transcript.index('') == len(self.transcript)-1
                #self.transcript = self.transcript[:-1]
        self.transcript_data_dict = transcript_data
        self.summary_data_dict = summary_data

        if infer:
            if os.path.exists(maybe_cached_infer_path:=f'SummScreen/transcripts/{epname}-inferred-scene-breaks.json'):
                print(f'loading cached inferred splits from {maybe_cached_infer_path}')
                with open(maybe_cached_infer_path) as f:
                    self.scenes = f.readlines()
            else:
                print(f'no cached found at {maybe_cached_infer_path}, inferring from scratch')
                self.scenes, _, had_markers = infer_scene_splits(self.transcript, force_infer=True)
                with open(maybe_cached_infer_path,'w') as f:
                    for sc in self.scenes:
                        f.write(sc+'\n')
        else:
            self.scenes, _, had_markers = infer_scene_splits(self.transcript, force_infer=force_infer)
            if not had_markers:
                with_explicit_breaks = '£[SCENE_BREAK]£'.join(self.scenes).split('£')
                assert split_by_marker(with_explicit_breaks,'\n[SCENE_BREAK]')[0]==self.scenes
                os.rename(f'SummScreen/transcripts/{epname}.json',f'SummScreen/transcripts/{epname}-without-explicit-breaks.json')
                with open(f'SummScreen/transcripts/{epname}.json','w') as f:
                    json.dump(dict(transcript_data, Transcript=with_explicit_breaks),f)

    def transcript_from_scenes(self):
        with_explicit_breaks = '£[SCENE_BREAK]£'.join(self.scenes).split('£')
        assert split_by_marker(with_explicit_breaks,'\n[SCENE_BREAK]')[0]==self.scenes
        return with_explicit_breaks

    def calc_rouge(self,pred):
        references = self.summaries.values()
        return rouge_from_multiple_refs(pred, references, return_full=True)

    def print_recap(self):
        for summ in self.summaries:
            for line in summ.split(' . '):
                print(line)
            print()

    def __repr__(self):
        return f'Episode object for {self.title}'

    def print_transcript(self):
        for line in self.transcript:
            print(line)

def get_char_names(tlines):
    return [x.split(':')[0] if not x.startswith('[') and ':' in x else -1 for x in tlines]

def split_by_marker(tlines, marker):
    splits = np.array([i for i,x in enumerate(tlines) if x == marker])
    return '\n'.join(tlines).split(marker+'\n'), splits, True

def infer_scene_splits(tlines, force_infer):
    if '[SCENE_BREAK]' in tlines and not force_infer:
        return split_by_marker(tlines, '[SCENE_BREAK]')
    if '--------' in ''.join(tlines) and not force_infer:
        tlines = ['£' if re.search(r'-{8,20}',x) else x for x in tlines] # number of -'s can vary
        return split_by_marker(tlines, '£') # so replace all - sequences with £ and split on that
    if '' in tlines and not force_infer:
        tlines = ['£' if x=='' else x for x in tlines]
        return split_by_marker(tlines, '£')
    #tlines = transcript.split('\n')
    char_names_with_fillers = get_char_names(tlines)
    fillers = [i for i,x in enumerate(char_names_with_fillers) if x==-1]
    char_names = [x for i,x in enumerate(char_names_with_fillers) if x!=-1]
    names_to_nums_dict = {c:i for i,c in enumerate(set(char_names))}
    nums_to_names_dict = {i:c for i,c in enumerate(char_names)}
    char_nums = [names_to_nums_dict[n] for n in char_names]
    splits_without_fillers = np.array([-1] + mdl_split(char_nums) + [len(char_nums)])
    #splits = splits_without_fillers+sum(np.array(splits_without_fillers)>=f for f in fillers)
    splits = splits_without_fillers.copy()
    for f in fillers:
        splits += (splits>=f)
    return ['\n'.join(tlines[splits[i]+1:splits[i+1]+1]) for i in range(len(splits)-1)], splits[1:-1], False

def dl_from_counts(x,tot_vocab_size):
    N = x.sum()
    safe_x = x + (x==0)
    #symbol_costs = np.log2(N+1) - np.log2(safe_x)
    symbol_costs = np.log2(N) - np.log2(safe_x)
    make_codebook_cost = np.log2(math.comb(tot_vocab_size, (x!=0).sum()))
    use_codebook_cost = np.dot(x, symbol_costs)
    #return make_codebook_cost + use_codebook_cost + np.log2(N+1) # 3rd thing to mark end of chunk
    return make_codebook_cost + use_codebook_cost

def mdl_split(x):
    N = len(x)
    vs = len(set(x))
    base_costs = np.array([[0 if j<i else dl_from_counts(np.bincount(x[i:j+1]),vs)
                            for j in range(N)] for i in range(N)])
    best_costs = np.empty([N,N])
    best_splits = np.empty([N,N], dtype='object')
    for length in range(1,N+1):
        #if length==N:
            #breakpoint()
        for start in range(N-length+1):
            stop = start+length-1
            #if (start,stop) == (6,11):
                #breakpoint()
            no_split = base_costs[start,stop], []
            options = [no_split]
            for k in range(start+1,stop):
                split_cost = best_costs[start,k] + best_costs[k+1,stop]
                split = best_splits[start,k] + [k] + best_splits[k+1,stop]
                options.append((split_cost,split))
            new_best_cost, new_best_splits = min(options, key=lambda x: x[0])
            best_costs[start,stop] = new_best_cost
            best_splits[start,stop] = new_best_splits
    #print(best_splits[0,N-1])
    return best_splits[0,N-1]

def labels_from_splits(splits, n):
    return (np.expand_dims(np.arange(n),1) > splits).sum(axis=1)


if __name__ == '__main__':
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from tqdm import tqdm
    #print(mdl_split([8,2,2,8,3,1,1,3,3,1,4,5,4,5,4,5]))
    #print(mdl_split([3,5,1,5,3,1,4,5,4,5,4,5]))
    #ep = episode_from_epname('atwt-01-02-03')
    #with open(f'SummScreen/transcripts/bb-10-06-14.json') as f:
    #    transcript_data = json.load(f)['Transcript']
    #scenes, splits = infer_scene_splits(transcript_data, False)
    #for sc in scenes:
    #    if '\n' in sc:
    #        print(sc)
    #    else:
    #        print('\n'.join(sc))
    #    print('[INFERRED_BREAK]\n')
    #epname_list = [x.rstrip('.json') for x in os.listdir('SummScreen/transcripts')][:10]
    info_df = pd.read_csv('dset_info.csv',index_col=0)
    epnames = info_df.loc[info_df['usable'] & (info_df['scene_breaks']=='explicit') & (info_df['split']=='test')].index
    nss = []
    for en in epnames:
        with open(f'SummScreen/transcripts/{en}.json') as f:
            td = json.load(f)['Transcript']
        assert any('[SCENE_BREAK]' in x for x in td)
        nss.append(len(''.join(td).split('[SCENE_BREAK]')))
    mean_num_scenes = np.array(nss).mean()
    all_scores = []
    all_rand_scores = []
    all_ro_scores = []
    for en in tqdm(epnames):
        print(en)
        with open(f'SummScreen/transcripts/{en}.json') as f:
            transcript_data = json.load(f)['Transcript']
        if not any('[SCENE_BREAK]' in x for x in transcript_data):
            continue
        N = len(transcript_data)
        gt_scenes, gt_splits, _ = infer_scene_splits(transcript_data, False)
        pred_scenes, pred_splits, _ = infer_scene_splits(transcript_data, True)
        random_oracle_splits = np.arange(0, N, N//len(gt_scenes))
        random_splits = np.arange(0, N, N//mean_num_scenes)
        preds = labels_from_splits(pred_splits, N)
        gts = labels_from_splits(gt_splits, N)
        ro_labels = labels_from_splits(random_oracle_splits, N)
        pcns = [[y for y in get_char_names(x.split('\n')) if y!=-1] for x in pred_scenes]
        gcns = [[y for y in get_char_names(x.split('\n')) if y!=-1] for x in gt_scenes]
        #if pcns[0] != gcns[0]:
            #breakpoint()
        rand_labels = labels_from_splits(random_splits, N)

        new_our_scores = {}
        new_rand_scores = {}
        new_ro_scores = {}
        for mname ,mfunc in zip(['acc','nmi','ari'], [accuracy, normalized_mutual_info_score, adjusted_rand_score]):
            new_our_scores[mname] = mfunc(preds, gts)
            new_rand_scores[mname] = mfunc(rand_labels, gts)
            new_ro_scores[mname] = mfunc(ro_labels, gts)
        if new_our_scores['acc']>.95:
            breakpoint()
        all_scores.append(new_our_scores)
        all_rand_scores.append(new_rand_scores)
        all_ro_scores.append(new_ro_scores)

    our_df = pd.DataFrame(all_scores)
    rand_df = pd.DataFrame(all_rand_scores)
    ro_df = pd.DataFrame(all_ro_scores)
    results_df = pd.DataFrame({'ours':our_df.mean(axis=0),'random':rand_df.mean(axis=0), 'random oracle':ro_df.mean(axis=0)}).T
    results_df.to_csv('scene_split_results.csv')
    results_df.to_latex('scene_split_results_latex_table.txt')
    print(results_df)
    breakpoint()

