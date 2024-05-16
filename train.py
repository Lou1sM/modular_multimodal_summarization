import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dl_utils.misc import set_experiment_dir
from datasets import load_dataset, load_from_disk
import argparse
import torch
import os
from os.path import join
from summarize_dialogue import SoapSummer
import sys
from utils import get_fn, display_rouges


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--bs',type=int,default=1)
parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions','kosmos-only','swinbert-only'], default='nocaptions')
parser.add_argument('--centralscenes', action='store_true')
parser.add_argument('--chkpt_path_reload_from',type=str,default='none')
parser.add_argument('--max_chunk_size',type=int,default=10000)
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--data_dir',type=str,default='./SummScreen')
parser.add_argument('--dbs',type=int,default=8)
parser.add_argument('--dont_save_new_scenes',action='store_true')
parser.add_argument('--early_stop_metric',type=int,default=2)
parser.add_argument('--eval_every',type=int,default=1)
parser.add_argument('--eval_first',action='store_true')
parser.add_argument('--expdir_prefix',type=str,default='experiments')
parser.add_argument('--expname',type=str)
parser.add_argument('--infer_splits_at_test',action='store_true')
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('-bt','--is_test_big_bart',action='store_true')
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn')
parser.add_argument('--n_dpoints',type=int,default=-1)
parser.add_argument('--n_epochs',type=int,default=10)
parser.add_argument('--n_iter',type=int,default=-1)
parser.add_argument('--no_pbar',action='store_true')
parser.add_argument('--order', type=str, choices=['identity','optimal','rand'], default='identity')
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--reload_from',type=str,default='none')
parser.add_argument('--resumm_scenes',action='store_true')
parser.add_argument('--retokenize',action='store_true')
parser.add_argument('--save_every',action='store_true')
parser.add_argument('--soft_scene_summs',action='store_true')
parser.add_argument('--startendscenes', action='store_true')
parser.add_argument('--uniform_breaks', action='store_true')
ARGS = parser.parse_args()


if ARGS.uniform_breaks:
    assert ARGS.caps == 'nocaptions'
if ARGS.startendscenes:
    assert not ARGS.uniform_breaks
    assert ARGS.order=='identity'
ARGS.is_test = ARGS.is_test or ARGS.is_test_big_bart
ARGS.retokenize = ARGS.retokenize or ARGS.resumm_scenes

if ARGS.expname is None and not ARGS.is_test:
    sys.exit('must set explicit expname when not in test mode')
elif ARGS.is_test:
    ARGS.expname='tmp'
    if ARGS.n_dpoints == -1:
        ARGS.n_dpoints = 10
    ARGS.n_epochs = 2

assert not (ARGS.reload_from!='none' and ARGS.chkpt_path_reload_from!='none'), "can't use two reload options"

expdir = join(ARGS.expdir_prefix,ARGS.expname)
if ARGS.reload_from != 'none':
    assert os.path.isdir(expdir)
else:
    set_experiment_dir(expdir, ARGS.overwrite, name_of_trials=join(ARGS.expdir_prefix,'tmp'))

device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

model_name = 'lucadiliello/bart-small' if ARGS.is_test and not ARGS.is_test_big_bart else ARGS.model_name

print(f'using model {model_name}')

if ARGS.reload_from!='none':
    chkpt_path = join(expdir,'checkpoints',ARGS.reload_from)
elif ARGS.chkpt_path_reload_from!='none':
    chkpt_path = join(ARGS.expdir_prefix, ARGS.chkpt_path_reload_from)
else:
    chkpt_path = model_name

print(f'loading model from {chkpt_path}')
if ARGS.startendscenes or ARGS.centralscenes:
    #model, tokenizer = None, None
    assert ARGS.model_name == 'facebook/bart-large-cnn', "can't set model name in startend/central, will be ignored"
#else:
    #model = AutoModelForSeq2SeqLM.from_pretrained(chkpt_path).to(device)
    #tokenizer = AutoTokenizer.from_pretrained(chkpt_path)
ss = SoapSummer(model_name=model_name,
                device=device,
                bs=ARGS.bs,
                dbs=ARGS.dbs,
                #tokenizer=tokenizer,
                caps=ARGS.caps,
                scene_order=ARGS.order,
                uniform_breaks=ARGS.uniform_breaks,
                startendscenes=ARGS.startendscenes,
                centralscenes=ARGS.centralscenes,
                soft_scene_summs=ARGS.soft_scene_summs,
                max_chunk_size=ARGS.max_chunk_size,
                expdir=expdir,
                data_dir=ARGS.data_dir,
                resumm_scenes=ARGS.resumm_scenes,
                do_save_new_scenes=not ARGS.dont_save_new_scenes,
                is_test=ARGS.is_test)

fn = get_fn(ARGS.caps, ARGS.order, ARGS.uniform_breaks, ARGS.startendscenes, ARGS.centralscenes, ARGS.soft_scene_summs, ARGS.is_test)

def train_preproc_fn(dpoint):
    if ARGS.soft_scene_summs:
        model_inputs = {'inputs_embeds': [torch.load(fpath) for fpath in dpoint['scene_summs']]}
        model_inputs['input_ids'] = model_inputs['inputs_embeds'] # makes checking input size easier later
        model_inputs['attention_mask'] = [[1]*len(x) for x in model_inputs['inputs_embeds']]
    else:
        inputs = [doc for doc in dpoint['scene_summs']]
        model_inputs = ss.tokenizer(inputs)

    # Setup the tokenizer for targets
    labels = ss.tokenizer(text_target=dpoint['summ'])

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def get_dsets():
    dsets = []
    splits = ('train','val','test-inferred') if ARGS.infer_splits_at_test else ('train','val','test')
    maybe_cache_paths = [join(ARGS.data_dir, f'cached_tokenized/{fn}_{ARGS.n_dpoints}dps_{s}_cache') for s in splits]
    if all(os.path.exists(p) for p in maybe_cache_paths) and not ARGS.retokenize:
        print(maybe_cache_paths[0], 'exists, loading from there')
        print('tokenized datasets have been cached, loading')
        return [load_from_disk(cp) for cp in maybe_cache_paths]
    json_paths = [join(ARGS.data_dir, f'json_datasets/{fn}_{ARGS.n_dpoints}dps_{s}_dset.json') for s in splits]
    if any(not os.path.exists(jp) for jp in json_paths) or ARGS.retokenize:
        print('building new dataset')
        for spl in splits:
            ss.build_dset(ARGS.caps, ARGS.n_dpoints, spl)


    assert all(os.path.exists(jp) for jp in json_paths)
    for split,jp,cp in zip(splits,json_paths,maybe_cache_paths):
        dset = load_dataset('json', data_files=jp, split='train')
        if split=='train':
            dset = dset.map(train_preproc_fn, batched=True, remove_columns=dset.column_names)
        assert cp == join(ARGS.data_dir, f'cached_tokenized/{fn}_{ARGS.n_dpoints}dps_{split}_cache')
        dset.save_to_disk(cp)
        dsets.append(dset)
    return dsets

trainset, valset, testset = get_dsets()


if ARGS.eval_first:
    rouges = ss.eval_epoch(0, testset)
    rouges_arr = np.array(rouges).mean(axis=0)
    print(f'Mean Rouge: {rouges_arr}')

test_rouges, best_val_rouges, all_val_rouges = ss.train_epochs(ARGS.n_epochs, trainset, valset, testset, ARGS.no_pbar, ARGS.early_stop_metric)

print(display_rouges(test_rouges))

results_path = join(expdir,'results.txt')
with open(results_path,'w') as f:
    f.write('\nTEST ROUGES:\n')
    for rname,rscore in display_rouges(test_rouges):
        f.write(f'{rname}: {rscore:.5f}\n')
    f.write('\nBEST VAL ROUGES:\n')
    for rname,rscore in display_rouges(best_val_rouges):
        f.write(f'{rname}: {rscore:.5f}\n')
    f.write('\nALL VAL ROUGES:\n')
    for r in all_val_rouges:
        for rname, rscore in display_rouges(r):
            f.write(f'{rname}: {rscore:.5f}\t')
        f.write('\n')

summary_path = join(expdir,'summary.txt')
with open(summary_path,'w') as f:
    f.write(f'Expname: {ARGS.expname}\n')
    f.write(f'captions: {ARGS.caps}\n')
    f.write(f'order: {ARGS.order}\n')
    f.write(f'N Epochs: {ARGS.n_epochs}\n')
    f.write(f'Batch size: {ARGS.bs}\n')
    f.write(f'Startendscenes: {ARGS.startendscenes}\n')
    f.write(f'Centralscenes: {ARGS.centralscenes}\n')
