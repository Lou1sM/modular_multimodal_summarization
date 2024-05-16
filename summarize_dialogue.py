from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from sklearn.feature_extraction.text import TfidfVectorizer
from dl_utils.misc import check_dir
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from reorder import optimal_order, identical_char_names
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode, episode_from_epname
from utils import chunkify, summ_short_scene, rouge_from_multiple_refs, get_fn
import numpy as np
from tqdm import tqdm


class SoapSummer():
    def __init__(self, model_name, device, bs, dbs, caps, scene_order, uniform_breaks, startendscenes, centralscenes, soft_scene_summs, max_chunk_size, expdir, data_dir, resumm_scenes=False, do_save_new_scenes=False, is_test=False):
        assert not (centralscenes and startendscenes)
        assert isinstance(expdir,str)
        self.device = device
        if is_test:
            self.model_name = self.dmodel_name = 'lucadiliello/bart-small'
        elif startendscenes or centralscenes or soft_scene_summs:
            self.model_name = self.dmodel_name = 'kabita-choudhary/finetuned-bart-for-conversation-summary'
        else:
            self.dmodel_name = 'kabita-choudhary/finetuned-bart-for-conversation-summary'
            self.model_name = model_name
        self.dmodel = AutoModelForSeq2SeqLM.from_pretrained(self.dmodel_name).to(device)
        self.dtokenizer = AutoTokenizer.from_pretrained(self.dmodel_name)
        if self.model_name == self.dmodel_name: # either test, or startend/central
            self.model = self.dmodel # avoid loading twice
            self.tokenizer = self.dtokenizer # avoid loading twice
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if caps.endswith('-only'):
            self.caps = caps[:-5]
            self.caps_only = True
        else:
            self.caps = caps
            self.caps_only = False
        self.expdir = expdir
        self.data_dir = data_dir
        self.n_epochs = 0
        self.scene_order = scene_order
        self.uniform_breaks = uniform_breaks
        self.startendscenes = startendscenes
        self.centralscenes = centralscenes
        self.soft_scene_summs = soft_scene_summs
        self.max_chunk_size = max_chunk_size
        self.resumm_scenes = resumm_scenes
        self.do_save_new_scenes = do_save_new_scenes
        self.is_test = is_test
        self.bs = bs
        self.dbs = dbs
        self.fn = get_fn(caps, self.scene_order, self.uniform_breaks, self.startendscenes, self.centralscenes, self.soft_scene_summs, self.is_test)

    def pad_batch(self,batch,tokenizer):
        N=max([len(c) for c in batch])
        attention_mask = torch.stack([torch.cat([torch.ones(len(c)),torch.zeros(N-len(c))]) for c in batch], dim=0).to(self.device)
        padded = [b+[tokenizer.eos_token_id]*(N-len(b)) for b in batch]
        padded = torch.tensor(padded).to(self.device)
        assert padded.shape == attention_mask.shape
        return padded, attention_mask

    def summ_scenes(self, epname, infer_splits):
        ep = episode_from_epname(epname, infer_splits)
        scenes = ['']*len(ep.scenes) if self.caps_only else ep.scenes
        max_chunk_size = min(self.max_chunk_size, self.dtokenizer.model_max_length)
        if len(scenes) == 1:
            print(f'no scene breaks for {epname}')
            breakpoint()
        if self.caps == 'nocaptions':
            caps = ['']*len(scenes)
        else: # prepend vid caps to the scene summ
            with open(join(self.data_dir,f'video_scenes/{epname}/{self.caps}_procced_scene_caps.json')) as f:
                caps_data = json.load(f)
            cdd = {c['scene_id']:c['with_names'] for c in caps_data}
            caps = [cdd.get(f'{epname}s{i}','') for i in range(len(scenes))]
            if not len(caps)==len(scenes):
                breakpoint()
        assert all('talking' not in x for x in caps)
        if self.scene_order=='optimal':
            order_idxs = optimal_order(scenes)
            optimally_ordered_scenes = [scenes[oi] for oi in order_idxs[:-1]]
            optimally_ordered_caps = [caps[oi] for oi in order_idxs[:-1]]
            combined_scenes = [optimally_ordered_scenes[0]]
            combined_caps = [optimally_ordered_caps[0]]
            for optscene, optcap in zip(optimally_ordered_scenes[1:],optimally_ordered_caps[1:]):
                if identical_char_names(optscene, combined_scenes[-1]):
                    combined_scenes[-1]+=optscene.lstrip()
                    combined_caps[-1]+=optcap.lstrip()
                else:
                    combined_scenes.append(optscene)
                    combined_caps.append(optcap)
        elif self.scene_order=='rand':
            idxs = sorted(range(len(scenes)), key=lambda x: np.random.rand())
            combined_scenes = [scenes[ri] for ri in idxs]
            combined_caps = [caps[ri] for ri in idxs]
        elif self.uniform_breaks:
            transcript_wo_scene_marks = '\n'.join([x for x in ep.transcript if x!='[SCENE_BREAK]'])
            combined_scenes = chunkify(transcript_wo_scene_marks, max_chunk_size)
            combined_caps = caps
        else:
            combined_scenes = scenes
            combined_caps = caps

        if self.startendscenes:
            start = ''
            end = ''
            newend = ''
            startidx = 0
            endidx = 0
            while True:
                if startidx == endidx:
                    newstart = start + scenes[startidx]
                    startidx += 1
                else:
                    assert startidx == endidx+1
                    newend = scenes[-endidx-1] + end
                    endidx += 1
                if len((newstart+newend).split()) > 3/4*self.tokenizer.model_max_length:
                    break
                else:
                    start, end = newstart, newend
            assert len((start+end).split())*4/3 <= self.tokenizer.model_max_length
            return start + end
        elif self.centralscenes:
            vect = TfidfVectorizer(min_df=1, stop_words='english')
            tfidf = vect.fit_transform(combined_scenes)
            scene_sims = (tfidf * tfidf.T).A
            np.fill_diagonal(scene_sims, -1)
            y = scene_sims
            best_scene_idxs = y.sum(axis=0).argsort()[::-1]
            for j in range(100):
                y = np.matmul(y,y.T)
                new_best_scene_idxs = y.sum(axis=0).argsort()[::-1]
                if (new_best_scene_idxs == best_scene_idxs).all():
                    break
                else:
                    best_scene_idxs = new_best_scene_idxs
            for go_up_to in range(len(best_scene_idxs)):
                if sum(len(scenes[sidx].split()) for sidx in best_scene_idxs[:go_up_to+1])*4/3 > self.dtokenizer.model_max_length:
                    break # next-one-up would push it over the edge
            idxs_to_use = sorted(best_scene_idxs[:go_up_to])

            best_in_order = [scenes[i] for i in idxs_to_use]
            assert sum(len(x.split()) for x in best_in_order)*4/3 <= self.dtokenizer.model_max_length
            return best_in_order
            #scene_sims = tfidf_sims(scenes)

        if self.caps_only:
            return combined_caps
        chunk_list = [chunkify(s, max_chunk_size) for s in combined_scenes]
        chunks = sum(chunk_list,[])
        assert (chunks==combined_scenes) or not self.uniform_breaks
        avg_scene_summ_len = self.tokenizer.model_max_length//len(chunks)

        tok_chunks = [self.dtokenizer(c)['input_ids'] for c in chunks]
        sort_idxs = np.argsort([len(x) for x in tok_chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_chunks = [chunks[i] for i in sort_idxs]
        sorted_tok_chunks = [tok_chunks[i] for i in sort_idxs]
        v_short_chunk_idxs = [i for i,sc in enumerate(sorted_tok_chunks) if len(sc) < avg_scene_summ_len]
        n_shorts = 0 if self.soft_scene_summs else len(v_short_chunk_idxs)
        assert self.soft_scene_summs or (v_short_chunk_idxs == list(range(n_shorts)))
        short_chunk_summs = [summ_short_scene(sc) for sc in sorted_chunks[:n_shorts]]
        remaining_chunks = sorted_tok_chunks[n_shorts:]
        assert all([sorted_tok_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(tok_chunks)])
        N = ceil(len(remaining_chunks)/self.dbs)
        remaining_chunk_summs = []
        for i in range(N):
            padded, attn = self.pad_batch(remaining_chunks[i*self.dbs:(i+1)*self.dbs],self.dtokenizer)
            max_len = min(padded.shape[1],avg_scene_summ_len+15)
            min_len = max(10,max_len-20)
            if padded.shape[1] > self.dtokenizer.model_max_length:
                #print('too long', padded.shape, self.dtokenizer.model_max_length)
                padded = padded[:,:self.dtokenizer.model_max_length]
                attn = attn[:,:self.dtokenizer.model_max_length]
            if self.soft_scene_summs:
                soft_tokens = self.soft_forward(padded, attention_mask=attn, n_new_tokens=max_len)
                summ = list(soft_tokens) # summ is a list of scene summaries in permuted order, will be depermuted later
            else:
                summ_tokens = self.dmodel.generate(padded, attention_mask=attn, min_length=min_len, max_length=max_len)
                assert summ_tokens.shape[1] <= max_len
                summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            len_first_unpadded = attn[0].argmin()
            if len_first_unpadded==0:
                assert attn[0].all()
                len_first_unpadded = attn.shape[1]
            remaining_chunk_summs += summ
        chunk_summs = short_chunk_summs + remaining_chunk_summs

        # return chunks to their original order
        desorted_chunk_summs = [chunk_summs[i] for i in reversed_sort_idxs]
        count = 0
        desplit = []
        # recombine scenes whose dialogue was split because of context size
        for cl in chunk_list: # take lens from original list, before it was sorted
            if self.soft_scene_summs:
                desplit.append(torch.cat(desorted_chunk_summs[count:count+len(cl)]))
            else:
                desplit.append(' '.join(desorted_chunk_summs[count:count+len(cl)]))
            count+=len(cl)
        if self.soft_scene_summs:
            return torch.cat(desplit)
        assert (desplit==desorted_chunk_summs) or (set([len(x) for x in chunk_list])!=set([1]))
        # if some were chunked together, may differ because of the join
        ss_with_caps = [f'{sc} {x}' for sc,x in zip(combined_caps,desplit)]
        if self.caps == 'nocaptions':
            assert self.tokenizer.model_max_length + 15*len(chunks) >= len(self.dtokenizer(''.join(ss_with_caps))[0])
        return ss_with_caps

    def get_scene_summs(self, epname, infer_splits):
        epname_ = f'{epname}-inferred' if infer_splits else epname
        maybe_scene_summ_path = join(self.data_dir,f'scene_summs/{epname_}_{self.fn}.pt') if self.soft_scene_summs else join(self.data_dir,f'scene_summs/{epname_}_{self.fn}.txt')
        if os.path.exists(maybe_scene_summ_path) and not self.resumm_scenes:
            if self.soft_scene_summs:
                ss = torch.load(maybe_scene_summ_path)
            else:
                with open(maybe_scene_summ_path) as f:
                    ss = [x.strip() for x in f.readlines()]
        else:
            ss = self.summ_scenes(epname, infer_splits)
            if self.do_save_new_scenes:
                if self.soft_scene_summs:
                    torch.save(ss, maybe_scene_summ_path)
                else:
                    with open(maybe_scene_summ_path,'w') as f:
                        f.write('\n'.join(ss))
                #print('saving to',fpath)
        return ss

    def summarize_from_epname(self, epname):
        scene_summs = self.get_scene_summs(epname)
        return self.summarize_scene_summs('\n'.join(scene_summs))

    def summarize_scene_summs(self, concatted_scene_summs):
        max_chunk_size = min(self.max_chunk_size, self.tokenizer.model_max_length)
        chunks = chunkify(concatted_scene_summs, max_chunk_size)
        tok_chunks = [self.tokenizer(c)['input_ids'] for c in chunks]
        pbatch, attn = self.pad_batch(tok_chunks,self.tokenizer)
        #if (self.caps=='nocaptions') and (pbatch.shape[1] > self.tokenizer.model_max_length):
        pbatch = pbatch[:,:self.tokenizer.model_max_length]
        attn = attn[:,:self.tokenizer.model_max_length]
        if self.caps_only:
            min_len = 80
            max_len = 100
        else:
            min_len = 360//len(chunks)
            max_len = 400//len(chunks)
        meta_chunk_summs = self.model.generate(pbatch, attention_mask=attn, min_length=min_len, max_length=max_len)
        final_summ = ' '.join(self.tokenizer.batch_decode(meta_chunk_summs,skip_special_tokens=True))
        print(len(self.tokenizer(final_summ)['input_ids']))
        return concatted_scene_summs, final_summ

    def dpoints_from_epnames(self, epname_list, scene_caps, infer_splits):
        assert not any(['reordered' in x for x in epname_list])
        data_list = []
        summ_dir = 'SummScreen/summaries'
        pbar = tqdm(epname_list)
        for epname in pbar:
            pbar.set_description(epname)
            unjoined_scene_summs = self.get_scene_summs(epname, infer_splits)
            if self.soft_scene_summs:
                #ss = torch.cat(unjoined_scene_summs)
                ss = unjoined_scene_summs
            else:
                ss = '\n'.join(unjoined_scene_summs)
            with open(os.path.join(summ_dir, f'{epname}.json')) as f:
                d = json.load(f)
            if len(d.items())==0:
                breakpoint()
            for k,v in d.items():
                if '[ RECAP AVAILABLE ]' in v or 'Episode summary coming soon.' in v:
                    continue
                assert (k=='tvmega_summary') == (v.startswith('Episode'))
                if self.soft_scene_summs:
                    ss = join(self.data_dir,f'scene_summs/{epname}_{self.fn}.pt')
                if len(v) > 0 and k not in ['soap_central','tvmega_summary']:
                    data_list.append({'scene_summs':ss, 'summ':v, 'summ_name':k, 'epname':epname})
        return data_list

    def build_dset(self, scene_caps, n_dpoints, dset_fragment_name):
        dset_info = pd.read_csv('dset_info.csv', index_col=0)
        base_dset_fragment_name = dset_fragment_name.removesuffix('-inferred')
        mask = dset_info['usable'] & (dset_info['split']==base_dset_fragment_name)
        epnames = dset_info.index[mask]
        epname_to_be_first = 'oltl-10-18-10'
        if n_dpoints != -1:
            if base_dset_fragment_name in ('val', 'test'):
                ndps_to_use = max(2,int(n_dpoints/10))
            else:
                assert dset_fragment_name == 'train'
                ndps_to_use = n_dpoints - 2*max(2,int(n_dpoints/10))
            assert ndps_to_use >= 2
            epnames = epnames[:ndps_to_use]
        if base_dset_fragment_name == 'test':
            epnames.insert(0, epname_to_be_first)
        else:
            epnames = [x for x in epnames if x!=epname_to_be_first]

        assert all([os.path.isdir(join(self.data_dir,f'video_scenes/{x}')) for x in epnames])
        infer_splits = dset_fragment_name.endswith('-inferred')
        dpoints = self.dpoints_from_epnames(epnames, scene_caps, infer_splits)
        if base_dset_fragment_name in ('val', 'test'):
            todump = []
            for tepn in epnames:
                dps_with_name = [t for t in dpoints if t['epname']==tepn]
                assert all(d['scene_summs']==dps_with_name[0]['scene_summs'] for d in dps_with_name[1:])
                combined = {'epname':tepn, 'scene_summs': dps_with_name[0]['scene_summs']}
                for dpsn in dps_with_name:
                    combined[dpsn['summ_name']] = dpsn['summ']
                todump.append(combined)
        else:
            todump = dpoints
        with open(join(self.data_dir,f'json_datasets/{self.fn}_{n_dpoints}dps_{dset_fragment_name}_dset.json','w')) as f:
            json.dump(todump, f)

    def train_one_epoch(self, epoch, trainloader, no_pbar):
        self.model.train()
        self.opt.zero_grad()
        epoch_loss = 0
        if no_pbar:
            trainiter = trainloader
        else:
            trainiter = tqdm(trainloader, dynamic_ncols=True, smoothing=0.01, leave=False)
        for i,batch in enumerate(trainiter):
            if (batch['input_ids'].shape[1]) > self.tokenizer.model_max_length*6/5 and not self.is_test:
                continue
            else:
                batch['input_ids'] = batch['input_ids'][:,:self.tokenizer.model_max_length]
                batch['attention_mask'] = batch['attention_mask'][:,:self.tokenizer.model_max_length]
                assert ('inputs_embeds' in batch.keys()) == self.soft_scene_summs
                if self.soft_scene_summs:
                    batch['inputs_embeds'] = batch['inputs_embeds'][:,:self.tokenizer.model_max_length]
            if (batch['labels'].shape[1]) > self.tokenizer.model_max_length:
                continue
            if max(batch['input_ids'].shape[1], batch['labels'].shape[1], batch['decoder_input_ids'].shape[1]) > self.tokenizer.model_max_length:
                breakpoint()
            cbatch = {k:v.to(self.device) for k,v in batch.items()}
            cbatch['labels'] = cbatch['labels'].contiguous()
            if self.soft_scene_summs:
                cbatch['input_ids'] = None
            try:
                outputs = self.model(**cbatch)
                loss = outputs[0]
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                x=batch['input_ids'].shape[1]
                y=batch['labels'].shape[1]
                print(f'got OOM with inputs {x} and labels {y}')
                continue
            epoch_loss = ((i*epoch_loss) + loss.item()) / (i+1) # running avg
            if not no_pbar:
                trainiter.set_description(f'Epoch: {epoch}/{self.n_epochs}'
                                 f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
            self.opt.step(); self.lr_scheduler.step()
            if i==10 and self.is_test:
                break
        return epoch_loss

    def save_to(self, fname):
        save_dir = join(self.expdir, 'checkpoints', fname)
        print('saving checkpoint to',save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def inference_epoch(self, epoch_num, dset, dset_fragment_name):
        self.model.eval()
        print('validating')
        prev = ''
        rouges = []
        epoch_rouge = np.zeros(4)
        check_dir(generations_dir := join(self.expdir, f'generations_{dset_fragment_name}'))
        for j,batch in enumerate(pbar := tqdm(dset, dynamic_ncols=True, smoothing=0.01, leave=False)):
            nl_inputs = batch['scene_summs']
            _, nl_outputs = self.summarize_scene_summs(nl_inputs)
            if (nl_outputs[:100] == prev[:100]):# and not (prev_inp[:100] == batch['input_ids'][:100]):
                print('repeat output')
            prev = nl_outputs
            references = [v for k,v in batch.items() if k not in ('epname','scene_summs') and v is not None]
            best_rouge = rouge_from_multiple_refs(nl_outputs, references, return_full=False, benchmark_rl=True)

            rouges.append(best_rouge)
            epoch_rouge = ((j*epoch_rouge) + best_rouge) / (j+1) # running avg
            pbar.set_description(f'Epoch: {epoch_num}/{self.n_epochs}'
                             f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f} {best_rouge[3]:.3f}  '
                             f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f} {epoch_rouge[3]:.3f}')
            epname = batch['epname']
            with open(f'{generations_dir}/{epname}.txt','w') as f:
                f.write(nl_outputs)
            if j==2 and self.is_test:
                break
        return np.array(rouges).mean(axis=0)

    def train_epochs(self, n_epochs, trainset, valset, testset, no_pbar, early_stop_metric):
        self.opt = AdamW(self.model.model.decoder.parameters(),lr=1e-6)
        dc = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainloader = DataLoader(trainset, batch_size=self.bs, shuffle=True, collate_fn=dc)
        self.n_epochs = n_epochs
        num_training_steps = self.n_epochs * len(trainloader)
        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.opt, num_warmup_steps=0, num_training_steps=num_training_steps)
        patience = 0
        alltime_best_rouges = np.zeros(4)
        all_rouges = []
        for epoch in range(self.n_epochs):
            print(f'training epoch {epoch}')
            epoch_loss = self.train_one_epoch(epoch, trainloader, no_pbar)
            print(f'Epoch: {epoch}\tLoss: {epoch_loss:.5f}')
            rouges_arr = self.inference_epoch(epoch, valset, 'val')
            #rouges = self.inference_epoch(epoch, valset, 'val')
            #rouges_arr = np.array(rouges).mean(axis=0)
            if len(all_rouges)>0 and (rouges_arr==all_rouges[-1]).all():
                print('WARNING: rouge unchanged since last epoch')
            else:
                assert not any((r==rouges_arr).all() for r in all_rouges)
            all_rouges.append(rouges_arr)
            if rouges_arr[early_stop_metric] > alltime_best_rouges[early_stop_metric]:
                patience = 0
                alltime_best_rouges = rouges_arr
                self.save_to('best')
            else:
                patience += 1
            print(f'Mean Rouge: {rouges_arr}\tPatience: {patience}')
            if patience == 2:
                break
        if self.n_epochs>0:
            best_chkpt = f'{self.expdir}/checkpoints/best'
            print('reloading', best_chkpt)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(best_chkpt).to(self.device)
        test_rouges = self.inference_epoch(self.n_epochs, testset, 'test')
        return test_rouges, alltime_best_rouges, all_rouges

    def soft_forward(self, input_ids, attention_mask, n_new_tokens):
        encoder_outputs = self.model.model.encoder(input_ids)[0]
        bos_token_id = 2
        bs = input_ids.shape[0]
        bos_token_id_tensor = torch.ones((bs, 1), dtype=torch.long, device=self.device)*bos_token_id
        embeds = self.model.model.decoder.embed_tokens(bos_token_id_tensor)
        #for i in range(n_new_tokens):
        for i in range(10):
            dec_out = self.model.model.decoder(input_ids=None, inputs_embeds=embeds, encoder_hidden_states=encoder_outputs, encoder_attention_mask=attention_mask)
            hiddens_for_each_word = dec_out.last_hidden_state
            hidden_for_last_word = hiddens_for_each_word[:,-1:,:] # vec that would normally be used to select next word
            embeds = torch.cat([embeds, hidden_for_last_word], dim=1)
        embeds = embeds.detach().cpu()
        return embeds

    def soft_decode(self, x):
        if x.ndim != 1:
            assert x.ndim == 2
            x = x.transpose(0,1)
        token_embed_logits = self.model.lm_head.weight @ x.to(self.device)
        nearest_token_embeds = token_embed_logits.argmax(axis=0)
        soft_decoding = self.tokenizer.decode(nearest_token_embeds)
        return soft_decoding
