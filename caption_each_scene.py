import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from dl_utils.misc import check_dir
import numpy as np
from tqdm import tqdm
from time import time
from os.path import join
from copy import copy
import json
from nltk.corpus import names
from SwinBERT.src.tasks.run_caption_VidSwinBert_inference import inference
from SwinBERT.src.datasets.caption_tensorizer import build_tensorizer
from SwinBERT.src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
import torch
import argparse
from episode import episode_from_epname
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


male_names = names.words('male.txt')
female_names = names.words('female.txt')


def gender(char_name):
    if char_name in male_names and char_name not in female_names:
        return 'm'
    if char_name in female_names and char_name not in male_names:
        return 'f'
    else:
        return 'a'

def maybe_replace_subj(char_name, indeternp, cap):
    if cap.startswith(indeternp):
        return char_name + cap[len(indeternp):]
    else:
        return cap

def get_frames(vid_paths_list,n_frames):
    frames_list = []
    for vp in vid_paths_list:
        new_frames, _ = extract_frames_from_video_path(
                    vp, target_fps=3, num_frames=n_frames,
                    multi_thread_decode=True, sampling_strategy="uniform",
                    safeguard_duration=False, start=None, end=None)

        frames_list.append(new_frames)
    frames = torch.stack(frames_list)
    return frames

class Captioner():
    def init_models(self, model_name):
        print('initializing', model_name)
        if model_name=='swinbert' and not hasattr(self,'swinbert_transformer'):
            self.bert_model, config, self.swin_tokenizer = get_bert_model(do_lower_case=True)
            self.img_res = 224
            self.n_frames = 32
            self.img_seq_len = int((self.n_frames/2)*(int(self.img_res)/32)**2)
            self.max_gen_len = 50

            self.swin_model = get_swin_model(self.img_res, 'base', '600', False, True)
            self.swin_transformer = VideoTransformer(True, config, self.swin_model, self.bert_model)
            self.swin_transformer.freeze_backbone(freeze=False)
            pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cuda'))
            self.swin_transformer.load_state_dict(pretrained_model, strict=False)
            self.swin_transformer.cuda()
            self.swin_transformer.eval()

            self.swin_tensorizer = build_tensorizer(self.swin_tokenizer, 150, self.img_seq_len, self.max_gen_len, is_train=False)

        elif model_name=='kosmos' and not hasattr(self,'kosmos_model'):
            self.kosmos_processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
            self.kosmos_model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
            self.kosmos_model.eval()
            self.kosmos_model.cuda()
        else:
            print(f'unrecognized model name: {model_name}')

    def kosmos_scene_caps(self,epname):
        ep_dir = os.path.join('SummScreen/keyframes',epname)
        scene_caps = []
        scene_locs = []
        n_frames_to_cap = 1
        scene_fnames = sorted(os.listdir(ep_dir), key=lambda x: int(x.split('_')[1][5:]))
        #scene_nums = sorted([int(x.split('_')[1][5:-4]) for x in scene_fnames])
        scene_nums = sorted([int(x.split('.')[0].split('_')[1][5:]) for x in scene_fnames])
        for scene_dir in scene_fnames:
            caps_for_this_scene = []
            locs_for_this_scene = []
            keyframes_files = [fn for fn in os.listdir(join(ep_dir,scene_dir)) if fn != 'middle_frame.jpg']
            keyframes_files = sorted(keyframes_files, key=lambda x: int(x[3:-5])) # by where the appear in scene
            if len(keyframes_files) == 0:
                keyframes_files = [fn for fn in os.listdir(join(ep_dir,scene_dir)) if fn == 'middle_frame.jpg']
                if len(keyframes_files) > 0:
                    print('middle frame to the rescue')
            if len(keyframes_files) > 0:
                select_every = len(keyframes_files)/(n_frames_to_cap+1)
                selected_frame_files = [keyframes_files[int(i*select_every)] for i in range(1,(n_frames_to_cap+1))]
                for fname in selected_frame_files:
                    keyframe = np.array(Image.open(join(ep_dir,scene_dir,fname)))
                    generated_text = self.run_kosmos(keyframe)
                    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
                    cap,l = self.kosmos_processor.post_process_generation(generated_text, cleanup_and_extract=True)
                    caps_for_this_scene.append(cap)
                    locs_for_this_scene.append(l)
            scene_caps.append(' '.join(caps_for_this_scene))
            scene_locs.append(locs_for_this_scene)

        to_dump = []
        for sn,c,l in zip(scene_nums,scene_caps,scene_locs):
            to_append = {'scene_id': f'{epname}s{sn}', 'raw_cap':c, 'locs':l}
            to_dump.append(to_append)
        check_dir(f'SummScreen/video_scenes/{epname}')
        with open(f'SummScreen/video_scenes/{epname}/kosmos_raw_scene_caps.json', 'w') as f:
            json.dump(to_dump,f)

    def swinbert_scene_caps(self,epname):
        scenes_dir = f'SummScreen/video_scenes/{epname}'
        #with open(f'SummScreen/transcripts/{epname}.json') as f:
        #    transcript_data = json.load(f)
        #if not '[SCENE_BREAK]' in transcript_data['Transcript']:
        #    print(f'There doesn\'t appear to be scene break markings for {epname}')
        #    return
        scene_fnames = [x for x in os.listdir(scenes_dir) if x.endswith('mp4')]
        scene_nums = sorted([int(x.split('_')[1][5:-4]) for x in scene_fnames])
        scene_vid_paths = [os.path.join(scenes_dir,f'{epname}_scene{sn}.mp4') for sn in scene_nums]
        scene_caps = []
        for vp in scene_vid_paths:
            frames, _ = extract_frames_from_video_path(
                        vp, target_fps=3, num_frames=self.n_frames,
                        multi_thread_decode=True, sampling_strategy="uniform",
                        safeguard_duration=False, start=None, end=None)
            if frames is None:
                newcap = ['']
                print(f'no scenes detected in {vp}, maybe it\'s v short')
            else:
                newcap = inference(frames, self.img_res, self.n_frames, self.swin_transformer, self.swin_tokenizer, self.swin_tensorizer)
            scene_caps += newcap

        to_dump = []
        for sn,c in zip(scene_nums, scene_caps):
            to_append = {'scene_id': f'{epname}s{sn}', 'raw_cap':c}
            to_dump.append(to_append)
        if to_dump == []:
            breakpoint()
        outpath = f'SummScreen/video_scenes/{epname}/swinbert_raw_scene_caps.json'
        with open(outpath, 'w') as f:
            json.dump(to_dump,f)

    def filter_and_namify_scene_captions(self, epname, model_name):
        scenes_dir = f'SummScreen/video_scenes/{epname}'
        ep = episode_from_epname(epname)
        with open(os.path.join(scenes_dir,f'{model_name}_raw_scene_caps.json')) as f:
            z = json.load(f)
            try:
                raw_caps = [x['raw_cap'] for x in z]
            except KeyError:
                breakpoint()
                raw_caps = [x['raw'] for x in z]
            scene_ids = [x['scene_id'] for x in z]
        #assert len(raw_caps) == len(ep.scenes)
        caps_per_scene = []
        for sid, raw_cap, scene_transcript in zip(scene_ids, raw_caps, ep.scenes):
            if isinstance(scene_transcript, list):
                breakpoint()
            cap = filter_single_caption(raw_cap, scene_transcript)
            caps_per_scene.append({'scene_id': sid, 'raw':raw_cap, 'with_names':cap})

        assert all('talking' not in x['with_names'] for x in caps_per_scene)
        if ARGS.verbose:
            print(f'{sid.upper()}: {raw_cap}\t{cap}')
        with open(f'{scenes_dir}/{model_name}_procced_scene_caps.json','w') as f:
            json.dump(caps_per_scene,f)

    def run_kosmos(self, image):
        inputs = self.kosmos_processor(text='<grounding>A scene from a TV show in which', images=image, return_tensors="pt")
        generated_ids = self.kosmos_model.generate(
            pixel_values=inputs["pixel_values"].cuda(),
            input_ids=inputs["input_ids"][:, :-1].cuda(),
            attention_mask=inputs["attention_mask"][:, :-1].cuda(),
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1].cuda(),
            use_cache=True,
            max_new_tokens=64,
        )
        return self.kosmos_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def filter_single_caption(scene_cap, scene_transcript):
    boring_list = ['a commercial','talking','is shown','sitting on a chair','sitting on a couch', 'sitting in a chair', 'walking around','announcer']
    if any(x in scene_cap for x in boring_list):
        return ''
    appearing_chars = set([x.split(':')[0] for x in scene_transcript.split('\n') if not x.startswith('[') and len(x) > 0 and not x.startswith('Announcer')])

    cap = scene_cap.lower()
    cap = cap.replace('is seen','is').replace('are seen','are')
    if cap.startswith('a scene from a tv show in which'):
        cap = cap[32:]
    appearing_maybe_males = [c for c in appearing_chars if gender(c) in ['m','a']]
    appearing_maybe_females = [c for c in appearing_chars if gender(c) in ['f','a']]

    if len(appearing_maybe_females)==1:
        femname = appearing_maybe_females.pop()
        if 'a woman' in cap:
            cap = cap.replace('a woman',femname, 1)
            if femname in appearing_maybe_males: # could be neut. name
                appearing_maybe_males.remove(femname)
        elif 'a girl' in cap:
            cap = cap.replace('a girl',femname, 1)
            if femname in appearing_maybe_males: # could be neut. name
                appearing_maybe_males.remove(femname)
    if len(appearing_maybe_males)==1:
        manname = appearing_maybe_males.pop()
        if 'a man' in cap:
            cap = cap.replace('a man',manname, 1)
            if manname in appearing_maybe_females:
                appearing_maybe_females.remove(manname)
        elif 'a boy' in cap:
            cap = cap.replace('a boy',manname, 1)
            if manname in appearing_maybe_females:
                appearing_maybe_females.remove(manname)
    if len(appearing_maybe_females)==1: # do again in case neut. removed when checking males
        if 'a woman' in cap:
            cap = cap.replace('a woman',appearing_maybe_females[0], 1)
        elif 'a girl' in cap:
            cap = cap.replace('a girl',appearing_maybe_females[0], 1)
    return cap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--do_filter',action='store_true')
    parser.add_argument('-v','--verbose',action='store_true')
    parser.add_argument('--filter_only',action='store_true')
    parser.add_argument('--refilter',action='store_true')
    parser.add_argument('--epname',type=str, default='oltl-10-18-10')
    parser.add_argument('--show_name',type=str, default='all')
    parser.add_argument('--model_name',type=str, choices=['swinbert','kosmos'], default='kosmos')
    parser.add_argument('--bs',type=int, default=1)
    ARGS = parser.parse_args()


    #bert_model, config, tokenizer_ = get_bert_model(do_lower_case=True)
    #swin_model = get_swin_model(img_res, 'base', '600', False, True)
    #vl_swin_transformer = VideoTransformer(True, config, swin_model, bert_model)
    #vl_swin_transformer.freeze_backbone(freeze=False)
    #pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cpu'))
    #vl_swin_transformer.load_state_dict(pretrained_model, strict=False)
    #vl_swin_transformer.cuda()
    #vl_swin_transformer.eval()

    #tensorizer_ = build_tensorizer(tokenizer_, 150, img_seq_len, max_gen_len, is_train=False)

    captioner = Captioner()
    if not ARGS.filter_only:
        captioner.init_models(ARGS.model_name)
    captioner_func = captioner.kosmos_scene_caps if ARGS.model_name=='kosmos' else captioner.swinbert_scene_caps
    if ARGS.epname == 'all':
        df = pd.read_csv('dset_info.csv',index_col=0)
        all_epnames = df.loc[(df['duration_raw']!='failed video read') & df['has_caps']].index
        #all_epnames = [fn for fn in os.listdir('SummScreen/video_scenes') if fn in os.listdir('SummScreen/keyframes')]
        #all_epnames = [x.split('.')[0] for x in os.listdir('SummScreen/summaries') if os.path.exists('SummScreen/closed_captions/{x}')]
        #all_epnames = [x.split('.')[0] for x in os.listdir('SummScreen/closed_captions')]
        if ARGS.show_name != 'all':
            all_epnames = [x for x in all_epnames if x.startswith(ARGS.show_name)]
        to_caption = []
        for en in all_epnames:
            if (not ARGS.refilter) and os.path.exists(f'SummScreen/video_scenes/{en}/{ARGS.model_name}_procced_scene_caps.json'):
                print(f'scene captions already exist for {en}')
                pass
            else:
                to_caption.append(en)

        for tc in tqdm(to_caption):
            if not ARGS.filter_only:
                captioner_func(tc)
            captioner.filter_and_namify_scene_captions(tc, ARGS.model_name)
    else:
        starttime = time()
        if not ARGS.filter_only:
            captioner_func(ARGS.epname)
        print(f'caption time: {time()-starttime:.2f}')
        starttime = time()
        captioner.filter_and_namify_scene_captions(ARGS.epname, ARGS.model_name)
        print(f'posproc time: {time()-starttime:.2f}')

