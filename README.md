Code for the paper [A Modular Approach for Multimodal Summarization of TV Shows](https://arxiv.org/abs/2403.03823), applied to the SummScreen3D dataset. 

## Data
The dataset consists of text transcripts and videos for tv show episodes, which are to be summarized as text. Our method first converts the videos to text as video captions. 

### Precomputed Captions
We include the video captions in this repo so you can run our model without access to the videos themselves. The captions are at 

`SummScreen/{episode-name}/{caption-method}_procced_scene_caps.json`,

where `caption-method` is one of 'kosmos' or 'swinbert', the two methods used in the paper.

### From Scratch 
If you want to process the videos from scratch, you can download them (>600GB) from <https://github.com/ppapalampidi/long_video_summarization>, and use the authors' public code for [kosmos](https://github.com/microsoft/unilm/tree/master/kosmos-2) or [swinbert](https://github.com/microsoft/SwinBERT). In that case, you must first split the videos into scenes, and then produce captions for each scene. To do this, use, in order `preproc/align_vid_and_transcript.py`, `preproc/frame_extractor.py` (for kosmos) and `preproc/caption_each_scene.py`.

## Model
To reproduce our model output, run 

`python train.py --caps kosmos --order optimal --n_epochs 10`.

## PREFS Metric
Our paper also introduces a new metric for factual precision and recall evaluation of summaries. This method makes multiple api calls to GPT4. To run it, place your openai api key at `PREFS/api.key`, then, assuming your output summaries are at `experiments/{experiment-name}/generations_text`, with each summary in a separate file, run

`python compute_metrics.py --expname {experiment-name}`.

## Citation
```
@article{mahon2024modular,
  title={A Modular Approach for Multimodal Summarization of TV Shows},
  author={Mahon, Louis and Lapata, Mirella},
  journal={arXiv preprint arXiv:2403.03823},
  year={2024}
}
```
