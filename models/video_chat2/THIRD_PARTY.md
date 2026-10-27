# Third-party code

The retained VideoChat2 backbones originate from
[OpenGVLab/Ask-Anything, video_chat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2),
through the authors' HumanEyeCue experimental source. OpenGVLab's
[MIT license](licenses/OpenGVLab-MIT.txt) is included. Original copyright and
license headers remain in the copied files.

The Q-Former is derived from Salesforce LAVIS/BLIP-2 (Copyright 2023,
salesforce.com, inc.) and Hugging Face's BERT implementation. Its Salesforce
[BSD-3-Clause license](licenses/LAVIS-BSD-3-Clause.txt) and Hugging Face's
[Apache-2.0 license](licenses/Transformers-Apache-2.0.txt) are included.
The custom Llama implementation derives from Hugging Face Transformers and is
covered by the included Apache-2.0 notice. The UMT vision implementation is
retained through VideoChat2 under its upstream MIT terms; it includes the
original sinusoidal-position implementation attribution in its source.

`SOURCE_MANIFEST.json` records the relative original file paths and SHA-256
hashes at extraction. The copied files have release adaptations. `model.py`
consolidates `models/videochat_vicuna/videochat2_it_vicuna.py` and
`models/blip2/blip2.py`; `data.py` consolidates the selected
`eyecue_dataset_clip.py` preprocessing and `json/convert_dataset_mv.ipynb`
conversion; `run.py` consolidates the Gautschi trainers/evaluators and Nano
per-layer NSS evaluation. `optimization.py` retains the source optimizer
grouping and cosine schedule behavior. Experimental alternatives are omitted.

No model weights, tokenizer files, or dataset records are redistributed here.
Users obtain these separately under their respective terms. This addition does
not select or grant a project-wide license for the authors' contributions.
