---
dataset_info:
  features:
  - name: question_id
    dtype: int8
  - name: question
    dtype: string
  - name: image
    dtype: image
  - name: caption
    dtype: string
  - name: gpt_answer
    dtype: string
  - name: category
    dtype: string
  - name: image_id
    dtype: string
  splits:
  - name: train
    num_bytes: 22333678.0
    num_examples: 60
  download_size: 9773451
  dataset_size: 22333678.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---



<p align="center" width="100%">
<img src="https://i.postimg.cc/g0QRgMVv/WX20240228-113337-2x.png"  width="100%" height="80%">
</p>

# Large-scale Multi-modality Models Evaluation Suite

> Accelerating the development of large-scale multi-modality models (LMMs) with `lmms-eval`

🏠 [Homepage](https://lmms-lab.github.io/) | 📚 [Documentation](docs/README.md) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab)

# This Dataset

This is a formatted version of [LLaVA-Bench(wild)](https://llava-vl.github.io/) that is used in LLaVA. It is used in our `lmms-eval` pipeline to allow for one-click evaluations of large multi-modality models.

```
  @misc{liu2023improvedllava,
          author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
          title={Improved Baselines with Visual Instruction Tuning}, 
          publisher={arXiv:2310.03744},
          year={2023},
  }

  @inproceedings{liu2023llava,
    author      = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    title       = {Visual Instruction Tuning},
    booktitle   = {NeurIPS},
    year        = {2023}
  }
```
