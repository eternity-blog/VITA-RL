# VITA-RL: Extending VITA-1.5 with Reinforcement Learning

> [!IMPORTANT]
> **This is not the official VITA repository.**
>
> This project is a fork of and an extension to [**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA)
> (VITA-1.5: *Towards GPT-4o Level Real-Time Vision and Speech Interaction*).
>
> - **Upstream repository**: https://github.com/VITA-MLLM/VITA
> - **Baseline commit**: [`35d064a`](https://github.com/VITA-MLLM/VITA/commit/35d064a6542a5d812136fcd66fa93d9beb27b03c) (2025-03-28)
> - **Upstream paper**: [VITA-1.5 (arXiv:2501.01957)](https://arxiv.org/pdf/2501.01957)
>
> All model architecture, training recipes, benchmark numbers, and pretrained weights
> described below are the work of the original **VITA team (Tencent Youtu Lab et al.)**.
> This repository adds no claim over them. Please cite the
> [original papers](#️-citation) and respect the [original license](./License.txt),
> which restricts use to **academic, research and educational purposes only**.

<p align="center">
    <img src="./asset/vita_newlog.jpg" width="100%" height="100%">
</p>

<font size=7><div align='center' > [[📖 VITA-1.5 Paper](https://arxiv.org/pdf/2501.01957)] [[🏠 Upstream Repo](https://github.com/VITA-MLLM/VITA)] [[🤖 Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)] [[🍎 VITA-1.0](https://vita-home.github.io/)]</div></font>

---

## 🎯 About This Fork

The goal of this repository is to **reproduce VITA-1.5 end-to-end**, and then to
**extend it with a reinforcement learning stage**, which upstream does not provide
(the original codebase contains only supervised fine-tuning).

### Roadmap

| Stage | Status | Description |
|---|---|---|
| 1. Reproduce inference | 🚧 In progress | Run the quick-start and web demos against the released VITA-1.5 checkpoint |
| 2. Reproduce training | 📋 Planned | Run continual training (Stage-3 `finetuneTaskNeg`) on a prepared dataset |
| 3. Add RL | 📋 Planned | Add a preference-optimization / RL stage on top of the SFT model |

### Changes relative to upstream

At the moment this fork is **functionally identical to upstream `35d064a`**, plus:

- Added a `.gitignore` (upstream has none) covering training outputs, model weights and secrets.
- Rewrote this README to attribute the work to the upstream project and to document the fork's goals.

Any further deviation from upstream will be recorded in this section.

> **Note on reproduction.** The upstream scripts contain hard-coded absolute paths from the
> original authors' cluster (`/mnt/cfs/lhj/...`), hard-coded multi-node addresses, and an
> empty dataset registry. These must be adapted locally before anything will run — see
> [Reproduction Notes](#-reproduction-notes).

---

<p align="center">
    <img src="./asset/vita_demo.jpg" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📽 VITA-1.5 Demo Show 🔥](https://youtu.be/tyi6SVFT5mM?si=fkMQCrwa5fVnmEe7)] </div></font>  
<font size=7><div align='center' > VITA-1.5 supports both **English** and **Chinese**.🌟 </div></font>  
You can try the upstream [Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo) on ModelScope directly. The Real-Time Interactive Demo needs to be configured according to the [instructions](#-real-time-interactive-demo).

## 🔥 Upstream News

*The following milestones are from the original VITA project.*

* **`2025.01.17`** 🌟 ModelScope has supported VITA-1.5! You can try the [Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo) on it!
* **`2025.01.06`** 🌟 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) of OpenCompass has supported both VITA-1.5 and VITA-1.0 models!
* **`2025.01.06`** 🌟 The [technical report](https://huggingface.co/VITA-MLLM) of VITA-1.5 has been released!
* **`2024.12.20`** 🌟 The VITA team introduced **VITA-1.5**, a more powerful and more real-time version!
* **`2024.08.12`** 🌟 The VITA team launched **VITA-1.0**, the first-ever open-source interactive omni multimodal LLM!


## Contents <!-- omit in toc -->

- [VITA-RL: Extending VITA-1.5 with Reinforcement Learning](#vita-rl-extending-vita-15-with-reinforcement-learning)
  - [🎯 About This Fork](#-about-this-fork)
  - [🔥 Upstream News](#-upstream-news)
  - [👀 VITA-1.5 Overview](#-vita-15-overview)
    - [🌟 What’s New in VITA-1.5?](#-whats-new-in-vita-15)
  - [📈 Experimental Results](#-experimental-results)
  - [🛠 Reproduction Notes](#-reproduction-notes)
  - [⭐ Training](#-training)
    - [Requirements and Installation](#requirements-and-installation)
    - [Data Preparation](#data-preparation)
    - [Continual Training](#continual-training)
  - [📐 Inference](#-inference)
    - [Quick Start](#quick-start)
    - [Demo](#demo)
      - [📍 Basic Demo](#-basic-demo)
      - [📍 Real-Time Interactive Demo](#-real-time-interactive-demo)
  - [📏Evaluating on MLLM Benchmarks](#evaluating-on-mllm-benchmarks)
    - [VLMEvalKit](#vlmevalkit)
    - [Video-MME](#video-mme)
      - [Data Preparation](#data-preparation-1)
      - [Evaluation](#evaluation)
  - [✒️ Citation](#️-citation)
  - [📣 Statement](#-statement)
  - [📜 Related Works](#-related-works)
  - [👍 Acknowledgement](#-acknowledgement)



## 👀 VITA-1.5 Overview

*This section describes the upstream model. All results below are reported by the original authors.*

On 2024.08.12, the VITA team launched **VITA-1.0**, the **first-ever open-source interactive omni-multimodal LLM**. On 2024.12.20, they released **VITA-1.5**.

### 🌟 What’s New in VITA-1.5?

**VITA-1.5** incorporates a series of advancements:

1. **Significantly Reduced Interaction Latency**. The end-to-end speech interaction latency has been reduced from about **4 seconds** to **1.5 seconds**, enabling near-instant interaction and greatly improving user experience.  

2. **Enhanced Multimodal Performance**.  The average performance on multimodal benchmarks such as *MME*, *MMBench*, and *MathVista* has been significantly increased from **59.8** to **70.8**.

3. **Improvement in Speech Processing**. The speech processing capabilities have been refined to a new level, with ASR WER (Word Error Rate, Test Other) reduced from **18.4** to **7.5**. Besides, we replace the independent TTS module of VITA-1.0 with an **end-to-end TTS module**, which accepts the LLM's embedding as input.  

4. **Progressive Training Strategy**. By this manner, the adding of speech has little effect on other multi-modal performance (vision-language). The average image understanding performance only drops from 71.3 to 70.8.


## 📈 Experimental Results

*All numbers below are reported by the upstream VITA team in the [VITA-1.5 paper](https://arxiv.org/pdf/2501.01957). They have **not** been independently re-measured in this fork; reproduction results will be added here as they become available.*

- **Evaluation on image and video understanding benchmarks.**

<p align="center">
    <img src="./asset/vita_mllm_performance.png" width="100%" height="100%">
</p>

- **VITA-1.5 outperforms professional speech models on ASR benchmarks.**

<p align="center">
    <img src="./asset/vita_15_audio_2.jpg" width="96%" height="96%">
</p>

- **Adding the audio modality has little effect on image and video understanding capability**.

<p align="center">
    <img src="./asset/vita_15_audio_training.png" width="68%" height="50%">
</p>

## 🛠 Reproduction Notes

*This section is specific to this fork and is not part of the upstream README.*

The upstream code was released as-is from the authors' internal cluster. The following
must be adapted before anything will run — none of these are bugs, they are simply
environment-specific values that were never parameterised:

1. **Hard-coded absolute paths.** Every script under `script/train/` references
   `/mnt/cfs/lhj/...`, `/mnt/cfs2/lhj/...` or `/mnt/shared/data1/lhj/...` for model
   weights and outputs. `GLOBAL_WEIGHTS_PATH` in
   [`vita/constants.py`](./vita/constants.py) is still the placeholder
   `/path/to/model_weights`.

2. **Hard-coded multi-node settings.** The `*_nodes.sh` scripts pin `INDEX` (node rank)
   and `MASTER_ADDR` to the authors' cluster, e.g. `INDEX=3` and
   `MASTER_ADDR="10.206.0.199"` in `finetuneTaskNeg_qwen_nodes.sh`. Each node needs a
   distinct `INDEX`. NCCL variables (`NCCL_SOCKET_IFNAME=eth0`, `NCCL_IB_GID_INDEX=3`)
   assume a specific interconnect.

3. **Empty dataset registry.** [`vita/config/dataset_config.py`](./vita/config/dataset_config.py)
   ships with empty strings for `AudioFolder`, `FolderDict` and `chat_path`. In addition,
   `DataConfig` in [`vita/config/__init__.py`](./vita/config/__init__.py) only defines the
   key `Pretrain_video`, while several scripts pass `--dataset_use Pretrain_video0` or
   `Pretrain_audio`; those keys must be added or the run fails with a `KeyError`.

4. **The data pipeline is selected in source, not on the CLI.** `train.py` imports one of
   seven `data_utils_*` variants via commented-out import lines near the top of
   [`vita/train/train.py`](./vita/train/train.py). The default (`..._neg_patch`) matches
   the documented continual-training recipe.

5. **Pinned, older dependencies.** `torch==2.3.1` and `transformers==4.41.1`.
   `vita/model/language_model/vita_qwen2.py` monkey-patches `Qwen2ForCausalLM.forward`,
   which couples it tightly to that `transformers` version — upgrading is likely to break it.

6. **`command.sh` is not a build script.** It is the original authors' scratch command
   history and references files that no longer exist in the repository. Do not use it as
   an entry point.

## ⭐ Training

*The recipe below is the upstream training procedure, reproduced here for convenience.*

### Requirements and Installation
```
git clone https://github.com/eternity-blog/VITA-RL
cd VITA-RL
conda create -n vita python=3.10 -y
conda activate vita
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Data Preparation
- An example json file of the training data:
```
[
    ...
    {
        "set": "sharegpt4",
        "id": "000000000164",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<audio>\n"
            },
            {
                "from": "gpt",  // follow the setting of llave, "gpt" is only used to indicate that this is the ground truth of the model output
                "value": "This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere. "
            }
        ],
        "image": "coco/images/train2017/000000000164.jpg",
        "audio": [
            "new_value_dict_0717/output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"
        ]
    },
    ...
]
```

- The `set` field is used to retrieve the image or video folder for data loading. You should add its key-value pair to the `FolderDict` in [./vita/config/dataset_config.py](./vita/config/dataset_config.py):
```
AudioFolder = ""
FolderDict = {
    #### NaturalCap
    "sharegpt4": "",
}
#### NaturalCap
ShareGPT4V = {"chat_path": ""}
```

- Set the JSON path for `"chat_path"` in the corresponding dictionary in [./vita/config/dataset_config.py](./vita/config/dataset_config.py).
- Set the audio folder path for `AudioFolder` in [./vita/config/dataset_config.py](./vita/config/dataset_config.py).
- Add the data class in `DataConfig` in [./vita/config/init.py](./vita/config/__init__.py):
```
from .dataset_config import *

NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap,
}
```


### Continual Training
- Download the required weights (all released by the upstream VITA team): (1) [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main), (2) [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px), and (3) [the pretrained audio encoder](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning) from Stage-2 audio-language alignment (refer to Fig. 3 in the paper).

- Replace the paths in [./script/train/finetuneTaskNeg_qwen_nodes.sh](./script/train/finetuneTaskNeg_qwen_nodes.sh):
```
    ...
    --model_name_or_path VITA1.5_ckpt \
    ...
    --vision_tower InternViT-300M-448px \
    ...
    --audio_encoder audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    ...
```

- Execute the following commands to start the training process (set `OUTPUT_DIR` to a path on your own machine):

```
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/path/to/your/outputs/vita_video_audio
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
```


## 📐 Inference
### Quick Start
- Text query
```
CUDA_VISIBLE_DEVICES=2 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this images."
```

- Audio query
```
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav
```

-  Noisy audio query
```
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav
```


### Demo

We have accelerated the model using [vLLM](https://github.com/vllm-project/vllm). 
Since VITA has not yet been integrated into vLLM, you need to make some modifications to the vLLM code to adapt it for VITA.


```bash
conda create -n vita_demo python==3.10
conda activate vita_demo
pip install -r web_demo/web_demo_requirements.txt

# Backup a new weight file
cp -rL  VITA_ckpt/ demo_VITA_ckpt/

mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json

cd ./web_demo/vllm_tools
cp -rf qwen2p5_model_weight_file/*  ../../demo_VITA_ckpt/
cp -rf vllm_file/*  your_anaconda/envs/vita_demo/lib/python3.10/site-packages/vllm/model_executor/models/
```




#### 📍 Basic Demo

https://github.com/user-attachments/assets/43edd44a-8c8d-43ea-9d2b-beebe909377a



```bash
python -m web_demo.web_ability_demo  demo_VITA_ckpt/
```



#### 📍 Real-Time Interactive Demo

To run the real-time interactive demo, you need to make the following preparations:

- Make sure that you have executed the above instructions under the [Demo](#demo) section (`cp` files out from the `vllm_tools`).

- Prepare a VAD (Voice Activity Detection) module. 
You can choose to download [silero_vad.onnx](https://github.com/snakers4/silero-vad/tree/v4.0/files) and [silero_vad.jit](https://github.com/snakers4/silero-vad/tree/v4.0/files), and place these files in the `./web_demo/wakeup_and_vad/resource/` directory.

- For a better real-time interactive experience, you need to set `max_dynamic_patch` to 1 in `demo_VITA_ckpt/config.json`. 
When you run the basic demo, you can set it to the default value of 12 to enhance the model's visual capabilities.

```bash
pip install flask==3.1.0 flask-socketio==5.5.0 cryptography==44.0.0 timm==1.0.12
python -m web_demo.server --model_path demo_VITA_ckpt --ip 0.0.0.0 --port 8081
```


## 📏Evaluating on MLLM Benchmarks
### [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
Modify the model path of `vita_qwen2` in `VLMEvalKit/vlmeval/config.py`
```
vita_series = { 
    'vita': partial(VITA, model_path='/path/to/model'),
    'vita_qwen2': partial(VITAQwen2, model_path='/path/to/model'),
}
```

Follow the [instuctions in VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) to set the GPT as the judge model.

If the openai api are not available, you can use a local model as the judge. The upstream authors found that a [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) judge works well compared to GPT-4, except on MM-Vet. To start the judge:
```
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /path/to/Qwen1.5-1.8B-Chat --server-port 23333
```
Then configure the `.env` file in the `VLMEvalKit` folder:
```
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=/path/to/Qwen1.5-1.8B-Chat
```
Evaluating on these benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME --model vita_qwen2 --verbose
```

### Video-MME
#### Data Preparation
Download the [Video-MME dataset](https://github.com/BradyFU/Video-MME) and extract the frames, saving them as images to improve IO efficiency.

#### Evaluation
```
cd ./videomme
```
Run the model on Video-MME in the setting of wo/ subtitles:
```
VIDEO_TYPE="s,m,l"
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=6 python yt_video_inference_qa_imgs.py \
        --model-path [vita/path] \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_wo_sub \
        --video_dir [Video-MME-imgs] | tee logs/infer.log
done

```
Run the model on Video-MME in the setting of w/ subtitles:
```
VIDEO_TYPE="s,m,l"
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=7 python yt_video_inference_qa_imgs.py \
        --model-path [vita/path] \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_w_sub \
        --video_dir [Video-MME-imgs] \
        --use_subtitles | tee logs/infer.log
done
```
Parse the results:
```
python parse_answer.py --video_types "s,m,l" --result_dir qa_wo_sub
python parse_answer.py --video_types "s,m,l" --result_dir qa_w_sub
```
## ✒️ Citation

**This fork introduces no new publication.** If you use this code, please cite the original
VITA papers — all credit for the model and method belongs to the upstream authors.

```bibtex
@article{fu2025vita,
  title={VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction},
  author={Fu, Chaoyou and Lin, Haojia and Wang, Xiong and Zhang, Yi-Fan and Shen, Yunhang and Liu, Xiaoyu and Li, Yangze and Long, Zuwei and Gao, Heting and Li, Ke and others},
  journal={arXiv preprint arXiv:2501.01957},
  year={2025}
}

@article{fu2024vita,
  title={Vita: Towards open-source interactive omni multimodal llm},
  author={Fu, Chaoyou and Lin, Haojia and Long, Zuwei and Shen, Yunhang and Zhao, Meng and Zhang, Yifan and Dong, Shaoqi and Wang, Xiong and Yin, Di and Ma, Long and others},
  journal={arXiv preprint arXiv:2408.05211},
  year={2024}
}
```


## &#x1F4E3; Statement

**The following statement is inherited from the upstream project and applies equally here:**

**VITA is trained on large-scale open-source corpus, and its output has randomness. Any content generated by VITA does not represent the views of the model developers. We are not responsible for any problems arising from the use, misuse, and dissemination of VITA, including but not limited to public opinion risks and data security issues.**

Additionally: this fork is an unofficial, research-only extension. It is not endorsed by,
affiliated with, or supported by the original VITA authors. Use of the code and the
upstream weights remains subject to [`License.txt`](./License.txt), which permits
**academic, research and educational use only** and prohibits commercial or production use.


## 📜 Related Works

Upstream related research from the original authors:
-  **[VITA-1.0]** [VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://vita-home.github.io/)
-  **[Awesome-MLLM]** [A Survey on Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
-  **[MME]** [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
-  **[Video-MME]** [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://github.com/BradyFU/Video-MME) 


## 👍 Acknowledgement

First and foremost, this repository is derived entirely from
[**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA) — thanks to the VITA team for
open-sourcing their work.

VITA itself is built with reference to the following outstanding works: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [Bunny](https://github.com/BAAI-DCAI/Bunny), [ChatUnivi](https://github.com/PKU-YuanGroup/Chat-UniVi), [InternVL](https://github.com/OpenGVLab/InternVL), [InternViT](https://huggingface.co/OpenGVLab/InternViT-300M-448px), [Qwen-2.5](https://github.com/QwenLM/Qwen2.5), [VLMEvalkit](https://github.com/open-compass/VLMEvalKit), and [Mixtral 8*7B](https://mistral.ai/news/mixtral-of-experts/).
Thanks！

