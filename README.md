# FlowerVLA-QC-FQL

[Paper](), [Project Page](),

[Moritz Reuss](https://mbreuss.github.io/)<sup>1</sup>,
[Hongyi Zhou](https://hongyizhoucn.github.io/)<sup>1</sup>,
[Marcel Ruehle]()<sup>1</sup>,
[Ömer Erdinç Yağmurlu](https://scholar.google.com/citations?user=I_Mxp5cAAAAJ&hl=en)<sup>1</sup>,
[Fabian Otto](https://ottofabian.github.io/)<sup>2</sup>,
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT)
<sup>2</sup>Microsoft Research


##  FLOWER: An efficient & versatile Flow-VLA + Q-Chunking RL

FLOWER VLA is a lightweight, efficient Vision-Language-Action (VLA) policy for robotic manipulation tasks that achieves state-of-the-art performance on multiple benchmarks. Built on a rectified flow architecture with several key architecture features:

- **Efficient Architecture**: At ~1B parameters, FLOWER is significantly smaller than most VLA models
- **Low Training Cost**: Only requires ~200 GPU hours of pretraining
- **Low Memory Footprint**: Uses <3GB of GPU memory for inference with single image setting
- **SOTA Performance**: Achieves sota results on CALVIN and LIBERO benchmarks

**NEW: Q-Chunking Reinforcement Learning Integration**
This repository now includes a complete integration of the Q-Chunking RL algorithm for improved action chunking performance:
- **Two-stage Training**: Offline behavior cloning → Online reinforcement learning
- **Action Chunking**: Enhanced multi-step action prediction with Q-learning
- **Dual Policies**: Separate BC and RL VLA models for robust training
- **Efficient Evaluation**: Compatible evaluation pipeline for both BC and RL models

For the pretraining of FLOWER and finetuning for Aloha check out our other codebase.


## Model Overview

FLOWER VLA uses a Florence-2-large-based VLM combined with a rectified flow architecture to predict robot actions from visual observations and language instructions.
The model efficiently handles multi-step action sequences through a chunking mechanism.

[Insert model architecture diagram here]

Key architectural components:
- Florence-2 DaVit Image Encoder [350M params]
- Language conditioning through cross-attention
- Half of the Florence-2 LLM layers for vision and language fusion [205M]
- Rectified Flow Predition for fast action generation (all results in CALVIN and LIBERO are achieved using just 4 denoising steps)
- Global AdaLN for parameter efficient conditioning
- Action chunking for multi-step action generation


## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:intuitive-robots/flower_vla_calvin.git
export flower_calvin_ROOT=$(pwd)/flower_vla_calvin

```
Install requirements
(Note we provided a changed verison of pyhash, given numerous problems we encountered when installing it manually on our slurm cluster)
You can also try to install setup tools using pip.

```bash
cd $flower_calvin_ROOT
conda create -n flower_cal python=3.9
conda activate flower_cal
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
```
Next we can install the rest of the missing packages

```
pip install -r requirements.txt
```

---

## Download
### CALVIN Dataset

If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $flower_calvin_ROOT/dataset
sh download_data.sh D | ABCD
```

## Training

### Original FLOWER VLA Training
To train the original FLOWER VLA with 4 GPUs, run:
```bash
python flower/training.py
```

### Q-Chunking Training (NEW)
To train with Q-Chunking reinforcement learning, you have three training modes:

#### 1. Offline Training (Behavior Cloning Only)
```bash
python flower/training_libero.py q_chunking.enabled=true q_chunking.training_stage=offline
```

#### 2. Online Training (Full RL Pipeline)
```bash
python flower/training_libero.py q_chunking.enabled=true q_chunking.training_stage=online
```

#### 3. Offline-to-Online Training (Recommended)
```bash
python flower/training_libero.py q_chunking.enabled=true q_chunking.training_stage=offline-to-online
```

#### Key Q-Chunking Configuration Parameters:
- `q_chunking.enabled`: Enable Q-chunking training (default: true)
- `q_chunking.training_stage`: Training mode - "offline", "online", or "offline-to-online"
- `q_chunking.chunk_size`: Action chunk size (default: same as act_seq_len)
- `q_chunking.discount_factor`: RL discount factor (default: 0.99)
- `q_chunking.replay_buffer_size`: Replay buffer capacity (default: 100000)
- `q_chunking.checkpoint.save_freq`: Checkpoint saving frequency in epochs (default: 5)

You can use the pretrained FLOWER checkpoint from [hf-link](https://huggingface.co/mbreuss/flower_vla_pret) to initialize your Q-chunking training.

Note that during training the full CALVIN eval or LIBERO rollouts will be called after _rollout_lh_skip_epochs_ and then every _callbacks.rollout_lh.rollout_freq_*1k training steps. Check out the training config for adopting the parameters.

## Evaluation

### Original FLOWER VLA Evaluation
```bash
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/model.ckpt
```

### Q-Chunking Model Evaluation (NEW)
Evaluate BC VLA model:
```bash
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/checkpoint.pt q_chunking_model_type=bc_vla
```

Evaluate RL VLA model:
```bash
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/checkpoint.pt q_chunking_model_type=rl_vla
```

The evaluation automatically detects checkpoint format (.ckpt vs .pt) and loads the appropriate model.

## Checkpoint Management

### Checkpoint Types and Formats

Q-Chunking training generates multiple checkpoint formats for maximum compatibility:

#### 1. PyTorch Lightning Checkpoints (.pt files)
- **BC VLA Only**: `bc_checkpoint_epoch_X.pt` - Contains only behavior cloning model
- **RL VLA Only**: `rl_checkpoint_epoch_X.pt` - Contains only RL policy model
- **Full Model**: `full_checkpoint_epoch_X.pt` - Contains BC VLA, RL VLA, Q-networks, replay buffer, and all optimizer states

#### 2. HuggingFace Format (Automatic)
For each .pt checkpoint, the system automatically creates HuggingFace-compatible model directories:
```
hf_models_epoch_X/
├── bc_vla/
│   ├── model.safetensors      # BC VLA weights
│   ├── model.pt               # Fallback format
│   ├── config.yaml            # Model configuration for evaluation
│   └── README.md              # Model documentation
└── rl_vla/
    ├── model.safetensors      # RL VLA weights
    ├── model.pt               # Fallback format
    ├── config.yaml            # Model configuration for evaluation
    └── README.md              # Model documentation
```

### Checkpoint Configuration

Configure checkpoint behavior in your training config:
```yaml
q_chunking:
  checkpoint:
    save_freq: 5                    # Save every N epochs
    keep_separate: true             # Keep BC, RL, and full checkpoints separate
    auto_resume: true               # Automatically resume from latest checkpoint
```

### Loading Checkpoints for Evaluation

The evaluation script automatically detects and loads the appropriate checkpoint format:

#### From PyTorch Lightning checkpoints:
```bash
# Load BC VLA model
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/bc_checkpoint_epoch_X.pt

# Load RL VLA model
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/rl_checkpoint_epoch_X.pt q_chunking_model_type=rl_vla

# Load from full checkpoint (defaults to BC VLA)
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/full_checkpoint_epoch_X.pt
```

#### From HuggingFace format:
```bash
# Load BC VLA from HuggingFace directory
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/hf_models_epoch_X/bc_vla/

# Load RL VLA from HuggingFace directory
python flower/evaluation/flower_eval_libero.py checkpoint=/path/to/hf_models_epoch_X/rl_vla/ q_chunking_model_type=rl_vla
```

### Resuming Training

To resume training from a saved checkpoint:
```bash
# Resume from specific checkpoint
python flower/training_libero.py q_chunking.enabled=true resume_from=/path/to/full_checkpoint_epoch_X.pt

# Auto-resume from latest checkpoint (if auto_resume=true)
python flower/training_libero.py q_chunking.enabled=true  # Automatically finds latest checkpoint
```

### Checkpoint Contents

**BC-only checkpoints** contain:
- BC VLA model weights and hyperparameters
- BC optimizer state
- Training metadata (epoch, global_step)

**RL-only checkpoints** contain:
- RL VLA model weights and hyperparameters
- RL optimizer state
- Training metadata

**Full checkpoints** contain:
- Both BC and RL VLA models
- Q-networks and target Q-networks
- All optimizer states (BC, RL, Q-network)
- Replay buffer state
- Complete training metadata

### Compatibility with Original FLOWER Evaluation

All saved models are fully compatible with the original FLOWER evaluation pipeline:
- Standard `state_dict` format for seamless loading
- Proper `hyper_parameters` preservation
- Compatible configuration structure
- No variable name conflicts with original implementation

Use `checkpoint_latest.pt` to resume training from the last saved state.

For replication of the orginial training results I recommend to use 4 GPUs with a batch_size of 8 and train them for 40k steps for ABC (ABCD) and evaluating after 19 epochs to get the best possible results.
See training configs for details.

#### Preprocessing with CALVIN
Since FLOWER uses action chunking, it needs to load multiple (~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually ~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:
```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```

```
python preprocess/extract_by_key.py -i /hkfs/work/workspace/scratch/ft4740-play3/data --in_task all
```

##### Params:
Run this command to see more detailed information:
```shell
python preprocess/extract_by_key.py -h
```

Important params:
* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)
Optional params:
* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data
Thanks to @ygtxr1997 for debugging the GPU utilization and providing a merge request.


## Evaluation

Download the pretrained FLOWER from Hugging Face:
You can find all checkpoints under:

- [FLOWER Collection](https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f)

We provide pretrained checkpoints for all CALVIN and LIBERO challenges.

## Performance Comparison on CALVIN Challenges (1000 chains)

Below is the average performance of FLOWER on CALVIN. It currenty is SOTA for all variants of CALVIN:

| Train→Test | Method | PrT | 1 | 2 | 3 | 4 | 5 | **Avg. Len.** |
|------------|---------|-----|---|---|---|---|---|---------------|
| ABCD→D | Diff-P-CNN | × | 86.3% | 72.7% | 60.1% | 51.2% | 41.7% | 3.16±0.06 |
| | Diff-P-T | × | 78.3% | 53.9% | 33.8% | 20.4% | 11.3% | 1.98±0.09 |
| | RoboFlamingo | ✓ | 96.4% | 89.6% | 82.4% | 74.0% | 66.0% | 4.09±0.00 |
| | GR-1 | ✓ | 94.9% | 89.6% | 84.4% | 78.9% | 73.1% | 4.21±0.00 |
| | MDT | × | 98.6% | 95.8% | 91.6% | 86.2% | 80.1% | 4.52±0.02 |
| | MoDE | ✓ | 97.1% | 92.5% | 87.9% | 83.5% | 77.9% | 4.39±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.49 |
| | **FLOWER (ours)** | × | **99.1%** | **97.8%** | **95.2%** | **92.4%** | **87.8%** | **4.67±0.04** |
| ABC→D | Diff-P-CNN | × | 63.5% | 35.3% | 19.4% | 10.7% | 6.4% | 1.35±0.05 |
| | Diff-P-T | × | 62.2% | 30.9% | 13.2% | 5.0% | 1.6% | 1.13±0.02 |
| | RoboFlamingo | ✓ | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.47±0.00 |
| | GR-1 | ✓ | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06±0.00 |
| | 3DDA | × | 93.8% | 80.3% | 66.2% | 53.3% | 41.2% | 3.35 |
| | MoDE | ✓ | 96.2% | 88.9% | 81.1% | 71.8% | 63.5% | 4.01±0.04 |
| | KomosVLA | ✓ | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.25 |
| | VPP | ✓ | 95.7% | 91.2% | 86.3% | 81.0% | 75.0% | 4.29 |
| | Seer | ✓ | 96.3% | 91.6% | 86.1% | 80.3% | 74.0% | 4.28 |
| | **FLOWER (ours)** | × | **99.3%** | **95.9%** | **90.5%** | **84.8%** | **77.5%** | **4.54±0.02** |
| D→D| **FLOWER (ours)** | × | **98.4%** | **94.0%** | **87.9%** | **81.7%** | **74.1%** | **4.36±0.04** |

## Performance on LIBERO Benchmarks

FLOWER achieves strong performance across all LIBERO benchmarks:

| Benchmark | FLOWER Success Rate |
|-----------|---------------------|
| LIBERO-10 | 94.5% |
| LIBERO-90 | 93.4% |
| LIBERO-SPATIAL | 97.2% |
| LIBERO-OBJECT | 99.3% |
| LIBERO-GOAL | 96.9% |


#### Common Issues

Sometimes this causes problems for the python env so just delete it:

```python
log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")
```
The path for this line is in the CALVIN env repo: https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/envs/play_table_env.py#L72

---

## Acknowledgements

This work is only possible because of the code from the following open-source projects and datasets. We thank all authors for their work:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)

License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### LIBERO

Original: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

License: [https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file)

#### Mimictest

Original: [mimictest](https://github.com/EDiRobotics/mimictest)
License: [license](https://github.com/EDiRobotics/mimictest?tab=Apache-2.0-1-ov-file)

#### HULC
Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)

License: [MIT](https://github.com/lukashermann/hulc/blob/main/LICENSE)

#### FLOWER Pretraining Codebase

Original: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy)

License: [https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE](https://github.com/intuitive-robots/FLOWER_Diffusion_Policy/blob/main/LICENSE)


## Citation

If you found the code usefull, please cite our work: (arxiv coming very soon)

```bibtex
@inproceedings{
    reuss2025flower,
    ???
}
```
