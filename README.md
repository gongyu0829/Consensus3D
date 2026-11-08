<h1 align="center">Consensus3D</h1>
<p align="center"><strong>Constructing Probabilistic Object Representations for 3D Scene Understanding</strong></p>
<p align="center">
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="docs/data_preparation.md">Data preparation</a> ·
  <a href="docs/configuration.md">Training settings</a>
</p>

Consensus3D learns probabilistic representations of an object's geometry and
appearance, and aligns their shared semantics before language reasoning.
The resulting object tokens support 3D visual grounding, dense captioning,
and question answering through a common language interface.

<p align="center">
  <img src="assets/framework.png" alt="Consensus3D framework: paired geometric and appearance features, distributional object projection, cross-modal consensus, and language-model responses." width="100%">
</p>

## Method

- **Distributional Object Projector (DOP).** Independent geometry and appearance
  branches learn object distributions in the language embedding space. Their
  means encode object semantics, while learned variances represent ambiguity
  in the visual evidence.
- **Cross-Modal Consensus (CMC).** An instance-level contrastive objective aligns
  the paired 2D and 3D representations of the same object. Together with response
  supervision and distributional regularization, it learns object tokens that
  preserve complementary evidence across modalities.
- **Stable probabilistic learning.** Low-variance initialization and deterministic
  warm-up establish the object-language mapping before sampled-token training.
  Inference uses the learned means.

The [architecture guide](docs/architecture.md) explains the complete flow from
scene observations and object proposals to language-model responses.

## Results

Results reported in the paper across five 3D scene-understanding benchmarks:

| Method | ScanRefer Acc@0.5 | Multi3DRefer F1@0.5 | Scan2Cap CIDEr@0.5 | ScanQA CIDEr | SQA3D EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Consensus3D | 52.2 | 55.1 | 77.4 | 89.6 | 55.0 |

Feature-removal experiments further show improved grounding robustness with
incomplete object evidence. Learned uncertainty increases as feature information
is removed and is associated with harder-to-ground objects.

## Getting started

Use Python 3.10 or newer and a PyTorch build appropriate for your device.

```bash
git clone https://github.com/gongyu0829/Consensus3D.git
cd Consensus3D
python -m pip install -r requirements.txt
python examples/inspect_object_encoder.py
```

The example runs on CPU with synthetic paired object features. It demonstrates
probabilistic projection, the transition from warm-up to sampling, mean-based
inference, and construction of the object-token sequence.

To use the paper's projector dimensions and run the module tests:

```bash
python examples/inspect_object_encoder.py --full-dimensions
python -m unittest discover -s tests -v
```

The tests cover distribution sampling, object masks, token ordering, and actual
LoRA adaptation and gradient routing on a small randomly initialized language model.

## Data and training

Mask3D provides object proposals, and frozen Uni3D and DINOv2 encoders supply
paired geometric and multi-view appearance features. Each object's identifier
and visual tokens are combined with the language instruction and passed to
[Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5).

Training jointly optimizes DOP, LoRA adapters, the input embedding matrix and
the language output head. The configuration uses two RTX 4090 GPUs, a global
batch of 64, two epochs and an effective base learning rate of `3.2e-4`.
Deterministic DOP warm-up and learning-rate warm-up have separate schedules.
See [data preparation](docs/data_preparation.md) for dataset and feature sources,
and [training settings](docs/configuration.md) for the optimization details.

## Repository guide

| Resource | Contents |
| --- | --- |
| [Object encoder](consensus3d/consensus3d.py) | DOP branches, distribution initialization, sampling, mean inference and object tokens |
| [Language-model adaptation](consensus3d/finetuning.py) | LoRA targets, trainable parameters and selective input gradients |
| [Configuration](configs/consensus3d_7b.json) | Architecture, batches, optimizer and training schedules |
| [Architecture](docs/architecture.md) | DOP, CMC, the joint objective and the complete model flow |
| [Data preparation](docs/data_preparation.md) | Dataset links, feature extraction and annotation formats |
| [Example](examples/inspect_object_encoder.py) and [tests](tests/test_modules.py) | Executable module examples and behavioral checks |

The code package contains the object encoder and language-model adaptation
modules. The end-to-end trainer, CMC loss implementation, benchmark evaluation
scripts and trained weights are not included.

## License

The project is available for noncommercial academic evaluation under the
[Academic Evaluation Source License](LICENSE). Third-party copyright and
license information is collected in [licenses](licenses/NOTICE.md).
