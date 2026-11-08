# Consensus3D

**Constructing Probabilistic Object Representations for 3D Scene Understanding**

Consensus3D represents each object's geometry and appearance as paired latent
distributions before supplying visual tokens to a language model. The Distributional
Object Projector (DOP) learns ambiguity within each modality, while Cross-Modal
Consensus (CMC) aligns representations of the same object across modalities.
The resulting object interface supports visual grounding, dense captioning and
question answering over 3D scenes.

## Selected implementation release

This release provides executable object-representation modules, the language-model
fine-tuning policy, reference configuration, and data-preparation documentation.
It supports inspection of the architecture and verification of module behavior.
The complete task trainer, CMC loss implementation, benchmark evaluation pipeline,
datasets and trained checkpoints are outside this release.

| Component | Location | What can be inspected |
| --- | --- | --- |
| Data and preprocessing | [Data preparation](docs/data_preparation.md) | Dataset sources, object proposals, paired features and annotation schema |
| Model architecture | [Architecture](docs/architecture.md) | The complete flow from scene observations to task responses |
| DOP and object tokens | [consensus3d.py](consensus3d/consensus3d.py) | Independent branches, distribution parameters, sampling, mean inference and token ordering |
| LLM fine-tuning | [finetuning.py](consensus3d/finetuning.py) | Frozen backbone, LoRA targets, trainable embeddings/output head and selective gradients |
| Optimization settings | [Reference configuration](configs/consensus3d_7b.json) | Batches, learning rate, warm-up schedules and objective settings |
| Module checks | [Tests](tests/test_modules.py) | Sampling behavior, token masks and actual LoRA/gradient routing on a small random LLM |

## Framework

1. Mask3D produces object proposals from a ScanNet scene. Frozen Uni3D and DINOv2
   encoders provide geometric and multi-view appearance features for each proposal.
2. Independent DOP branches transform the paired features into distributions in
   the language embedding space. Training uses mean tokens during deterministic
   warm-up and sampled tokens during probabilistic adaptation; inference uses means.
3. CMC aligns the two modalities for the same object, while separating unmatched
   objects. Response generation and KL regularization jointly supervise the object
   interface with the consensus objective.
4. Each object is represented by its identifier followed by geometric and appearance
   tokens. These tokens and the textual instruction enter Vicuna-7B-v1.5,
   which generates object references, descriptions or answers for the requested task.

See [Architecture and optimization](docs/architecture.md) for the training policy
and the roles of DOP and CMC in this flow.

## Installation and module inspection

Use Python 3.10 or newer. Install a PyTorch build appropriate for the local device,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python examples/inspect_object_encoder.py
python -m unittest discover -s tests -v
```

The example uses small dimensions and synthetic features by default. To inspect
the manuscript's object-projector dimensions, run:

```bash
python examples/inspect_object_encoder.py --full-dimensions
```

Both examples run on CPU without downloading a language model or a dataset.
They check the released modules, rather than evaluating task benchmark accuracy.
Language-model adaptation accepts a locally supplied causal language model whose
vocabulary has been extended with object identifiers. The reference backbone is
[Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5); obtain its weights
under the model provider's terms.

## Data and configuration

Follow [Data preparation](docs/data_preparation.md) for official dataset links,
the upstream preprocessing tools and the paired-feature format. The released
configuration uses two RTX 4090 GPUs, 32 training samples per GPU, a global batch
of 64 and two epochs. Evaluation uses 16 samples per GPU. The effective base
learning rate is `3.2e-4`; the configuration records the `5e-6` argument and its
batch scaling explicitly. [Configuration notes](docs/configuration.md) distinguish
learning-rate warm-up from deterministic DOP warm-up.

## Acknowledgments and provenance

Consensus3D builds on [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene).
We retain its object-identifier interface, feature preparation and language-model
adaptation policy, and develop probabilistic object representations and paired
cross-modal consensus at this interface. We also acknowledge ScanNet, Mask3D,
Uni3D, DINOv2, the task datasets, Transformers and PEFT.
See [Third-party notices](THIRD_PARTY_NOTICES.md) for source attribution.

## License

The Consensus3D-specific material is made available under the
[Academic Evaluation Source License](LICENSE), for noncommercial academic
evaluation. Redistribution and commercial use require separate permission.
Third-party material retains its original license; the upstream Chat-Scene MIT
notice is preserved in [licenses/Chat-Scene-MIT.txt](licenses/Chat-Scene-MIT.txt).
This is a source-available research preview, not an unrestricted open-source release.

Questions about the released modules can be raised through this repository's issues.

