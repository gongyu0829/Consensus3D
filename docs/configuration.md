# Training configuration

The [JSON configuration](../configs/consensus3d_7b.json) records the Consensus3D
manuscript's architecture, fine-tuning policy and optimization settings.
The executable examples read the projector settings, fine-tuning policy and
random seed from this file. The table below specifies the training setup.

| Setting | Value |
| --- | --- |
| Language model | Vicuna-7B-v1.5 |
| Visual features | Frozen Uni3D geometry and DINOv2 appearance |
| Training hardware | 2 NVIDIA RTX 4090 GPUs |
| Training batch | 16 per GPU, 32 globally; no gradient accumulation |
| Evaluation batch | 16 per GPU |
| Training duration | 3 epochs (indices 0, 1 and 2) |
| Optimizer | AdamW, betas 0.9/0.999, weight decay 0.02 |
| Effective base learning rate | 1.6e-4 |
| Learning-rate schedule | Cosine; 0.1-epoch warm-up; minimum multiplier 0.01 |
| Gradient norm clipping | 0.01 |
| LoRA | Rank 16, alpha 16, dropout 0.05 |
| Deterministic DOP warm-up | 2,000 optimizer updates |
| Auxiliary weight ramp | 1,000 optimizer updates |
| Final KL / CMC weights | 1e-6 / 0.02 |
| CMC temperature | 0.07 |
| CMC negative pool | Valid objects gathered across devices |
| Candidate limit / generated response limit | 100 objects / 64 new tokens |
| Random seed | 42 |

The learning-rate argument `5e-6` is multiplied by the global batch size of 32
to obtain `1.6e-4`. A trainer that consumes the effective rate directly must use
`1.6e-4` without scaling it again. `load_configuration` checks this relationship.

Learning-rate warm-up changes the optimizer step size. Deterministic DOP warm-up
controls representation sampling and the activation of KL/CMC. These schedules
have different purposes and units. At step 2,000, sampling is enabled and the
auxiliary ramp starts at zero; at step 3,000, both auxiliary weights reach their
configured maximum.

The KL reduction sums over latent dimensions, then averages over valid objects
and both modalities. The CMC settings specify the temperature, objective weight
and cross-device negative pool used by the joint objective.

Exact projector dimensions and low-variance initialization appear in the JSON
and model source. The full-dimensional example verifies the projector interface at the paper's
1024-dimensional input and 4096-dimensional output sizes.

The package versions in `requirements.txt` specify the tested environment for
the released module example and tests, separately from the training
hyperparameters above.

