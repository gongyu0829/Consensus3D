# Architecture and optimization

This guide describes the complete model flow. Executable code currently covers
DOP, object-token construction and language-model adaptation; see
[release contents](../README.md#release-contents) for implementation coverage.

## Scene observations to paired object features

The input consists of a 3D scene, its associated multi-view images, and a language
instruction. Mask3D proposals provide a common object index. Uni3D encodes the
points belonging to each proposal, and DINOv2 features from the corresponding
image regions are aggregated across views. Matching the two feature records by
object identity preserves the geometry/appearance correspondence. Visual encoders
are frozen and features are extracted before language-model training.

## Probabilistic object representation

DOP has separate trainable branches for geometry and appearance. Each branch
normalizes its object feature and maps it to a Gaussian distribution in the language
embedding space. Its mean carries the object semantics; its variance learns the
dispersion of the representation under task supervision and distributional
regularization. Independent branches allow the two modalities to represent
different ambiguities while sharing the language interface.

In the released implementation, normalized 1024-dimensional inputs pass through
a 4096-dimensional nonlinear mapping with GELU, followed by independent mean and
log-variance projections. The exact layers are visible in
[`DistributionalObjectProjector`](../consensus3d/consensus3d.py).
The variance head starts with zero weights and bias -4.5; a forward offset of
-4.5 gives initial log variance -9. The log variance has an upper bound of -1.5.

The training representation is sampled with reparameterization after deterministic
warm-up. This lets the task objective train the object-language interface under
learned perturbations. During deterministic warm-up and inference, the projected
mean is used directly. Calling `eval()` therefore disables sampling regardless
of the supplied step number.

## Cross-modal consensus and the joint objective

CMC encourages the paired 2D and 3D representations of the same physical object
to agree while distinguishing them from representations of other objects. Its
bidirectional contrastive objective uses normalized features and temperature 0.07.
A valid-object mask excludes padding. The manuscript configuration uses a global
pool of valid object representations across the two devices.

The full training objective combines response-token cross-entropy, a standard
Gaussian KL regularizer, and CMC. Thus object representations are learned through
language-task supervision, within-modality distribution learning, and explicit
cross-modal correspondence. DOP addresses ambiguity in each modality; CMC links
the complementary evidence belonging to the same object.

The reference KL reduction sums latent dimensions for each object and then averages
over valid objects and the two modalities, matching the manuscript equations.
Its numerical coefficient should be interpreted together with this reduction.

## Object tokens to task responses

Each candidate receives an object identifier such as `<OBJ003>`. In the reference
interface the scene sequence interleaves an identifier, a 3D token, and a 2D token
for every valid candidate. The same ordering must be used for the paired features,
identifiers and target references. Padding is omitted from the sequence.

The scene sequence is combined with the system/task instruction and the query.
Vicuna-7B-v1.5 autoregressively produces the response: object identifiers for
grounding, object descriptions for dense captioning, or answer text for question
answering. Object identifiers are resolved back to the scene's candidate boxes
by the task pipeline. Training supervises the response tokens while masking the
input prefix in the language loss.

[`object_token_sequences`](../consensus3d/consensus3d.py) constructs the visual
sequence for combination with task-specific prompts and target responses.

## Language-model fine-tuning

The original language Transformer is frozen. LoRA adapts its attention projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward projections (`gate_proj`,
`up_proj`, `down_proj`), using rank 16, alpha 16 and dropout 0.05.
The DOP branches, language output head and input embedding matrix are trainable.

On the text-input path, original-vocabulary and system-prompt activations are
detached. Added object-identifier activations retain gradients so their embeddings
can learn explicit instance references. This routing acts on input activations;
the embedding matrix is still included in the optimizer and can be affected by
weight decay. It is different from clipping the global gradient norm to 0.01.

[`prepare_language_model`](../consensus3d/finetuning.py) exposes the parameter
policy, and `selective_input_embeddings` implements the input-gradient routing.
The caller first adds the object identifiers to the tokenizer and resizes the
language model's token embeddings, then calls `prepare_language_model` before
constructing the optimizer. During input assembly, the caller applies
`selective_input_embeddings` with the added identifier range (end exclusive),
and with `system_prompt=True` for the system prompt. This input routing is a
separate call; installing LoRA alone does not apply it. The preparation utility
keeps input embeddings and the output head in float32; mixed-precision training
must also handle the forward computation through these modules.

## Training schedule

The two-epoch setup uses two RTX 4090 GPUs, a global batch of 64, AdamW, an effective
base learning rate of 3.2e-4 and a cosine schedule. Learning-rate warm-up lasts
0.1 epoch. Separately, the first 2,000 updates use deterministic DOP means with
KL and CMC disabled. Sampling then begins, and the auxiliary weights increase
linearly over 1,000 updates to 1e-6 for KL and 0.02 for CMC.

The reference settings are in [the configuration](../configs/consensus3d_7b.json).
The module tests check architecture behavior and gradient routing with synthetic
inputs. In the object encoder, `global_step` is the number of completed optimizer
updates and must be supplied by the caller during training. The encoder controls
sampling only; the auxiliary-loss ramp and learning-rate schedule belong to the
full training pipeline.

