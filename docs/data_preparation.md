# Data preparation

## Dataset sources

Obtain data through its original providers and follow their access and usage terms.
No dataset is redistributed in this repository.

| Resource | Purpose | Source |
| --- | --- | --- |
| ScanNet | RGB-D scenes, camera poses and 3D reconstructions | [ScanNet](https://github.com/ScanNet/ScanNet) |
| ScanRefer | Single-object language grounding | [ScanRefer](https://github.com/daveredrum/ScanRefer) |
| Multi3DRefer | Zero-, single- and multiple-object grounding | [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) |
| Scan2Cap | Object captioning | [Scan2Cap](https://github.com/daveredrum/Scan2Cap) |
| ScanQA | Scene question answering | [ScanQA](https://github.com/ATR-DBI/ScanQA) |
| SQA3D | Situated question answering | [SQA3D](https://github.com/SilongYong/SQA3D) |
| Nr3D | Object descriptions used by the caption training stream | [ReferIt3D](https://github.com/referit3d/referit3d) |

The training mixture contains `scanrefer`, `scan2cap`, `scanqa`, `sqa3d`,
`multi3dref`, `nr3d_caption`, and `obj_align`. The last stream pairs object
identifiers with object descriptions in the upstream task preparation.
The configuration uses task annotations without additional scene-description or
grounded chain-of-thought supervision.

## Preparation from scenes

Follow the [Chat-Scene preprocessing instructions](https://github.com/ZzZZCHS/Chat-Scene/tree/dev/preprocess)
for the common feature and annotation format:

1. Prepare ScanNet scenes and calibrated multi-view images.
2. Obtain object masks/proposals with [Mask3D](https://github.com/JonasSchult/Mask3D).
   Keep proposal identifiers consistent across features and annotation records.
3. Extract geometric object features with [Uni3D](https://github.com/baaivision/Uni3D).
   The upstream [object-feature extraction implementation](https://github.com/ZzZZCHS/Uni3D)
   uses the Uni3D-g model.
4. Project object masks into images using camera calibration, extract DINOv2
   features for the visible regions, and aggregate them into one appearance
   vector per object. See [DINOv2](https://github.com/facebookresearch/dinov2) and
   the upstream [multi-view feature preparation](https://github.com/ZzZZCHS/openscene/blob/main/scripts/feature_fusion/scannet_videofeat.py).
5. Prepare task prompts, target responses and object attributes using the common
   candidate indexing. Preserve the official train/validation splits.

Alternatively, the upstream project supplies
[prepared annotations and features](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene/tree/main/annotations).
These resources belong to their original providers and are linked here for convenience.

## Feature and annotation interface

The reference loader consumes the following records:

| Record | Format |
| --- | --- |
| Geometric feature cache | Object-keyed tensors, 1024 values per object |
| Appearance feature cache | Object-keyed tensors, 1024 values per object |
| Object key | Scene identifier plus candidate index, for example `scene0000_00_03` |
| Scene attributes | Candidate identifiers and `locs` with center and box extent |
| Task annotation | `scene_id`, `prompt`, `caption`, and task-dependent object identifiers |
| Model feature tensors | `[batch, objects, feature_dimension]` for each modality |
| Valid-object mask | Boolean `[batch, objects]`, excluding padding |

Typical upstream files include `scannet_mask3d_uni3d_feats.pt`,
`scannet_mask3d_videofeats.pt`, `scannet_mask3d_train_attributes.pt`,
`scannet_mask3d_val_attributes.pt`, and task-specific JSON annotations.
The candidate limit is 100. A missing feature is represented by a zero vector
without deleting the corresponding candidate or the other modality.

Object identifiers, both modalities and target references must follow the same
permutation. The released object encoder assumes its inputs are already aligned
and does not infer correspondence from independently shuffled feature arrays.

The synthetic example demonstrates this tensor interface without requiring
ScanNet access. It does not perform preprocessing or train a task model.

