# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data.datasets.register_coco import register_coco_instances


_CUSTOM_SPLITS_COCO = {
    "marine_train_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/class_level_train.json",
    ),
    "marine_val_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/class_level_val.json",
    ),
    "marine_val_seen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/class_level_val_seen.json",
    ),
    "marine_val_unseen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/class_level_val_unseen.json",
    ),
    "marine_pseduo": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/imagenet_lvis_v1_decola_phase1_r50_21k_zeroshot_4x.json",
    ),
    "marine_intra_train_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/intra_class_train.json",
    ),
    "marine_intra_val_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/intra_class_val.json",
    ),
    "marine_intra_val_seen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/intra_class_val_seen.json",
    ),
    "marine_intra_val_unseen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/intra_class_val_unseen.json",
    ),
    "marine_inter_train_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/inter_class_train.json",
    ),
    "marine_inter_val_all": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/inter_class_val.json",
    ),
    "marine_inter_val_seen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/inter_class_val_seen.json",
    ),
    "marine_inter_val_unseen": (
        "marine/MarineDet",
        "marine/MarineDet/annotations/inter_class_val_unseen.json",
    ),
}

for key, (image_root, json_file) in _CUSTOM_SPLITS_COCO.items():
    # Assume pre-defined datasets live in `./datasets`.
    register_coco_instances(
        key,
        {},  # empty metadata, it will be overwritten in load_coco_json() function
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
