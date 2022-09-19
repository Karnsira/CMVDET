.. _detection:

Detection Stage
=======================

Description of detection...

Preprocessing
---------------------------------
::

    python3 detection/preprocessing/gen_patches.py \
        --output data/detection/ \
        --img-size 256

Training
---------------------------------------
::

    python3 mmdetection/tools/train.py  detection/training/configs/config.py \
        --work-dir detection/training/checkpoints/test

Inference
---------------------------------------
::

    python3 detection\inference\inference_script.py

