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
The model, dataset, and training schedule can be modified in ``config.py`` located at training/configs/config.py
::

    python3 mmdetection/tools/train.py  detection/training/configs/config.py \
        --work-dir detection/training/checkpoints/test

Inference
---------------------------------------
::

    python3 detection\inference\inference_script.py

