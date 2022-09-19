.. _classification:

Classification Stage
=======================

Description of classification...

Preprocessing
---------------------------------
::

    python3 classification/preprocessing/gen_patches.py \
        --input detection/inference/output/test.pkl \
        --output data/classification/ \
        --img-size 224

Training
---------------------------------------
::

    python3 mmclassification/tools/train.py  classification/training/configs/config.py \
        --work-dir classification/training/checkpoints/test

Inference
---------------------------------------
::

    python3 classification/inference/inference.py \
        --config classification/training/configs/config.py \
        --checkpoint detection/training/checkpoints/test/lastest.pth \
        --input detection/inference/output/test_nms_0.36.pkl \
        --output classification/inference/output/test.pkl \
        --gpu-id 0

