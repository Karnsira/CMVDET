# Classification Stage

The socure code of the classification stage.

### Preprocessing

The socure code of preprocessing can be found in [preprocessing/](preprocessing/).
```
python3 classification/preprocessing/gen_patches.py \
        --input detection/inference/output/test.pkl \
        --output data/classification/ \
        --img-size 224
```

### Training

The model, dataset, and training schedule can be modified in [config.py](training/configs/config.py).
```
python3 mmclassification/tools/train.py  classification/training/configs/config.py \
        --work-dir classification/training/checkpoints/test
```

### Inference

The socure code of inference can be found in [inference.py](inference/inference.py). The output is in a pickle format and will be used in the ensemble stage.
```
python3 classification/inference/inference.py \
        --config classification/training/configs/config.py \
        --checkpoint detection/training/checkpoints/test/lastest.pth \
        --input detection/inference/output/test_nms_0.36.pkl \
        --output classification/inference/output/test.pkl \
        --gpu-id 0
```