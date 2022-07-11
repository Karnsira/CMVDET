# Detection Stage

The socure code of the detection stage.

### Preprocessing

The socure code of preprocessing can be found in [preprocessing/](preprocessing/)
```
python3 detection/preprocessing/gen_patches.py \
        --output data/detection/ \
        --img-size 256
```

### Training

The model, dataset, and training schedule can be modified in [config.py](training/configs/config.py)
```
python3 mmdetection/tools/train.py  detection/training/configs/config.py \
        --work-dir detection/training/checkpoints/test
```

### Inference

The inference script can be found in [inference_script.py](detection\inference\inference_script.py) , which you can modify configuration via [config.json](detection\inference\config.json)

```
python3 detection\inference\inference_script.py
```