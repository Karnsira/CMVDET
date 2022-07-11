# Ensemble Stage

The socure code of the ensemble stage can be found in [ensemble.py](ensemble.py).
```
python3 ensemble/ensemble.py \
        --det detection/inference/output/test_nms_0.36.pkl \
        --cls classification/inference/output/test.pkl \
        --output ensemble/output/test.pkl \
        --det-weight 0.5
```