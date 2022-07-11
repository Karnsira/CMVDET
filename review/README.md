# Review

The socure code for exporting Top False Positive. The output is in SQLITE format.
```
python review/export_top_fp.py \
       --input ensemble/output/test.pkl \
       --output review/output/test.sqlite \
       --db data/database/CU_DB.sqlite \
       --top 100 \
       --conf-thres 0.5
```