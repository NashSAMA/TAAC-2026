```shell
TRAIN_DATA_PATH="$PWD/data_sample_1000" \
TRAIN_CKPT_PATH="$PWD/outputs/local_ckpt" \
TRAIN_LOG_PATH="$PWD/outputs/local_logs" \
TRAIN_TF_EVENTS_PATH="$PWD/outputs/local_tb" \
python3 -u "Model Training/train.py" \
  --schema_path "$PWD/data_sample_1000/schema.json" \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --num_workers 0 \
  --batch_size 16 \
  --device cuda
```
