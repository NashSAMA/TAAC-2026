#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"


# echo "SCRIPT_DIR=${SCRIPT_DIR}"
# echo "TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
# echo "TRAIN_CKPT_PATH=${TRAIN_CKPT_PATH}"
# echo "TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
# echo "TRAIN_TF_EVENTS_PATH=${TRAIN_TF_EVENTS_PATH}"

# echo "Data dir files:"
# ls -lah "${TRAIN_DATA_PATH}"

# echo "Schema candidates:"
# find "${TRAIN_DATA_PATH}" -maxdepth 2 -name "schema.json" -print

# echo "schema.json compact chunks:"
# python3 - <<'PY'
# import os, json

# path = os.path.join(os.environ["TRAIN_DATA_PATH"], "schema.json")
# with open(path, "r", encoding="utf-8") as f:
#     s = json.dumps(json.load(f), separators=(",", ":"))

# chunk_size = 800
# for i in range(0, len(s), chunk_size):
#     print(f"SCHEMA_CHUNK_{i // chunk_size:03d}: {s[i:i + chunk_size]}")
# PY
# echo "schema.json compact chunks end"

# exit 0

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --use_rope \
#     "$@"
