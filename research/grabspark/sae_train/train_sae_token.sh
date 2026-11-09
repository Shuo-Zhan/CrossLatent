#!/usr/bin/env bash
#
# Token-level SAE training launcher for train_sae_token_base.py
#
# Examples:
#   bash train_sae_token.sh
#   bash train_sae_token.sh --nohup --gpu 0
#   bash train_sae_token.sh --sae_type relu --expansion_factor 12 --l1_coefficient 0.06
#   bash train_sae_token.sh --nohup --save_path /path/to/output --epochs 30
#

# 1003901 relu
# 1005414 relu_tied
# 1010492 topk 64000
# 1016619 jump_relu 64000
# 1018497 batch_topk 64000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_sae_token_base.py"

# ---------- defaults (override via CLI) ----------
NOHUP=0
GPU=""
PYTHON_BIN="${PYTHON}"

# training
EPOCHS=10
BATCH_SIZE=64000
NUM_WORKERS=0
SAVE_EPOCH=1
SAVE_PATH=""
OUTPUT_ROOT="/data1/zshuo23/GraBSpArk/ckpt/ERNIE-RNA/layer_12_token"

# model
# SAEType = Literal[
#     "relu",
#     "relu_tied",
#     "topk",
#     "jump_relu",
#     "batch_topk",
#     "identity",
# ]
SAE_TYPE="relu"
ACTIVATION_DIM=768
EXPANSION_FACTOR=8
DICT_SIZE=""
TOPK=64
NORMALIZE_TO_SQRT_D=0

# optimizer / scheduler
LEARNING_RATE="2e-4"
WEIGHT_DECAY="1e-3"
GRAD_CLIP_NORM=1.0
WARMUP_STEP_RATIO=0.05
DECAY_START_RATIO=0.7
SCHEDULER_POWER=2.0
MIN_LR="1e-7"
STEPS_PER_EPOCH=""

# loss
L1_COEFFICIENT=0.06
L1_WARMUP_RATIO=0.05
DEAD_DIM_COEFFICIENT=0.2
DEAD_DIM_EPSILON="1e-7"

# resample
RESAMPLE_UNTIL_EPOCH_RATIO=0.8
RESAMPLE_EVERY_N_BATCHES=100
RESAMPLE_INACTIVE_RATIO=0.5
RESAMPLE_NOISE_SCALE=0.01

# data paths (override if your embedding dataset lives elsewhere)
DATA_ROOT="${DATA_ROOT:-/data1/zshuo23/GraBSpArk/data/RNA/rnacentral_sae_train}"
EMBED_LAYER_DIR="${DATA_ROOT}/ERNIERNA_embeddings/layer_12"
TRAIN_FASTA_PREFIX="${DATA_ROOT}/val_8_1_1"
VAL_FASTA_PREFIX="${DATA_ROOT}/val_8_1_1"
TEST_FASTA_PREFIX="${DATA_ROOT}/test_8_1_1"
TRAIN_EMBEDDING_H5="${EMBED_LAYER_DIR}/val_0_205367.h5"
VAL_EMBEDDING_H5="${EMBED_LAYER_DIR}/val_0_205367.h5"
TEST_EMBEDDING_H5="${EMBED_LAYER_DIR}/test_0_205380.h5"
LOGITS_HEAD_PATH="/data1/zshuo23/GraBSpArk/ckpt/ERNIERNA_LogitsHead/model_weights.pt"

# logging
LOG_EVERY_N_BATCHES=10
LOG_DIR="${OUTPUT_ROOT}/logs"

usage() {
    cat <<'EOF'
Usage: bash train_sae_token.sh [OPTIONS]

Launch options:
  --nohup                   Run in background with nohup
  --gpu ID                  CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1
  --python PATH             Python interpreter (default: python3)
  --log_dir PATH            Directory for nohup log files

Training:
  --epochs N                Training epochs (default: 50)
  --batch_size N            Token batch size (default: 128000)
  --num_workers N           DataLoader workers (default: 0)
  --save_epoch N            Save checkpoint every N epochs (default: 1)
  --save_path PATH          Explicit output directory
  --output_root PATH        Auto output parent dir if save_path omitted

Model:
  --sae_type TYPE           relu | relu_tied | topk | jump_relu | batch_topk | identity
  --activation_dim N        Embedding dim (default: 768)
  --expansion_factor F      dict_size = activation_dim * F (default: 12)
  --dict_size N             Override dict_size
  --topk N                  For topk / batch_topk (default: 64)
  --normalize_to_sqrt_d     Enable sqrt(d) normalization

Optimizer / scheduler:
  --learning_rate F         Default: 2e-4
  --weight_decay F          Default: 1e-3
  --grad_clip_norm F        Default: 1.0
  --warmup_step_ratio F     Default: 0.1
  --decay_start_ratio F     Default: 0.7
  --scheduler_power F       Default: 2.0
  --min_lr F                Default: 1e-7
  --steps_per_epoch N       Optional; default auto from train loader

Loss:
  --l1_coefficient F        Default: 0.06
  --l1_warmup_ratio F       Default: 0.05
  --dead_dim_coefficient F  Default: 0.2
  --dead_dim_epsilon F      Default: 1e-7

Data:
  --data_root PATH          Dataset root prefix
  --train_embedding_h5 PATH
  --val_embedding_h5 PATH
  --test_embedding_h5 PATH

Other:
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup) NOHUP=1; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;

        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --save_epoch) SAVE_EPOCH="$2"; shift 2 ;;
        --save_path) SAVE_PATH="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;

        --sae_type) SAE_TYPE="$2"; shift 2 ;;
        --activation_dim) ACTIVATION_DIM="$2"; shift 2 ;;
        --expansion_factor) EXPANSION_FACTOR="$2"; shift 2 ;;
        --dict_size) DICT_SIZE="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        --normalize_to_sqrt_d) NORMALIZE_TO_SQRT_D=1; shift ;;

        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --grad_clip_norm) GRAD_CLIP_NORM="$2"; shift 2 ;;
        --warmup_step_ratio) WARMUP_STEP_RATIO="$2"; shift 2 ;;
        --decay_start_ratio) DECAY_START_RATIO="$2"; shift 2 ;;
        --scheduler_power) SCHEDULER_POWER="$2"; shift 2 ;;
        --min_lr) MIN_LR="$2"; shift 2 ;;
        --steps_per_epoch) STEPS_PER_EPOCH="$2"; shift 2 ;;

        --l1_coefficient) L1_COEFFICIENT="$2"; shift 2 ;;
        --l1_warmup_ratio) L1_WARMUP_RATIO="$2"; shift 2 ;;
        --dead_dim_coefficient) DEAD_DIM_COEFFICIENT="$2"; shift 2 ;;
        --dead_dim_epsilon) DEAD_DIM_EPSILON="$2"; shift 2 ;;

        --resample_until_epoch_ratio) RESAMPLE_UNTIL_EPOCH_RATIO="$2"; shift 2 ;;
        --resample_every_n_batches) RESAMPLE_EVERY_N_BATCHES="$2"; shift 2 ;;
        --resample_inactive_ratio) RESAMPLE_INACTIVE_RATIO="$2"; shift 2 ;;
        --resample_noise_scale) RESAMPLE_NOISE_SCALE="$2"; shift 2 ;;

        --data_root)
            DATA_ROOT="$2"
            EMBED_LAYER_DIR="${DATA_ROOT}/ERNIERNA_embeddings/layer_12"
            TRAIN_FASTA_PREFIX="${DATA_ROOT}/train_8_1_1"
            VAL_FASTA_PREFIX="${DATA_ROOT}/val_8_1_1"
            TEST_FASTA_PREFIX="${DATA_ROOT}/test_8_1_1"
            shift 2
            ;;
        --train_embedding_h5) TRAIN_EMBEDDING_H5="$2"; shift 2 ;;
        --val_embedding_h5) VAL_EMBEDDING_H5="$2"; shift 2 ;;
        --test_embedding_h5) TEST_EMBEDDING_H5="$2"; shift 2 ;;
        --logits_head_path) LOGITS_HEAD_PATH="$2"; shift 2 ;;
        --log_every_n_batches) LOG_EVERY_N_BATCHES="$2"; shift 2 ;;

        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --save_epoch "${SAVE_EPOCH}"
    --output_root "${OUTPUT_ROOT}"
    --sae_type "${SAE_TYPE}"
    --activation_dim "${ACTIVATION_DIM}"
    --expansion_factor "${EXPANSION_FACTOR}"
    --topk "${TOPK}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip_norm "${GRAD_CLIP_NORM}"
    --warmup_step_ratio "${WARMUP_STEP_RATIO}"
    --decay_start_ratio "${DECAY_START_RATIO}"
    --scheduler_power "${SCHEDULER_POWER}"
    --min_lr "${MIN_LR}"
    --l1_coefficient "${L1_COEFFICIENT}"
    --l1_warmup_ratio "${L1_WARMUP_RATIO}"
    --dead_dim_coefficient "${DEAD_DIM_COEFFICIENT}"
    --dead_dim_epsilon "${DEAD_DIM_EPSILON}"
    --resample_enabled_until_epoch_ratio "${RESAMPLE_UNTIL_EPOCH_RATIO}"
    --resample_every_n_batches "${RESAMPLE_EVERY_N_BATCHES}"
    --resample_inactive_ratio "${RESAMPLE_INACTIVE_RATIO}"
    --resample_noise_scale "${RESAMPLE_NOISE_SCALE}"
    --train_fasta_prefix "${TRAIN_FASTA_PREFIX}"
    --val_fasta_prefix "${VAL_FASTA_PREFIX}"
    --test_fasta_prefix "${TEST_FASTA_PREFIX}"
    --train_embedding_h5 "${TRAIN_EMBEDDING_H5}"
    --val_embedding_h5 "${VAL_EMBEDDING_H5}"
    --test_embedding_h5 "${TEST_EMBEDDING_H5}"
    --logits_head_path "${LOGITS_HEAD_PATH}"
    --log_every_n_batches "${LOG_EVERY_N_BATCHES}"
)

if [[ -n "${GPU}" ]]; then
    CMD+=(--cuda_visible_devices "${GPU}")
fi
if [[ -n "${SAVE_PATH}" ]]; then
    CMD+=(--save_path "${SAVE_PATH}")
fi
if [[ -n "${DICT_SIZE}" ]]; then
    CMD+=(--dict_size "${DICT_SIZE}")
fi
if [[ -n "${STEPS_PER_EPOCH}" ]]; then
    CMD+=(--steps_per_epoch "${STEPS_PER_EPOCH}")
fi
if [[ "${NORMALIZE_TO_SQRT_D}" -eq 1 ]]; then
    CMD+=(--normalize_to_sqrt_d)
fi

echo "Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "${NOHUP}" -eq 1 ]]; then
    mkdir -p "${LOG_DIR}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${LOG_DIR}/train_sae_token_${SAE_TYPE}_${TIMESTAMP}.out"
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Started in background."
    echo "  PID:      ${PID}"
    echo "  Log file: ${LOG_FILE}"
    echo "  Tail log: tail -f ${LOG_FILE}"
else
    exec "${CMD[@]}"
fi
