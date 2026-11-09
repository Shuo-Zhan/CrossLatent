#!/usr/bin/env bash
# 批量调用 ESM2_eval_neuron_boosting_seq_multi_json.py：
#   High-quality-dataset/results/<label>/ 下的 boosting JSON 系列
#   → 在 top_pfam_by_label_ratio 中同标签 CSV 上评估
#   → 结果写入 top_pfam_by_label_ratio/eval_results/
#
# 用法:
#   bash batch_eval_hq_on_top_pfam.sh
#   DEVICE=cuda:0 bash batch_eval_hq_on_top_pfam.sh
#   LABELS="mask_protein_domains mask_disulfide_bonds" bash batch_eval_hq_on_top_pfam.sh
#   DRY_RUN=1 bash batch_eval_hq_on_top_pfam.sh
#   SKIP_EXISTING=0 bash batch_eval_hq_on_top_pfam.sh   # 默认跳过已有输出
# cd /data1/zshuo23/GraBSpArk/grabspark/boost_pipeline
# nohup bash batch_eval_hq_on_top_pfam.sh > batch_eval_hq_on_top_pfam.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/ESM2_eval_neuron_boosting_seq_multi_json.py"

ROOT="/data1/zshuo23/GraBSpArk"
RESULTS_ROOT="${ROOT}/data/Protein/High-quality-dataset/results"
EVAL_CSV_DIR="${ROOT}/data/Protein/top_pfam_by_label_ratio"
OUT_DIR="${EVAL_CSV_DIR}/eval_results"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
INDEX_START="${INDEX_START:-0}"
# 为空则自动取该系列最大下标
INDEX_END="${INDEX_END:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# 空格分隔的标签名；为空则遍历 RESULTS_ROOT 下所有子目录
LABELS="${LABELS:-}"

mkdir -p "${OUT_DIR}"

discover_json_stems() {
  local json_dir="$1"
  # 仅 indexed checkpoint：*_neuron_parameters_<int>.json
  find "${json_dir}" -maxdepth 1 -type f -name '*_neuron_parameters_[0-9]*.json' \
    | sed -E 's|.*/||; s|_[0-9]+\.json$||' \
    | sort -u
}

max_index_for_stem() {
  local json_dir="$1"
  local stem="$2"
  local max_idx=""
  local f idx
  for f in "${json_dir}/${stem}_"[0-9]*.json; do
    [[ -e "${f}" ]] || continue
    idx="${f##*_}"
    idx="${idx%.json}"
    [[ "${idx}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${max_idx}" || "${idx}" -gt "${max_idx}" ]]; then
      max_idx="${idx}"
    fi
  done
  echo "${max_idx}"
}

list_labels() {
  if [[ -n "${LABELS}" ]]; then
    # shellcheck disable=SC2086
    printf '%s\n' ${LABELS}
  else
    find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  fi
}

echo "=============================================="
echo "EVAL_PY      = ${EVAL_PY}"
echo "RESULTS_ROOT = ${RESULTS_ROOT}"
echo "EVAL_CSV_DIR = ${EVAL_CSV_DIR}"
echo "OUT_DIR      = ${OUT_DIR}"
echo "PYTHON       = ${PYTHON}"
echo "DEVICE       = ${DEVICE}"
echo "INDEX_START  = ${INDEX_START}"
echo "INDEX_END    = ${INDEX_END:-<auto>}"
echo "SKIP_EXISTING= ${SKIP_EXISTING}"
echo "DRY_RUN      = ${DRY_RUN}"
echo "=============================================="

total=0
ok=0
skipped=0
failed=0

while IFS= read -r label; do
  [[ -z "${label}" ]] && continue
  json_dir="${RESULTS_ROOT}/${label}"
  if [[ ! -d "${json_dir}" ]]; then
    echo "[WARN] 跳过：结果目录不存在 ${json_dir}"
    continue
  fi

  mapfile -t stems < <(discover_json_stems "${json_dir}")
  if [[ ${#stems[@]} -eq 0 ]]; then
    echo "[WARN] ${label}: 未找到 *_neuron_parameters_<i>.json"
    continue
  fi

  mapfile -t csvs < <(
    find "${EVAL_CSV_DIR}" -maxdepth 1 -type f -name "*_${label}.csv" | sort
  )
  if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "[WARN] ${label}: ${EVAL_CSV_DIR} 下无匹配 CSV (*_${label}.csv)"
    continue
  fi

  for stem in "${stems[@]}"; do
    auto_end="$(max_index_for_stem "${json_dir}" "${stem}")"
    if [[ -z "${auto_end}" ]]; then
      echo "[WARN] ${label}/${stem}: 无法解析 index 范围"
      continue
    fi
    end_idx="${INDEX_END:-${auto_end}}"

    for csv_path in "${csvs[@]}"; do
      csv_stem="$(basename "${csv_path}" .csv)"
      out_json="${OUT_DIR}/ESM2_eval_multi_${stem}_idx${INDEX_START}-${end_idx}_on_${csv_stem}.json"
      total=$((total + 1))

      if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
        echo "[SKIP] ${out_json}"
        skipped=$((skipped + 1))
        continue
      fi

      echo "----------------------------------------------"
      echo "[RUN] label=${label}"
      echo "      json_dir=${json_dir}"
      echo "      json_name=${stem}"
      echo "      indices=${INDEX_START}..${end_idx}"
      echo "      csv=${csv_path}"
      echo "      out=${out_json}"

      cmd=(
        "${PYTHON}" "${EVAL_PY}"
        --csv_path "${csv_path}"
        --json_dir "${json_dir}"
        --json_name "${stem}"
        --index_start "${INDEX_START}"
        --index_end "${end_idx}"
        --feature "${label}"
        --device "${DEVICE}"
        --output_json "${out_json}"
      )

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[DRY]';
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
      fi

      if "${cmd[@]}"; then
        ok=$((ok + 1))
      else
        echo "[FAIL] ${label} / ${csv_stem}"
        failed=$((failed + 1))
      fi
    done
  done
done < <(list_labels)

echo "=============================================="
echo "done: total=${total} ok=${ok} skipped=${skipped} failed=${failed}"
echo "outputs: ${OUT_DIR}"
echo "=============================================="

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi

# 234485   