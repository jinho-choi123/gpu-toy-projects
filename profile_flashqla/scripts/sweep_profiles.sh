#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${VIRTUAL_ENV:-${project_dir}/.venv}/bin/python"
profiles_dir="${project_dir}/profiles"
ncu_root_tmpdir="/tmp/ncu-root"
ncu_sudo_bin="/usr/local/cuda/bin/ncu"

backends=(flash_qla fla_triton)
profilers=(nsys ncu)
sequence_lengths=(16384 32768 65536)
strong_decay_ratios=(0.0 0.5 1.0)
strong_decay_head_counts=(0 8 16)

batch_size=1
num_heads=16
seed=42
nsys_iterations=10
ncu_iterations=1

dry_run=0
force=0
sudo_ncu=0
current_job=0
skipped_jobs=0
ncu_prepared=0

usage() {
    cat <<'EOF'
Usage: sweep_profiles.sh [OPTIONS]

Run the FlashQLA and FLA Triton Nsight Systems/Compute sweep for the synthetic
GDN workload and Qwen3.8-27B checkpoint prefill at 16K, 32K, and 64K.

Options:
  --dry-run    Print commands without running them.
  --force      Overwrite existing reports and snapshots instead of skipping.
  --sudo-ncu   Run Nsight Compute with sudo and TMPDIR=/tmp/ncu-root.
  -h, --help   Show this help message.
EOF
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

print_command() {
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

report_exists() {
    local profiler="$1"
    local output_base="$2"

    case "${profiler}" in
        nsys)
            [[ -e "${output_base}.nsys-rep" ]]
            ;;
        ncu)
            [[ -e "${output_base}.ncu-rep" ]]
            ;;
        *)
            fail "unknown profiler: ${profiler}"
            ;;
    esac
}

prepare_sudo_ncu() {
    ((sudo_ncu == 1)) || return 0
    ((ncu_prepared == 0)) || return 0

    local -a setup_command=(sudo install -d -m 700 "${ncu_root_tmpdir}")
    printf 'Preparing Nsight Compute temporary directory:\n'
    print_command "${setup_command[@]}"
    if ((dry_run == 0)); then
        "${setup_command[@]}"
    fi
    ncu_prepared=1
}

finalize_snapshot() {
    local profiler="$1"
    local report_path="$2"
    local snapshot_source="$3"
    local snapshot_target="$4"

    [[ -e "${snapshot_source}" ]] || fail "memory snapshot not found: ${snapshot_source}"
    if ((sudo_ncu == 1)) && [[ "${profiler}" == ncu ]]; then
        sudo chown -- "$(id -u):$(id -g)" "${report_path}" "${snapshot_source}"
    fi
    mv -- "${snapshot_source}" "${snapshot_target}"
}

run_profile() {
    local profiler="$1"
    local workload="$2"
    local backend="$3"
    local entrypoint="$4"
    local seq_len="$5"
    local ratio="${6:-}"
    local strong_decay_heads="${7:-}"
    local artifact_prefix
    local job_description
    local iterations
    local output_base
    local report_path
    local snapshot_source
    local snapshot_target
    local -a cmd
    local -a profile_args

    case "${workload}" in
        gdn)
            artifact_prefix="${profiles_dir}/${backend}_b${batch_size}_t${seq_len}_h${num_heads}_sdh${strong_decay_heads}"
            job_description="${backend} | T=${seq_len} | strong-decay=${ratio} (${strong_decay_heads}/${num_heads} heads)"
            ;;
        qwen38_27b)
            artifact_prefix="${profiles_dir}/qwen38_27b_${backend}_b1_t${seq_len}"
            job_description="Qwen3.8-27B ${backend} | T=${seq_len}"
            ;;
        *)
            fail "unknown workload: ${workload}"
            ;;
    esac

    output_base="${artifact_prefix}_${profiler}"
    snapshot_source="${artifact_prefix}_memory_snapshot.pickle"
    snapshot_target="${artifact_prefix}_${profiler}_memory_snapshot.pickle"
    case "${profiler}" in
        nsys)
            report_path="${output_base}.nsys-rep"
            iterations="${nsys_iterations}"
            ;;
        ncu)
            report_path="${output_base}.ncu-rep"
            iterations="${ncu_iterations}"
            ;;
        *)
            fail "unknown profiler: ${profiler}"
            ;;
    esac

    if [[ "${workload}" == gdn ]]; then
        profile_args=(
            --batch-size "${batch_size}"
            --seq-len "${seq_len}"
            --num-heads "${num_heads}"
            --strong-decay-head-ratio "${ratio}"
            --seed "${seed}"
            --iterations "${iterations}"
        )
    else
        profile_args=(--seq-len "${seq_len}")
    fi

    current_job=$((current_job + 1))
    printf '[%02d/%d] %s | %s\n' \
        "${current_job}" "${total_jobs}" "${job_description}" "${profiler}"

    if ((force == 0)); then
        if report_exists "${profiler}" "${output_base}"; then
            if [[ -e "${snapshot_target}" ]]; then
                printf '  skip: report and snapshot already exist\n'
            elif [[ -e "${snapshot_source}" ]]; then
                printf '  recover: moving unfinished snapshot to %s\n' "${snapshot_target}"
                if ((dry_run == 0)); then
                    finalize_snapshot \
                        "${profiler}" \
                        "${report_path}" \
                        "${snapshot_source}" \
                        "${snapshot_target}"
                fi
            else
                fail "report exists but snapshot is missing: ${report_path}; rerun with --force"
            fi
            skipped_jobs=$((skipped_jobs + 1))
            return
        fi

        [[ ! -e "${snapshot_source}" ]] || fail \
            "unclaimed memory snapshot exists: ${snapshot_source}; rerun with --force"
        [[ ! -e "${snapshot_target}" ]] || fail \
            "snapshot exists without its report: ${snapshot_target}; rerun with --force"
    fi

    case "${profiler}" in
        nsys)
            cmd=(
                nsys profile
                --trace=cuda,nvtx
                --sample=none
            )
            if [[ "${workload}" == gdn ]]; then
                cmd+=(--cuda-graph-trace=node)
            fi
            cmd+=(
                --capture-range=cudaProfilerApi
                --capture-range-end=stop
            )
            if ((force == 1)); then
                cmd+=(--force-overwrite=true)
            fi
            cmd+=(
                "--output=${output_base}"
                "${python_bin}"
                "${entrypoint}"
                "${profile_args[@]}"
            )
            ;;
        ncu)
            prepare_sudo_ncu
            if ((sudo_ncu == 1)); then
                cmd=(sudo env "TMPDIR=${ncu_root_tmpdir}" "${ncu_sudo_bin}")
            else
                cmd=(ncu)
            fi
            cmd+=(
                --set basic
            )
            if [[ "${workload}" == gdn ]]; then
                cmd+=(--graph-profiling node --nvtx --nvtx-include 'gdn_forward/')
            else
                cmd+=(
                    --nvtx
                    --nvtx-include 'regex:qwen38_gdn_decoder_layer_[0-9]+_gdn_ordinal_[0-9]+/'
                )
            fi
            cmd+=(--profile-from-start=off)
            if ((force == 1)); then
                cmd+=(--force-overwrite)
            fi
            cmd+=(
                "--export=${output_base}"
                "${python_bin}"
                "${entrypoint}"
                "${profile_args[@]}"
            )
            ;;
    esac

    print_command "${cmd[@]}"
    printf '  snapshot: %s\n' "${snapshot_target}"
    if ((dry_run == 0)); then
        "${cmd[@]}"
        finalize_snapshot \
            "${profiler}" \
            "${report_path}" \
            "${snapshot_source}" \
            "${snapshot_target}"
    fi
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        --force)
            force=1
            ;;
        --sudo-ncu)
            sudo_ncu=1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
    shift
done

gdn_jobs=$((${#profilers[@]} * ${#backends[@]} * ${#sequence_lengths[@]} * ${#strong_decay_ratios[@]}))
qwen_jobs=$((${#profilers[@]} * ${#backends[@]} * ${#sequence_lengths[@]}))
total_jobs=$((gdn_jobs + qwen_jobs))

if ((dry_run == 0)); then
    [[ -x "${python_bin}" ]] || fail \
        "Python environment not found: ${python_bin}; run 'uv run --locked ./scripts/sweep_profiles.sh' from ${project_dir}"
    require_command nsys
    if ((sudo_ncu == 1)); then
        require_command sudo
        [[ -x "${ncu_sudo_bin}" ]] || fail "Nsight Compute not found: ${ncu_sudo_bin}"
    else
        require_command ncu
    fi
    mkdir -p "${profiles_dir}"
fi

printf 'Profiling sweep: %d jobs (%d GDN + %d Qwen3.8-27B)\n' \
    "${total_jobs}" "${gdn_jobs}" "${qwen_jobs}"
if ((dry_run == 1)); then
    printf 'Dry-run mode: no commands will be executed.\n'
fi

for profiler in "${profilers[@]}"; do
    for backend in "${backends[@]}"; do
        case "${backend}" in
            flash_qla)
                gdn_entrypoint="${script_dir}/profile_gdn_flash_qla.py"
                qwen_entrypoint="${script_dir}/profile_qwen38_27b_flash_qla.py"
                ;;
            fla_triton)
                gdn_entrypoint="${script_dir}/profile_gdn_fla_triton.py"
                qwen_entrypoint="${script_dir}/profile_qwen38_27b_fla_triton.py"
                ;;
        esac

        for seq_len in "${sequence_lengths[@]}"; do
            for ratio_index in "${!strong_decay_ratios[@]}"; do
                run_profile \
                    "${profiler}" \
                    gdn \
                    "${backend}" \
                    "${gdn_entrypoint}" \
                    "${seq_len}" \
                    "${strong_decay_ratios[${ratio_index}]}" \
                    "${strong_decay_head_counts[${ratio_index}]}"
            done
            run_profile \
                "${profiler}" \
                qwen38_27b \
                "${backend}" \
                "${qwen_entrypoint}" \
                "${seq_len}"
        done
    done
done

if ((dry_run == 1)); then
    printf 'Dry run complete: %d jobs shown, %d existing reports skipped.\n' \
        "${total_jobs}" "${skipped_jobs}"
else
    printf 'Sweep complete: %d jobs scheduled, %d existing reports skipped.\n' \
        "${total_jobs}" "${skipped_jobs}"
fi
