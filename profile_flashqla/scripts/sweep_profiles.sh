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

Run the FlashQLA and FLA Triton Nsight Systems/Compute sweep for sequence
lengths 16K, 32K, and 64K and strong-decay head ratios 0.0, 0.5, and 1.0.

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
    local backend="$2"
    local entrypoint="$3"
    local seq_len="$4"
    local ratio="$5"
    local strong_decay_heads="$6"
    local artifact_prefix
    local output_base
    local report_path
    local snapshot_source
    local snapshot_target
    local -a cmd

    artifact_prefix="${profiles_dir}/${backend}_b${batch_size}_t${seq_len}_h${num_heads}_sdh${strong_decay_heads}"
    output_base="${artifact_prefix}_${profiler}"
    snapshot_source="${artifact_prefix}_memory_snapshot.pickle"
    snapshot_target="${artifact_prefix}_${profiler}_memory_snapshot.pickle"
    case "${profiler}" in
        nsys)
            report_path="${output_base}.nsys-rep"
            ;;
        ncu)
            report_path="${output_base}.ncu-rep"
            ;;
        *)
            fail "unknown profiler: ${profiler}"
            ;;
    esac

    current_job=$((current_job + 1))
    printf '[%02d/%d] %s | T=%s | strong-decay=%s (%s/%s heads) | %s\n' \
        "${current_job}" "${total_jobs}" "${backend}" "${seq_len}" "${ratio}" \
        "${strong_decay_heads}" "${num_heads}" "${profiler}"

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
                --cuda-graph-trace=node
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
                --batch-size "${batch_size}"
                --seq-len "${seq_len}"
                --num-heads "${num_heads}"
                --strong-decay-head-ratio "${ratio}"
                --seed "${seed}"
                --iterations "${nsys_iterations}"
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
                --graph-profiling node
                --nvtx
                --nvtx-include 'gdn_forward/'
                --profile-from-start=off
            )
            if ((force == 1)); then
                cmd+=(--force-overwrite)
            fi
            cmd+=(
                "--export=${output_base}"
                "${python_bin}"
                "${entrypoint}"
                --batch-size "${batch_size}"
                --seq-len "${seq_len}"
                --num-heads "${num_heads}"
                --strong-decay-head-ratio "${ratio}"
                --seed "${seed}"
                --iterations "${ncu_iterations}"
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

total_jobs=$((${#profilers[@]} * ${#backends[@]} * ${#sequence_lengths[@]} * ${#strong_decay_ratios[@]}))

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

printf 'Profiling sweep: %d jobs (2 backends x 3 sequence lengths x 3 ratios x 2 profilers)\n' "${total_jobs}"
if ((dry_run == 1)); then
    printf 'Dry-run mode: no commands will be executed.\n'
fi

for profiler in "${profilers[@]}"; do
    for backend in "${backends[@]}"; do
        case "${backend}" in
            flash_qla)
                entrypoint="${script_dir}/profile_gdn_flash_qla.py"
                ;;
            fla_triton)
                entrypoint="${script_dir}/profile_gdn_fla_triton.py"
                ;;
        esac

        for seq_len in "${sequence_lengths[@]}"; do
            for ratio_index in "${!strong_decay_ratios[@]}"; do
                run_profile \
                    "${profiler}" \
                    "${backend}" \
                    "${entrypoint}" \
                    "${seq_len}" \
                    "${strong_decay_ratios[${ratio_index}]}" \
                    "${strong_decay_head_counts[${ratio_index}]}"
            done
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
