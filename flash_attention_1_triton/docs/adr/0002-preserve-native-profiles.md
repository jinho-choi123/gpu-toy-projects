# Preserve native profiles as the source of truth

Each profiling run preserves its native Nsight Systems or Nsight Compute report, while a single long-form CSV is derived for cross-tool analysis. This costs more storage than CSV-only output but retains the complete evidence needed to regenerate summaries or inspect metrics that the initial schema did not select.
