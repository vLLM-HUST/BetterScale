# Shared prototype host admission

`admit_subset.py` is the unchanged helper formerly under
`attention-client/device-service`. It is retained for joint-wave, owned-wave
and Qwen serving launchers after retirement of the AE prototype.

Its external `PROBE_HELPERS` dependency and behavior are unchanged. Current
repository/host allocation instructions remain authoritative; this relocation
does not authorize a new accelerator run or broaden a device lease.
