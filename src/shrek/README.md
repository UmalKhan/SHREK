# Planned modules

Create these as their behavior is implemented; avoid empty implementation stubs.

- `config.py`: parse and validate experimental configuration.
- `schemas.py`: rules, interactions, proposals, memory versions, and reports.
- `model_client.py`: provider interface and complete usage accounting.
- `baseline.py`: faithful paper-based defender implementation.
- `feedback.py`: online feedback without access to evaluation labels.
- `proposals.py`: add, revise, archive, and atomic replacement proposals.
- `memory.py`: versioned state, preview, atomic commit, archival, and rollback.
- `probes.py`: time-appropriate development and historical probe selection.
- `validation.py`: frozen before/after replay with learning disabled.
- `maintenance.py`: bounded periodic revision/archival.
- `runner.py`: chronological experimental execution.
- `metrics.py`: security, utility, memory, delay, and cost measurements.
