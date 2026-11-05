# Implemented and planned modules

Implemented in Phases 1-2: `config.py`, `schemas.py`, `model_client.py`, `memory.py`, `data.py`, `baseline.py`, `backends.py`, `runner.py`, `hardware.py`, and labeled offline fixtures. Feedback and rule induction currently live in `baseline.py`.

The following decomposition remains a guide for later work; no empty implementation stubs are created:

- `config.py`: parse and validate experimental configuration.
- `schemas.py`: rules, interactions, proposals, memory versions, and reports.
- `model_client.py`: provider interface and complete usage accounting.
- `baseline.py`: paper-based defender adaptation with documented fidelity differences.
- `feedback.py`: online feedback without access to evaluation labels.
- `proposals.py`: add, revise, archive, and atomic replacement proposals.
- `memory.py`: versioned state, preview, atomic commit, archival, and rollback.
- `probes.py`: time-appropriate development and historical probe selection.
- `validation.py`: frozen before/after replay with learning disabled.
- `maintenance.py`: bounded periodic revision/archival.
- `runner.py`: chronological experimental execution.
- `metrics.py`: security, utility, memory, delay, and cost measurements.
