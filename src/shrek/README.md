# Implemented and planned modules

Implemented in Phases 1-4: the initial foundation/baseline modules plus `probes.py`, `validation.py`, `controller.py`, `system.py`, `prepare.py`, `study.py`, and `scoring.py`. The controller owns candidate acceptance and periodic maintenance; baseline induction and feedback remain in `baseline.py`. Offline fixtures are explicitly marked synthetic.

The following original decomposition explains responsibilities; some are combined in the implemented modules rather than empty standalone stubs:

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
