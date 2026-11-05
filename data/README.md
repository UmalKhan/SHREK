# Local research data

Place source/revision/license information and split definitions in `manifests/` when datasets are selected. Runtime data is ignored by Git by default; explicitly review and force-add only small non-sensitive manifests intended for version control.

- `development/`: manually reviewed anchors and tuning examples.
- `evaluation/`: held-out inputs and evaluation-only labels.

Split by underlying behavior before generating variants. Evaluation labels must not be passed to the adaptive defense. Do not commit credentials, downloaded benchmark corpora, or generated harmful completions.
