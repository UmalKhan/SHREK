# Input JSONL contract

Each line must contain exactly: `request_id`, `text`, `behavior_group`, `split`, `source`, `evaluation_label`.

- Split: `development`, `adaptation`, or `holdout`.
- Evaluation label: `harmful` or `legitimate`.
- IDs/groups: letters, numbers, `_`, `.`, `:`, `-`.
- `source` should identify the benchmark/revision; maintain licensing and provenance separately.

Audit the complete file before selecting a split. Group related variants together. Only `request_id` and `text` become runtime requests; labels and source metadata are never passed to model roles. CSV review files deliberately include labels for independent manual scoring after execution.

`smoke.jsonl` has only two authored legitimate inputs. It tests integration, not jailbreak resistance or learned-rule effectiveness. The baseline-demo command uses separate synthetic fixtures with fictional harmful markers, not real attacks.
