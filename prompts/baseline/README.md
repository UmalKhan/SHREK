# Baseline prompt location and provenance

Executable role prompts live in `src/shrek/prompt_templates/` so they are included in installed wheels. They are newly worded JSON adaptations of Hu and Hooi's Section 3 and Appendix A.6 (https://arxiv.org/html/2608.26008v1), not official copied prompt files. Fidelity differences are tracked in `docs/baseline.md`. Changing a prompt changes the method and must be recorded in the run manifest/source revision.
