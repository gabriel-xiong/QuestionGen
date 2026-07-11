# QuestionGen — Misconception-Tagged AP Bio Item Generator

Fine-tune a small open model (**Qwen3-1.7B**, QLoRA) to do **one narrow thing
reliably**: given a topic and a set of target misconceptions, generate an AP
Biology multiple-choice question where **every wrong answer is a deliberate,
named misconception** — output as clean JSON with each distractor tagged.

**Thesis:** *behavior comes from data, not model size.* A prompted 1.7B can't do
this reliably; a by-construction dataset makes it. Start with
[`docs/behavior_spec.md`](docs/behavior_spec.md) (the falsifiable spec) and
[`docs/brainlift_generator.md`](docs/brainlift_generator.md) (thesis + evidence).

## Results (distractor mapping, 0–2, real gpt-4o judge)
| | base 1.7B | **tuned 1.7B** | gpt-4o (prompted) |
|---|---|---|---|
| in-distribution | 1.05 | **1.80** | 1.90 |
| out-of-distribution (unseen topic) | 0.70 | 1.40 | 1.75 |

- In-distribution, the tuned 1.7B **ties prompted gpt-4o** (and beats it on task
  quality) — despite gpt-4o also being the judge, and genetics scored objectively.
- Genetics (objective recompute): **base 0/40 → tuned 40/40.**
- **Data-iteration win:** v1 (3 topics) overfit — negative OOD transfer; adding 2
  topics flipped OOD to **+0.65 over base**. Fixed in data, not hyperparameters.

## Quickstart
- **Full pipeline (train → eval → push → demo):** open
  [`notebooks/run_all_pipeline.ipynb`](notebooks/run_all_pipeline.ipynb) in Colab
  (GPU runtime), run top to bottom.
- **Deploy the persistent demo:** run
  [`notebooks/deploy_space.ipynb`](notebooks/deploy_space.ipynb) → free CPU
  Hugging Face Space with a permanent URL.
- **Guides:** [`docs/summary.md`](docs/summary.md) (one-pager),
  [`docs/demo.md`](docs/demo.md) (demo + hosting).

## How the dataset is built (the real artifact)
By construction, not scraped (harvesting gave ~51% "no-fit" filler distractors).
Each misconception is applied as an **error operator** to generate its distractor,
so the tag is ground-truth:
- **Genetics (procedural):** Punnett solver + error operators → objectively
  verifiable by recomputation.
- **Conceptual (cell resp, enzymes, membrane transport, evolution):** curated
  frames, each with ≥3 competing misconceptions.

**Corpus: 2,046 items, 50/50 procedural/conceptual, 5 topics.** Validated by
independent recompute (genetics 1023/1023) + human review.

## Folder layout
```
scripts/
  gen_genetics.py            procedural (Punnett) generator + error operators
  gen_cellresp.py  gen_enzymes.py  gen_membrane.py  gen_evolution.py
  conceptual_engine.py       shared frame-based generator
  build_dataset.py           orchestrates corpus + held-out split
  gen_spec.py                the generation prompt (shared by train + eval)
  train_gen_sft.py           QLoRA SFT (Unsloth), completion-only masking
  eval_generation.py         base-vs-tuned harness (hf / openai / mock generators)
  score_rubric.py  judge.py  rubric scorer + calibrated LLM judge
  validate_corpus.py         independent validation + human-review loop
  app.py                     Gradio inference demo (for the HF Space)
  set_api_key.py             write API key to gitignored .local/.env
data/
  gen_train.jsonl            the corpus (items)
  gen_sft_train.jsonl  gen_sft_val.jsonl   SFT chat pairs
  eval_scenarios.jsonl       held-out in-distribution scenarios
  eval_scenarios_ood.jsonl   out-of-distribution (photosynthesis) scenarios
  apbio_misconceptions.json  misconception taxonomy (ids + definitions)
docs/
  behavior_spec.md  brainlift_generator.md  summary.md  demo.md
notebooks/
  run_all_pipeline.ipynb     end-to-end (train, eval, demo)
  deploy_space.ipynb         create the free HF Space
```

## Eval design
Rubric per output (0/1/2): spec adherence, **distractor mapping** (the core
metric), task quality, plus reliability / forbidden-failure aggregates. Genetics
scored **objectively by recomputation**; conceptual scored by an LLM judge
**calibrated to human labels** (gpt-4o-mini failed calibration — caught 0/15
injected errors; gpt-4o hit 15/15). Held-out scenarios are disjoint from training;
photosynthesis is held out entirely for the OOD test.

## Secrets
API keys load from a **gitignored** `.local/.env` (or env vars). Never paste keys
into notebook cells — use the `getpass`/`login()` prompts. See `set_api_key.py`.

---
*Note: this repo began as an algebra error-type classifier (see `docs/spec.md`,
`docs/brainlift.md`); those legacy artifacts are retained.*
