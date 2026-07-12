# QuestionGen: AI-generated AP Biology questions with meaningful wrong answers

A fine-tuned small language model that writes AP Biology multiple-choice
questions where **every wrong answer reflects a specific, well-known student
misconception**, not a random distractor. The model returns a clean, structured
JSON item (question, four options, answer key, and the misconception behind each
wrong choice), which is exactly what an adaptive learning or test-prep tool needs
to diagnose *why* a student got something wrong.

The interesting result: a **1.7B-parameter model**, small enough to run locally
and for free, was trained to do this reliably and performs on par with a much
larger frontier model on the task, at a tiny fraction of the size and cost.

## Highlights
- **Purpose-built dataset** of ~2,000 questions across 5 biology topics, generated
  *by construction* so every distractor provably maps to a named misconception.
- **Evaluation harness built before training**, scoring generations on four
  dimensions (below) with programmatic checks plus an LLM judge validated
  against human labels.
- **Large, measured gains over the base model** on every dimension.
- **A data-iteration case study:** diagnosed a generalization failure and fixed it
  by improving the *data* (adding topic coverage), not the training settings.
- Trained with **QLoRA (Unsloth)** on a single GPU; packaged with a Gradio demo.

## How it's evaluated (four dimensions)
Every generated question is scored 0 to 2 on:

1. **Spec adherence**: is the output a single, valid, well-formed JSON item
   (four distinct options, exactly one correct answer, every wrong option tagged)?
2. **Distractor mapping**: does each wrong answer genuinely embody the
   misconception it's labeled with? *(the core metric)*
3. **Task quality**: is the biology correct, with the right answer keyed and the
   distractors actually wrong and plausible?
4. **Consistency**: does the model behave reliably across similar prompts?

The eval was written *before* any training. Outputs are scored with programmatic
structural checks plus an LLM judge that was validated against human labels before
being trusted. Models are compared on held-out prompts the model never trained on.

That validation mattered. On a calibration set seeded with deliberately
mislabeled distractors, a cheaper judge (gpt-4o-mini) missed every planted error
while gpt-4o matched the human labels exactly, so gpt-4o was chosen as the judge.

## Results (base vs. fine-tuned vs. frontier, 0 to 2)
| Dimension | Base Qwen3-1.7B | Fine-tuned | GPT-4o (prompted) |
|---|---|---|---|
| Spec adherence | 1.80 | **2.00** | 2.00 |
| Distractor mapping | 1.05 | **1.80** | 1.90 |
| Task quality | 1.71 | **2.00** | 1.95 |
| Consistency (fully-correct rate) | 15% | **80%** | 90% |

The fine-tuned 1.7B model reaches near-parity with prompted GPT-4o on this task,
at a fraction of the size and cost.

- On a **topic held out of training entirely**, fine-tuning still improved
  misconception mapping over the base model, and improving the dataset's topic
  coverage is what made that generalization possible.

## Base vs. fine-tuned on the same request
Both models were asked to write a **photosynthesis** item (a topic held out of
training) embedding three specific misconceptions, one per wrong answer.

**Base Qwen3-1.7B** tags only one wrong answer; the other two are off-spec:
```
Q: In photosynthesis, where does the oxygen (O2) released come from?
  A. CO2            misconception: ps_o2_comes_from_co2
  B. water (H2O)    [correct]
  C. ATP            (untagged)
  D. NADPH          (untagged)
```

**Fine-tuned** puts a requested misconception behind every wrong answer:
```
Q: What is the direct source of reduction power (ATP and NADPH) in the Calvin cycle?
  A. The oxygen released at the end of respiration comes from the CO2.
       misconception: ps_o2_comes_from_co2
  B. The light reactions produce it directly.    [correct]
  C. The Calvin cycle itself requires light directly.
       misconception: ps_calvin_needs_light_directly
  D. It comes from the light reactions, which fix CO2 into organic molecules.
       misconception: ps_light_dark_reaction_swap
```

## How it works
Good exam distractors are *engineered*, not scraped. Real question banks are full
of filler wrong answers that don't correspond to any coherent misconception, so the
training data is generated **by construction**: each misconception is turned into
an "error operator" that produces the exact wrong answer a student holding that
belief would choose, which guarantees the label is correct. The dataset spans
five biology topics (cellular respiration, enzymes, membrane transport, evolution,
and genetics), each item pairing three competing misconceptions, and is validated
with automated checks and human spot-checks before an unseen split is held out for
evaluation.

## A sample training pair
The model is fine-tuned on pairs of a **request** (topic plus the misconceptions to
embed) and the **JSON item** it should produce:

*Input (abridged):*
```
Write one AP Biology multiple-choice item on the topic: enzymes.
Embed EXACTLY these misconceptions, one per wrong answer choice:
  - enz_change_equilibrium: enzymes shift the equilibrium / change dG
  - enz_raise_activation_energy: enzymes increase activation energy
  - enz_specificity_any_substrate: an enzyme can act on any substrate
Output a single JSON object (stem, choices A-D, correct, distractor_tags).
```

*Output:*
```json
{"stem": "How does an enzyme increase the rate of a chemical reaction?",
 "choices": {"A": "It lowers the activation energy required for the reaction to proceed.",
             "B": "It increases the activation-energy barrier of the reaction.",
             "C": "It binds and converts any available molecule to force the reaction.",
             "D": "It shifts the reaction's equilibrium toward the products."},
 "correct": "A",
 "distractor_tags": {"B": {"misconception_id": "enz_raise_activation_energy"},
                     "C": {"misconception_id": "enz_specificity_any_substrate"},
                     "D": {"misconception_id": "enz_change_equilibrium"}}}
```

## Tech stack
Python · Qwen3-1.7B · QLoRA / LoRA (Unsloth, PEFT) · Hugging Face Transformers &
Hub · Gradio · LLM-as-judge plus programmatic scoring for evaluation.

## Quickstart
- **Full pipeline** (train and evaluate): `notebooks/run_all_pipeline.ipynb`
  (open in Colab with a GPU runtime).
- **Model:** [`gabriel-xiong/apbio-item-generator-qwen3-1.7b-lora`](https://huggingface.co/gabriel-xiong/apbio-item-generator-qwen3-1.7b-lora)
  on the Hugging Face Hub. Load it on top of `Qwen/Qwen3-1.7B` with PEFT.
- **Read more:** [`docs/summary.md`](docs/summary.md) (one-page overview),
  [`docs/brainlift_generator.md`](docs/brainlift_generator.md) (full write-up),
  [`docs/behavior_spec.md`](docs/behavior_spec.md) (the target behavior definition).

## Repository layout
```
scripts/
  gen_genetics.py                 procedural (Punnett) generator + error operators
  gen_cellresp/enzymes/membrane/evolution.py   conceptual topic generators
  conceptual_engine.py            shared frame-based generator
  build_dataset.py                assembles the corpus + train/eval split
  gen_spec.py                     the generation prompt (shared by train + eval)
  train_gen_sft.py                QLoRA fine-tuning (Unsloth)
  eval_generation.py              base-vs-tuned evaluation harness
  score_rubric.py / judge.py      rubric scorer + validated LLM judge
  validate_corpus.py              independent validation + human-review loop
  app.py                          Gradio inference demo
data/                             the dataset (items, train/val splits, eval sets)
docs/                             overview, full write-up, behavior spec, demo guide
notebooks/                        end-to-end pipeline + demo/deploy notebooks
```

## A note on secrets
API keys load from a git-ignored `.local/.env` (or environment variables) and are
never committed. Use the notebooks' `getpass` / `login()` prompts rather than
pasting keys into cells.

---
*This repository previously hosted an algebra error-classification project; those
earlier files (`docs/spec.md`, `docs/brainlift.md`, legacy `data/`) are retained.*
