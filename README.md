# Learning Theory → Optimization

**A self-study field guide to generalization, gradient methods, and stochastic approximation.**

[中文详解与证明](notes/zh/README.md) · [English reading path](#reading-path)

中文部分按学习进度补到课程前三章，含风险与双下降、GD/SGD、渐近推断、集中不等式和一致收敛。正文共九篇，另附线性代数工具及作业 4、5 的推导。关键计算写到可逐行核对，条件与反例放在相应结论旁。

The English chapters provide a shorter reading path; the Chinese edition develops the proofs and later material in greater detail. Experiments use synthetic data and fixed seeds.

![Four views of learning and optimization](assets/overview.png)

## Start here

1. Read the [study map](notes/00-study-map.md) and [notation](notes/notation.md).
2. Work through [one scalar gradient-descent problem](exercises/worked-example.md).
3. Follow the chapters below. Attempt the checkpoint before opening its solution.
4. Run a lab, explain what its plot does **and does not** establish, and record evidence in the [learning ledger](study/LEARNING_LOG.md).

## Reading path

| Chapter | The question it answers | Source coverage |
|---|---|---|
| [00 · Study map](notes/00-study-map.md) | How do the results fit together? | Included Lectures 1–2 |
| [01 · Risk and representation](notes/01-risk-and-representation.md) | What are we minimizing, and what do we actually want? | §§1.1–1.3 |
| [02 · Capacity, validation, and ridge](notes/02-capacity-and-ridge.md) | Why can a better fit predict worse? | §1.4 |
| [03 · Interpolation and double descent](notes/03-double-descent.md) | How can more parameters help after interpolation? | §1.5 |
| [04 · Geometry and exact GD](notes/04-geometry-and-gd.md) | Which step sizes shrink the error? | §2.1 |
| [05 · General GD proofs](notes/05-gd-proof-patterns.md) | Where do geometric and inverse-time rates come from? | §§2.2–2.3 |
| [06 · Stochastic gradient descent](notes/06-sgd.md) | What changes when the gradient is noisy? | §2.4 |
| [07 · Uncertainty and averaging](notes/07-asymptotic-uncertainty.md) | What does a distributional limit tell us beyond an error bound? | §2.5 |
| [08 · Momentum](notes/08-momentum.md) | Why does remembering past motion sometimes accelerate optimization? | §2.6 |
| [09 · Adam and preconditioning](notes/09-adam.md) | Which parts of adaptive optimization can we actually prove? | §2.7 |
| [10 · Probability toolkit](notes/10-probability.md) | How do convergence modes, stochastic orders, and the delta method work? | Appendix A |

These opening lectures concern supervised learning and optimization. They are prerequisites for deep learning and reinforcement learning; this repository does **not** claim to teach a full neural-network or reinforcement-learning curriculum.

## Practice and reproduce

- [Worked example: from a derivative to a convergence proof](exercises/worked-example.md)
- [Problem set with hints](exercises/problems.md) · [Complete solutions](exercises/solutions.md)
- [Proof selection guide](study/PROOF_PLAYBOOK.md) · [Rate comparison](study/RATE_SHEET.md)
- [Experiments and interpretation](labs/README.md)
- [Technical qualifications and corrections](notes/precision-notes.md)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_repository.py
python -m unittest discover -s tests -v
python labs/run_all.py
```

CI uses Python 3.12; the latest recorded tail diagnostic used Python 3.13.15. Experiments use NumPy and Matplotlib, synthetic data, and fixed random seeds. Each run writes its plots, measurements, and report into a new dated directory under `runs/`, leaving the published reference results unchanged. Use `python labs/run_all.py --output-dir runs/my-experiment` to choose a new directory; existing paths are refused. See [the reproducibility report](labs/RESULTS.md) for measured results, runtime versions, and limitations. Numerical checks supplement proofs; they do not prove universal convergence claims.

## Learning record and scope

The 2026-10-26 update consolidates the documented study of Chapters 1–3 from the course's 2026-10-14 version, including SGD inference and Homework 4–5. The [Chinese edition](notes/zh/README.md) contains the expanded material; the English path remains a concise companion to the older course version. Reading and worked solutions are recorded separately from independent mastery in the [ledger](study/LEARNING_LOG.md).

## Source, attribution, and reuse

The [original lecture PDF](reference/lecture_note2.pdf), *Deep Learning and Reinforcement Learning: Lecture notes — Lectures 1–2 with Probability Appendix*, is included with redistribution permission confirmed by the repository maintainer. Its author and institution are not identified in the supplied copy. See [source provenance and reading references](SOURCES.md) and the [section map](reference/SECTION_MAP.md).

The expanded Chinese notes also consult the 91-page Lectures 1–4 version compiled on 2026-10-14; that later source PDF is not distributed here. See the [two-version crosswalk](reference/SECTION_MAP.md).

This companion is an independently written explanation, not an official course edition, verbatim translation, or set of official homework answers. New prose, exercises, code, and generated figures are provided under the [MIT License](LICENSE). The original PDF is explicitly excluded from that license; see [its rights notice](reference/README.md). References retain their own rights.

Contributions should improve understanding and preserve assumptions. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [validation record](VALIDATION.md). To publish your copy, follow [PUBLISHING.md](PUBLISHING.md).
