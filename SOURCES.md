# Sources, provenance, and editorial scope

## The course document

Primary course source: *Deep Learning and Reinforcement Learning: Lecture notes — Lectures 1–2 with Probability Appendix*, supplied as `lecture_note2.pdf`, 40 PDF pages. The document identifies no author or institution; this repository does not invent either. The original is in [reference/](reference/README.md), with a [section crosswalk](reference/SECTION_MAP.md).

Public redistribution permission was confirmed by the repository maintainer. The original PDF remains outside the repository's MIT license. The new companion explains standard mathematical material in new prose, reorganizes it for self-study, adds worked examples and original exercises, and generates all experiment figures from synthetic data. It does not reproduce private dialogue or claim the original lecture's authorship.

AI assistance was used in drafting and coding. Numerical invariants, calculations, links, and generated plots were checked locally; independent expert review has not been claimed. The 2026-10-26 revision consolidates the documented study through Chapter 3. Reading coverage and supplied solutions do not certify independent mastery.

## Expanded Chinese edition: source version

The Chinese edition also uses *Deep Learning and Reinforcement Learning: Lecture notes — Lectures 1–4*, compiled on **October 14, 2026**, 91 PDF pages. The supplied version does not identify an author or institution. Only Chapters 1–3 are consolidated here; Chapter 4 and a full reinforcement-learning course are outside this update. The later source PDF is not redistributed in this repository.

Version fingerprint (SHA-256): `5760cc2b5ea1722994650a0cd1ba1caf71eff39045ee3d5f681c401c5c197af8`.

The new prose combines the documented mathematical discussion with separately developed calculations, examples, proofs and the assumptions needed to make them valid. The inference chapter's equal-length, late-trajectory batch-means argument uses explicitly stronger sufficient conditions. Fundamental results such as martingale CLTs and functional CLTs are stated as inputs, with the required verification distinguished from a proof of the foundational theorem itself.

Homework 4 and 5 are reconstructed mathematical statements and worked solutions, not copies of assignment PDFs or official solutions. Their textbook exercise attributions appear in the [first](exercises/zh/homework04.md) and [second](exercises/zh/homework05.md) solution sets. No private conversation text, personal study questions, or identifying local paths are published.

## Further reading linked in the companion

- Guillaume Garrigos and Robert M. Gower, *Handbook of Convergence Theorems for (Stochastic) Gradient Methods*, arXiv:2301.11235. [Author preprint](https://arxiv.org/abs/2301.11235). A proof reference for gradient methods, minibatches, and momentum.
- Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J. Tibshirani, *Surprises in High-Dimensional Ridgeless Least Squares Interpolation*, Annals of Statistics 50(2), 2022. [Author preprint](https://arxiv.org/abs/1903.08560). Background for risk in overparameterized regression.
- B. T. Polyak and A. B. Juditsky, *Acceleration of Stochastic Approximation by Averaging*, SIAM Journal on Control and Optimization 30(4), 1992. [Publisher record](https://doi.org/10.1137/0330046). Classical trajectory averaging.
- Diederik P. Kingma and Jimmy Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015. [Author preprint](https://arxiv.org/abs/1412.6980). Algorithm definition and original motivation.
- Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar, *On the Convergence of Adam and Beyond*, ICLR 2018. [Author preprint](https://arxiv.org/abs/1904.09237). Counterexamples and the AMSGrad modification.

- Ilya Loshchilov and Frank Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. [Author preprint](https://arxiv.org/abs/1711.05101). Algorithm definition and the difference between adaptive regularization and decoupled decay.
- Keller Jordan and collaborators, *Muon*. [Author implementation](https://github.com/KellerJordan/Muon). Matrix orthogonalization and the finite-step polynomial used in the optimizer. The classical Newton–Schulz proof in the notes is not a convergence theorem for that modified polynomial or for network training.

These landing pages were checked while preparing their respective editions. The companion's detailed derivations primarily follow the supplied course material and explicit calculations here; no claim is made to have re-proved every result in these research works. Additional textbook and research references appear on PDF pages 39–40 of the source.

## Citation suggestion

Cite this repository descriptively as *Learning Theory to Optimization: A Self-Study Companion*, 2026 edition. Do not fabricate an author list, DOI, institutional affiliation, or course endorsement. A future tagged release can add a `CITATION.cff` with verified repository metadata.
