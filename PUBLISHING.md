# Publish this repository

The directory is prepared for GitHub but does not assume an account, repository URL, or remote. Publishing is a separate action from generating these files.

1. Read `README.md`, `SOURCES.md`, and `reference/README.md` for scope and attribution. Permission to distribute the original PDF has been confirmed; its existing rights notice is preserved separately from the companion's MIT license.
2. Run the validation commands in the README.
3. Inspect `git status --short` and the files selected for upload. The `.venv` directory and private drafts must remain excluded.
4. Create an empty GitHub repository with a name such as `learning-theory-to-optimization`. Use the account and visibility you intend.
5. If this folder is not yet a Git repository, run `git init -b main`. Then run `git add .` and `git commit -m "Add self-study companion and reproducible labs"` using your own configured Git identity.
6. Add the remote URL supplied by GitHub using `git remote add origin YOUR_REPOSITORY_URL`, then `git push -u origin main`. Replace the placeholder rather than running it literally.

Suggested description: “A self-study journey through learning theory and optimization: derivations, solved exercises, and reproducible NumPy experiments.”

Suggested topics: `machine-learning`, `optimization`, `self-study`, `gradient-descent`, `stochastic-gradient-descent`, `probability`, `learning-notes`.

No website hosting is required: GitHub renders the Markdown and math directly. The CI workflow checks links, mathematical invariants, and lab reproducibility after a push. Local validation and hosted CI are different; a local pass does not imply GitHub Actions has run.
