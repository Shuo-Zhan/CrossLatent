# GitHub upload preparation

Upload the contents of the CrossLatent directory as the repository root. Do not upload the surrounding research workspace. This directory contains no figures, result snapshots, datasets, credentials or checkpoints.

Before public release, confirm the authors' chosen license and any inherited source obligations, then add LICENSE and author-approved citation information. A license has deliberately not been invented. Add a public manuscript/DOI link when available.

Using an empty GitHub repository named CrossLatent, run locally inside this directory:

```sh
git init
git add .
git commit -m "Add CrossLatent core implementation"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/CrossLatent.git
git push -u origin main
```

Replace YOUR_ACCOUNT first. These instructions have not been executed; no repository has been created or uploaded automatically. Review `git diff --cached --stat` before committing. Tests and the demo are documented in README.md. Keep generated demo outputs and build artifacts out of Git.
