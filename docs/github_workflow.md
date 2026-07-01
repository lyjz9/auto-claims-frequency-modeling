# GitHub Workflow

## Recommended repo name

`auto-claims-frequency-modeling`

## Before publishing

For a public repo, do **not** publish `private_reference_do_not_publish/`. It may include original class-submission materials, names, student IDs, and local file paths. The `.gitignore` already excludes it if you use the terminal workflow below.

## Terminal commands

Open PowerShell in this folder and run:

```powershell
git init
git add .
git commit -m "Organize auto claims frequency modeling project"
git branch -M main
git remote add origin https://github.com/lyjz9/auto-claims-frequency-modeling.git
git push -u origin main
```

The repository is published under the `lyjz9` GitHub account.

## Better workflow going forward

Use one branch per improvement:

```powershell
git checkout -b improve-readme
# edit files
git add .
git commit -m "Improve project documentation"
git push -u origin improve-readme
```

Then open a pull request on GitHub.

## Suggested first issues

- Add more model diagnostics.
- Add a confusion-free explanation of Poisson vs. Negative Binomial regression.
- Add a live portfolio case study page.
- Add more features to the synthetic dataset, such as years of driving experience or vehicle type.
- Add tests for the data cleaning function.
