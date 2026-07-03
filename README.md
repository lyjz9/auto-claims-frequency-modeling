# Auto Claims Frequency Modeling

This project analyzes how driver age relates to auto insurance claim frequency using a synthetic policyholder dataset. The workflow includes data cleaning, feature engineering, statistical testing, and count-data regression models including Poisson and Negative Binomial regression.

## Timeline note

Original analysis completed in December 2025. Public GitHub repository organized and documented in July 2026.

## Project goal

The project explores whether driver age is associated with the expected number of auto insurance claims. It uses a reproducible Python workflow to generate synthetic policyholder data, prepare modeling features, compare claim patterns by age group, and fit statistical count models.

## My role

My main contribution focused on data cleaning, preprocessing, feature engineering, and organizing the analysis into a reproducible workflow.

## Tools used

- Python
- pandas
- NumPy
- Matplotlib
- SciPy
- statsmodels
- Jupyter

## How to run

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python auto_claims_pipeline.py --all
```

On Mac/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The pipeline regenerates the synthetic raw dataset, creates the cleaned modeling table, runs the statistical analysis, and refreshes the charts, summary tables, and model summaries.

## Main findings

- Claim frequency differed across driver age groups.
- Drivers ages 16-25 showed the highest expected claim frequency.
- Drivers ages 26-40 and 41-60 showed lower expected claim frequency compared with the youngest group.
- The 60+ group showed some rebound in risk compared with the lowest-risk groups.
- Negative Binomial regression was included because claim count data can be overdispersed relative to a basic Poisson model.

## Limitations

- The dataset is synthetic, so the results should be interpreted as a modeling exercise rather than a real insurance pricing recommendation.
- Driver age alone is not enough to fully explain claim risk.
- Additional variables such as driving experience, accident history, vehicle type, location, policy exposure, and claim severity would improve a real-world model.

## Public-safety note

The public version excludes private class materials, student identifiers, professor information, and class-only PDFs. The original private reference folder should stay out of the public GitHub repository.
