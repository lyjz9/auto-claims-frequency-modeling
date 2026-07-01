# Portfolio Project Page: Auto Insurance Claim Frequency Modeling

## Page title

Auto Insurance Claim Frequency Modeling

## One-sentence intro

Built a Python-based statistical analysis project to examine whether driver age affects auto insurance claim frequency using simulated policyholder data and count-regression models.

## Hero section

**Problem:** Insurance companies need a way to understand which driver segments are more likely to file claims so pricing and risk management decisions can be more data-informed.

**Question:** Does driver age affect the expected number of auto insurance claims?

**My role:** Data cleaning, preprocessing, feature engineering, and organizing the project into a reproducible GitHub-ready workflow.

**Tools:** Python, pandas, NumPy, SciPy, statsmodels, Matplotlib, Jupyter Notebook

**Timeline note:** Original analysis completed in December 2025. Public GitHub repository organized and documented in July 2026 as a portfolio-ready version.

## What the project click-through page should show

When someone clicks the project card, the page should not just show screenshots. It should read like a mini case study:

1. **Overview:** What the project is and why it matters.
2. **Problem / research question:** The main question you answered.
3. **Data:** What the dataset contained and whether it was real or simulated.
4. **Your contribution:** What you personally worked on.
5. **Methods:** Cleaning, feature engineering, statistical tests, and models.
6. **Visuals:** 2–3 charts with short explanations.
7. **Results:** The main findings in plain English.
8. **Limitations:** What the project cannot fully prove.
9. **Links:** GitHub repo, single-file pipeline, generated outputs, and any live preview.

## Data

The project used 10,000 simulated auto insurance policy records. Each record represented a policyholder and included fields such as driver age, gender, vehicle age, territory, annual mileage, prior claims, and current claim count.

Because the original project did not use a real-world insurance dataset, the simulated data made it possible to practice actuarial-style claim frequency modeling while keeping the workflow reproducible.

## Workflow

1. Generated a synthetic auto insurance dataset.
2. Removed irrelevant fields and incomplete records.
3. Engineered age groups: 16–25, 26–40, 41–60, and 60+.
4. Engineered mileage categories: low, medium, high, and very high.
5. Compared claim frequency across age groups.
6. Ran ANOVA, Tukey HSD, and Pearson correlation analysis.
7. Fit Poisson and Negative Binomial regression models.
8. Interpreted age-group rate ratios and model limitations.
9. Organized the final workflow into `auto_claims_pipeline.py`, which can be run with `python auto_claims_pipeline.py --all`.

## Key findings

- Ages **16–25** showed the highest expected claim frequency.
- Ages **26–40** and **41–60** showed lower claim frequency compared with the youngest group.
- The **60+** group remained below the 16–25 group but showed some rebound in risk.
- Age was statistically significant, but the overall model fit was still limited, showing that claim frequency likely depends on more variables than age alone.

## Limitations

- The dataset was simulated, not real insurance company data.
- Age alone is not enough to build a strong predictive pricing model.
- Important real-world variables such as driving experience, accident severity, vehicle type, credit/rating factors, weather, and road conditions were not included.
- The project is best presented as a statistical modeling and data-cleaning project, not a production insurance pricing model.

## Resume bullet options

- Cleaned and feature-engineered a 10,000-record synthetic auto insurance dataset using Python and pandas to support claim frequency modeling by driver age group.
- Applied ANOVA, Tukey HSD, Pearson correlation, Poisson regression, and Negative Binomial regression to evaluate age-based differences in auto insurance claim frequency.
- Organized a class-based analysis into a reproducible GitHub-ready workflow with a single-file pipeline, generated outputs, and portfolio documentation.

## Short project card copy

**Auto Insurance Claim Frequency Modeling**  
Python statistical modeling project analyzing how driver age affects auto insurance claim frequency using synthetic policyholder data, feature engineering, ANOVA/Tukey tests, and count-regression models.

**Tags:** Python · pandas · Statistical Modeling · Poisson Regression · Negative Binomial Regression · Data Cleaning
