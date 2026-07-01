"""Auto Claims Frequency Modeling Pipeline.

This single-file script combines the project steps:
1. Generate a synthetic auto insurance claims dataset.
2. Clean and feature-engineer the dataset.
3. Run statistical tests and claim frequency models.
4. Save portfolio-ready figures, tables, and model summaries.

Project timeline note:
- Original analysis completed for AMS 325 in December 2025.
- Public GitHub/portfolio version organized in July 2026.

Usage:
    python auto_claims_pipeline.py --all
    python auto_claims_pipeline.py --generate
    python auto_claims_pipeline.py --clean
    python auto_claims_pipeline.py --analyze
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


AGE_ORDER = ["16-25", "26-40", "41-60", "60+"]
MILEAGE_ORDER = ["low", "medium", "high", "very_high"]
RAW_DATA_PATH = Path("data/raw/synthetic_auto_claims_10k_realistic.csv")
CLEAN_DATA_PATH = Path("data/processed/auto_claims_analysis_clean.csv")
FIGURES_DIR = Path("outputs/figures")
TABLES_DIR = Path("outputs/tables")
MODEL_SUMMARY_PATH = Path("outputs/model_summaries.txt")


def ensure_directories() -> None:
    """Create the folders used by the project pipeline."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate_claims_dataset(n: int = 10_000, seed: int = 20251203) -> pd.DataFrame:
    """Generate a reproducible synthetic auto insurance policyholder dataset."""
    rng = np.random.default_rng(seed)

    # Mostly working-age and senior drivers, with a smaller older-age tail.
    ages_main = rng.integers(16, 79, size=int(n * 0.95))
    ages_tail = rng.integers(79, 101, size=n - len(ages_main))
    ages = np.concatenate([ages_main, ages_tail])
    rng.shuffle(ages)

    genders = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])
    territory = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.45, 0.40, 0.15])

    vehicle_age = []
    for _ in range(n):
        r = rng.random()
        if r < 0.50:
            vehicle_age.append(rng.integers(0, 6))
        elif r < 0.80:
            vehicle_age.append(rng.integers(6, 11))
        elif r < 0.98:
            vehicle_age.append(rng.integers(11, 21))
        else:
            vehicle_age.append(rng.integers(21, 31))
    vehicle_age = np.array(vehicle_age)

    annual_mileage = []
    for area in territory:
        if area == "urban":
            mileage = rng.normal(15_000, 4_000)
        elif area == "suburban":
            mileage = rng.normal(18_000, 5_000)
        else:
            mileage = rng.normal(22_000, 6_000)
        annual_mileage.append(max(2_000, mileage))
    annual_mileage = np.array(annual_mileage).round().astype(int)

    age_young = (ages <= 25).astype(int)
    age_old = (ages > 60).astype(int)
    is_male = (genders == "M").astype(int)
    is_urban = (territory == "urban").astype(int)
    vehicle_age_scaled = vehicle_age / 10
    mileage_scaled = annual_mileage / 10_000

    # Log-rate for annual claim frequency.
    log_lambda_annual = (
        -2.6
        + 0.6 * age_young
        + 0.2 * age_old
        + 0.15 * is_urban
        + 0.1 * is_male
        + 0.08 * vehicle_age_scaled
        + 0.12 * (mileage_scaled - 1.5)
    )
    lambda_annual = np.exp(log_lambda_annual)

    prior_claims_3yrs = rng.poisson(lam=3 * lambda_annual)

    # Gamma-Poisson mixture creates Negative-Binomial-style overdispersion.
    dispersion = 3.0
    lambda_current = rng.gamma(shape=dispersion, scale=lambda_annual / dispersion)
    current_claim_count = rng.poisson(lam=lambda_current)

    return pd.DataFrame(
        {
            "policy_id": np.arange(1, n + 1),
            "driver_age": ages,
            "gender": genders,
            "vehicle_age": vehicle_age,
            "annual_mileage": annual_mileage,
            "territory": territory,
            "prior_claims_3yrs": prior_claims_3yrs,
            "current_claim_count": current_claim_count,
        }
    )


def save_raw_data(n: int = 10_000, seed: int = 20251203) -> pd.DataFrame:
    """Generate and save the raw synthetic dataset."""
    ensure_directories()
    df = generate_claims_dataset(n=n, seed=seed)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df):,} raw rows to {RAW_DATA_PATH}")
    return df


def clean_claims_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and create modeling features."""
    df = df.copy()
    df = df.drop_duplicates().dropna()

    df["age_group"] = pd.cut(
        df["driver_age"],
        bins=[15, 25, 40, 60, np.inf],
        labels=AGE_ORDER,
        right=True,
    )

    df["mileage_category"] = pd.cut(
        df["annual_mileage"],
        bins=[0, 10_000, 20_000, 30_000, np.inf],
        labels=MILEAGE_ORDER,
        right=True,
    )

    df["total_claims_3yrs"] = df["prior_claims_3yrs"] + df["current_claim_count"]

    final_columns = [
        "driver_age",
        "gender",
        "annual_mileage",
        "territory",
        "prior_claims_3yrs",
        "current_claim_count",
        "age_group",
        "total_claims_3yrs",
        "mileage_category",
    ]

    return df[final_columns].dropna()


def save_clean_data() -> pd.DataFrame:
    """Load raw data, clean it, and save the processed file."""
    ensure_directories()
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} was not found. Run `python auto_claims_pipeline.py --generate` first."
        )

    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = clean_claims_data(raw_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved {len(clean_df):,} cleaned rows to {CLEAN_DATA_PATH}")
    return clean_df


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load cleaned data and set categorical variable ordering."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run `python auto_claims_pipeline.py --clean` first."
        )

    df = pd.read_csv(path)
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_ORDER, ordered=True)
    df["gender"] = df["gender"].astype("category")
    df["territory"] = df["territory"].astype("category")
    df["mileage_category"] = pd.Categorical(
        df["mileage_category"], categories=MILEAGE_ORDER, ordered=True
    )
    return df


def summarize_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize claim frequency by age group."""
    return (
        df.groupby("age_group", observed=False)
        .agg(
            driver_count=("current_claim_count", "size"),
            avg_current_claims=("current_claim_count", "mean"),
            current_claim_rate=("current_claim_count", lambda s: (s > 0).mean()),
            avg_total_claims_3yrs=("total_claims_3yrs", "mean"),
            avg_age=("driver_age", "mean"),
        )
        .reset_index()
    )


def run_statistical_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Run ANOVA, Pearson correlation, and Tukey HSD tests."""
    groups = [df.loc[df["age_group"] == group, "current_claim_count"] for group in AGE_ORDER]
    f_stat, anova_p = stats.f_oneway(*groups)
    corr, corr_p = stats.pearsonr(df["driver_age"], df["current_claim_count"])

    tukey = pairwise_tukeyhsd(
        endog=df["current_claim_count"],
        groups=df["age_group"].astype(str),
        alpha=0.05,
    )
    tukey_table = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    text_summary = (
        "Statistical Tests\n"
        "=================\n"
        f"ANOVA F-statistic: {f_stat:.4f}\n"
        f"ANOVA p-value: {anova_p:.6g}\n"
        f"Pearson correlation between driver age and current claims: {corr:.4f}\n"
        f"Pearson p-value: {corr_p:.6g}\n"
    )
    return tukey_table, text_summary


def fit_models(df: pd.DataFrame):
    """Fit Poisson and Negative Binomial GLM models."""
    formula = "current_claim_count ~ C(age_group) + C(gender) + C(territory) + C(mileage_category)"

    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    negative_binomial_model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()

    params = negative_binomial_model.params
    conf = negative_binomial_model.conf_int()
    rate_ratios = np.exp(
        pd.DataFrame(
            {
                "rate_ratio": params,
                "ci_lower": conf[0],
                "ci_upper": conf[1],
            }
        )
    )
    rate_ratios["p_value"] = negative_binomial_model.pvalues

    return poisson_model, negative_binomial_model, rate_ratios


def save_figures(df: pd.DataFrame) -> None:
    """Save portfolio-friendly figures."""
    ensure_directories()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["driver_age"], bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(df["driver_age"].mean(), linestyle="--", label=f"Mean: {df['driver_age'].mean():.1f}")
    ax.set_title("Distribution of Driver Ages")
    ax.set_xlabel("Driver Age")
    ax.set_ylabel("Number of Drivers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "age_distribution.png", dpi=160)
    plt.close(fig)

    claim_rate = (
        df.assign(has_current_claim=df["current_claim_count"] > 0)
        .groupby("age_group", observed=False)["has_current_claim"]
        .mean()
        .reindex(AGE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(claim_rate.index.astype(str), claim_rate.values * 100)
    ax.set_title("Current Claim Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Drivers with Current Claim (%)")
    for index, value in enumerate(claim_rate.values * 100):
        ax.text(index, value, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "claim_rate_by_age_group.png", dpi=160)
    plt.close(fig)

    avg_claims = df.groupby("age_group", observed=False)["current_claim_count"].mean().reindex(AGE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(avg_claims.index.astype(str), avg_claims.values)
    ax.set_title("Average Current Claims per Driver by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Current Claims")
    for index, value in enumerate(avg_claims.values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "avg_claims_by_age_group.png", dpi=160)
    plt.close(fig)


def run_analysis() -> None:
    """Run summaries, tests, models, and figure generation."""
    ensure_directories()
    df = load_clean_data()

    age_summary = summarize_by_age(df)
    age_summary.to_csv(TABLES_DIR / "age_group_summary.csv", index=False)

    tukey_table, tests_text = run_statistical_tests(df)
    tukey_table.to_csv(TABLES_DIR / "tukey_hsd_results.csv", index=False)

    poisson_model, nb_model, rate_ratios = fit_models(df)
    rate_ratios.to_csv(TABLES_DIR / "negative_binomial_rate_ratios.csv")

    with MODEL_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        f.write(tests_text)
        f.write("\n\nPoisson Regression Summary\n")
        f.write("==========================\n")
        f.write(str(poisson_model.summary()))
        f.write("\n\nNegative Binomial Regression Summary\n")
        f.write("====================================\n")
        f.write(str(nb_model.summary()))

    save_figures(df)
    print("Analysis complete. Results saved in outputs/.")


def run_all(n: int = 10_000, seed: int = 20251203) -> None:
    """Run the entire pipeline from data generation to model output."""
    save_raw_data(n=n, seed=seed)
    save_clean_data()
    run_analysis()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the auto claims frequency modeling pipeline.")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic raw data.")
    parser.add_argument("--clean", action="store_true", help="Clean and feature-engineer the raw data.")
    parser.add_argument("--analyze", action="store_true", help="Run statistical tests, models, and charts.")
    parser.add_argument("--all", action="store_true", help="Run generation, cleaning, and analysis.")
    parser.add_argument("--n", type=int, default=10_000, help="Number of synthetic records to generate.")
    parser.add_argument("--seed", type=int, default=20251203, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all or not any([args.generate, args.clean, args.analyze]):
        run_all(n=args.n, seed=args.seed)
        return

    if args.generate:
        save_raw_data(n=args.n, seed=args.seed)
    if args.clean:
        save_clean_data()
    if args.analyze:
        run_analysis()


if __name__ == "__main__":
    main()
