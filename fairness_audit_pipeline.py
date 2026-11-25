"""
Fairness Audit of an AI Model for Pre-Approved Credit-Card Offers
Tasks 1 to 6 Implementation using Scikit-Learn, Fairlearn, and Matplotlib.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)


def main():
    base_dir = r"C:\Users\yugm chaudhary\Desktop\practical RSA C\practrical C"
    os.chdir(base_dir)

    print("=" * 70)
    print("FAIRNESS AUDIT: PRE-APPROVED CREDIT-CARD OFFERS PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Task 1: Load, Clean, One-Hot Encode, Scale, Train Logistic Regression
    # -------------------------------------------------------------
    csv_file = os.path.join(base_dir, "adult.csv")
    df = pd.read_csv(csv_file)

    # Clean missing values '?'
    df_clean = df.replace('?', np.nan).dropna()

    # Target: High Income (>50K) -> 1, Low Income (<=50K) -> 0
    y = (df_clean['income'].str.strip() == '>50K').astype(int)

    # Sensitive attribute: Sex ('Male', 'Female')
    A = df_clean['sex'].str.strip()

    # Features: Exclude target and sensitive feature ("sex")
    X = df_clean.drop(columns=['income', 'sex'])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    # Split 75:25 (stratified on y, random_state=42)
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.25, random_state=42, stratify=y
    )

    clf = Pipeline([
        ('prep', preprocessor),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    clf.fit(X_train, y_train)

    print("\n[Task 1] Data cleaned and encoded successfully.")
    print(f"Total instances: {len(df_clean)} | Features (without sex): {X.shape[1]}")
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

    # -------------------------------------------------------------
    # Task 2: Baseline Performance & MetricFrame by Sex
    # -------------------------------------------------------------
    y_pred_base = clf.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)

    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'tpr': true_positive_rate,
        'fpr': false_positive_rate
    }

    mf_base = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_base,
        sensitive_features=A_test
    )

    # -------------------------------------------------------------
    # Task 3: Demographic Parity Difference, Ratio & Equalized Odds
    # -------------------------------------------------------------
    dp_diff_base = demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_test)
    dp_ratio_base = demographic_parity_ratio(y_test, y_pred_base, sensitive_features=A_test)
    eo_diff_base = equalized_odds_difference(y_test, y_pred_base, sensitive_features=A_test)

    print("\n" + "=" * 70)
    print("Expected Output — Part A: Baseline Fairness Audit (sensitive feature = sex)")
    print("=" * 70)
    print(f"Baseline model : Logistic Regression   Accuracy : {base_acc:.2f}\n")
    print("GROUP     ACCURACY   SELECTION_RATE    TPR     FPR")
    print("-" * 51)
    for group in ['Male', 'Female']:
        acc = mf_base.by_group.loc[group, 'accuracy']
        sel = mf_base.by_group.loc[group, 'selection_rate']
        tpr = mf_base.by_group.loc[group, 'tpr']
        fpr = mf_base.by_group.loc[group, 'fpr']
        print(f"{group:<9} {acc:.2f}         {sel:.2f}          {tpr:.2f}    {fpr:.2f}")
    print("-" * 51)
    bias_tag = "   (< 0.80 -> BIAS)" if dp_ratio_base < 0.80 else ""
    print(f"Demographic parity difference : {dp_diff_base:.2f}")
    print(f"Demographic parity ratio (DI)  : {dp_ratio_base:.2f}{bias_tag}")
    print(f"Equalized odds difference      : {eo_diff_base:.2f}")

    # -------------------------------------------------------------
    # Task 4: Mitigation with ThresholdOptimizer (Demographic Parity)
    # -------------------------------------------------------------
    to_dp = ThresholdOptimizer(
        estimator=clf,
        constraints='demographic_parity',
        predict_method='predict_proba',
        prefit=True
    )
    to_dp.fit(X_train, y_train, sensitive_features=A_train)
    y_pred_dp = to_dp.predict(X_test, sensitive_features=A_test)

    acc_dp = accuracy_score(y_test, y_pred_dp)
    mf_dp = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_dp,
        sensitive_features=A_test
    )
    dp_diff_dp = demographic_parity_difference(y_test, y_pred_dp, sensitive_features=A_test)
    eo_diff_dp = equalized_odds_difference(y_test, y_pred_dp, sensitive_features=A_test)

    # -------------------------------------------------------------
    # Task 5: Mitigation with ThresholdOptimizer (Equalized Odds)
    # -------------------------------------------------------------
    to_eo = ThresholdOptimizer(
        estimator=clf,
        constraints='equalized_odds',
        predict_method='predict_proba',
        prefit=True
    )
    to_eo.fit(X_train, y_train, sensitive_features=A_train)
    y_pred_eo = to_eo.predict(X_test, sensitive_features=A_test)

    acc_eo = accuracy_score(y_test, y_pred_eo)
    mf_eo = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_eo,
        sensitive_features=A_test
    )
    dp_diff_eo = demographic_parity_difference(y_test, y_pred_eo, sensitive_features=A_test)
    eo_diff_eo = equalized_odds_difference(y_test, y_pred_eo, sensitive_features=A_test)

    # -------------------------------------------------------------
    # Task 6: Accuracy vs Fairness Table & Selection Rate Plot
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Expected Output — Part B: Mitigation with ThresholdOptimizer")
    print("=" * 70)
    print("MODEL                 ACC    SEL_M   SEL_F   DP_DIFF   EO_DIFF")
    print("-" * 62)

    sel_m_base = mf_base.by_group.loc['Male', 'selection_rate']
    sel_f_base = mf_base.by_group.loc['Female', 'selection_rate']

    sel_m_dp = mf_dp.by_group.loc['Male', 'selection_rate']
    sel_f_dp = mf_dp.by_group.loc['Female', 'selection_rate']

    sel_m_eo = mf_eo.by_group.loc['Male', 'selection_rate']
    sel_f_eo = mf_eo.by_group.loc['Female', 'selection_rate']

    print(f"{'Baseline':<21} {base_acc:.2f}   {sel_m_base:.2f}    {sel_f_base:.2f}    {dp_diff_base:.2f}      {eo_diff_base:.2f}")
    print(f"{'TO: demographic_par.':<21} {acc_dp:.2f}   {sel_m_dp:.2f}    {sel_f_dp:.2f}    {dp_diff_dp:.2f}      {eo_diff_dp:.2f}")
    print(f"{'TO: equalized_odds':<21} {acc_eo:.2f}   {sel_m_eo:.2f}    {sel_f_eo:.2f}    {dp_diff_eo:.2f}      {eo_diff_eo:.2f}")
    print("-" * 62)

    # Plot selection rate by sex
    models = ['Baseline', 'TO: demographic_par.', 'TO: equalized_odds']
    male_rates = [sel_m_base, sel_m_dp, sel_m_eo]
    female_rates = [sel_f_base, sel_f_dp, sel_f_eo]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, male_rates, width, label='Male', color='#2980b9')
    rects2 = ax.bar(x + width/2, female_rates, width, label='Female', color='#e74c3c')

    ax.set_ylabel('Selection Rate (P(Offer = 1))', fontsize=12)
    ax.set_title('Selection Rate by Sex across Models (Fairness Audit)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 0.35)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, weight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, weight='bold')

    plot_path = os.path.join(base_dir, 'selection_rate_by_sex.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved : selection_rate_by_sex.png")
    print("No single model satisfies both fairness definitions.")
    print("=" * 70)


if __name__ == '__main__':
    main()
