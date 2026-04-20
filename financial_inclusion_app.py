# ============================================================
# Financial Inclusion of Rural Women in India
# Streamlit App — ABA Final Project
# Adapted from reference: RandomForestRegressor + Smoothing
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Inclusion — Rural Women India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3c5e 0%, #2e7d52 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; font-size: 1.6rem; margin: 0; }
    .main-header p  { color: #c8e6d4; font-size: 0.9rem; margin: 0.3rem 0 0; }
    .metric-card {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-card h2 { color: #1a3c5e; font-size: 1.8rem; margin: 0; }
    .metric-card p  { color: #555; font-size: 0.85rem; margin: 0.2rem 0 0; }
    .section-header {
        border-left: 4px solid #2e7d52; padding-left: 0.8rem;
        color: #1a3c5e; margin-bottom: 1rem;
    }
    .insight-box {
        background: #e8f4e8; border-left: 3px solid #2e7d52;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.9rem;
    }
    .warn-box {
        background: #fff8e1; border-left: 3px solid #f9a825;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.9rem;
    }
    .hypothesis-card {
        background: white; border: 1px solid #ddd;
        border-radius: 8px; padding: 0.9rem; margin: 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & PREPARATION ────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    """Load NFHS-5 + RBI merged dataset and compute branch density."""

    # ── core data ─────────────────────────────────────────────────────────
    data = {
        'State': [
            'Andaman & Nicobar Islands','Andhra Pradesh','Arunachal Pradesh',
            'Assam','Bihar','Chhattisgarh',
            'Dadra and Nagar Haveli & Daman and Diu','Goa','Gujarat',
            'Haryana','Himachal Pradesh','Jammu & Kashmir','Jharkhand',
            'Karnataka','Kerala','Ladakh','Madhya Pradesh','Maharashtra',
            'Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Puducherry',
            'Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana','Tripura',
            'Uttar Pradesh','Uttarakhand','West Bengal'
        ],
        'BankAccount': [
            89.75,79.63,76.74,77.86,76.21,81.14,89.28,92.38,67.48,
            72.39,82.23,83.52,79.75,87.68,78.22,88.65,73.34,70.91,
            70.71,68.22,74.97,55.41,87.38,96.72,82.10,79.00,76.70,
            91.72,85.21,77.67,74.06,79.75,73.24
        ],
        'Literacy': [
            85.58,63.84,71.63,75.39,54.54,71.30,67.90,93.43,69.03,
            78.75,91.23,74.71,59.34,70.95,97.47,76.60,63.30,79.51,
            84.84,85.53,87.67,82.74,69.22,89.07,80.04,61.63,86.24,
            81.50,58.10,76.88,65.55,79.78,72.54
        ],
        'MobileOwn': [
            80.86,40.93,75.28,53.87,49.26,33.94,45.53,87.06,36.18,
            43.41,77.77,73.25,43.67,53.42,86.90,81.22,31.44,43.07,
            68.19,64.29,70.64,76.33,47.95,76.56,54.90,45.30,83.25,
            68.93,50.60,48.03,42.37,55.65,39.07
        ],
        'InternetUse': [
            27.87,15.35,49.61,24.36,17.00,20.78,23.84,68.28,17.53,
            42.80,45.23,38.90,22.69,24.76,57.54,53.98,20.08,23.71,
            40.39,27.95,47.98,40.26,21.33,50.37,48.75,30.76,68.08,
            39.20,15.78,17.66,24.48,39.35,13.98
        ],
        'HHDecision': [
            95.52,84.30,86.64,91.77,87.02,91.54,87.43,98.60,90.67,
            86.15,93.90,81.66,89.84,80.52,94.61,80.27,84.13,89.15,
            95.00,91.98,98.02,99.82,90.32,96.55,90.25,86.79,93.94,
            93.65,86.16,89.47,86.78,90.64,85.81
        ],
        'PaidCash': [
            17.12,44.50,23.55,19.27,12.80,42.62,39.47,27.86,34.05,
            17.03,17.71,18.50,17.74,41.36,25.79,28.25,28.00,39.60,
            44.03,39.11,29.00,20.67,25.58,44.84,20.21,17.52,29.25,
            45.51,55.54,25.93,14.83,22.22,20.17
        ],
        'AssetOwn': [
            19.36,50.63,70.25,43.85,55.67,45.45,59.74,24.07,43.27,
            41.03,23.38,60.77,66.47,69.65,29.18,72.95,41.27,24.50,
            58.93,70.11,28.36,28.86,45.47,38.30,67.07,26.61,50.57,
            52.04,74.46,17.28,53.53,25.13,22.46
        ],
        'ChildMarriage': [
            15.27,32.90,19.33,33.35,43.36,13.20,26.16,3.18,26.91,
            13.65,5.14,5.26,36.09,24.66,8.22,3.13,26.55,27.57,
            17.61,19.12,14.00,7.25,21.71,0.97,8.68,28.28,12.54,
            15.18,27.38,42.40,17.85,9.76,48.13
        ],
        'BankOffices': [
            71,7252,164,2957,7463,2823,110,697,8626,
            5224,1652,1767,3169,10778,6801,68,7234,13578,
            203,364,203,176,5223,267,6735,7775,162,
            11829,5388,565,18041,2194,9251
        ],
        'Population_2021': [
            417000,53903000,1570000,35607000,124800000,29436000,
            894000,1539000,70400000,29506000,7380000,13635000,38593000,
            67562000,35125000,290000,85358000,122153000,3436000,
            3443000,1239000,2150000,44395000,1694000,30141000,
            79502000,671000,76396000,37220000,4169000,
            224979000,11439000,96906000
        ]
    }

    df = pd.DataFrame(data)
    df['BranchDensity'] = (df['BankOffices'] / (df['Population_2021'] / 1000)).round(4)

    # ── SMOOTHING ENCODING on State (Target Encoding — adapted from reference) ─
    # In reference file: TargetEncoder used for 'brand' and 'model'
    # Here: State is our categorical identifier — encode with mean target smoothing
    global_mean = df['BankAccount'].mean()
    smoothing_factor = 5  # controls how much we smooth toward global mean

    state_stats = df.groupby('State')['BankAccount'].agg(['mean', 'count'])
    state_stats['smoothed'] = (
        (state_stats['mean'] * state_stats['count'] + global_mean * smoothing_factor) /
        (state_stats['count'] + smoothing_factor)
    )
    df['State_Encoded'] = df['State'].map(state_stats['smoothed'])

    return df

df = load_and_prepare_data()

# ── FEATURE / TARGET DEFINITION ───────────────────────────────────────────
FEATURE_COLS = ['Literacy','MobileOwn','InternetUse','HHDecision',
                'PaidCash','AssetOwn','ChildMarriage','BranchDensity']
FEATURE_LABELS = {
    'Literacy':      'Female Literacy %',
    'MobileOwn':     'Mobile Ownership %',
    'InternetUse':   'Internet Usage %',
    'HHDecision':    'HH Decision Making %',
    'PaidCash':      'Women Paid in Cash %',
    'AssetOwn':      'Asset Ownership %',
    'ChildMarriage': 'Child Marriage %',
    'BranchDensity': 'Branch Density (per 1,000)'
}
TARGET_COL = 'BankAccount'

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── MODEL TRAINING (following reference structure) ─────────────────────────
@st.cache_resource
def train_models(X, y):
    """Train M1 (base RF), M2 (OLS), M3 (Ridge), M4 (RF + outlier treatment)."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── M1: Base RandomForestRegressor (same as reference) ─────────────────
    m1 = RandomForestRegressor(random_state=42, n_estimators=100)
    m1.fit(X_train, y_train)
    y_pred_m1 = m1.predict(X_test)
    r2_m1   = r2_score(y_test, y_pred_m1)
    rmse_m1 = np.sqrt(mean_squared_error(y_test, y_pred_m1))

    # ── M2: OLS Regression (statsmodels — for p-values and hypothesis testing)
    X_train_c = sm.add_constant(X_train)
    X_test_c  = sm.add_constant(X_test)
    m2 = sm.OLS(y_train, X_train_c).fit()
    y_pred_m2 = m2.predict(X_test_c)
    r2_m2   = r2_score(y_test, y_pred_m2)
    rmse_m2 = np.sqrt(mean_squared_error(y_test, y_pred_m2))

    # ── M3: Ridge Regression (handles multicollinearity) ──────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    m3 = Ridge(alpha=1.0, random_state=42)
    m3.fit(X_train_sc, y_train)
    y_pred_m3 = m3.predict(X_test_sc)
    r2_m3   = r2_score(y_test, y_pred_m3)
    rmse_m3 = np.sqrt(mean_squared_error(y_test, y_pred_m3))

    # ── M4: RF with IQR Outlier Treatment (same logic as reference df_m4) ─
    X_m4 = X.copy()
    y_m4_series = y.copy()
    df_m4 = X_m4.copy()
    df_m4['BankAccount'] = y_m4_series.values

    median_vals = {}
    for col in FEATURE_COLS:
        Q1 = df_m4[col].quantile(0.25)
        Q3 = df_m4[col].quantile(0.75)
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR
        non_out_median = df_m4[(df_m4[col] >= lb) & (df_m4[col] <= ub)][col].median()
        median_vals[col] = non_out_median
        df_m4.loc[(df_m4[col] < lb) | (df_m4[col] > ub), col] = non_out_median

    X_m4_clean = df_m4[FEATURE_COLS]
    y_m4_clean = df_m4['BankAccount']

    X_train_m4, X_test_m4, y_train_m4, y_test_m4 = train_test_split(
        X_m4_clean, y_m4_clean, test_size=0.2, random_state=42
    )
    m4 = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8)
    m4.fit(X_train_m4, y_train_m4)
    y_pred_m4 = m4.predict(X_test_m4)
    r2_m4   = r2_score(y_test_m4, y_pred_m4)
    rmse_m4 = np.sqrt(mean_squared_error(y_test_m4, y_pred_m4))

    # Cross-validation on best model (M4)
    cv_scores = cross_val_score(m4, X_m4_clean, y_m4_clean, cv=5, scoring='r2')

    return {
        'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
        'scaler': scaler,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_m1': y_pred_m1, 'y_pred_m2': y_pred_m2,
        'y_pred_m3': y_pred_m3, 'y_pred_m4': y_pred_m4,
        'y_test_m4': y_test_m4,
        'r2_m1': r2_m1,   'rmse_m1': rmse_m1,
        'r2_m2': r2_m2,   'rmse_m2': rmse_m2,
        'r2_m3': r2_m3,   'rmse_m3': rmse_m3,
        'r2_m4': r2_m4,   'rmse_m4': rmse_m4,
        'cv_scores': cv_scores,
        'median_vals': median_vals,
    }

results = train_models(X, y)

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📋 Dataset Explorer",
    "🔍 Exploratory Analysis",
    "🤖 Model Comparison (M1–M4)",
    "📈 OLS Regression & Hypotheses",
    "🌟 Feature Importance",
    "🔮 Predict Financial Inclusion",
    "📊 State-wise Insights",
    "📚 Methodology"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- NFHS-5 (2019–21)")
st.sidebar.markdown("- RBI Handbook 2021")
st.sidebar.markdown("- Population Projections, MoHFW")
st.sidebar.markdown("**n = 33 States/UTs**")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>Financial Inclusion of Rural Women in India</h1>
        <p>Applied Business Analytics — ABA Final Project | NFHS-5 (2019–21) + RBI Branch Data (2021)</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h2>{df['BankAccount'].mean():.1f}%</h2>
            <p>Avg Rural Women Bank Account Usage</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h2>{df['BankAccount'].min():.1f}%</h2>
            <p>Lowest State (Nagaland)</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h2>{df['BankAccount'].max():.1f}%</h2>
            <p>Highest State (Puducherry)</p></div>""", unsafe_allow_html=True)
    with c4:
        best_r2 = max(results['r2_m1'], results['r2_m2'], results['r2_m3'], results['r2_m4'])
        st.markdown(f"""<div class="metric-card">
            <h2>{best_r2:.3f}</h2>
            <p>Best Model R² Score</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3 class="section-header">Problem Statement</h3>', unsafe_allow_html=True)
    st.info("""
    **"What socioeconomic factors most strongly predict financial exclusion among rural women
    in India, and how can these insights guide targeted policy or banking interventions
    to bridge the inclusion gap?"**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dependent Variable")
        st.success("🎯 Women's Bank Account Usage % (Rural)")

        st.markdown("### Independent Variables (8)")
        for col, label in FEATURE_LABELS.items():
            st.markdown(f"- {label}")

    with col2:
        st.markdown("### 5 Hypotheses Being Tested")
        hypotheses = [
            ("H1", "States with lower female literacy have significantly lower bank account ownership"),
            ("H2", "Mobile phone ownership positively predicts financial inclusion independently of income"),
            ("H3", "Higher child marriage rates are associated with lower financial inclusion"),
            ("H4", "Women with greater household decision-making power have higher bank account usage"),
            ("H5", "Higher bank branch density (supply-side) positively predicts financial inclusion"),
        ]
        for code, text in hypotheses:
            st.markdown(f"""<div class="hypothesis-card">
                <strong style="color:#1a3c5e">{code}:</strong> {text}</div>""",
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.markdown('<h2 class="section-header">Dataset Explorer</h2>', unsafe_allow_html=True)

    st.markdown(f"**Shape:** {df.shape[0]} states × {len(FEATURE_COLS)+2} variables | **Missing values:** 0")

    display_cols = ['State','BankAccount'] + FEATURE_COLS
    rename_map = {'State':'State/UT','BankAccount':'Bank Account %',
                  **{k: v for k,v in FEATURE_LABELS.items()}}
    st.dataframe(
        df[display_cols].rename(columns=rename_map).set_index('State/UT').style.format("{:.2f}"),
        use_container_width=True, height=500
    )

    st.markdown("### Descriptive Statistics")
    desc = df[['BankAccount']+FEATURE_COLS].describe().round(2)
    desc.index.name = "Statistic"
    st.dataframe(desc.rename(columns=rename_map), use_container_width=True)

    st.markdown("### Smoothing Encoding Applied to State")
    st.markdown("""
    **Adapted from reference file:** The reference used `TargetEncoder` (smoothing) on
    `brand` and `model` columns. Here, we apply the same smoothing concept to `State`
    — encoding each state by a smoothed average of its bank account usage toward the
    global mean. This prevents overfitting on small states (e.g., Ladakh, Sikkim).
    """)
    smooth_df = df[['State','BankAccount','State_Encoded']].copy()
    smooth_df.columns = ['State','Actual Bank Account %','Smoothed Encoding']
    st.dataframe(smooth_df.set_index('State').style.format("{:.3f}"),
                 use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Distribution", "Scatter Analysis"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 7))
        corr_data = df[['BankAccount']+FEATURE_COLS].copy()
        corr_data.columns = ['Bank Account']+[FEATURE_LABELS[c] for c in FEATURE_COLS]
        mask = np.zeros_like(corr_data.corr(), dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(corr_data.corr(), annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, ax=ax,
                    linewidths=0.5, annot_kws={'size':8})
        ax.set_title('Correlation Matrix — All Variables', fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        corr_with_target = corr_data.corr()['Bank Account'].drop('Bank Account').sort_values(ascending=False)
        st.markdown("### Correlation with Bank Account Usage")
        for var, val in corr_with_target.items():
            color = "🟢" if val > 0.3 else ("🔴" if val < -0.3 else "🟡")
            st.markdown(f"{color} **{var}**: r = {val:.3f}")

    with tab2:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        all_cols = ['BankAccount'] + FEATURE_COLS
        colors = ['#1a3c5e'] + ['#2e7d52']*8
        for i, (col, ax) in enumerate(zip(all_cols, axes.flatten())):
            label = 'Bank Account %' if col == 'BankAccount' else FEATURE_LABELS[col]
            ax.hist(df[col], bins=12, color=colors[i], edgecolor='white', alpha=0.85)
            ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2,
                       label=f'Mean: {df[col].mean():.1f}')
            ax.set_title(label, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_ylabel('Count', fontsize=8)
        fig.suptitle('Distribution of All Variables', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        x_var = st.selectbox("Select X variable", FEATURE_COLS,
                              format_func=lambda c: FEATURE_LABELS[c])
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(df[x_var], df['BankAccount'],
                             c=df['BankAccount'], cmap='RdYlGn',
                             s=80, alpha=0.85, edgecolors='white', linewidth=0.5)
        # trend line
        z = np.polyfit(df[x_var], df['BankAccount'], 1)
        p = np.poly1d(z)
        xline = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        ax.plot(xline, p(xline), 'k--', linewidth=1.5, alpha=0.6, label='Trend line')
        # annotate states
        for _, row in df.iterrows():
            ax.annotate(row['State'][:8], (row[x_var], row['BankAccount']),
                        fontsize=5.5, alpha=0.7, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')
        plt.colorbar(scatter, ax=ax, label='Bank Account %')
        ax.set_xlabel(FEATURE_LABELS[x_var], fontsize=11)
        ax.set_ylabel('Bank Account Usage % (Rural Women)', fontsize=11)
        ax.set_title(f'{FEATURE_LABELS[x_var]} vs Bank Account Usage', fontsize=12, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        r = df[[x_var,'BankAccount']].corr().iloc[0,1]
        direction = "positive" if r > 0 else "negative"
        strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
        st.markdown(f"""<div class="insight-box">
            📊 Pearson r = <strong>{r:.3f}</strong> — {strength} {direction} correlation
            between <em>{FEATURE_LABELS[x_var]}</em> and bank account usage.</div>""",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON M1–M4
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison (M1–M4)":
    st.markdown('<h2 class="section-header">Model Comparison — M1 to M4</h2>', unsafe_allow_html=True)
    st.markdown("""
    Following the same structure as the reference file, we train **4 models progressively**:
    - **M1** — Base RandomForestRegressor (raw data, same as reference)
    - **M2** — OLS Multiple Regression (for interpretability and p-values)
    - **M3** — Ridge Regression (handles multicollinearity between variables)
    - **M4** — RandomForestRegressor with **IQR Outlier Treatment** (same as reference df_m4)
    """)

    # ── model comparison table ─────────────────────────────────────────────
    models_df = pd.DataFrame({
        'Model': ['M1 — Base RF', 'M2 — OLS Regression', 'M3 — Ridge Regression', 'M4 — RF + Outlier Treatment'],
        'R² Score': [results['r2_m1'], results['r2_m2'], results['r2_m3'], results['r2_m4']],
        'RMSE (% points)': [results['rmse_m1'], results['rmse_m2'], results['rmse_m3'], results['rmse_m4']],
        'Method': ['RandomForest', 'Statsmodels OLS', 'Ridge + Scaling', 'RF + IQR Outlier Fix']
    })

    best_idx = models_df['R² Score'].idxmax()

    st.dataframe(
        models_df.style
        .highlight_max(subset=['R² Score'], color='#c8f0d4')
        .highlight_min(subset=['RMSE (% points)'], color='#c8f0d4')
        .format({'R² Score': '{:.4f}', 'RMSE (% points)': '{:.4f}'}),
        use_container_width=True
    )

    best_model = models_df.loc[best_idx, 'Model']
    best_r2    = models_df.loc[best_idx, 'R² Score']
    best_rmse  = models_df.loc[best_idx, 'RMSE (% points)']
    st.markdown(f"""<div class="insight-box">
        🏆 <strong>Best Model: {best_model}</strong><br>
        R² = {best_r2:.4f} — explains {best_r2*100:.1f}% of variance in bank account usage<br>
        RMSE = {best_rmse:.4f} percentage points — average prediction error</div>""",
        unsafe_allow_html=True)

    # ── RMSE bar chart ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#90caf9','#a5d6a7','#ffcc80','#ef9a9a']
        bars = ax.bar(['M1\nBase RF','M2\nOLS','M3\nRidge','M4\nRF+Outlier'],
                      [results['r2_m1'],results['r2_m2'],results['r2_m3'],results['r2_m4']],
                      color=colors, edgecolor='white', linewidth=0.7)
        for bar, val in zip(bars, [results['r2_m1'],results['r2_m2'],results['r2_m3'],results['r2_m4']]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title('R² Score — Model Comparison', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='0.5 threshold')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        rmse_vals = [results['rmse_m1'],results['rmse_m2'],results['rmse_m3'],results['rmse_m4']]
        bars = ax.bar(['M1\nBase RF','M2\nOLS','M3\nRidge','M4\nRF+Outlier'],
                      rmse_vals, color=colors, edgecolor='white', linewidth=0.7)
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('RMSE (% points)', fontsize=10)
        ax.set_title('RMSE — Lower is Better', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Cross-Validation ───────────────────────────────────────────────────
    st.markdown("### 5-Fold Cross-Validation (M4 Best Model)")
    cv = results['cv_scores']
    cv_df = pd.DataFrame({'Fold': [f'Fold {i+1}' for i in range(5)], 'R² Score': cv})
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(cv_df['Fold'], cv_df['R² Score'], color='#2e7d52', alpha=0.8, edgecolor='white')
    ax.axhline(cv.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean R² = {cv.mean():.4f}')
    ax.set_ylabel('R² Score')
    ax.set_title('5-Fold Cross-Validation Results (M4)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown(f"""<div class="insight-box">
        Cross-validation mean R² = <strong>{cv.mean():.4f}</strong>
        (std = {cv.std():.4f}) — model is {'stable' if cv.std() < 0.1 else 'slightly variable'} across folds.
        </div>""", unsafe_allow_html=True)

    # ── Actual vs Predicted ────────────────────────────────────────────────
    st.markdown("### Actual vs Predicted — M4")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results['y_test_m4'], results['y_pred_m4'],
               color='#1a3c5e', alpha=0.8, edgecolors='white', s=70)
    mn = min(results['y_test_m4'].min(), results['y_pred_m4'].min()) - 2
    mx = max(results['y_test_m4'].max(), results['y_pred_m4'].max()) + 2
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Bank Account Usage %', fontsize=11)
    ax.set_ylabel('Predicted Bank Account Usage %', fontsize=11)
    ax.set_title('Actual vs Predicted — M4 (RF + Outlier Treatment)', fontsize=11, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — OLS REGRESSION & HYPOTHESES
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 OLS Regression & Hypotheses":
    st.markdown('<h2 class="section-header">OLS Regression Results & Hypothesis Testing</h2>', unsafe_allow_html=True)

    m2 = results['m2']

    # ── full OLS summary ───────────────────────────────────────────────────
    with st.expander("📄 Full OLS Regression Summary (M2)", expanded=True):
        st.text(str(m2.summary()))

    # ── coefficient table ──────────────────────────────────────────────────
    st.markdown("### Coefficient Interpretation")
    coef_df = pd.DataFrame({
        'Variable':    [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
        'Coefficient': m2.params[1:].values,
        'p-value':     m2.pvalues[1:].values,
        'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in m2.pvalues[1:].values],
        'Direction':   ['↑ Increases bank usage' if c > 0 else '↓ Decreases bank usage'
                        for c in m2.params[1:].values]
    })
    st.dataframe(coef_df.style.format({'Coefficient':'{:.4f}','p-value':'{:.4f}'}),
                 use_container_width=True)

    # ── hypothesis verdict ─────────────────────────────────────────────────
    st.markdown("### Hypothesis Testing Results")
    pvals = dict(zip(FEATURE_COLS, m2.pvalues[1:].values))
    coefs = dict(zip(FEATURE_COLS, m2.params[1:].values))

    hypotheses_full = [
        ("H1", "Literacy", "States with lower female literacy have lower bank account ownership"),
        ("H2", "MobileOwn", "Mobile phone ownership positively predicts financial inclusion"),
        ("H3", "ChildMarriage", "Higher child marriage rates → lower financial inclusion"),
        ("H4", "HHDecision", "Higher HH decision-making power → higher bank account usage"),
        ("H5", "BranchDensity", "Higher bank branch density → higher financial inclusion"),
    ]

    for code, feat, text in hypotheses_full:
        p = pvals[feat]
        c = coefs[feat]
        supported = p < 0.05
        verdict   = "✅ Supported" if supported else "❌ Not Supported at 5% level"
        bg_color  = "#e8f4e8" if supported else "#fff3e0"
        st.markdown(f"""
        <div style="background:{bg_color};border-radius:8px;padding:0.8rem 1rem;margin:0.4rem 0;
                    border-left:4px solid {'#2e7d52' if supported else '#f9a825'}">
            <strong>{code}: {verdict}</strong><br>
            {text}<br>
            <small>Coefficient = {c:.4f} | p-value = {p:.4f}</small>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌟 Feature Importance":
    st.markdown('<h2 class="section-header">Feature Importance — M4 (Best RF Model)</h2>', unsafe_allow_html=True)

    m4 = results['m4']
    importances = pd.Series(m4.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    labels = [FEATURE_LABELS[c] for c in importances.index]
    colors_imp = ['#ef9a9a' if importances[c] < 0.10 else
                  '#fff176' if importances[c] < 0.15 else '#a5d6a7'
                  for c in importances.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, importances.values, color=colors_imp, edgecolor='white', linewidth=0.7)
    for bar, val in zip(bars, importances.values):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title('Feature Importance — RandomForest M4\n(Higher = More Predictive Power)',
                 fontsize=12, fontweight='bold')
    patches = [
        mpatches.Patch(color='#a5d6a7', label='High (>0.15)'),
        mpatches.Patch(color='#fff176', label='Moderate (0.10–0.15)'),
        mpatches.Patch(color='#ef9a9a', label='Low (<0.10)'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    top3 = importances.sort_values(ascending=False).head(3)
    st.markdown("### Top 3 Predictors of Financial Inclusion")
    for rank, (feat, score) in enumerate(top3.items(), 1):
        st.markdown(f"""<div class="insight-box">
            <strong>#{rank} {FEATURE_LABELS[feat]}</strong> — Importance Score: {score:.4f}<br>
            This variable explains the most variance in rural women's bank account usage.</div>""",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PREDICTION TOOL
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Financial Inclusion":
    st.markdown('<h2 class="section-header">Predict Financial Inclusion for a New State/Region</h2>', unsafe_allow_html=True)
    st.markdown("""
    Enter socioeconomic values for any region to predict its expected **rural women's
    bank account usage %**. Useful for policymakers and banks to estimate inclusion gaps.
    """)

    col1, col2 = st.columns(2)
    inputs = {}
    with col1:
        inputs['Literacy']     = st.slider("Female Literacy %",      40.0, 100.0, 70.0, 0.1)
        inputs['MobileOwn']    = st.slider("Mobile Ownership %",      20.0, 100.0, 55.0, 0.1)
        inputs['InternetUse']  = st.slider("Internet Usage %",        10.0, 80.0,  30.0, 0.1)
        inputs['HHDecision']   = st.slider("HH Decision Making %",    70.0, 100.0, 88.0, 0.1)
    with col2:
        inputs['PaidCash']     = st.slider("Women Paid in Cash %",    10.0, 60.0,  25.0, 0.1)
        inputs['AssetOwn']     = st.slider("Asset Ownership %",       10.0, 80.0,  45.0, 0.1)
        inputs['ChildMarriage']= st.slider("Child Marriage %",         0.0, 55.0,  20.0, 0.1)
        inputs['BranchDensity']= st.slider("Branch Density (per 1K)", 0.05, 0.50,  0.13, 0.01)

    if st.button("🔮 Predict Bank Account Usage %", type="primary"):
        input_df = pd.DataFrame([inputs])[FEATURE_COLS]
        pred_m1  = results['m1'].predict(input_df)[0]
        pred_m4  = results['m4'].predict(input_df)[0]
        X_scaled = results['scaler'].transform(input_df)
        pred_m3  = results['m3'].predict(X_scaled)[0]
        input_c  = sm.add_constant(input_df, has_constant='add')
        pred_m2  = results['m2'].predict(input_c)[0]

        ensemble = np.mean([pred_m1, pred_m2, pred_m3, pred_m4])

        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in zip(
            [c1,c2,c3,c4,c5],
            ['M1 RF','M2 OLS','M3 Ridge','M4 RF+IQR','🎯 Ensemble'],
            [pred_m1, pred_m2, pred_m3, pred_m4, ensemble]
        ):
            col.metric(label, f"{val:.2f}%")

        nat_avg = df['BankAccount'].mean()
        gap = ensemble - nat_avg
        status = "above" if gap > 0 else "below"
        color  = "#e8f4e8" if gap > 0 else "#fff3e0"
        st.markdown(f"""<div style="background:{color};border-radius:8px;padding:1rem;margin-top:1rem;">
            <strong>Ensemble Prediction: {ensemble:.2f}%</strong><br>
            This is <strong>{abs(gap):.2f}% {status}</strong> the national average of {nat_avg:.1f}%.<br>
            {'✅ This region is performing well on financial inclusion.' if gap > 0
             else '⚠️ This region needs targeted intervention to improve financial inclusion.'}
        </div>""", unsafe_allow_html=True)

        # policy recommendations
        st.markdown("### Suggested Policy Interventions")
        recs = []
        if inputs['Literacy'] < 65:       recs.append("📚 Literacy is low — prioritise adult literacy programs for women")
        if inputs['MobileOwn'] < 50:      recs.append("📱 Mobile ownership is low — smartphone distribution scheme recommended")
        if inputs['ChildMarriage'] > 30:  recs.append("⚖️ High child marriage — enforce legal age of marriage strictly")
        if inputs['BranchDensity'] < 0.1: recs.append("🏦 Low branch density — open Business Correspondent (BC) points")
        if inputs['PaidCash'] < 20:       recs.append("💼 Low paid employment — promote rural livelihood programs (MGNREGA)")
        if not recs:
            recs.append("✅ All indicators are at acceptable levels — maintain current programs")
        for r in recs:
            st.markdown(f"- {r}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 8 — STATE-WISE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 State-wise Insights":
    st.markdown('<h2 class="section-header">State-wise Insights</h2>', unsafe_allow_html=True)

    selected_state = st.selectbox("Select a State", sorted(df['State'].tolist()))
    state_row = df[df['State'] == selected_state].iloc[0]
    nat_avg = df.mean(numeric_only=True)

    st.markdown(f"### {selected_state} — Profile")
    cols = st.columns(4)
    metrics = [
        ('Bank Account Usage', 'BankAccount', '%'),
        ('Female Literacy',    'Literacy',    '%'),
        ('Mobile Ownership',   'MobileOwn',   '%'),
        ('Branch Density',     'BranchDensity', ' per 1K'),
    ]
    for col, (label, feat, unit) in zip(cols, metrics):
        delta = state_row[feat] - nat_avg[feat]
        col.metric(label, f"{state_row[feat]:.2f}{unit}", f"{delta:+.2f} vs national avg")

    # radar-style comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    compare_feats = FEATURE_COLS
    state_vals = [state_row[c] for c in compare_feats]
    natavg_vals = [nat_avg[c] for c in compare_feats]
    x = np.arange(len(compare_feats))
    w = 0.38
    ax.bar(x - w/2, state_vals, w, label=selected_state, color='#1a3c5e', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, natavg_vals, w, label='National Avg', color='#2e7d52', alpha=0.55, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[c] for c in compare_feats], rotation=35, ha='right', fontsize=8)
    ax.set_title(f'{selected_state} vs National Average', fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ranking
    rank = int(df['BankAccount'].rank(ascending=False)[df['State']==selected_state].values[0])
    st.markdown(f"""<div class="insight-box">
        <strong>{selected_state}</strong> ranks <strong>#{rank} out of 33</strong>
        states/UTs in rural women's bank account usage.</div>""", unsafe_allow_html=True)

    st.markdown("### All States Ranked")
    ranked = df[['State','BankAccount']+FEATURE_COLS].sort_values('BankAccount', ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked.columns = ['State','Bank Account %'] + [FEATURE_LABELS[c] for c in FEATURE_COLS]
    st.dataframe(ranked.style.format({c: '{:.2f}' for c in ranked.columns if c != 'State'})
                 .highlight_max(subset=['Bank Account %'], color='#c8f0d4')
                 .highlight_min(subset=['Bank Account %'], color='#ffd5d5'),
                 use_container_width=True, height=400)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 9 — METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Methodology":
    st.markdown('<h2 class="section-header">Methodology & Technical Documentation</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### Data Sources
    | Source | Year | Variables |
    |--------|------|-----------|
    | NFHS-5 (National Family Health Survey) | 2019–21 | Bank account usage, literacy, mobile ownership, internet use, HH decision making, paid employment, asset ownership, child marriage |
    | RBI Handbook of Statistics on Indian States | 2021 | Bank offices count (state-wise) |
    | MoHFW Population Projections | 2021 | State-wise population (for branch density calculation) |

    ### Data Preprocessing (adapted from reference file)
    1. **Missing Value Handling** — NFHS-5 state fact sheets are complete; zero missing values
    2. **Smoothing Encoding** — Applied to State column (same concept as TargetEncoder in reference for brand/model)
    3. **Outlier Treatment (M4)** — IQR method replaces outliers with non-outlier median (same as reference df_m4)
    4. **Feature Engineering** — Branch Density = Bank Offices ÷ (Population ÷ 1,000)

    ### Model Architecture
    | Model | Type | Key Feature |
    |-------|------|-------------|
    | M1 | RandomForestRegressor (base) | Same as reference — 100 trees, default params |
    | M2 | OLS Regression (statsmodels) | p-values for hypothesis testing |
    | M3 | Ridge Regression | Handles multicollinearity |
    | M4 | RandomForestRegressor + IQR | Same as reference df_m4 — best model |

    ### RMSE Interpretation
    > **Note:** Unlike the reference file where RMSE is in Rupees, here RMSE is in **percentage points**
    > (because the dependent variable is bank account usage %). A lower RMSE means the model
    > predicts closer to the actual state-level percentage.

    ### Limitations
    - n = 33 states — small sample, which limits statistical power
    - NFHS-5 is from 2019–21; conditions may have changed
    - Branch density uses total population (not rural-only)
    - Cross-sectional study — causal inference requires longitudinal data
    """)

    st.markdown("---")
    st.markdown("### Model Performance Summary")
    summary_df = pd.DataFrame({
        'Model': ['M1 — Base RF', 'M2 — OLS', 'M3 — Ridge', 'M4 — RF+Outlier'],
        'R²':   [results['r2_m1'],results['r2_m2'],results['r2_m3'],results['r2_m4']],
        'RMSE (% pts)': [results['rmse_m1'],results['rmse_m2'],results['rmse_m3'],results['rmse_m4']],
    })
    st.dataframe(summary_df.style.format({'R²':'{:.4f}','RMSE (% pts)':'{:.4f}'}),
                 use_container_width=True)

    cv = results['cv_scores']
    st.markdown(f"**5-Fold CV (M4):** Mean R² = {cv.mean():.4f} | Std = {cv.std():.4f}")
