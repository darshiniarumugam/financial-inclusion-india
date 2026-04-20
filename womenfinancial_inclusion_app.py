# ============================================================
# Financial Inclusion of Rural Women in India
# Professional Dashboard — ABA Final Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinInclude India",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }

.topbar {
    background: #0f1923; padding: 1rem 2rem;
    margin: -1rem -2rem 2rem -2rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #1e3a52;
}
.topbar-logo { color: #38bdf8; font-weight: 600; font-size: 1.1rem; letter-spacing: -0.02em; }
.topbar-sub  { color: #64748b; font-size: 0.75rem; margin-top: 1px; }
.topbar-badge {
    background: #1e3a52; color: #38bdf8; font-size: 0.7rem;
    padding: 3px 10px; border-radius: 99px; font-weight: 500;
    border: 1px solid #38bdf833;
}

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 2rem; }
.kpi-card {
    background: #0f1923; border: 1px solid #1e3a52;
    border-radius: 12px; padding: 1.25rem 1.5rem;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.kpi-label { color: #64748b; font-size: 0.72rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
.kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1; }
.kpi-delta { color: #38bdf8; font-size: 0.75rem; margin-top: 0.4rem; }

.sec-head {
    color: #f1f5f9; font-size: 1rem; font-weight: 600;
    letter-spacing: -0.02em; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b;
}

.insight {
    background: #0d2137; border: 1px solid #1e3a52;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    color: #94a3b8; font-size: 0.85rem; line-height: 1.6; margin: 0.5rem 0;
}
.insight strong { color: #e2e8f0; }
.warn {
    background: #1a1200; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    color: #fbbf24; font-size: 0.85rem; line-height: 1.6; margin: 0.4rem 0;
}
.success-msg {
    background: #052e16; border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    color: #4ade80; font-size: 0.85rem; line-height: 1.6; margin: 0.4rem 0;
}

.pred-card {
    background: #0d2137; border: 1px solid #38bdf844;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.pred-value { color: #38bdf8; font-size: 3.5rem; font-weight: 700; letter-spacing: -0.04em; }
.pred-label { color: #64748b; font-size: 0.85rem; margin-top: 0.3rem; }

.hyp-row {
    display: flex; align-items: flex-start; gap: 12px;
    background: #0f1923; border: 1px solid #1e3a52;
    border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
}
.hyp-code { color: #38bdf8; font-weight: 600; font-size: 0.8rem; min-width: 28px; font-family: 'DM Mono', monospace; }
.hyp-text { color: #94a3b8; font-size: 0.83rem; line-height: 1.5; flex: 1; }
.hyp-verdict { font-size: 0.75rem; font-weight: 600; padding: 2px 10px; border-radius: 99px; white-space: nowrap; }
.v-yes { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.v-no  { background: #1a0000; color: #f87171; border: 1px solid #7f1d1d; }

.model-row {
    display: flex; align-items: center; gap: 12px;
    padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 6px;
    background: #0f1923; border: 1px solid #1e3a52;
}
.model-name { color: #e2e8f0; font-weight: 500; font-size: 0.85rem; flex: 1; }
.model-r2   { color: #38bdf8; font-weight: 600; font-size: 0.9rem; font-family: 'DM Mono', monospace; min-width: 70px; text-align: right; }
.model-rmse { color: #818cf8; font-size: 0.8rem; font-family: 'DM Mono', monospace; min-width: 70px; text-align: right; }
.model-best { background: #0d2137; border-color: #38bdf844; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB DARK THEME ─────────────────────────────────────────────────
DARK_BG = '#0f1923'
ACCENT  = '#38bdf8'
ACCENT2 = '#818cf8'
ACCENT3 = '#4ade80'
DANGER  = '#f87171'
AMBER   = '#f59e0b'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': DARK_BG,
    'axes.edgecolor': '#1e3a52', 'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b', 'ytick.color': '#64748b',
    'text.color': '#e2e8f0', 'grid.color': '#1e3a52',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'axes.titlecolor': '#f1f5f9',
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'figure.dpi': 120,
})

# ── DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        'State': [
            'Andaman & Nicobar','Andhra Pradesh','Arunachal Pradesh',
            'Assam','Bihar','Chhattisgarh','Dadra & NH + DD',
            'Goa','Gujarat','Haryana','Himachal Pradesh',
            'Jammu & Kashmir','Jharkhand','Karnataka','Kerala','Ladakh',
            'Madhya Pradesh','Maharashtra','Manipur','Meghalaya',
            'Mizoram','Nagaland','Odisha','Puducherry','Punjab',
            'Rajasthan','Sikkim','Tamil Nadu','Telangana','Tripura',
            'Uttar Pradesh','Uttarakhand','West Bengal'
        ],
        'BankAccount':   [89.75,79.63,76.74,77.86,76.21,81.14,89.28,92.38,67.48,72.39,82.23,83.52,79.75,87.68,78.22,88.65,73.34,70.91,70.71,68.22,74.97,55.41,87.38,96.72,82.10,79.00,76.70,91.72,85.21,77.67,74.06,79.75,73.24],
        'Literacy':      [85.58,63.84,71.63,75.39,54.54,71.30,67.90,93.43,69.03,78.75,91.23,74.71,59.34,70.95,97.47,76.60,63.30,79.51,84.84,85.53,87.67,82.74,69.22,89.07,80.04,61.63,86.24,81.50,58.10,76.88,65.55,79.78,72.54],
        'MobileOwn':     [80.86,40.93,75.28,53.87,49.26,33.94,45.53,87.06,36.18,43.41,77.77,73.25,43.67,53.42,86.90,81.22,31.44,43.07,68.19,64.29,70.64,76.33,47.95,76.56,54.90,45.30,83.25,68.93,50.60,48.03,42.37,55.65,39.07],
        'InternetUse':   [27.87,15.35,49.61,24.36,17.00,20.78,23.84,68.28,17.53,42.80,45.23,38.90,22.69,24.76,57.54,53.98,20.08,23.71,40.39,27.95,47.98,40.26,21.33,50.37,48.75,30.76,68.08,39.20,15.78,17.66,24.48,39.35,13.98],
        'HHDecision':    [95.52,84.30,86.64,91.77,87.02,91.54,87.43,98.60,90.67,86.15,93.90,81.66,89.84,80.52,94.61,80.27,84.13,89.15,95.00,91.98,98.02,99.82,90.32,96.55,90.25,86.79,93.94,93.65,86.16,89.47,86.78,90.64,85.81],
        'PaidCash':      [17.12,44.50,23.55,19.27,12.80,42.62,39.47,27.86,34.05,17.03,17.71,18.50,17.74,41.36,25.79,28.25,28.00,39.60,44.03,39.11,29.00,20.67,25.58,44.84,20.21,17.52,29.25,45.51,55.54,25.93,14.83,22.22,20.17],
        'AssetOwn':      [19.36,50.63,70.25,43.85,55.67,45.45,59.74,24.07,43.27,41.03,23.38,60.77,66.47,69.65,29.18,72.95,41.27,24.50,58.93,70.11,28.36,28.86,45.47,38.30,67.07,26.61,50.57,52.04,74.46,17.28,53.53,25.13,22.46],
        'ChildMarriage': [15.27,32.90,19.33,33.35,43.36,13.20,26.16,3.18,26.91,13.65,5.14,5.26,36.09,24.66,8.22,3.13,26.55,27.57,17.61,19.12,14.00,7.25,21.71,0.97,8.68,28.28,12.54,15.18,27.38,42.40,17.85,9.76,48.13],
        'BankOffices':   [71,7252,164,2957,7463,2823,110,697,8626,5224,1652,1767,3169,10778,6801,68,7234,13578,203,364,203,176,5223,267,6735,7775,162,11829,5388,565,18041,2194,9251],
        'Population_2021':[417000,53903000,1570000,35607000,124800000,29436000,894000,1539000,70400000,29506000,7380000,13635000,38593000,67562000,35125000,290000,85358000,122153000,3436000,3443000,1239000,2150000,44395000,1694000,30141000,79502000,671000,76396000,37220000,4169000,224979000,11439000,96906000],
    }
    df = pd.DataFrame(data)
    df['BranchDensity'] = (df['BankOffices'] / (df['Population_2021'] / 1000)).round(4)
    return df

df = load_data()

FEATURES = ['Literacy','MobileOwn','InternetUse','HHDecision',
            'PaidCash','AssetOwn','ChildMarriage','BranchDensity']
LABELS = {
    'Literacy':      'Female Literacy %',
    'MobileOwn':     'Mobile Ownership %',
    'InternetUse':   'Internet Usage %',
    'HHDecision':    'HH Decision Making %',
    'PaidCash':      'Women Paid in Cash %',
    'AssetOwn':      'Asset Ownership %',
    'ChildMarriage': 'Child Marriage %',
    'BranchDensity': 'Branch Density (per 1K)',
}
TARGET = 'BankAccount'

# ── MODEL TRAINING ────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    m1 = RandomForestRegressor(n_estimators=100, random_state=42)
    m1.fit(Xtr, ytr); yp1 = m1.predict(Xte)

    m2 = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    yp2 = m2.predict(sm.add_constant(Xte))

    sc = StandardScaler()
    m3 = Ridge(alpha=1.0)
    m3.fit(sc.fit_transform(Xtr), ytr)
    yp3 = m3.predict(sc.transform(Xte))

    df_m4 = X.copy(); df_m4['BankAccount'] = y.values
    for col in FEATURES:
        df_m4[col] = df_m4[col].astype(float)
        Q1, Q3 = df_m4[col].quantile(0.25), df_m4[col].quantile(0.75)
        IQR = Q3 - Q1
        med = df_m4[(df_m4[col] >= Q1-1.5*IQR) & (df_m4[col] <= Q3+1.5*IQR)][col].median()
        df_m4.loc[(df_m4[col] < Q1-1.5*IQR)|(df_m4[col] > Q3+1.5*IQR), col] = med
    Xm4, ym4 = df_m4[FEATURES], df_m4['BankAccount']
    Xtr4, Xte4, ytr4, yte4 = train_test_split(Xm4, ym4, test_size=0.2, random_state=42)
    m4 = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    m4.fit(Xtr4, ytr4); yp4 = m4.predict(Xte4)
    cv = cross_val_score(m4, Xm4, ym4, cv=5, scoring='r2')

    def met(yt, yp): return r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp))

    return dict(
        m1=m1, m2=m2, m3=m3, m4=m4, sc=sc,
        Xte=Xte, yte=yte, yte4=yte4,
        yp1=yp1, yp2=yp2, yp3=yp3, yp4=yp4,
        r2m1=met(yte,yp1)[0],  rmse1=met(yte,yp1)[1],
        r2m2=met(yte,yp2)[0],  rmse2=met(yte,yp2)[1],
        r2m3=met(yte,yp3)[0],  rmse3=met(yte,yp3)[1],
        r2m4=met(yte4,yp4)[0], rmse4=met(yte4,yp4)[1],
        cv=cv,
        imp=pd.Series(m4.feature_importances_, index=FEATURES).sort_values(ascending=False)
    )

R = train_models()

# ── TOP NAV BAR ───────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-logo">🏦 FinInclude India</div>
    <div class="topbar-sub">Financial Inclusion of Rural Women · NFHS-5 (2019–21) · 33 States/UTs</div>
  </div>
  <div class="topbar-badge">ABA Final Project</div>
</div>
""", unsafe_allow_html=True)

# ── PAGE SELECTOR ─────────────────────────────────────────────────────────
PAGES = ["🏠 Dashboard", "🔮 Predict", "📊 Analysis", "🧪 Models", "📋 States"]
page = st.radio("Navigate", PAGES, horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:0.5rem 0 1.5rem'>",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">National Average</div>
        <div class="kpi-value">{:.1f}%</div>
        <div class="kpi-delta">Rural women with bank account</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Top Performer</div>
        <div class="kpi-value">{:.1f}%</div>
        <div class="kpi-delta">Puducherry — highest inclusion</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Needs Attention</div>
        <div class="kpi-value">{:.1f}%</div>
        <div class="kpi-delta">Nagaland — lowest inclusion</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Best Model R²</div>
        <div class="kpi-value">{:.3f}</div>
        <div class="kpi-delta">Variance explained by model</div>
      </div>
    </div>
    """.format(
        df['BankAccount'].mean(),
        df['BankAccount'].max(),
        df['BankAccount'].min(),
        max(R['r2m1'], R['r2m2'], R['r2m3'], R['r2m4'])
    ), unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="sec-head">State-wise Bank Account Usage — Rural Women</div>', unsafe_allow_html=True)
        sdf = df.sort_values('BankAccount', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 9))
        bar_colors = [ACCENT3 if v >= 85 else ACCENT if v >= 75 else DANGER for v in sdf['BankAccount']]
        ax.barh(sdf['State'], sdf['BankAccount'], color=bar_colors, edgecolor='none', height=0.7)
        ax.axvline(df['BankAccount'].mean(), color=AMBER, linestyle='--', linewidth=1.2)
        for i, (val, state) in enumerate(zip(sdf['BankAccount'], sdf['State'])):
            ax.text(val + 0.4, i, f'{val:.1f}', va='center', fontsize=7, color='#94a3b8')
        ax.set_xlim(45, 105)
        ax.tick_params(axis='y', labelsize=7.5)
        legend_elems = [
            mpatches.Patch(color=ACCENT3, label='High ≥ 85%'),
            mpatches.Patch(color=ACCENT, label='Medium 75–85%'),
            mpatches.Patch(color=DANGER, label='Low < 75%'),
            plt.Line2D([0],[0], color=AMBER, linestyle='--', linewidth=1.2,
                       label=f'Avg {df["BankAccount"].mean():.1f}%'),
        ]
        ax.legend(handles=legend_elems, fontsize=7.5, loc='lower right',
                  facecolor=DARK_BG, edgecolor='#1e3a52', labelcolor='#94a3b8')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel('Bank Account Usage %')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="sec-head">What Drives Inclusion?</div>', unsafe_allow_html=True)
        imp = R['imp']
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        colors_pie = [ACCENT, ACCENT2, ACCENT3, AMBER, DANGER, '#e879f9', '#fb923c', '#34d399']
        ax2.pie(imp.values, labels=[LABELS[k][:16] for k in imp.index],
                colors=colors_pie, autopct='%1.0f%%', pctdistance=0.75, startangle=90,
                textprops={'fontsize': 7, 'color': '#94a3b8'},
                wedgeprops={'linewidth': 1.5, 'edgecolor': DARK_BG})
        ax2.set_title('Feature Importance (RF Model)', pad=10)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown('<div class="sec-head" style="margin-top:1.5rem">Hypotheses Status</div>', unsafe_allow_html=True)
        pvals = dict(zip(FEATURES, R['m2'].pvalues[1:].values))
        hyps = [
            ("H1", "Literacy drives inclusion", "Literacy"),
            ("H2", "Mobile ownership predicts inclusion", "MobileOwn"),
            ("H3", "Child marriage reduces inclusion", "ChildMarriage"),
            ("H4", "HH decision-making → inclusion", "HHDecision"),
            ("H5", "Branch density → inclusion", "BranchDensity"),
        ]
        for code, text, feat in hyps:
            sup = pvals[feat] < 0.05
            st.markdown(f"""
            <div class="hyp-row">
              <span class="hyp-code">{code}</span>
              <span class="hyp-text">{text}</span>
              <span class="hyp-verdict {'v-yes' if sup else 'v-no'}">{'✓' if sup else '✗'}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="sec-head">Financial Inclusion Predictor — Policy Scenario Tool</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="insight">
        <strong>For Researchers & Policymakers:</strong> Adjust the sliders below to simulate
        any region's socioeconomic profile. The model predicts expected bank account usage %.
        Use the <strong>Quick What-If</strong> buttons to instantly see the impact of policy changes
        like reducing child marriage or improving mobile access.
    </div>""", unsafe_allow_html=True)

    col_s, col_r = st.columns([1, 1])

    with col_s:
        st.markdown("#### Set Region Profile")
        literacy    = st.slider("📚 Female Literacy %",       40.0, 100.0, 72.0, 0.5)
        mobile      = st.slider("📱 Mobile Ownership %",       20.0, 100.0, 55.0, 0.5)
        internet    = st.slider("🌐 Internet Usage %",          5.0,  80.0, 28.0, 0.5)
        hhdecision  = st.slider("🏠 HH Decision Making %",    60.0, 100.0, 88.0, 0.5)
        paidcash    = st.slider("💼 Women Paid in Cash %",     10.0,  60.0, 25.0, 0.5)
        assetowner  = st.slider("🏡 Asset Ownership %",        10.0,  80.0, 44.0, 0.5)
        childmarr   = st.slider("👧 Child Marriage %",           0.0,  55.0, 20.0, 0.5,
                                 help="Key policy lever — reducing this improves inclusion")
        branchdense = st.slider("🏦 Branch Density (per 1K)",   0.05,  0.50, 0.13, 0.01,
                                 help="Number of bank offices per 1,000 people")

    with col_r:
        st.markdown("#### Prediction Result")

        inp = pd.DataFrame([{
            'Literacy': literacy, 'MobileOwn': mobile, 'InternetUse': internet,
            'HHDecision': hhdecision, 'PaidCash': paidcash, 'AssetOwn': assetowner,
            'ChildMarriage': childmarr, 'BranchDensity': branchdense
        }])[FEATURES]

        p1 = float(R['m1'].predict(inp)[0])
        p2 = float(R['m2'].predict(sm.add_constant(inp, has_constant='add'))[0])
        p3 = float(R['m3'].predict(R['sc'].transform(inp))[0])
        p4 = float(R['m4'].predict(inp)[0])
        ensemble = float(np.clip(np.mean([p1, p2, p3, p4]), 0, 100))

        nat_avg = df['BankAccount'].mean()
        gap     = ensemble - nat_avg
        arrow   = '▲' if gap > 0 else '▼'
        status  = 'above' if gap > 0 else 'below'

        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-label">Predicted Bank Account Usage</div>
          <div class="pred-value">{ensemble:.1f}%</div>
          <div class="pred-label" style="margin-top:.5rem">
            {arrow} {abs(gap):.1f}% {status} national average ({nat_avg:.1f}%)
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Model Breakdown**")
        for label, val, best in [
            ("M1 — Base RF", p1, False), ("M2 — OLS", p2, False),
            ("M3 — Ridge", p3, False),   ("M4 — RF+IQR", p4, False),
            ("⭐ Ensemble", ensemble, True)
        ]:
            st.markdown(f"""
            <div class="model-row {'model-best' if best else ''}">
              <span class="model-name">{label}</span>
              <span class="model-r2">{val:.2f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Quick What-If Scenarios**")
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            if st.button("👧 Marriage −10%", use_container_width=True):
                ni = inp.copy(); ni['ChildMarriage'] = max(0, childmarr - 10)
                np1 = np.clip(np.mean([R['m1'].predict(ni)[0], R['m4'].predict(ni)[0]]), 0, 100)
                st.metric("New Prediction", f"{np1:.1f}%", f"{np1-ensemble:+.2f}%")

        with sc2:
            if st.button("📱 Mobile +15%", use_container_width=True):
                ni = inp.copy(); ni['MobileOwn'] = min(100, mobile + 15)
                np2 = np.clip(np.mean([R['m1'].predict(ni)[0], R['m4'].predict(ni)[0]]), 0, 100)
                st.metric("New Prediction", f"{np2:.1f}%", f"{np2-ensemble:+.2f}%")

        with sc3:
            if st.button("📚 Literacy +10%", use_container_width=True):
                ni = inp.copy(); ni['Literacy'] = min(100, literacy + 10)
                np3 = np.clip(np.mean([R['m1'].predict(ni)[0], R['m4'].predict(ni)[0]]), 0, 100)
                st.metric("New Prediction", f"{np3:.1f}%", f"{np3-ensemble:+.2f}%")

        st.markdown("**Policy Recommendations**")
        recs = []
        if literacy    < 65:  recs.append(("📚", "Literacy critically low — adult education programs needed"))
        if mobile      < 45:  recs.append(("📱", "Low mobile ownership — smartphone access schemes recommended"))
        if childmarr   > 30:  recs.append(("⚖️", "High child marriage — enforce legal protections strictly"))
        if branchdense < 0.1: recs.append(("🏦", "Sparse bank access — open Business Correspondent points"))
        if paidcash    < 18:  recs.append(("💼", "Low employment — expand MGNREGA in this region"))
        if internet    < 20:  recs.append(("🌐", "Low internet access — digital literacy programs needed"))
        if not recs:
            st.markdown('<div class="success-msg">✅ All indicators healthy. Maintain current programs.</div>',
                        unsafe_allow_html=True)
        else:
            for icon, text in recs:
                st.markdown(f'<div class="warn">{icon} {text}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analysis":
    st.markdown('<div class="sec-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔥 Correlation Heatmap", "📈 Scatter Analysis"])

    with tab1:
        corr_df = df[['BankAccount'] + FEATURES].copy()
        corr_df.columns = ['Bank Account'] + [LABELS[f] for f in FEATURES]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={'size': 8, 'color': '#e2e8f0'},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Pearson Correlation — All Variables', pad=15)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Correlation strength with Bank Account Usage:**")
        corr_vals = corr_df.corr()['Bank Account'].drop('Bank Account').sort_values(ascending=False)
        for var, val in corr_vals.items():
            bar_pct = int(abs(val) * 100)
            color = ACCENT if val > 0 else DANGER
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0">
              <span style="color:#94a3b8;font-size:.8rem;min-width:175px">{var}</span>
              <div style="flex:1;background:#1e3a52;border-radius:4px;height:8px;overflow:hidden">
                <div style="width:{bar_pct}%;background:{color};height:100%;border-radius:4px"></div>
              </div>
              <span style="font-family:'DM Mono',monospace;font-size:.8rem;color:{color};min-width:52px;text-align:right">
                {val:+.3f}</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        feat_sel = st.selectbox("Select variable", FEATURES, format_func=lambda x: LABELS[x])
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(df[feat_sel], df['BankAccount'],
                             c=df['BankAccount'], cmap='cool',
                             s=90, alpha=0.9, edgecolors='#1e3a52', linewidth=0.8, zorder=3)
        z = np.polyfit(df[feat_sel], df['BankAccount'], 1)
        xl = np.linspace(df[feat_sel].min(), df[feat_sel].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=AMBER, linewidth=1.5,
                linestyle='--', label='Trend line', zorder=2)
        for _, row in df.iterrows():
            ax.annotate(row['State'][:7], (row[feat_sel], row['BankAccount']),
                        fontsize=5.5, color='#64748b', ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
        plt.colorbar(scatter, ax=ax, label='Bank Account %')
        ax.set_xlabel(LABELS[feat_sel]); ax.set_ylabel('Bank Account Usage %')
        ax.set_title(f'{LABELS[feat_sel]}  vs  Bank Account Usage')
        ax.legend(facecolor=DARK_BG, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        r = df[[feat_sel, 'BankAccount']].corr().iloc[0, 1]
        direction = "positive" if r > 0 else "negative"
        strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
        st.markdown(f"""<div class="insight">
            Pearson r = <strong>{r:.3f}</strong> — <strong>{strength} {direction}</strong> relationship.
            {'Higher ' + LABELS[feat_sel] + ' is associated with higher bank account usage.'
             if r > 0 else 'Higher ' + LABELS[feat_sel] + ' is associated with lower bank account usage.'}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Models":
    st.markdown('<div class="sec-head">Model Comparison — M1 to M4</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("**Model Performance**")
        model_rows = [
            ("M1 — Base RandomForest", R['r2m1'], R['rmse1'], False),
            ("M2 — OLS Regression",    R['r2m2'], R['rmse2'], False),
            ("M3 — Ridge Regression",  R['r2m3'], R['rmse3'], False),
            ("M4 — RF + IQR Outlier ⭐",R['r2m4'], R['rmse4'], True),
        ]
        for name, r2, rmse, best in model_rows:
            st.markdown(f"""
            <div class="model-row {'model-best' if best else ''}">
              <span class="model-name">{name}</span>
              <span class="model-r2">R² {r2:.4f}</span>
              <span class="model-rmse">±{rmse:.3f}</span>
            </div>""", unsafe_allow_html=True)

        best_r2 = max(R['r2m1'], R['r2m2'], R['r2m3'], R['r2m4'])
        st.markdown(f"""<div class="insight" style="margin-top:1rem">
            Best model explains <strong>{best_r2*100:.1f}%</strong> of variance.
            RMSE is in percentage points — lower = more accurate.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Feature Importance — M4**")
        for feat, score in R['imp'].items():
            bar_w = int(score * 500)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0">
              <span style="color:#94a3b8;font-size:.78rem;min-width:150px">{LABELS[feat]}</span>
              <div style="flex:1;background:#1e3a52;border-radius:4px;height:8px;overflow:hidden">
                <div style="width:{min(bar_w,100)}%;background:{ACCENT};height:100%;border-radius:4px"></div>
              </div>
              <span style="font-family:'DM Mono',monospace;font-size:.78rem;color:{ACCENT};min-width:42px;text-align:right">
                {score:.3f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**OLS Hypothesis Results (M2)**")
        pvals = dict(zip(FEATURES, R['m2'].pvalues[1:].values))
        coefs  = dict(zip(FEATURES, R['m2'].params[1:].values))
        for feat in FEATURES:
            p, c = pvals[feat], coefs[feat]
            sig  = p < 0.05
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                        border-bottom:1px solid #1e293b;">
              <span style="color:{'#4ade80' if sig else '#f87171'};font-size:.8rem">
                {'✓' if sig else '✗'}</span>
              <span style="color:#94a3b8;font-size:.78rem;flex:1">{LABELS[feat]}</span>
              <span style="font-family:'DM Mono',monospace;font-size:.75rem;
                           color:{'#38bdf8' if c>0 else '#f87171'}">{c:+.3f}</span>
              <span style="font-family:'DM Mono',monospace;font-size:.72rem;color:#64748b">
                p={p:.3f}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Actual vs Predicted — M4**")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(R['yte4'], R['yp4'], color=ACCENT, s=80,
                   edgecolors='#1e3a52', linewidth=0.8, alpha=0.9, zorder=3)
        mn = min(float(R['yte4'].min()), float(min(R['yp4']))) - 2
        mx = max(float(R['yte4'].max()), float(max(R['yp4']))) + 2
        ax.plot([mn, mx], [mn, mx], color=AMBER, linestyle='--',
                linewidth=1.5, label='Perfect prediction', zorder=2)
        ax.set_xlabel('Actual %'); ax.set_ylabel('Predicted %')
        ax.set_title('Actual vs Predicted (M4)')
        ax.legend(facecolor=DARK_BG, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**5-Fold Cross Validation (M4)**")
        cv = R['cv']
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        cv_colors = [ACCENT3 if v > cv.mean() else ACCENT for v in cv]
        bars2 = ax2.bar([f'Fold {i+1}' for i in range(5)], cv,
                        color=cv_colors, edgecolor='none', width=0.55)
        ax2.axhline(cv.mean(), color=AMBER, linestyle='--', linewidth=1.5,
                    label=f'Mean: {cv.mean():.4f}')
        for bar, val in zip(bars2, cv):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{val:.3f}', ha='center', fontsize=8, color='#94a3b8')
        ax2.set_ylabel('R² Score'); ax2.set_ylim(0, 1.15)
        ax2.set_title(f'Cross-validation  |  Std = {cv.std():.4f}')
        ax2.legend(facecolor=DARK_BG, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STATES
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 States":
    st.markdown('<div class="sec-head">State-wise Deep Dive</div>', unsafe_allow_html=True)

    state = st.selectbox("Select State / UT", sorted(df['State'].tolist()))
    row   = df[df['State'] == state].iloc[0]
    avg   = df.mean(numeric_only=True)
    rank  = int(df['BankAccount'].rank(ascending=False)[df['State'] == state].values[0])

    c1, c2, c3, c4 = st.columns(4)
    for col_obj, label, feat, fmt in [
        (c1, 'Bank Account %',   'BankAccount',   '{:.1f}%'),
        (c2, 'Female Literacy',  'Literacy',      '{:.1f}%'),
        (c3, 'Mobile Ownership', 'MobileOwn',     '{:.1f}%'),
        (c4, 'Branch Density',   'BranchDensity', '{:.4f}/1K'),
    ]:
        delta = row[feat] - avg[feat]
        col_obj.metric(label, fmt.format(row[feat]), f"{delta:+.2f} vs avg")

    st.markdown(f"""<div class="insight">
        <strong>{state}</strong> ranks <strong>#{rank} out of 33</strong> states/UTs.
        Bank account usage: <strong>{row['BankAccount']:.1f}%</strong>
        ({'above' if row['BankAccount'] > avg['BankAccount'] else 'below'} national average of {avg['BankAccount']:.1f}%)
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(FEATURES)); w = 0.38
        sv = [row[f] for f in FEATURES]; nv = [avg[f] for f in FEATURES]
        ax.bar(x-w/2, sv, w, label=state[:15], color=ACCENT, alpha=0.9, edgecolor='none')
        ax.bar(x+w/2, nv, w, label='National Avg', color=ACCENT2, alpha=0.55, edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[f][:14] for f in FEATURES], rotation=35, ha='right', fontsize=7.5)
        ax.set_title(f'{state[:20]} vs National Average')
        ax.legend(facecolor=DARK_BG, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**All Indicators vs National Average**")
        for feat, label in LABELS.items():
            val   = row[feat]
            navg  = avg[feat]
            delta = val - navg
            above = delta >= 0
            max_v = df[feat].max()
            bar_w = int((val / max_v) * 100)
            color = ACCENT3 if above else DANGER
            st.markdown(f"""
            <div style="margin:7px 0">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                <span style="color:#94a3b8;font-size:.75rem">{label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:{color}">
                  {val:.2f} ({delta:+.2f})</span>
              </div>
              <div style="background:#1e3a52;border-radius:4px;height:6px;overflow:hidden">
                <div style="width:{bar_w}%;background:{color};height:100%;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="margin-top:2rem">All States — Ranked Table</div>',
                unsafe_allow_html=True)
    ranked = df[['State', 'BankAccount'] + FEATURES].sort_values(
        'BankAccount', ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked.columns = ['State', 'Bank Account %'] + [LABELS[f] for f in FEATURES]
    st.dataframe(
        ranked.style
              .format({c: '{:.2f}' for c in ranked.columns if c != 'State'})
              .highlight_max(subset=['Bank Account %'], color='#052e16')
              .highlight_min(subset=['Bank Account %'], color='#1a0000'),
        use_container_width=True, height=400
    )
