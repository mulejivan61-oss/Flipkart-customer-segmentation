"""
=========================================================
  Customer Segmentation Intelligence Dashboard
  Built with: Streamlit + scikit-learn + Plotly
=========================================================
Run with:
    pip install streamlit pandas numpy scikit-learn plotly xlsxwriter openpyxl
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM DARK THEME CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #1a1c2e 0%, #0e0f1a 45%, #0a0b12 100%);
        color: #eaeaf0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14162a 0%, #0d0e1a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    h1, h2, h3, h4 {
        color: #f5f5ff !important;
        font-weight: 700 !important;
    }

    .hero {
        padding: 26px 32px;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(124,58,237,0.25), rgba(56,189,248,0.15));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(80, 40, 200, 0.15);
    }
    .hero h1 {
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #b7b9d0;
        margin-top: 6px;
        font-size: 0.95rem;
    }

    .kpi-card {
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-4px); }
    .kpi-label {
        font-size: 0.78rem;
        color: #9ca0c0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
    }
    .kpi-sub { font-size: 0.75rem; color: #7de3a3; margin-top: 4px;}

    .badge-high {
        background: rgba(34,197,94,0.15); color:#4ade80; padding:4px 12px;
        border-radius:999px; font-weight:600; font-size:0.85rem; border:1px solid rgba(74,222,128,0.4);
    }
    .badge-medium {
        background: rgba(250,204,21,0.15); color:#facc15; padding:4px 12px;
        border-radius:999px; font-weight:600; font-size:0.85rem; border:1px solid rgba(250,204,21,0.4);
    }
    .badge-low {
        background: rgba(248,113,113,0.15); color:#f87171; padding:4px 12px;
        border-radius:999px; font-weight:600; font-size:0.85rem; border:1px solid rgba(248,113,113,0.4);
    }

    .insight-box {
        background: rgba(124,58,237,0.08);
        border-left: 4px solid #a78bfa;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        color: #dcdcf0;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #7c3aed, #38bdf8);
        color: white; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    div.stButton > button:hover { opacity: 0.9; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        color: #b7b9d0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.25) !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def to_excel_bytes(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Clustered_Customers")
    return output.getvalue()


def badge_html(customer_type):
    if customer_type == "High Value Customer":
        return "<span class='badge-high'>🟢 High Value</span>"
    elif customer_type == "Medium Value Customer":
        return "<span class='badge-medium'>🟡 Medium Value</span>"
    else:
        return "<span class='badge-low'>🔴 Low Value</span>"


def assign_customer_type(sorted_cluster_index):
    """Generic Low/Medium/High labelling that works for any K, based on
    relative rank of cluster mean value (ascending)."""
    n = len(sorted_cluster_index)
    mapping = {}
    for i, idx in enumerate(sorted_cluster_index):
        pct = i / (n - 1) if n > 1 else 1.0
        if pct < 0.34:
            mapping[idx] = "Low Value Customer"
        elif pct < 0.67:
            mapping[idx] = "Medium Value Customer"
        else:
            mapping[idx] = "High Value Customer"
    return mapping


def kpi_card(label, value, sub=""):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🛍️ Customer Segmentation Intelligence Dashboard</h1>
    <p>Upload your customer data, auto-cluster with KMeans, and explore who your best customers really are.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR — UPLOAD
# =========================================================
st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started. "
            "Your file should contain numeric columns like **Annual_spending** and **Orders_count**.")
    st.stop()

df = load_data(uploaded_file)

# =========================================================
# SIDEBAR — CLEANING & FEATURE SELECTION
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧹 Data Cleaning")

rows_before = df.shape[0]
df = df.drop_duplicates()
dup_removed = rows_before - df.shape[0]

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Your file needs at least 2 numeric columns to run clustering.")
    st.stop()

default_features = [c for c in ["Annual_spending", "Orders_count"] if c in df.columns]
if len(default_features) < 2:
    default_features = numeric_cols[:2]

features = st.sidebar.multiselect(
    "Select features for clustering",
    numeric_cols,
    default=default_features
)

if len(features) < 2:
    st.warning("⚠️ Please select at least 2 numeric features to continue.")
    st.stop()

missing_filled = 0
for f in features:
    missing_filled += df[f].isnull().sum()
    df[f] = df[f].fillna(df[f].mean())

st.sidebar.success(f"✔ Removed {dup_removed} duplicate rows")
st.sidebar.success(f"✔ Filled {int(missing_filled)} missing values with mean")

# Optional ID / name column for search
id_col_options = ["(row index)"] + [c for c in df.columns if c not in numeric_cols] + numeric_cols
id_col = st.sidebar.selectbox("🔍 Column to use as Customer ID / Name (for search)", id_col_options)

# =========================================================
# SIDEBAR — CLUSTERING SETTINGS
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔢 Clustering Settings")
elbow_max_k = st.sidebar.slider("Elbow method: max K to test", 3, 12, 6)
k_value = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

# =========================================================
# SCALING
# =========================================================
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# ELBOW METHOD
# =========================================================
wcss = []
k_list = list(range(1, elbow_max_k + 1))
for k in k_list:
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    m.fit(X_scaled)
    wcss.append(m.inertia_)

# =========================================================
# FINAL KMEANS MODEL
# =========================================================
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

score = silhouette_score(X_scaled, clusters) if k_value > 1 and k_value < len(df) else 0.0

summary = df.groupby("Cluster")[features].mean().round(2)
sort_feature = features[0]
sorted_clusters = summary[sort_feature].sort_values()
customer_type_map = assign_customer_type(sorted_clusters.index)
df["Customer_Type"] = df["Cluster"].map(customer_type_map)

# =========================================================
# SIDEBAR — FILTERS
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Filters")
type_filter = st.sidebar.multiselect(
    "Customer Type",
    options=["High Value Customer", "Medium Value Customer", "Low Value Customer"],
    default=["High Value Customer", "Medium Value Customer", "Low Value Customer"]
)

range_filters = {}
for f in features:
    fmin, fmax = float(df[f].min()), float(df[f].max())
    if fmin == fmax:
        fmax = fmin + 1
    range_filters[f] = st.sidebar.slider(f"{f} range", fmin, fmax, (fmin, fmax))

filtered_df = df[df["Customer_Type"].isin(type_filter)].copy()
for f in features:
    lo, hi = range_filters[f]
    filtered_df = filtered_df[(filtered_df[f] >= lo) & (filtered_df[f] <= hi)]

# =========================================================
# KPI CARDS
# =========================================================
st.markdown("### 📊 Key Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Total Customers", f"{len(df):,}")
with c2:
    kpi_card(f"Avg {features[0]}", f"{df[features[0]].mean():,.1f}")
with c3:
    kpi_card(f"Avg {features[1]}", f"{df[features[1]].mean():,.1f}")
with c4:
    kpi_card("Silhouette Score", f"{score:.3f}", "closer to 1 = better")
with c5:
    high_pct = (df["Customer_Type"] == "High Value Customer").mean() * 100
    kpi_card("High Value %", f"{high_pct:.1f}%", "of total customers")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Elbow & Clusters", "🎨 Explore & Search", "👤 Predict a Customer",
    "📋 Data & Downloads", "📊 Business Insights"
])

# ---------------------------------------------------------
# TAB 1 — ELBOW + CLUSTER PLOT
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Elbow Method")
        elbow_fig = go.Figure()
        elbow_fig.add_trace(go.Scatter(
            x=k_list, y=wcss, mode="lines+markers",
            line=dict(color="#a78bfa", width=3),
            marker=dict(size=9, color="#38bdf8", line=dict(width=2, color="#7c3aed"))
        ))
        elbow_fig.add_vline(x=k_value, line_dash="dash", line_color="#4ade80",
                             annotation_text=f"Chosen K={k_value}", annotation_font_color="#4ade80")
        elbow_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Number of Clusters (K)", yaxis_title="WCSS (Inertia)",
            margin=dict(l=10, r=10, t=10, b=10), height=380
        )
        st.plotly_chart(elbow_fig, use_container_width=True)

    with col2:
        st.markdown("#### ⭐ Silhouette Score")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(score, 3),
            number={'font': {'color': '#ffffff'}},
            gauge={
                'axis': {'range': [-1, 1], 'tickcolor': '#ffffff'},
                'bar': {'color': "#a78bfa"},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [-1, 0], 'color': 'rgba(248,113,113,0.25)'},
                    {'range': [0, 0.5], 'color': 'rgba(250,204,21,0.25)'},
                    {'range': [0.5, 1], 'color': 'rgba(74,222,128,0.25)'},
                ],
            }
        ))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=10, r=10, t=30, b=10), height=380)
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("#### 🎨 Interactive Cluster Plot")
    scatter_fig = px.scatter(
        df, x=features[0], y=features[1],
        color="Customer_Type",
        color_discrete_map={
            "High Value Customer": "#4ade80",
            "Medium Value Customer": "#facc15",
            "Low Value Customer": "#f87171"
        },
        size=[10] * len(df),
        hover_data=df.columns,
        opacity=0.85
    )
    scatter_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Customer Type", height=480, margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("#### 📋 Cluster Summary")
    display_summary = summary.copy()
    display_summary["Customer_Type"] = [customer_type_map[i] for i in display_summary.index]
    display_summary["Count"] = df["Cluster"].value_counts().reindex(display_summary.index)
    st.dataframe(display_summary, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — SEARCH + EXPLORE
# ---------------------------------------------------------
with tab2:
    st.markdown("#### 🔍 Search Customer")
    search_term = st.text_input("Search by ID / Name / row value")

    display_df = filtered_df.copy()
    display_df["Segment"] = display_df["Customer_Type"].apply(badge_html)

    if search_term:
        if id_col == "(row index)":
            try:
                display_df = display_df.loc[[int(search_term)]]
            except (ValueError, KeyError):
                display_df = display_df.iloc[0:0]
        else:
            display_df = display_df[
                display_df[id_col].astype(str).str.contains(search_term, case=False, na=False)
            ]

    st.markdown(f"**{len(display_df)}** customer(s) found")
    st.write(display_df.drop(columns=["Segment"]).to_html(escape=False, index=False) +
             "<br>", unsafe_allow_html=True)

    # Show pretty badges below for the filtered/search results (first 20)
    st.markdown("##### Segment Preview")
    for _, row in display_df.head(20).iterrows():
        label = row[id_col] if id_col != "(row index)" else f"Row {row.name}"
        st.markdown(
            f"{badge_html(row['Customer_Type'])} &nbsp; **{label}** — "
            f"{features[0]}: {row[features[0]]:.1f}, {features[1]}: {row[features[1]]:.1f}",
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# TAB 3 — SINGLE CUSTOMER PREDICTION
# ---------------------------------------------------------
with tab3:
    st.markdown("#### 👤 Predict a New Customer's Segment")
    st.write("Enter values for a new or hypothetical customer to see which segment they fall into.")

    input_cols = st.columns(len(features))
    input_vals = {}
    for i, f in enumerate(features):
        with input_cols[i]:
            input_vals[f] = st.number_input(
                f, value=float(df[f].mean()), step=1.0, format="%.2f"
            )

    if st.button("🔮 Predict Segment"):
        new_point = pd.DataFrame([input_vals])[features]
        new_scaled = scaler.transform(new_point)
        pred_cluster = kmeans.predict(new_scaled)[0]
        pred_type = customer_type_map[pred_cluster]

        st.markdown(f"### Result: {badge_html(pred_type)}", unsafe_allow_html=True)

        if pred_type == "High Value Customer":
            st.success("This customer belongs to your top segment — prioritize retention & loyalty offers! 🟢")
        elif pred_type == "Medium Value Customer":
            st.warning("This customer has growth potential — consider upsell campaigns. 🟡")
        else:
            st.error("This customer is low value — consider engagement or win-back campaigns. 🔴")

# ---------------------------------------------------------
# TAB 4 — DATA TABLE + DOWNLOADS
# ---------------------------------------------------------
with tab4:
    st.markdown("#### 📋 Filtered Cluster Data")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes,
                            file_name="clustered_customers.csv", mime="text/csv")
    with dl2:
        excel_bytes = to_excel_bytes(filtered_df)
        st.download_button("📥 Download Excel", data=excel_bytes,
                            file_name="clustered_customers.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------
# TAB 5 — BUSINESS INSIGHTS
# ---------------------------------------------------------
with tab5:
    st.markdown("#### 📊 Auto-Generated Business Insights")

    total = len(df)
    high_n = (df["Customer_Type"] == "High Value Customer").sum()
    med_n = (df["Customer_Type"] == "Medium Value Customer").sum()
    low_n = (df["Customer_Type"] == "Low Value Customer").sum()

    high_val = df.loc[df["Customer_Type"] == "High Value Customer", features[0]].sum()
    total_val = df[features[0]].sum()
    high_share = (high_val / total_val * 100) if total_val else 0

    st.markdown(f"""
    <div class="insight-box">💡 <b>{high_n}</b> customers ({high_n/total*100:.1f}%) are classified as
    <b>High Value</b>, and together they contribute <b>{high_share:.1f}%</b> of total {features[0]}.</div>

    <div class="insight-box">💡 <b>{med_n}</b> customers ({med_n/total*100:.1f}%) fall into the
    <b>Medium Value</b> segment — a strong target for upsell and cross-sell campaigns.</div>

    <div class="insight-box">💡 <b>{low_n}</b> customers ({low_n/total*100:.1f}%) are <b>Low Value</b> —
    consider re-engagement offers, discounts, or churn-prevention outreach.</div>

    <div class="insight-box">💡 Clustering quality (Silhouette Score) is <b>{score:.3f}</b> —
    {"a strong, well-separated segmentation." if score > 0.5 else "a reasonable segmentation; try adjusting K or features for tighter clusters." if score > 0.25 else "clusters overlap significantly; consider different features or K."}</div>
    """, unsafe_allow_html=True)

    st.markdown("##### Segment Distribution")
    pie_fig = px.pie(
        df, names="Customer_Type", hole=0.55,
        color="Customer_Type",
        color_discrete_map={
            "High Value Customer": "#4ade80",
            "Medium Value Customer": "#facc15",
            "Low Value Customer": "#f87171"
        }
    )
    pie_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("##### Average Feature Value by Segment")
    bar_fig = px.bar(
        summary.reset_index().assign(Customer_Type=lambda d: d["Cluster"].map(customer_type_map)),
        x="Customer_Type", y=features, barmode="group",
        color_discrete_sequence=["#a78bfa", "#38bdf8", "#34d399", "#facc15"]
    )
    bar_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("<br><center style='color:#5f6180; font-size:0.8rem;'>Built with ❤️ using Streamlit • KMeans Customer Segmentation</center>", unsafe_allow_html=True)
