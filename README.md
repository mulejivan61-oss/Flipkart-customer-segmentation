# 🛍️ Customer Segmentation Intelligence Dashboard

A Streamlit-based dashboard that uses **KMeans clustering** to segment customers into
**Low / Medium / High Value** groups based on their spending and order behavior.

---

## 📁 Project Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit dashboard (all logic + UI) |
| `requirements.txt` | Python packages needed to run the app |
| `test_customers.csv` | Sample dataset (1015 rows) to test the dashboard |

---

## ⚙️ How It Works (Behind the Scenes)

1. **Upload** — User uploads a CSV or Excel file.
2. **Cleaning** — Duplicate rows are removed, missing values in numeric columns are
   filled with the column mean.
3. **Feature Selection** — By default the app looks for `Annual_spending` and
   `Orders_count` columns. If they don't exist, you can pick any 2+ numeric columns
   manually from the sidebar.
4. **Scaling** — Features are standardized using `StandardScaler` (mean=0, std=1) so
   that columns with different ranges (e.g. spending in thousands vs orders in units)
   don't unfairly dominate the clustering.
5. **Elbow Method** — KMeans is run for K = 1 to N (you choose N) and the WCSS
   (inertia) is plotted. The "elbow" point is where adding more clusters stops giving
   big improvements — that's usually the best K.
6. **Final Clustering** — KMeans runs again with your chosen K (default 3) to assign
   every customer to a cluster.
7. **Silhouette Score** — Measures how well-separated the clusters are (range -1 to 1,
   higher is better). Shown as a gauge chart.
8. **Customer Type Labeling** — Clusters are automatically ranked by average spending
   (or your first selected feature) and labeled:
   - 🔴 **Low Value Customer** — bottom ~34%
   - 🟡 **Medium Value Customer** — middle ~33%
   - 🟢 **High Value Customer** — top ~33%

   This works for *any* number of clusters (K), not just 3.

---

## 🖥️ Dashboard Sections

### 📊 KPI Cards (top of page)
Quick numbers: total customers, average spending, average orders, silhouette score,
% of high-value customers.

### 📈 Elbow & Clusters tab
- Elbow graph (with your chosen K marked)
- Silhouette score gauge
- Interactive Plotly scatter plot of clusters (hover to see full customer details)
- Cluster summary table (mean values + count per cluster)

### 🎨 Explore & Search tab
- Search any customer by ID/name/value (or row number if no ID column exists)
- Color-coded segment badges for quick scanning

### 👤 Predict a Customer tab
- Enter spending/order values for a **new/hypothetical customer**
- Instantly predicts which segment (Low/Medium/High) they'd fall into, using the
  same trained scaler + KMeans model

### 📋 Data & Downloads tab
- Full filtered data table
- Download results as **CSV** or **Excel**

### 📊 Business Insights tab
- Auto-generated text insights (e.g. "X% of customers are High Value and drive Y%
  of total revenue")
- Pie chart of segment distribution
- Bar chart comparing average feature values across segments

### 🎛️ Sidebar Filters
- Filter by customer type (Low/Medium/High)
- Filter by value ranges for each selected feature
- Change number of clusters (K) live and see everything update

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open the link Streamlit gives you (usually http://localhost:8501)
# 4. Upload test_customers.csv (or your own file) from the sidebar
```

---

## 🧪 Testing It

Use the included `test_customers.csv`:
- 1015 rows (includes 15 intentional duplicates to test cleaning)
- Two columns: `Annual_spending`, `Orders_count`
- ~5% missing values in each column (to test null-handling)
- Data is built from 3 real underlying segments, so the elbow graph and clusters
  will show clear, meaningful structure — good for verifying the app works correctly.

---

## 📌 Notes / Tips

- If your dataset has different column names, just select them manually from the
  **"Select features for clustering"** dropdown in the sidebar — the app isn't
  hardcoded to `Annual_spending`/`Orders_count`.
- The **Predict a Customer** tab always uses the model trained on your currently
  uploaded data — re-upload or change K to retrain.
- Silhouette Score interpretation:
  - **> 0.5** → strong, well-separated clusters
  - **0.25 – 0.5** → reasonable, could be improved
  - **< 0.25** → clusters overlap a lot, try different features or K

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **scikit-learn** — StandardScaler, KMeans, silhouette_score
- **Plotly** — interactive charts (elbow, scatter, gauge, pie, bar)
- **pandas / numpy** — data handling
- **xlsxwriter / openpyxl** — Excel export/read support
