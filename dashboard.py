
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px

# === Load data (use sample_data as default) ===
CSV_PATH = "sample_data/retail_sales_raw.csv"

def _to_number(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return s
    if isinstance(s, str):
        s = s.strip().replace("$","").replace(",","")
        try: return float(s)
        except: return np.nan
    return np.nan

def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    # Basic cleaning for visuals only
    if "Unit Price" in df.columns: df["Unit Price"] = df["Unit Price"].map(_to_number)
    if "Sales Amount" in df.columns: df["Sales Amount"] = df["Sales Amount"].map(_to_number)
    if "Quantity" in df.columns: df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    return df

df = load_data()

# === KPI Tracker (state of data) ===
total_rows = int(len(df))
total_cols = int(df.shape[1])
missing_cells = int(df.isna().sum().sum())
total_cells = int(df.shape[0] * df.shape[1]) or 1
missing_pct = round(100 * missing_cells / total_cells, 2)
dup_rows = int(df.duplicated().sum())
num_cols = int(df.select_dtypes(include=[np.number]).shape[1])

kpis = [
    {"label":"Rows", "value": f"{total_rows:,}"},
    {"label":"Columns", "value": f"{total_cols}"},
    {"label":"Missing (%)", "value": f"{missing_pct}%"},
    {"label":"Duplicates", "value": f"{dup_rows:,}"},
    {"label":"Numeric Cols", "value": f"{num_cols}"},
]

# === Figures (keep histograms & pie only) ===
fig_hist_qty = None
if "Quantity" in df.columns:
    fig_hist_qty = px.histogram(df, x="Quantity", nbins=30, title="Quantity Distribution")

fig_hist_sales = None
if "Sales Amount" in df.columns:
    fig_hist_sales = px.histogram(df, x="Sales Amount", nbins=30, title="Sales Amount Distribution")

# Pie by Customer (Top 5) if available; else by SKU
def pie_series(frame):
    if "Customer" in frame.columns:
        s = frame["Customer"].fillna("Unknown").value_counts().nlargest(5)
        return "Customer", s
    if "SKU" in frame.columns:
        s = frame["SKU"].fillna("Unknown").value_counts().nlargest(5)
        return "SKU", s
    return None, None

pie_label, pie_counts = pie_series(df)
fig_pie = None
if pie_counts is not None:
    fig_pie = px.pie(
        values=pie_counts.values,
        names=pie_counts.index,
        title=f"Top {len(pie_counts)} by {pie_label} (share of rows)"
    )

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Data Cleaning Portfolio — Analytics"

kpi_items = [
    html.Div([html.Div(k["label"], className="kpi-label"), html.Div(k["value"], className="kpi-value")], className="kpi-card")
    for k in kpis
]

graphs = []
if fig_hist_qty is not None: graphs.append(dcc.Graph(figure=fig_hist_qty))
if fig_hist_sales is not None: graphs.append(dcc.Graph(figure=fig_hist_sales))
if fig_pie is not None: graphs.append(dcc.Graph(figure=fig_pie))

app.layout = html.Div(children=[
    html.H1("Data Health & Insights"),
    html.Div(kpi_items, className="kpi-row"),
    html.Hr(),
    html.Div(graphs, className="charts"),
], className="container")

# Simple inline CSS (flat, minimal)
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Data Cleaning Portfolio — Analytics</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {%metas%}
    {%css%}
    <style>
        .container { max-width: 1100px; margin: 20px auto; padding: 0 12px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 12px; }
        .kpi-card { padding: 12px; border: 1px solid #e5e7eb; border-radius: 12px; }
        .kpi-label { font-size: 12px; color: #6b7280; }
        .kpi-value { font-size: 22px; font-weight: 700; }
        .charts { display: grid; grid-template-columns: 1fr; gap: 18px; }
        @media (min-width: 900px) { .charts { grid-template-columns: 1fr 1fr; } }
        h1 { font-size: 24px; margin: 8px 0 16px; }
    </style>
    {%favicon%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=True)
