from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px

from src.cleaner import run_cleaning
from src.report import missing_by_column, overall_missing_pct, numeric_summary, outlier_counts_iqr, negative_counts

def load_or_clean(input_csv: Path, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    cleaned = outdir / "cleaned.csv"
    if not cleaned.exists():
        run_cleaning(input_csv, outdir)
    return pd.read_csv(cleaned)

def build_app(df: pd.DataFrame, outdir: Path):
    app = Dash(__name__)
    app.title = "Data Cleaning Report"

    miss = missing_by_column(df)
    numsum = numeric_summary(df)
    outliers = outlier_counts_iqr(df)
    negatives = negative_counts(df)
    overall_miss = overall_missing_pct(df)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    app.layout = html.Div([
        html.H2("Data Cleaning & Quality Dashboard"),
        html.Div([
            html.Div(className="card", children=[html.H4("Rows"), html.H3(f"{len(df):,}")], style={"padding":"10px","border":"1px solid #ddd","borderRadius":"10px","marginRight":"10px"}),
            html.Div(className="card", children=[html.H4("Overall Missing %"), html.H3(f"{overall_miss:.2f}%")], style={"padding":"10px","border":"1px solid #ddd","borderRadius":"10px","marginRight":"10px"}),
        ], style={"display":"flex","marginBottom":"10px"}),

        html.H3("Missing by Column"),
        dcc.Graph(figure=px.bar(miss, x="column", y="missing", title="Missing values by column")),

        html.H3("Numeric Summary"),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in numsum.columns],
            data=numsum.to_dict("records"),
            page_size=10,
            sort_action="native",
            style_table={"overflowX":"auto"}
        ),

        html.H3("Outlier Counts (IQR)"),
        dcc.Graph(figure=px.bar(outliers, x="column", y="outliers", title="Outliers by numeric column")),

        html.H3("Negative Value Counts"),
        dcc.Graph(figure=px.bar(negatives, x="column", y="negatives", title="Negative values by numeric column")),

        html.H3("Histogram"),
        dcc.Dropdown(id="num-col", options=[{"label":c,"value":c} for c in num_cols], value=(num_cols[0] if num_cols else None), clearable=False, style={"maxWidth":"300px"}),
        dcc.Graph(id="hist-plot")
    ], style={"fontFamily":"Arial, sans-serif", "padding":"20px"})

    @app.callback(Output("hist-plot","figure"), Input("num-col","value"))
    def _histogram(col):
        if not col:
            return px.histogram(title="No numeric columns found")
        return px.histogram(df, x=col, nbins=30, title=f"Histogram — {col}")

    return app

def main():

    ap = argparse.ArgumentParser(description="Run the cleaning dashboard.")
    ap.add_argument("--config", default="data_cleaning_config.yaml", help="Path to YAML config")
    ap.add_argument("--input", default="sample_data/retail_sales_raw.csv", help="Path to raw CSV (will be cleaned first time)")
    ap.add_argument("--outdir", default="outputs", help="Outputs directory")
    args = ap.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    import yaml
    cfg = {}
    from pathlib import Path as _P
    cfg_path = _P(args.config)
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        # trigger cleaning with config
        from src.cleaner import run_cleaning
        run_cleaning(input_csv, outdir, config=cfg)
        df = pd.read_csv(outdir / (cfg.get('io', {}).get('cleaned_filename', 'cleaned.csv')))
        app = build_app(df, outdir)
        app.run(debug=True)

if __name__ == "__main__":
    main()
