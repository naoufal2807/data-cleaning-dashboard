# Data Cleaning & Automation — Portfolio Project

This is a ready-to-run template for a **Data Cleaning & Automation** portfolio entry.
It demonstrates automatic cleaning (duplicates removal, type parsing, missing value imputation),
and produces **before/after** table snapshots plus a small **cleaning log**.

## Quickstart

```bash
# 1) (Optional) Create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run on the sample dataset
python main.py --input sample_data/retail_sales_raw.csv --outdir outputs

# 4) Check results in ./outputs/
#    - sample_before.png
#    - sample_after.png
#    - cleaning_log.txt
#    - cleaned.csv
```

## What it does
- Standardizes column names
- Parses dates and numeric amounts (removes currency symbols)
- Removes duplicates
- Fills missing values (numeric → median, categorical → mode/"Unknown")
- Saves cleaned CSV
- Exports BEFORE/AFTER previews as PNGs
- Writes a simple **cleaning_log.txt** for your portfolio description

## Customize
Open `src/cleaner.py` and tweak:
- `NUMERIC_HINTS` to match your dataset amount/price columns
- Column selection or advanced rules per column
- Add validation rules (e.g., non-negative quantities)

## Generate Portfolio Screenshots
Use the images in `outputs/` (before/after) in your Upwork portfolio entry.
Optionally, open the cleaned CSV in Excel/Power BI for additional visuals.

## License
MIT


---
## 📊 Run the Interactive Dashboard (Plotly Dash)

This app shows **statistics after cleaning** (missing values, outliers, numeric summary) and plots.

```bash
pip install -r requirements.txt
python app_dash.py --input sample_data/retail_sales_raw.csv --outdir outputs
# open the local URL printed in the terminal (usually http://127.0.0.1:8050)
```
