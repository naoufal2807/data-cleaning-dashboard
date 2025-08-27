import pandas as pd
from src.cleaner import clean_dataframe

def test_basic_cleaning():
    df = pd.DataFrame({
        "Order Date": ["2024-01-01", "2024-01-01", None],
        "Amount $": ["$10", "$10", None],
        "Category": ["A", "A", None],
    })
    res = clean_dataframe(df)
    # duplicates removed (first two rows are duplicates after standardization & parsing)
    assert res.duplicates_removed >= 1
    assert "parsed dates" in " ".join([n.lower() for n in res.notes])
    assert "filled numeric" in " ".join([n.lower() for n in res.notes])
    assert "filled categorical" in " ".join([n.lower() for n in res.notes])
    assert len(res.df) == res.rows_after
