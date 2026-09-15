"""Load and clean the UCI Default of Credit Card Clients dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_CSV, RAW_XLS

SEX_MAP = {1: "Male", 2: "Female"}
EDU_MAP = {
    1: "Graduate school",
    2: "University",
    3: "High school",
}
MARRIAGE_MAP = {1: "Married", 2: "Single"}


def _read_xls(path: Path) -> pd.DataFrame:
    import xlrd

    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    headers = [str(sheet.cell_value(1, c)).strip() for c in range(sheet.ncols)]
    rows = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(2, sheet.nrows)]
    return pd.DataFrame(rows, columns=headers)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the Yeh (2009) UCI credit-card default file."""
    if path is not None:
        src = Path(path)
        if src.suffix.lower() in {".xls", ".xlsx"}:
            df = _read_xls(src)
        else:
            df = pd.read_csv(src)
    elif RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
    elif RAW_XLS.exists():
        df = _read_xls(RAW_XLS)
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Place the UCI file at {RAW_CSV} or {RAW_XLS}."
        )

    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": "default"})
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Recode rare education/marriage buckets and attach readable labels."""
    out = df.copy()
    out["default"] = out["default"].astype(int)
    out["education"] = out["education"].astype(int).map(lambda x: EDU_MAP.get(x, "Other"))
    out["marriage"] = out["marriage"].astype(int).map(lambda x: MARRIAGE_MAP.get(x, "Other"))
    out["sex"] = out["sex"].astype(int).map(SEX_MAP).fillna("Other")
    pay_cols = [c for c in out.columns if c.startswith("pay_") and c[4:].isdigit()]
    for col in pay_cols:
        out[col] = out[col].astype(int)
    return out
