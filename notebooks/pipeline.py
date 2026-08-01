"""
Shared cleaning + feature-engineering logic for the House Price project.

This module is imported by both the training notebook and the FastAPI backend so
that a request is transformed exactly the way the training rows were. Keeping it
in one place is what stops the classic "works in the notebook, 500s in the API"
failure.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SQM_TO_SQFT = 10.7639

# Columns that must never reach the model.
#   Price (in rupees) is price-PER-SQFT: price = price_per_sqft * area, so leaving
#   it in leaks the target and inflates R^2 to ~0.99.
#   Plot Area / Dimensions are 100% empty. Status is constant ('Ready to Move').
LEAKY_OR_USELESS = [
    "Index",
    "Price (in rupees)",
    "Plot Area",
    "Dimensions",
    "Status",
    "Description",
]

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathroom",
    "balcony",
    "car_parking",
    "floor_num",
    "total_floors",
    "floor_ratio",
]

CATEGORICAL_FEATURES = [
    "location",
    "area_type",
    "property_type",
    "furnishing",
    "transaction",
    "ownership",
    "facing",
    "overlooking",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_amount(value) -> float | None:
    """'42 Lac' -> 4200000.0, '1.2 Cr' -> 12000000.0, 'Call for Price' -> None."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace(",", "")
    if not text:
        return None
    try:
        if "lac" in text:
            return float(text.replace("lac", "").strip()) * 1e5
        if "cr" in text:
            return float(text.replace("cr", "").strip()) * 1e7
        return float(text)
    except ValueError:
        return None


def parse_area(value) -> float | None:
    """'1200 sqft' -> 1200.0, '140 sqm' -> 1506.9. Returns sqft."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace(",", "")
    match = re.search(r"([\d.]+)", text)
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    if number <= 0:
        return None
    if "sqm" in text or "sq m" in text:
        return number * SQM_TO_SQFT
    if "sqyrd" in text or "yard" in text:
        return number * 9.0
    if "acre" in text:
        return number * 43560.0
    return number


def parse_floor(value) -> tuple[float | None, float | None]:
    """'3 out of 10' -> (3, 10). 'Ground out of 5' -> (0, 5). 'Basement' -> (-1, None)."""
    if not isinstance(value, str):
        return None, None
    text = value.strip().lower()

    def word_to_num(word: str) -> float | None:
        word = word.strip()
        if "ground" in word:
            return 0.0
        if "upper basement" in word or "lower basement" in word or "basement" in word:
            return -1.0
        found = re.search(r"(-?\d+)", word)
        return float(found.group(1)) if found else None

    if "out of" in text:
        left, right = text.split("out of", 1)
        return word_to_num(left), word_to_num(right)
    return word_to_num(text), None


def parse_count(value) -> float | None:
    """'2' -> 2.0, '> 10' -> 11.0, '3+' -> 3.0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text.startswith(">"):
        found = re.search(r"(\d+)", text)
        return float(found.group(1)) + 1 if found else None
    found = re.search(r"(\d+)", text)
    return float(found.group(1)) if found else None


def parse_bedrooms(title) -> float | None:
    """Extract BHK/bedroom count out of the listing title."""
    if not isinstance(title, str):
        return None
    match = re.search(r"(\d+)\s*(?:bhk|bedroom|bed room|rk)", title.lower())
    return float(match.group(1)) if match else None


def parse_property_type(title) -> str:
    """Coarse property type from the title text."""
    if not isinstance(title, str):
        return "unknown"
    text = title.lower()
    for needle, label in (
        ("villa", "villa"),
        ("independent house", "house"),
        ("independent floor", "builder_floor"),
        ("builder floor", "builder_floor"),
        ("penthouse", "penthouse"),
        ("studio", "studio"),
        ("plot", "plot"),
        ("land", "plot"),
        ("house", "house"),
        ("apartment", "flat"),
        ("flat", "flat"),
    ):
        if needle in text:
            return label
    return "unknown"


def normalise_facing(value) -> str:
    """'South -West' and 'South - West' both -> 'south_west'."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = re.sub(r"[\s\-]+", "_", value.strip().lower())
    return text.strip("_") or "unknown"


def normalise_overlooking(value) -> str:
    """The column is multi-valued ('Garden/Park, Main Road'); keep the primary one."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    primary = value.split(",")[0].strip().lower()
    return re.sub(r"[\s/]+", "_", primary) or "unknown"


def simple_category(value, fallback: str = "unknown") -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.strip().lower().replace(" ", "_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw scraped frame into the model's feature frame (+ target)."""
    out = pd.DataFrame(index=df.index)

    # --- target -----------------------------------------------------------
    if "Amount(in rupees)" in df.columns:
        out["price"] = df["Amount(in rupees)"].apply(parse_amount)

    # --- area: coalesce carpet + super, they are mostly mutually exclusive --
    carpet = df["Carpet Area"].apply(parse_area) if "Carpet Area" in df else pd.Series(index=df.index, dtype=float)
    super_area = df["Super Area"].apply(parse_area) if "Super Area" in df else pd.Series(index=df.index, dtype=float)
    out["area_sqft"] = carpet.fillna(super_area)
    out["area_type"] = np.where(carpet.notna(), "carpet", np.where(super_area.notna(), "super", "unknown"))

    # --- floor ------------------------------------------------------------
    if "Floor" in df.columns:
        parsed = df["Floor"].apply(parse_floor)
        out["floor_num"] = [p[0] for p in parsed]
        out["total_floors"] = [p[1] for p in parsed]
    else:
        out["floor_num"] = np.nan
        out["total_floors"] = np.nan
    out["floor_ratio"] = np.where(
        (out["total_floors"].notna()) & (out["total_floors"] > 0),
        out["floor_num"] / out["total_floors"].replace(0, np.nan),
        np.nan,
    )

    # --- counts -----------------------------------------------------------
    for src, dest in (("Bathroom", "bathroom"), ("Balcony", "balcony"), ("Car Parking", "car_parking")):
        out[dest] = df[src].apply(parse_count) if src in df else np.nan

    # --- derived from the title ------------------------------------------
    title = df["Title"] if "Title" in df else pd.Series(index=df.index, dtype=object)
    out["bedrooms"] = title.apply(parse_bedrooms)
    out["property_type"] = title.apply(parse_property_type)

    # --- plain categoricals ----------------------------------------------
    out["location"] = df["location"].apply(lambda v: simple_category(v, "unknown")) if "location" in df else "unknown"
    out["furnishing"] = df["Furnishing"].apply(simple_category) if "Furnishing" in df else "unknown"
    out["transaction"] = df["Transaction"].apply(simple_category) if "Transaction" in df else "unknown"
    out["ownership"] = df["Ownership"].apply(simple_category) if "Ownership" in df else "unknown"
    out["facing"] = df["facing"].apply(normalise_facing) if "facing" in df else "unknown"
    out["overlooking"] = df["overlooking"].apply(normalise_overlooking) if "overlooking" in df else "unknown"

    return out


def clean_training_frame(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full training-time cleaning: features, then row filtering."""
    feats = build_features(df)
    before = len(feats)

    feats = feats.dropna(subset=["price", "area_sqft"])
    if verbose:
        print(f"dropped {before - len(feats):,} rows with unusable price or area")

    # Physically implausible listings.
    feats = feats[(feats["area_sqft"] >= 100) & (feats["area_sqft"] <= 25_000)]
    feats = feats[(feats["price"] >= 1e5) & (feats["price"] <= 1e9)]

    # Price-per-sqft outliers: keep the middle 98%.
    pps = feats["price"] / feats["area_sqft"]
    low, high = pps.quantile([0.01, 0.99])
    feats = feats[(pps >= low) & (pps <= high)]
    if verbose:
        print(f"price-per-sqft kept between Rs {low:,.0f} and Rs {high:,.0f}/sqft")

    # Near-duplicate listings would otherwise land in both train and test and
    # silently inflate the score.
    dupe_cols = ["location", "area_sqft", "price", "bedrooms", "floor_num"]
    before_dupes = len(feats)
    feats = feats.drop_duplicates(subset=dupe_cols)
    if verbose:
        print(f"dropped {before_dupes - len(feats):,} near-duplicate listings")
        print(f"final training rows: {len(feats):,}")

    return feats.reset_index(drop=True)
