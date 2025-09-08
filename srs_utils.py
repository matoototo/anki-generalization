from __future__ import annotations

import html
import json
import re
import sqlite3
import os
import zipfile
import tempfile

from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

# --------------------
# Text and cue helpers
# --------------------

def strip_markup(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"[`*_#>~\\-]+", " ", text)  # light markdown cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_fields(flds: str) -> List[str]:
    return (flds or "").split("\x1f")


def cue_text_from_note(sfld: str, flds: str, prefer_field_index: Optional[int] = None) -> str:
    fields = parse_fields(flds)
    if prefer_field_index is not None and 0 <= prefer_field_index < len(fields):
        return strip_markup(fields[prefer_field_index])
    return strip_markup(sfld)


def compute_cue_length(sfld: str, flds: str, prefer_field_index: Optional[int] = None) -> int:
    return len(cue_text_from_note(sfld, flds, prefer_field_index))


# -----------------
# Time conversions
# -----------------

def ms_to_s(ms: int | float) -> float:
    try:
        return float(ms) / 1000.0
    except Exception:
        return float("nan")


def ts_ms_to_dt(ms: int | float) -> datetime:
    return datetime.utcfromtimestamp(int(ms) / 1000.0)


# -----------------
# DB and data model
# -----------------

def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    try:
        row = con.execute(
            'select name from sqlite_master where type="table" and name=?', (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False


def build_deck_and_model_maps(con: sqlite3.Connection) -> Tuple[Dict[int, str], Dict[int, str]]:
    col = con.execute("select * from col").fetchone()
    decks_blob, models_blob = {}, {}

    if col:
        try:
            if col["decks"]:
                decks_blob = json.loads(col["decks"])  # type: ignore[index]
        except Exception:
            decks_blob = {}
        try:
            if "models" in col.keys() and col["models"]:
                models_blob = json.loads(col["models"])  # type: ignore[index]
            elif "notetypes" in col.keys() and col["notetypes"]:
                models_blob = json.loads(col["notetypes"])  # type: ignore[index]
        except Exception:
            models_blob = {}

    deck_map: Dict[int, str] = {}
    if isinstance(decks_blob, dict) and decks_blob:
        for k, v in decks_blob.items():
            if not isinstance(v, dict):
                continue
            did = v.get("id")
            if did is None:
                try:
                    did = int(k)
                except Exception:
                    continue
            try:
                did = int(did)
            except Exception:
                continue
            name = v.get("name") or str(did)
            deck_map[did] = name
    elif _table_exists(con, "decks"):
        try:
            df_decks = pd.read_sql_query("select id, name from decks", con)
            for did, name in zip(df_decks["id"], df_decks["name"]):
                try:
                    deck_map[int(did)] = str(name)
                except Exception:
                    pass
        except Exception:
            pass

    model_map: Dict[int, str] = {}
    if isinstance(models_blob, dict):
        for k, v in models_blob.items():
            if isinstance(v, dict):
                try:
                    model_map[int(k)] = v.get("name", str(k))
                except Exception:
                    pass

    return deck_map, model_map


def load_reviews(con: sqlite3.Connection, deck_map: Dict[int, str], model_map: Dict[int, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    query = (
        """
        select
          r.id          as review_ts_ms,
          r.cid         as cid,
          r.ease        as ease,
          r.ivl         as ivl_days,
          r.lastIvl     as last_ivl_days,
          r.factor      as factor,
          r.time        as review_time_ms,
          r.type        as rtype,
          c.did         as did,
          c.ord         as ord,
          c.odid        as odid,
          n.id          as nid,
          n.sfld        as sfld,
          n.flds        as flds,
          n.mid         as mid
        from revlog r
        join cards c on c.id = r.cid
        join notes n on n.id = c.nid
        """
    ).strip()

    rows = con.execute(query).fetchall()
    if not rows:
        raise RuntimeError("No revlog rows found. Check that your collection has reviews.")

    df = pd.DataFrame(rows, columns=rows[0].keys())

    # Enrichment
    df["review_s"] = df["review_time_ms"].apply(ms_to_s)
    df["dt"] = df["review_ts_ms"].apply(ts_ms_to_dt)
    # Prefer original deck for filtered decks
    df["deck_id"] = df["odid"].where(df["odid"] > 0, df["did"])
    df["deck_name"] = df["deck_id"].map(deck_map)
    df["deck_name"] = df["deck_name"].fillna(df["did"].astype(str))
    df["deck"] = df["deck_name"]  # alias
    df["model"] = df["mid"].map(model_map).fillna(df["mid"].astype(str))
    df["cue_length"] = [compute_cue_length(s, f) for s, f in zip(df["sfld"], df["flds"])]
    df["is_correct"] = df["ease"] != 1  # Again == 1

    # Filter implausible times
    df["valid_time"] = (df["review_s"] >= 0.25) & (df["review_s"] <= 30.0)
    df_valid = df[df["valid_time"]].copy()

    return df, df_valid


def _is_sqlite_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        return header.startswith(b"SQLite format 3\x00")
    except Exception:
        return False


def open_collection(db_path: Path | str) -> sqlite3.Connection:
    """
    Open an Anki collection given either:
      - a direct path to a SQLite DB (collection.anki2 / collection.sqlite / collection.anki21), or
      - an export package (.apkg / .colpkg), from which the collection file will be extracted to a temp file.

    Returns a sqlite3.Connection with row_factory set. If a temp file was created, it will be placed in the
    system temp directory and not automatically deleted while the process is running.
    """
    p = Path(db_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Collection not found: {p}")

    # If a directory is given, try to locate the collection file inside it
    if p.is_dir():
        for candidate in ("collection.anki21", "collection.anki2", "collection.sqlite"):
            cand_path = p / candidate
            if cand_path.exists():
                p = cand_path
                break
        else:
            raise FileNotFoundError(
                f"No collection file found in directory {p}. Expected one of collection.anki21/anki2/sqlite"
            )

    # Try a direct SQLite connection first (matches the single-notebook behavior)
    try:
        con = sqlite3.connect(str(p))
        con.row_factory = sqlite3.Row
        # Execute a trivial query to ensure the file is a valid SQLite DB
        con.execute("PRAGMA schema_version;").fetchone()
        return con
    except sqlite3.DatabaseError:
        # If direct open fails, check if sibling known collection files exist (anki21 vs anki2 vs sqlite)
        parent = p.parent
        for candidate in ("collection.anki21", "collection.anki2", "collection.sqlite"):
            cand = parent / candidate
            if cand.exists() and _is_sqlite_file(cand):
                con = sqlite3.connect(str(cand))
                con.row_factory = sqlite3.Row
                return con
        # Fall through and try handling as a package next
        pass

    # Package formats: .apkg/.colpkg are zip files containing collection.anki2/anki21
    if p.is_file() and zipfile.is_zipfile(str(p)):
        with zipfile.ZipFile(str(p), 'r') as zf:
            candidates = [
                'collection.anki21',
                'collection.anki2',
                'collection.sqlite',
                'collection.anki',
            ]
            inner = None
            for name in zf.namelist():
                base = name.split('/')[-1]
                if base in candidates:
                    inner = name
                    break
            if inner is None:
                raise FileNotFoundError(
                    f"No collection file found inside package {p}. Expected one of {candidates}."
                )
            # Extract to a temp file
            tmpdir = Path(tempfile.gettempdir())
            out_path = tmpdir / f"anki_collection_extracted_{os.getpid()}.anki2"
            with zf.open(inner) as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            if not _is_sqlite_file(out_path):
                raise RuntimeError(
                    f"Extracted file from {p} is not a valid SQLite DB: {inner} -> {out_path}"
                )
            con = sqlite3.connect(str(out_path))
            con.row_factory = sqlite3.Row
            return con

    # If we got here, the file exists but isn't a SQLite DB or known package
    raise sqlite3.DatabaseError(
        f"File is not a SQLite database or Anki package: {p}.\n"
        f"Hint: point to your profile's collection.anki2, or provide an .apkg/.colpkg export."
    )


# --------------
# Select helpers
# --------------

def available_decks(df_valid: pd.DataFrame) -> List[str]:
    return sorted(df_valid["deck_name"].astype(str).unique())


def filter_by_decks(df_valid: pd.DataFrame, decks: Iterable[str]) -> pd.DataFrame:
    decks = list(decks)
    if not decks:
        return df_valid.copy()
    return df_valid[df_valid["deck_name"].isin(decks)].copy()


# ---------
# Plotting
# ---------

def plot_population_distributions(df_sel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # Overall review time
    axes[0].hist(df_sel["review_s"], bins=100, color="#4c78a8", alpha=0.9)
    axes[0].set_title("Review Time (s) distribution")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("count")
    # Cue length
    axes[1].hist(df_sel["cue_length"], bins=100, color="#f58518", alpha=0.9)
    axes[1].set_title("Cue Length (chars) distribution")
    axes[1].set_xlabel("characters")
    axes[1].set_ylabel("count")
    # Review time by correctness
    if len(df_sel):
        bins = np.linspace(df_sel["review_s"].min(), df_sel["review_s"].max(), 100)
    else:
        bins = 50
    axes[2].hist(df_sel[df_sel["is_correct"]]["review_s"], bins=bins, color="#54a24b", alpha=0.6, label="Correct")
    axes[2].hist(df_sel[~df_sel["is_correct"]]["review_s"], bins=bins, color="#e45756", alpha=0.6, label="Incorrect")
    axes[2].set_title("Review Time by correctness")
    axes[2].set_xlabel("seconds")
    axes[2].set_ylabel("count")
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def _scatter_with_fit(x: np.ndarray, y: np.ndarray, title: str) -> Tuple[float, float]:
    if len(x) >= 2:
        b, a = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, 0.0
    xx = np.linspace(x.min() if len(x) else 0, x.max() if len(x) else 1, 100)
    yy = a + b * xx
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=6, alpha=0.2, color="#4c78a8", edgecolors="none")
    plt.plot(xx, yy, color="black", linewidth=2, label=f"y = {a:.2f} + {b:.3f} x")
    plt.xlabel("Cue length (chars)")
    plt.ylabel("Review time (s)")
    plt.title(title)
    plt.legend()
    plt.show()
    return a, b
