"""
Core data structures and data loading for the Entity Resolution pipeline.

This module defines the canonical Record object and provides functions
to load TSV data from all sources.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class Record:
    """Canonical internal representation of a business entity record.
    Uses ultra-compact __slots__ and dynamic on-the-fly properties for tokens
    to avoid allocating 80M+ Python heap objects across millions of records.
    """
    __slots__ = (
        "record_id", "source_id", "original_name", "normalized_name",
        "compact_name", "original_address", "normalized_address",
        "country", "state", "city", "locality", "postal_code",
        "house_number", "road", "landmark", "source_metadata"
    )

    def __init__(
        self,
        record_id: str,
        source_id: str,
        original_name: str = "",
        normalized_name: str = "",
        compact_name: str = "",
        name_tokens: Optional[List[str]] = None,
        original_address: str = "",
        normalized_address: str = "",
        address_tokens: Optional[List[str]] = None,
        country: str = "",
        state: str = "",
        city: str = "",
        locality: str = "",
        postal_code: str = "",
        house_number: str = "",
        road: str = "",
        landmark: str = "",
        numeric_tokens: Optional[List[str]] = None,
        record_text: str = "",
        source_metadata: Optional[Dict] = None,
    ):
        self.record_id = record_id
        self.source_id = source_id
        self.original_name = original_name
        self.normalized_name = normalized_name if normalized_name else (" ".join(name_tokens) if name_tokens else "")
        self.compact_name = compact_name
        self.original_address = original_address
        self.normalized_address = normalized_address if normalized_address else (" ".join(address_tokens) if address_tokens else "")
        self.country = country
        self.state = state
        self.city = city
        self.locality = locality
        self.postal_code = postal_code
        self.house_number = house_number
        self.road = road
        self.landmark = landmark
        self.source_metadata = source_metadata if source_metadata is not None else {}

    @property
    def name_tokens(self) -> List[str]:
        return self.normalized_name.split() if self.normalized_name else []

    @name_tokens.setter
    def name_tokens(self, val: Any) -> None:
        if val and not self.normalized_name:
            self.normalized_name = " ".join(val)

    @property
    def address_tokens(self) -> List[str]:
        return self.normalized_address.split() if self.normalized_address else []

    @address_tokens.setter
    def address_tokens(self, val: Any) -> None:
        if val and not self.normalized_address:
            self.normalized_address = " ".join(val)

    @property
    def numeric_tokens(self) -> List[str]:
        tokens = (self.normalized_name + " " + self.normalized_address).split()
        return [t for t in tokens if any(c.isdigit() for c in t)]

    @numeric_tokens.setter
    def numeric_tokens(self, val: Any) -> None:
        pass  # Derived on-the-fly

    @property
    def record_text(self) -> str:
        return f"{self.normalized_name} | {self.normalized_address} | {self.country}"

    @record_text.setter
    def record_text(self, val: Any) -> None:
        pass


@dataclass(slots=True)
class CandidatePair:
    """A candidate pair linking an S1 entity to an S2/S3 candidate."""
    s1_id: str
    candidate_id: str
    candidate_source: str = ""

    found_by_exact: bool = False
    found_by_char_tfidf: bool = False
    found_by_word_tfidf: bool = False
    found_by_address_tfidf: bool = False
    found_by_bm25: bool = False
    found_by_name_ann: bool = False
    found_by_address_ann: bool = False
    found_by_record_ann: bool = False

    name_ann_rank: int = 0
    address_ann_rank: int = 0
    record_ann_rank: int = 0
    char_tfidf_rank: int = 0
    word_tfidf_rank: int = 0
    address_tfidf_rank: int = 0
    bm25_rank: int = 0

    name_ann_score: float = 0.0
    address_ann_score: float = 0.0
    record_ann_score: float = 0.0
    char_tfidf_score: float = 0.0
    word_tfidf_score: float = 0.0
    address_tfidf_score: float = 0.0
    bm25_score: float = 0.0

    rrf_score: float = 0.0
    retrieval_votes: int = 0


@dataclass
class PairFeatures:
    """Complete feature vector for a candidate pair."""
    s1_id: str
    candidate_id: str
    features: Dict[str, float] = field(default_factory=dict)
    label: int = -1


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


# =============================================================================
# Data Loading
# =============================================================================

def load_source_file(filepath: str) -> pd.DataFrame:
    """Load a single source TSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df = df.fillna("")
    expected = {"entity_id", "business_name", "business_address", "country"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in {filepath}: {expected - set(df.columns)}")
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Load ground truth file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df = df.fillna("")
    logger.info(f"Loaded {len(df)} ground truth entries from {filepath}")
    return df


def load_all_sources(config: dict, mode: str = "train") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three source files for train or test mode."""
    base = config["paths"]["base_dir"]
    if mode == "train":
        s1 = load_source_file(os.path.join(base, config["paths"]["train_source1"]))
        s2 = load_source_file(os.path.join(base, config["paths"]["train_source2"]))
        s3 = load_source_file(os.path.join(base, config["paths"]["train_source3"]))
    elif mode == "test":
        s1 = load_source_file(os.path.join(base, config["paths"]["test_source1"]))
        s2 = load_source_file(os.path.join(base, config["paths"]["test_source2"]))
        s3 = load_source_file(os.path.join(base, config["paths"]["test_source3"]))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return s1, s2, s3


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Parse ground truth into dict mapping S1 IDs to matched IDs."""
    gt_dict = {}
    s1_ids = gt_df["source1_entity_id"].astype(str).values
    matched_col = gt_df["matched_entity_ids"].fillna("").astype(str).values
    for s1_id, matched_str in zip(s1_ids, matched_col):
        s_str = matched_str.strip()
        if not s_str:
            gt_dict[s1_id] = []
        else:
            gt_dict[s1_id] = [m.strip() for m in s_str.split(",") if m.strip()]
    return gt_dict


def build_records(df: pd.DataFrame) -> Dict[str, Record]:
    """Convert a DataFrame into a dictionary of Record objects efficiently."""
    records = {}
    eids = df["entity_id"].astype(str).values
    names = df["business_name"].fillna("").astype(str).values
    addrs = df["business_address"].fillna("").astype(str).values
    countries = df["country"].fillna("").astype(str).values

    for eid, name, addr, country in zip(eids, names, addrs, countries):
        sid = eid.split("-")[0]
        records[eid] = Record(
            record_id=eid,
            source_id=sid,
            original_name=name,
            original_address=addr,
            country=country.strip().lower(),
        )
    logger.info(f"Built {len(records)} Record objects")
    return records


def get_country_partition_records(filepath: str, country: str, chunksize: int = 250000, fallback_on_empty_unknown: bool = True, existing_dict: Optional[Dict[str, Record]] = None) -> Dict[str, Record]:
    """Stream-load TSV records for a specific country partition directly into Record objects.
    
    Unified for both training and inference:
    - If country is a known country (e.g. 'us', 'in', 'fr'), matches rows where country == c.
    - If country is 'unknown' or empty:
        First filters for rows where country is empty ('').
        If 0 records are found and fallback_on_empty_unknown is True, falls back to loading
        all records from the file to ensure candidate recall for unlabelled S1 entities.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    records = existing_dict if existing_dict is not None else {}
    country_target = country.strip().lower()
    is_unknown = (country_target == "unknown" or not country_target)

    for chunk in pd.read_csv(filepath, sep="\t", dtype=str, chunksize=chunksize):
        chunk = chunk.fillna("")
        c_series = chunk["country"].str.strip().str.lower()
        if is_unknown:
            mask = (c_series == "")
        else:
            mask = (c_series == country_target)

        sub_df = chunk[mask]
        if len(sub_df) == 0:
            continue

        eids = sub_df["entity_id"].values
        names = sub_df["business_name"].values
        addrs = sub_df["business_address"].values
        countries = sub_df["country"].values

        for eid, name, addr, c_val in zip(eids, names, addrs, countries):
            sid = eid.split("-")[0]
            records[eid] = Record(
                record_id=eid,
                source_id=sid,
                original_name=name,
                original_address=addr,
                country=c_val.strip().lower(),
            )

    if is_unknown and len(records) == 0 and fallback_on_empty_unknown:
        logger.warning(
            f"Candidate source {os.path.basename(filepath)} has 0 records with empty country; "
            f"falling back to loading all records for unknown country partition."
        )
        for chunk in pd.read_csv(filepath, sep="\t", dtype=str, chunksize=chunksize):
            chunk = chunk.fillna("")
            eids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            countries = chunk["country"].values

            for eid, name, addr, c_val in zip(eids, names, addrs, countries):
                sid = eid.split("-")[0]
                records[eid] = Record(
                    record_id=eid,
                    source_id=sid,
                    original_name=name,
                    original_address=addr,
                    country=c_val.strip().lower(),
                )

    logger.info(f"Loaded {len(records)} partition records for country='{country}' from {os.path.basename(filepath)}")
    return records


# Alias for backward compatibility
load_source_records_for_country = get_country_partition_records


def get_source_prefix(entity_id: str) -> str:
    """Extract source prefix (S1, S2, S3) from entity_id."""
    return entity_id.split("-")[0]
