"""
Business name normalization pipeline.

Handles Unicode normalization, abbreviation expansion, legal suffix
standardization, tokenization, and character n-gram generation.
Supports Hindi/Devanagari transliterations.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Tuple

from preprocessing.schema import Record

logger = logging.getLogger(__name__)

# =============================================================================
# Abbreviation Dictionaries
# =============================================================================

LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "llp": "limited liability partnership",
    "llc": "limited liability company",
    "plc": "public limited company",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "assoc": "associates",
    "bros": "brothers",
    "mfg": "manufacturing",
    "intl": "international",
    "natl": "national",
    "grp": "group",
    "svcs": "services",
    "svc": "service",
    "tech": "technology",
    "engg": "engineering",
    "engr": "engineering",
    "infra": "infrastructure",
    "ent": "enterprises",
    "soln": "solutions",
    "solns": "solutions",
    "ind": "industries",
    "inds": "industries",
    "pharma": "pharmaceuticals",
    "hosp": "hospital",
    "univ": "university",
    "inst": "institute",
    "org": "organization",
    "fdn": "foundation",
    "govt": "government",
    "dept": "department",
    "mgmt": "management",
    "mgt": "management",
    "sys": "systems",
    "dev": "development",
    "labs": "laboratories",
    "lab": "laboratory",
    "fin": "financial",
    "prop": "properties",
    "props": "properties",
    "const": "construction",
    "constr": "construction",
    "consult": "consultants",
    "distrib": "distributors",
    "mktg": "marketing",
    "advt": "advertising",
    "advtg": "advertising",
    "comm": "communications",
    "telecom": "telecommunications",
    "agri": "agriculture",
    "auto": "automobile",
    "edu": "education",
    "med": "medical",
    "elec": "electrical",
    "electr": "electronics",
    "chem": "chemicals",
    "petro": "petroleum",
    "transpt": "transport",
    "logist": "logistics",
    "sec": "securities",
    "inv": "investments",
    "cap": "capital",
    "hldg": "holdings",
    "hldgs": "holdings",
    "res": "resources",
    "ntwk": "network",
    "assn": "association",
    # French
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "sci": "sci",
    "eurl": "eurl",
    "ste": "societe",
    "sté": "societe",
    "cie": "compagnie",
    "grpe": "groupe",
    "étab": "etablissement",
    "ets": "etablissements",
}

# Suffixes to REMOVE when creating compact_name
LEGAL_SUFFIXES_TO_STRIP = {
    "private", "limited", "pvt", "ltd", "llp", "llc", "plc",
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited liability partnership", "limited liability company",
    "public limited company",
    "sarl", "sas", "sa", "sci", "eurl", "societe", "compagnie",
    "groupe", "group", "associates", "association",
}

PUNCTUATION_MAP = {
    "&": " and ",
    "@": " at ",
    "+": " plus ",
    "#": " number ",
    "–": " ",
    "—": " ",
    "'": "'",
    "'": "'",
    """: '"',
    """: '"',
    "«": '"',
    "»": '"',
}


def _unicode_normalize(text: str) -> str:
    """Apply NFKD Unicode normalization and remove combining marks."""
    # NFKD decomposes characters but keep original for Devanagari
    if any('\u0900' <= c <= '\u097F' for c in text):
        # Contains Devanagari - just normalize whitespace
        return text
    normalized = unicodedata.normalize("NFKD", text)
    # Remove combining diacritical marks (accents) for Latin text
    result = ""
    for c in normalized:
        if unicodedata.category(c) != "Mn":
            result += c
    return result


def _normalize_punctuation(text: str) -> str:
    """Replace special punctuation with standard equivalents."""
    for old, new in PUNCTUATION_MAP.items():
        text = text.replace(old, new)
    # Remove remaining punctuation except apostrophes and hyphens within words
    text = re.sub(r"[.,;:!?(){}[\]<>/\\|`~^*_=]", " ", text)
    return text


def _expand_abbreviations(tokens: List[str]) -> List[str]:
    """Expand known abbreviations in token list."""
    result = []
    for token in tokens:
        # Strip trailing period (common in abbreviations)
        clean = token.rstrip(".")
        if clean in LEGAL_SUFFIX_MAP:
            expanded = LEGAL_SUFFIX_MAP[clean]
            result.extend(expanded.split())
        else:
            result.append(token)
    return result


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _generate_char_ngrams(text: str, ngram_range: Tuple[int, int] = (3, 4)) -> List[str]:
    """Generate character n-grams from text."""
    ngrams = []
    clean = text.replace(" ", "")
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for i in range(len(clean) - n + 1):
            ngrams.append(clean[i:i + n])
    return ngrams


def _make_compact_name(normalized_name: str) -> str:
    """Create compact name by removing legal suffixes and whitespace."""
    tokens = normalized_name.split()
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES_TO_STRIP]
    return "".join(filtered)


def normalize_name(record: Record, config: dict = None) -> Record:
    """Apply full name normalization pipeline to a record.
    
    Args:
        record: Record to normalize (mutated in-place).
        config: Optional config dict with normalization settings.
    
    Returns:
        The same Record with normalized name fields populated.
    """
    name = record.original_name
    if not name or not name.strip():
        record.normalized_name = ""
        record.compact_name = ""
        record.name_tokens = []
        return record

    # Pipeline
    text = name
    text = _unicode_normalize(text)
    text = text.lower()
    text = _normalize_punctuation(text)
    text = _normalize_whitespace(text)

    # Tokenize
    tokens = text.split()

    # Expand abbreviations
    tokens = _expand_abbreviations(tokens)

    normalized = " ".join(tokens)

    record.normalized_name = normalized
    record.compact_name = _make_compact_name(normalized)
    record.name_tokens = tokens

    return record


def normalize_all_names(records: Dict[str, Record], config: dict = None) -> Dict[str, Record]:
    """Normalize names for all records."""
    for rid, record in records.items():
        normalize_name(record, config)
    logger.info(f"Normalized names for {len(records)} records")
    return records


def extract_normalized_name_tokens(name: str) -> List[str]:
    """Tokenize and normalize name string matching exact pipeline."""
    if not name or not str(name).strip():
        return []
    text = _unicode_normalize(str(name)).lower()
    text = _normalize_punctuation(text)
    text = _normalize_whitespace(text)
    tokens = text.split()
    return _expand_abbreviations(tokens)

