"""
Business address normalization and component extraction.

Handles address standardization, abbreviation expansion, and extraction
of structured components (postal code, city, house number, etc.).
Supports US, India, and France address patterns with open-set fallback.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional

from preprocessing.schema import Record

logger = logging.getLogger(__name__)

# =============================================================================
# Address Abbreviation Dictionary
# =============================================================================

ADDRESS_ABBREV_MAP = {
    # Road types
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "fwy": "freeway", "pkwy": "parkway",
    "cir": "circle", "trl": "trail", "way": "way", "ter": "terrace",
    "expy": "expressway", "ext": "extension",
    # Building types
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "rm": "room", "dept": "department",
    # Directions
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # India-specific
    "opp": "opposite", "nr": "near", "dist": "district",
    "tal": "taluka", "taluk": "taluka", "tq": "taluka",
    "marg": "road", "nagar": "nagar", "vihar": "vihar",
    "gali": "lane", "mohalla": "locality", "chowk": "square",
    "wadi": "wadi", "peth": "peth", "ganj": "ganj",
    "kh": "khasra", "moh": "mohalla",
    # French
    "r": "rue", "bd": "boulevard", "av": "avenue", "imp": "impasse",
    "chem": "chemin", "rte": "route", "pl": "place", "all": "allee",
    "fbg": "faubourg", "qt": "quartier", "ht": "haut", "bt": "batiment",
    # General
    "po": "post office", "gpd": "general post delivery",
    "no": "number", "ph": "phase",
}

# =============================================================================
# Postal Code Patterns
# =============================================================================

# US: 5 digits or 5+4
US_POSTAL_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
# India: 6 digits
INDIA_POSTAL_RE = re.compile(r'\b(\d{6})\b')
# France: 5 digits (starts with 0-9, typically 01-98)
FRANCE_POSTAL_RE = re.compile(r'\b(\d{5})\b')
# Generic fallback
GENERIC_POSTAL_RE = re.compile(r'\b(\d{5,6})\b')

# House number patterns
HOUSE_NUMBER_RE = re.compile(
    r'(?:^|\b)(?:no\.?\s*|plot\s*(?:no\.?\s*)?|kh\.?\s*(?:no\.?\s*)?|'
    r'#\s*|flat\s*(?:no\.?\s*)?|door\s*(?:no\.?\s*)?)?'
    r'(\d+(?:[/-]\d+)*(?:\s*[a-zA-Z])?)\b',
    re.IGNORECASE
)

# Landmark patterns
LANDMARK_RE = re.compile(
    r'(?:near|opp(?:osite)?|behind|beside|adjacent\s+to|next\s+to|'
    r'in\s+front\s+of|close\s+to)\s+(.+?)(?:,|$)',
    re.IGNORECASE
)

# Numeric tokens
NUMERIC_RE = re.compile(r'\b(\d+(?:[/-]\d+)*)\b')


def _unicode_normalize_address(text: str) -> str:
    """Normalize Unicode for addresses, preserving Devanagari."""
    if any('\u0900' <= c <= '\u097F' for c in text):
        return text
    normalized = unicodedata.normalize("NFKD", text)
    result = ""
    for c in normalized:
        if unicodedata.category(c) != "Mn":
            result += c
    return result


def _normalize_punctuation_address(text: str) -> str:
    """Normalize address punctuation."""
    text = text.replace("&", " and ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"[;:!?(){}[\]<>\\|`~^*_=]", " ", text)
    # Keep commas, periods, hyphens, slashes (meaningful in addresses)
    text = re.sub(r"\.(?!\d)", " ", text)  # Remove periods not before digits
    return text


def _expand_address_abbreviations(tokens: List[str]) -> List[str]:
    """Expand known address abbreviations."""
    result = []
    for token in tokens:
        clean = token.rstrip(".,")
        if clean in ADDRESS_ABBREV_MAP:
            result.append(ADDRESS_ABBREV_MAP[clean])
        else:
            result.append(token)
    return result


def _extract_postal_code(text: str, country: str) -> str:
    """Extract postal code based on country patterns."""
    country_lower = country.lower().strip() if country else ""

    if country_lower == "india":
        m = INDIA_POSTAL_RE.search(text)
        if m:
            return m.group(1)
    elif country_lower == "us":
        m = US_POSTAL_RE.search(text)
        if m:
            return m.group(1)
    elif country_lower == "france":
        m = FRANCE_POSTAL_RE.search(text)
        if m:
            return m.group(1)

    # Generic fallback
    m = GENERIC_POSTAL_RE.search(text)
    if m:
        return m.group(1)
    return ""


def _extract_house_number(text: str) -> str:
    """Extract house/building number from address."""
    m = HOUSE_NUMBER_RE.search(text)
    if m:
        num = m.group(1).strip()
        # Filter out likely postal codes (5-6 digits alone)
        if num.isdigit() and len(num) >= 5:
            return ""
        return num
    return ""


def _extract_landmark(text: str) -> str:
    """Extract landmark references from address."""
    m = LANDMARK_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_numeric_tokens(text: str) -> List[str]:
    """Extract all numeric sequences from address."""
    return NUMERIC_RE.findall(text)


def _extract_city_state(text: str, country: str) -> Dict[str, str]:
    """Extract city and state from address text.
    
    Uses comma-separated component heuristics since addresses
    vary widely across countries.
    """
    result = {"city": "", "state": ""}

    # Split by commas and work backwards (city/state often at end)
    parts = [p.strip() for p in text.split(",") if p.strip()]

    if len(parts) >= 2:
        # Last part is often state/region, second-to-last is often city
        last = parts[-1].strip()
        second_last = parts[-2].strip()

        # Remove postal code from components
        last_clean = re.sub(r'\b\d{5,6}\b', '', last).strip()
        second_last_clean = re.sub(r'\b\d{5,6}\b', '', second_last).strip()

        # Remove state abbreviations (2-letter codes)
        state_match = re.search(r'\b([A-Z]{2})\b', last)

        if state_match:
            result["state"] = state_match.group(1).lower()
            # City might be in the same part or previous
            city_text = re.sub(r'\b[A-Z]{2}\b', '', last).strip()
            if city_text:
                result["city"] = city_text.lower()
            elif second_last_clean:
                result["city"] = second_last_clean.lower()
        elif last_clean:
            # Try to determine if last is state or city
            if len(last_clean.split()) <= 2:
                result["state"] = last_clean.lower()
                if second_last_clean:
                    result["city"] = second_last_clean.lower()
            else:
                result["city"] = last_clean.lower()

    elif len(parts) == 1:
        # Single part — try to extract city
        clean = re.sub(r'\b\d{5,6}\b', '', parts[0]).strip()
        if clean:
            result["city"] = clean.lower()

    return result


def normalize_address(record: Record, config: dict = None) -> Record:
    """Apply full address normalization pipeline to a record.
    
    Args:
        record: Record to normalize (mutated in-place).
        config: Optional config dict with normalization settings.
    
    Returns:
        The same Record with normalized address fields populated.
    """
    address = record.original_address
    if not address or not str(address).strip() or address == "nan":
        record.normalized_address = ""
        record.address_tokens = []
        record.postal_code = ""
        record.house_number = ""
        record.city = ""
        record.state = ""
        record.locality = ""
        record.landmark = ""
        record.numeric_tokens = []
        return record

    address = str(address)
    country = record.country

    # Extract components BEFORE normalization (using original casing)
    record.postal_code = _extract_postal_code(address, country)
    record.house_number = _extract_house_number(address)
    record.landmark = _extract_landmark(address)
    record.numeric_tokens = _extract_numeric_tokens(address)

    # Normalize
    text = _unicode_normalize_address(address)
    text = text.lower()
    text = _normalize_punctuation_address(text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and expand abbreviations
    tokens = text.split()
    tokens = _expand_address_abbreviations(tokens)
    normalized = " ".join(tokens)

    record.normalized_address = normalized
    record.address_tokens = tokens

    # Extract city/state from normalized text
    city_state = _extract_city_state(address, country)
    record.city = city_state["city"]
    record.state = city_state["state"]
    # Free original address string to save memory across millions of records
    record.original_address = ""

    return record


def normalize_all_addresses(records: Dict[str, Record], config: dict = None) -> Dict[str, Record]:
    """Normalize addresses for all records."""
    for rid, record in records.items():
        normalize_address(record, config)
    logger.info(f"Normalized addresses for {len(records)} records")
    return records


def extract_normalized_address_tokens(address: str) -> List[str]:
    """Tokenize and normalize address string matching exact pipeline."""
    if not address or not str(address).strip():
        return []
    text = _unicode_normalize_address(str(address)).lower()
    text = _normalize_punctuation_address(text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return _expand_address_abbreviations(tokens)

