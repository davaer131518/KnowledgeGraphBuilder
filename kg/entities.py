"""
Lightweight entity extraction layer.

Pure-Python rule/regex extractors + optional spaCy NER (PERSON, ORG, LOCATION)
when spaCy is installed. Designed to never block the pipeline: if spaCy isn't
available, the rule-based extractors run alone.

Public entrypoint:
    extract_entities(blocks, doc_id) -> (entities, mention_edges)

Both return values are dicts/lists ready for Neo4j writers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import config


# ── Optional spaCy import ─────────────────────────────────────────────────────

try:
    import spacy  # type: ignore
    _SPACY_IMPORT_OK = True
    _SPACY_IMPORT_ERR: str | None = None
except Exception as e:  # pragma: no cover — exercised only when spaCy missing
    spacy = None  # type: ignore
    _SPACY_IMPORT_OK = False
    _SPACY_IMPORT_ERR = str(e)


_nlp_singleton = None
_nlp_status_logged = False


def _get_spacy_nlp():
    """Lazily load the configured spaCy model. Returns None on any failure."""
    global _nlp_singleton, _nlp_status_logged
    if not config.ENTITY_USE_SPACY:
        if not _nlp_status_logged:
            print("Entity extraction: spaCy disabled (ENTITY_USE_SPACY=False) — using rule-based NER only.")
            _nlp_status_logged = True
        return None
    if not _SPACY_IMPORT_OK:
        if not _nlp_status_logged:
            print(
                "Entity extraction: spaCy not installed — using rule-based NER only. "
                f"(Install with: pip install \"spacy>=3.7,<4.0\" && python -m spacy download {config.ENTITY_SPACY_MODEL})"
            )
            _nlp_status_logged = True
        return None
    if _nlp_singleton is not None:
        return _nlp_singleton
    try:
        _nlp_singleton = spacy.load(  # type: ignore[union-attr]
            config.ENTITY_SPACY_MODEL,
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
        if not _nlp_status_logged:
            print(f"Entity extraction: spaCy active (model={config.ENTITY_SPACY_MODEL}).")
            _nlp_status_logged = True
        return _nlp_singleton
    except OSError:
        if not _nlp_status_logged:
            print(
                f"Entity extraction: spaCy installed but model '{config.ENTITY_SPACY_MODEL}' is missing — "
                f"using rule-based NER only. "
                f"Tip: python -m spacy download {config.ENTITY_SPACY_MODEL}"
            )
            _nlp_status_logged = True
        return None


def is_spacy_active() -> bool:
    """Return True iff spaCy is installed AND the model loaded successfully."""
    return _get_spacy_nlp() is not None


# ── Patterns ──────────────────────────────────────────────────────────────────

_RE_DATE_LONG = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{2,4}\b",
    re.IGNORECASE,
)
_RE_DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_RE_DATE_QUARTER = re.compile(r"\bQ[1-4]\s+\d{4}\b", re.IGNORECASE)

_RE_MONEY_SYM = re.compile(
    r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|thousand|trillion|m|bn|k)?\b",
    re.IGNORECASE,
)
_RE_MONEY_CODE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY)\b",
)

_RE_PERCENT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|(?:percent|pct)\b)",
    re.IGNORECASE,
)

_RE_NUMBER_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:kg|mg|g|m|km|cm|mm|GB|MB|KB|TB|ms|s|min|hours?|days?|weeks?|months?|years?|"
    r"miles?|feet|ft|inches|in|tons?|°[CF])\b",
    re.IGNORECASE,
)

_RE_ACRONYM_STANDALONE = re.compile(r"(?:^|[\s(\[])([A-Z]{2,6})(?=[\s.,;:!?)\]]|$)")
# Capture an N-word capitalised expansion immediately followed by "(ACR)".
_RE_ACRONYM_EXPANSION = re.compile(
    r"\b((?:[A-Z][\w-]*\s+){1,5})\(([A-Z]{2,6})\)",
)

_RE_TABLE_REF = re.compile(r"\bTable\s+(\d+(?:\.\d+)*)\b")
_RE_FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)*)\b")
_RE_SECTION_REF = re.compile(r"\bSection\s+(\d+(?:\.\d+)*)\b")
_RE_DOC_REF = re.compile(r"\b(?:Appendix|Exhibit|Schedule)\s+[A-Z0-9]+(?:\.\d+)?\b")

_RE_LAW_USC = re.compile(r"\b\d+\s+U\.S\.C\.?\s?§?\s?\d+[a-z]?\b")
_RE_LAW_SECTION = re.compile(r"\bSection\s+\d+\([a-z0-9]\)\b", re.IGNORECASE)
_RE_LAW_NAMED = re.compile(
    r"\b(?:GDPR|HIPAA|SOX|ITAR|CCPA|PCI-DSS|FERPA|FOIA|GLBA|FCPA|"
    r"EU\s+\d+/\d+)\b"
)

_RE_QUOTED_TERM = re.compile(r'["“]([^"“”\n]{3,60})["”]')

# ORG suffix terms — used both for rule-based ORG detection and for canonical
# name preservation when spaCy reports an ORG.
_ORG_SUFFIXES = (
    "Inc.", "Inc", "Ltd", "LLC", "Corp.", "Corp", "Corporation", "Company", "Co.",
    "Group", "Holdings", "GmbH", "AG", "S.A.", "SA", "PLC", "PLLC", "LLP",
    "L.P.", "LP", "BV", "NV", "Pty",
)
_ORG_SUFFIX_RE = "|".join(
    re.escape(s.rstrip(".")) + r"\.?"  # accept either "Inc" or "Inc."
    for s in dict.fromkeys(s.rstrip(".") for s in _ORG_SUFFIXES)
)
_RE_ORG = re.compile(
    rf"\b((?:[A-Z][\w&'-]*\s+){{0,5}}[A-Z][\w&'-]*)\s+({_ORG_SUFFIX_RE})\b"
)

_RE_PERSON_INITIAL = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b")
_RE_PERSON_TITLE = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"
)

_RE_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b"
)

# Very small built-in location lookup — kept tiny on purpose. Anything beyond
# this needs spaCy.
_LOCATION_LOOKUP: frozenset[str] = frozenset({
    # Selected countries
    "United States", "United Kingdom", "Canada", "Mexico", "Brazil", "Argentina",
    "Germany", "France", "Italy", "Spain", "Portugal", "Netherlands", "Belgium",
    "Switzerland", "Austria", "Sweden", "Norway", "Finland", "Denmark", "Iceland",
    "Poland", "Russia", "Ukraine", "Turkey", "Greece", "Ireland", "Czechia",
    "China", "Japan", "South Korea", "India", "Pakistan", "Indonesia", "Vietnam",
    "Thailand", "Philippines", "Singapore", "Malaysia", "Australia", "New Zealand",
    "Egypt", "Nigeria", "South Africa", "Kenya", "Morocco",
    # US states (a representative subset)
    "California", "New York", "Texas", "Florida", "Illinois", "Pennsylvania",
    "Ohio", "Georgia", "Michigan", "North Carolina", "Virginia", "Washington",
    "Arizona", "Massachusetts", "Tennessee", "Colorado", "Oregon", "Nevada",
    "Utah", "Connecticut", "Maryland", "New Jersey", "Minnesota", "Wisconsin",
})


# ── Normalisation helpers ────────────────────────────────────────────────────

_PUNCT_STRIP_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    s = text.strip().lower()
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _strip_org_suffix(name: str) -> str:
    """Strip a trailing legal suffix (Inc., LLC, etc.) for canonical comparison."""
    parts = name.strip().split()
    if not parts:
        return name
    last = parts[-1].rstrip(".")
    suffixes_norm = {s.rstrip(".") for s in _ORG_SUFFIXES}
    if last in suffixes_norm:
        return " ".join(parts[:-1]).strip()
    return name.strip()


def _slug(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] if s else "x"


# ── Per-block extraction ─────────────────────────────────────────────────────

def _add_candidate(
    out: dict[tuple[str, str], dict],
    type_: str,
    canonical_name: str,
    method: str,
    confidence: float,
    span: tuple[int, int],
    normalized_override: str | None = None,
) -> None:
    """Accumulate one candidate keyed by (type, normalized_name)."""
    if len(canonical_name.strip()) < config.ENTITY_MIN_TERM_LEN:
        return
    norm = normalized_override if normalized_override is not None else _normalize(canonical_name)
    if not norm or norm in config.ENTITY_STOPWORD_BLOCKLIST:
        return
    if type_ == "TERM" and norm in config.ENTITY_GENERIC_TERMS:
        return
    key = (type_, norm)
    rec = out.get(key)
    if rec is None:
        rec = {
            "type":           type_,
            "canonical_name": canonical_name.strip(),
            "normalized_name": norm,
            "methods":        [method],
            "confidence":     confidence,
            "spans":          [span],
        }
        out[key] = rec
    else:
        if method not in rec["methods"]:
            rec["methods"].append(method)
        rec["confidence"] = max(rec["confidence"], confidence)
        if len(rec["spans"]) < config.ENTITY_MAX_SPANS_PER_MENTION:
            rec["spans"].append(span)


def _extract_regex_entities(text: str, out: dict[tuple[str, str], dict]) -> None:
    """Run all regex/rule extractors on a text and accumulate candidates."""
    for m in _RE_DATE_LONG.finditer(text):
        _add_candidate(out, "DATE", m.group(0), "regex:date", 0.95, m.span())
    for m in _RE_DATE_ISO.finditer(text):
        _add_candidate(out, "DATE", m.group(0), "regex:date", 0.95, m.span())
    for m in _RE_DATE_QUARTER.finditer(text):
        _add_candidate(out, "DATE", m.group(0), "regex:date", 0.9, m.span())

    for m in _RE_MONEY_SYM.finditer(text):
        _add_candidate(out, "MONEY", m.group(0), "regex:money", 0.95, m.span())
    for m in _RE_MONEY_CODE.finditer(text):
        _add_candidate(out, "MONEY", m.group(0), "regex:money", 0.95, m.span())

    for m in _RE_PERCENT.finditer(text):
        _add_candidate(out, "PERCENT", m.group(0), "regex:percent", 0.95, m.span())

    for m in _RE_NUMBER_UNIT.finditer(text):
        _add_candidate(out, "NUMBER", m.group(0), "regex:number_unit", 0.9, m.span())

    for m in _RE_TABLE_REF.finditer(text):
        _add_candidate(out, "TABLE_REF", f"Table {m.group(1)}", "regex:ref", 0.85, m.span())
    for m in _RE_FIGURE_REF.finditer(text):
        _add_candidate(out, "FIGURE_REF", f"Figure {m.group(1)}", "regex:ref", 0.85, m.span())
    for m in _RE_SECTION_REF.finditer(text):
        _add_candidate(out, "SECTION_REF", f"Section {m.group(1)}", "regex:ref", 0.85, m.span())
    for m in _RE_DOC_REF.finditer(text):
        _add_candidate(out, "DOCUMENT_REF", m.group(0), "regex:ref", 0.85, m.span())

    for m in _RE_LAW_USC.finditer(text):
        _add_candidate(out, "LAW_OR_REGULATION", m.group(0), "regex:law", 0.85, m.span())
    for m in _RE_LAW_SECTION.finditer(text):
        _add_candidate(out, "LAW_OR_REGULATION", m.group(0), "regex:law", 0.8, m.span())
    for m in _RE_LAW_NAMED.finditer(text):
        _add_candidate(out, "LAW_OR_REGULATION", m.group(0), "regex:law", 0.85, m.span())

    for m in _RE_QUOTED_TERM.finditer(text):
        _add_candidate(out, "TERM", m.group(1).strip(), "rule:quoted", 0.6, m.span(1))

    for m in _RE_ORG.finditer(text):
        full = m.group(0).strip()
        suffix = m.group(2).strip()
        if not suffix.endswith("."):
            suffix_canon = next((s for s in _ORG_SUFFIXES if s.rstrip(".") == suffix.rstrip(".")), suffix)
        else:
            suffix_canon = suffix
        canonical = f"{m.group(1).strip()} {suffix_canon}"
        norm = _normalize(_strip_org_suffix(canonical))
        if norm:
            _add_candidate(out, "ORG", canonical, "rule:org", 0.85, m.span(), normalized_override=norm)

    for m in _RE_PERSON_TITLE.finditer(text):
        _add_candidate(out, "PERSON", m.group(0), "rule:person_title", 0.7, m.span())
    for m in _RE_PERSON_INITIAL.finditer(text):
        _add_candidate(out, "PERSON", m.group(0), "rule:person_initial", 0.7, m.span())

    for m in _RE_CAPITALIZED_PHRASE.finditer(text):
        phrase = m.group(1).strip()
        # Skip if any token is a stopword or all tokens are very short.
        tokens = [t for t in phrase.split() if t]
        if all(t.lower() in config.ENTITY_STOPWORD_BLOCKLIST for t in tokens):
            continue
        _add_candidate(out, "TERM", phrase, "rule:capitalized", 0.55, m.span(1))

    for loc in _LOCATION_LOOKUP:
        # Only match whole-word occurrences.
        idx = text.find(loc)
        while idx >= 0:
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            end = idx + len(loc)
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok:
                _add_candidate(out, "LOCATION", loc, "rule:location_lookup", 0.7, (idx, end))
            idx = text.find(loc, idx + 1)


def _extract_acronym_pairs(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Return [(expansion, acronym, expansion_span)] for in-block 'Expansion (ACR)' hits."""
    results: list[tuple[str, str, tuple[int, int]]] = []
    for m in _RE_ACRONYM_EXPANSION.finditer(text):
        expansion = m.group(1).strip()
        acronym = m.group(2).strip()
        # Acronym must be plausibly initials of the expansion.
        exp_tokens = [t for t in expansion.split() if t and t[0].isupper()]
        if not exp_tokens:
            continue
        if len(acronym) > len(exp_tokens) + 1 or len(acronym) < 2:
            continue
        # Basic initials check: acronym letters should be a subsequence of expansion-word initials.
        initials = "".join(t[0] for t in exp_tokens)
        if not _is_subsequence(list(acronym.upper()), list(initials.upper())):
            continue
        results.append((expansion, acronym, m.span(1)))
    return results


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(c == h for h in it) for c in needle)


def _extract_spacy_entities(text: str, out: dict[tuple[str, str], dict]) -> None:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return
    truncated = text[: config.ENTITY_SPACY_TEXT_MAX_CHARS]
    try:
        doc = nlp(truncated)
    except Exception as e:  # pragma: no cover — defensive
        print(f"  [warn] spaCy NER failed on a block: {e}")
        return
    for ent in doc.ents:
        label = ent.label_
        canonical = ent.text.strip()
        span = (ent.start_char, ent.end_char)
        if label == "PERSON":
            _add_candidate(out, "PERSON", canonical, "spacy:PERSON", 0.75, span)
        elif label == "ORG":
            norm = _normalize(_strip_org_suffix(canonical))
            if norm:
                _add_candidate(out, "ORG", canonical, "spacy:ORG", 0.8, span, normalized_override=norm)
        elif label in ("GPE", "LOC"):
            _add_candidate(out, "LOCATION", canonical, f"spacy:{label}", 0.75, span)
        elif label == "DATE":
            _add_candidate(out, "DATE", canonical, "spacy:DATE", 0.8, span)
        elif label == "MONEY":
            _add_candidate(out, "MONEY", canonical, "spacy:MONEY", 0.85, span)
        elif label == "PERCENT":
            _add_candidate(out, "PERCENT", canonical, "spacy:PERCENT", 0.85, span)
        elif label == "LAW":
            _add_candidate(out, "LAW_OR_REGULATION", canonical, "spacy:LAW", 0.8, span)


# ── Document-level entity assembly ───────────────────────────────────────────

def extract_entities(
    blocks: list[dict[str, Any]],
    doc_id: str,
) -> tuple[list[dict], list[dict]]:
    """
    Build document-scoped Entity records and Block->Entity MENTIONS edges.

    Returns:
        entities      — list of Entity dicts (one per (type, normalized_name))
        mention_edges — list of MentionEdge dicts ready for the Neo4j writer
    """
    if not blocks:
        return [], []

    # Trigger spaCy load (or fallback) once, for the startup log message.
    _ = _get_spacy_nlp()

    # Per-block candidate dicts: list of {key: (type, norm) -> rec}.
    per_block: list[dict[tuple[str, str], dict]] = []
    # Document-level acronym registry: acronym -> set of expansion normalized names.
    acronym_to_expansions: dict[str, set[str]] = defaultdict(set)
    # expansion-norm -> (type, canonical_name, expansion_norm)
    expansion_records: dict[str, dict] = {}

    # Priority types that "claim" their text spans — TERM candidates overlapping
    # one of these are dropped to avoid duplicate entities like "Apple Inc" (TERM)
    # alongside "Apple Inc." (ORG).
    priority_types = frozenset({
        "ORG", "PERSON", "LOCATION", "DATE", "MONEY", "PERCENT", "NUMBER",
        "LAW_OR_REGULATION", "TABLE_REF", "FIGURE_REF", "SECTION_REF", "DOCUMENT_REF",
    })

    for blk in blocks:
        text = blk["text"] or ""
        block_out: dict[tuple[str, str], dict] = {}
        _extract_regex_entities(text, block_out)
        _extract_spacy_entities(text, block_out)

        # Drop TERMs whose first span lies inside a priority entity's span.
        claimed_spans = [
            span
            for (t, _), rec in block_out.items()
            if t in priority_types
            for span in rec["spans"]
        ]
        if claimed_spans:
            to_drop = []
            for key, rec in block_out.items():
                if key[0] != "TERM":
                    continue
                # Any TERM span fully inside a claimed span → drop the TERM entirely.
                if any(
                    cs <= ts[0] and ts[1] <= ce
                    for ts in rec["spans"]
                    for (cs, ce) in claimed_spans
                ):
                    to_drop.append(key)
            for k in to_drop:
                del block_out[k]

        # Acronym + expansion pairs (in-block co-occurrence)
        for expansion, acronym, exp_span in _extract_acronym_pairs(text):
            exp_canon = expansion
            exp_norm = _normalize(_strip_org_suffix(exp_canon)) or _normalize(exp_canon)
            # Decide entity type for the expansion: ORG if expansion has an ORG suffix nearby,
            # otherwise TERM (the safest general bucket).
            is_org = any(
                tok.rstrip(".") in {s.rstrip(".") for s in _ORG_SUFFIXES}
                for tok in exp_canon.split()
            )
            etype = "ORG" if is_org else "TERM"
            rec_key = (etype, exp_norm)
            existing = block_out.get(rec_key)
            if existing:
                if "rule:acronym" not in existing["methods"]:
                    existing["methods"].append("rule:acronym")
                if "aliases" not in existing:
                    existing["aliases"] = []
                if acronym not in existing["aliases"]:
                    existing["aliases"].append(acronym)
            else:
                _add_candidate(
                    block_out, etype, exp_canon, "rule:acronym", 0.85, exp_span,
                    normalized_override=exp_norm,
                )
                rec = block_out[rec_key]
                rec["aliases"] = [acronym]
            acronym_to_expansions[acronym].add(exp_norm)
            expansion_records[exp_norm] = {"type": etype, "canonical": exp_canon}

        # Cap entities per block (highest-confidence first)
        if len(block_out) > config.ENTITY_MAX_ENTITIES_PER_BLOCK:
            kept = sorted(block_out.items(), key=lambda kv: kv[1]["confidence"], reverse=True)
            block_out = dict(kept[: config.ENTITY_MAX_ENTITIES_PER_BLOCK])

        per_block.append(block_out)

    # Standalone-acronym resolution (second pass).
    #
    # When a bare acronym was extracted via _RE_ACRONYM_STANDALONE, we don't add it
    # eagerly above (we only run _RE_ACRONYM_STANDALONE for confidence-0.5 ACRONYM
    # records). Do that here, with disambiguation against acronym_to_expansions.
    for i, blk in enumerate(blocks):
        text = blk["text"] or ""
        for m in _RE_ACRONYM_STANDALONE.finditer(text):
            acr = m.group(1)
            if len(acr) < 2:
                continue
            expansions = acronym_to_expansions.get(acr, set())
            if len(expansions) == 1:
                # Resolve to the single expansion — bump that entity's mention.
                exp_norm = next(iter(expansions))
                rec_info = expansion_records[exp_norm]
                key = (rec_info["type"], exp_norm)
                rec = per_block[i].get(key)
                if rec is None:
                    _add_candidate(
                        per_block[i], rec_info["type"], rec_info["canonical"],
                        "rule:acronym_resolved", 0.8, m.span(1),
                        normalized_override=exp_norm,
                    )
                    rec = per_block[i][key]
                    rec.setdefault("aliases", []).append(acr)
                else:
                    if "rule:acronym_resolved" not in rec["methods"]:
                        rec["methods"].append("rule:acronym_resolved")
                    if len(rec["spans"]) < config.ENTITY_MAX_SPANS_PER_MENTION:
                        rec["spans"].append(m.span(1))
            else:
                # 0 expansions known, or > 1 (ambiguous) — emit ambiguous ACRONYM entity.
                key = ("ACRONYM", _normalize(acr))
                rec = per_block[i].get(key)
                if rec is None:
                    _add_candidate(
                        per_block[i], "ACRONYM", acr, "rule:acronym_standalone",
                        0.5, m.span(1),
                    )
                    rec = per_block[i].get(key)
                    if rec is not None and len(expansions) > 1:
                        rec["ambiguous"] = True

    # ── Aggregate to document-level Entity records ───────────────────────────
    total_blocks = len(blocks)
    aggregate: dict[tuple[str, str], dict] = {}
    block_count_per_entity: dict[tuple[str, str], int] = defaultdict(int)
    mention_edges: list[dict] = []

    for blk, block_recs in zip(blocks, per_block):
        for key, rec in block_recs.items():
            if rec["confidence"] < config.ENTITY_MIN_CONFIDENCE and key[0] != "ACRONYM":
                # Allow ambiguous ACRONYMs through at lower confidence.
                continue
            agg = aggregate.get(key)
            if agg is None:
                agg = {
                    "type":            rec["type"],
                    "canonical_name":  rec["canonical_name"],
                    "normalized_name": rec["normalized_name"],
                    "methods":         list(rec["methods"]),
                    "confidence":      rec["confidence"],
                    "aliases":         list(rec.get("aliases", [])),
                    "ambiguous":       bool(rec.get("ambiguous", False)),
                }
                aggregate[key] = agg
            else:
                for m in rec["methods"]:
                    if m not in agg["methods"]:
                        agg["methods"].append(m)
                if rec["confidence"] > agg["confidence"]:
                    agg["confidence"] = rec["confidence"]
                for alias in rec.get("aliases", []):
                    if alias not in agg["aliases"]:
                        agg["aliases"].append(alias)
                if rec.get("ambiguous"):
                    agg["ambiguous"] = True

            block_count_per_entity[key] += 1

            # Build the MENTIONS edge for this (block, entity) pair.
            spans = rec["spans"][: config.ENTITY_MAX_SPANS_PER_MENTION]
            spans_flat: list[int] = []
            for s in spans:
                spans_flat.append(int(s[0]))
                spans_flat.append(int(s[1]))
            ent_id_placeholder = key  # filled below
            first_start, first_end = spans[0]
            evidence_src = blk["text"] or ""
            ev_start = max(0, first_start - 20)
            ev_end = min(len(evidence_src), first_end + 20)
            evidence = evidence_src[ev_start:ev_end].replace("\n", " ").strip()
            if len(evidence) > config.ENTITY_EVIDENCE_MAX_CHARS:
                evidence = evidence[: config.ENTITY_EVIDENCE_MAX_CHARS - 1] + "…"
            mention_edges.append({
                "src":         blk["block_id"],
                "_ent_key":    ent_id_placeholder,   # resolved to ent_id below
                "count":       len(rec["spans"]),
                "spans_flat":  spans_flat,
                "evidence":    evidence,
                "methods":     list(rec["methods"]),
                "confidence":  rec["confidence"],
            })

    # Build final Entity list with IDs + document-frequency ratio + noisy-term demotion.
    entities: list[dict] = []
    ent_id_by_key: dict[tuple[str, str], str] = {}
    high_freq_filtered = 0
    for key, agg in aggregate.items():
        type_, norm = key
        df_count = block_count_per_entity[key]
        df_ratio = df_count / total_blocks if total_blocks else 0.0
        # Noisy-term demotion (only TERMs)
        if type_ == "TERM" and df_ratio > config.ENTITY_MAX_DOCUMENT_FREQUENCY_RATIO:
            agg["confidence"] = 0.2
            if "filtered:high_doc_freq" not in agg["methods"]:
                agg["methods"].append("filtered:high_doc_freq")
            high_freq_filtered += 1
        ent_id = f"ent_{doc_id[:12]}_{type_.lower()}_{_slug(norm)}"
        ent_id_by_key[key] = ent_id
        entities.append({
            "id":                  ent_id,
            "doc_id":              doc_id,
            "type":                type_,
            "canonical_name":      agg["canonical_name"],
            "normalized_name":     norm,
            "methods":             agg["methods"],
            "confidence":          agg["confidence"],
            "aliases":             agg["aliases"],
            "ambiguous":           agg["ambiguous"],
            "doc_frequency_ratio": round(df_ratio, 4),
        })

    # Resolve mention edge entity IDs.
    resolved: list[dict] = []
    for m in mention_edges:
        key = m.pop("_ent_key")
        if key not in ent_id_by_key:
            continue  # entity was dropped (e.g. low-confidence ACRONYM that didn't qualify)
        m["ent_id"] = ent_id_by_key[key]
        resolved.append(m)

    return entities, resolved


def compute_shares_entity_with(
    entities: list[dict],
    mention_edges: list[dict],
) -> list[dict]:
    """
    Build optional capped (:Block)-[:SHARES_ENTITY_WITH]->(:Block) edges, one
    per (block_a, block_b, entity) triple, gated by CREATE_SHARES_ENTITY_WITH.

    Returns a list of pair dicts; empty when the feature is disabled.
    """
    if not config.CREATE_SHARES_ENTITY_WITH:
        return []
    if not entities or not mention_edges:
        return []

    ent_by_id = {e["id"]: e for e in entities}
    # Group blocks per entity
    blocks_per_entity: dict[str, list[str]] = defaultdict(list)
    for m in mention_edges:
        blocks_per_entity[m["ent_id"]].append(m["src"])

    pairs: list[dict] = []
    allowed_types = set(config.SHARED_ENTITY_ALLOWED_TYPES)
    for ent_id, block_ids in blocks_per_entity.items():
        ent = ent_by_id.get(ent_id)
        if ent is None:
            continue
        if ent["type"] not in allowed_types:
            continue
        if ent["confidence"] < config.SHARED_ENTITY_MIN_ENTITY_CONFIDENCE:
            continue
        if "filtered:high_doc_freq" in ent["methods"]:
            continue
        unique_blocks = list(dict.fromkeys(block_ids))[: config.SHARED_ENTITY_MAX_BLOCKS_PER_ENTITY]
        if len(unique_blocks) < 2:
            continue
        for i in range(len(unique_blocks)):
            for j in range(i + 1, len(unique_blocks)):
                a, b = unique_blocks[i], unique_blocks[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pairs.append({
                    "src":          a,
                    "tgt":          b,
                    "entity_id":    ent_id,
                    "entity_name":  ent["canonical_name"],
                    "confidence":   ent["confidence"],
                })
                if len(pairs) >= config.SHARED_ENTITY_MAX_PAIRS:
                    return pairs
    return pairs
