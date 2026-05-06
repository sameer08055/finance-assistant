import re
import hashlib
import logging
from datetime import datetime
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
audit_log = logging.getLogger("pii_audit")

# ── Engines ───────────────────────────────────────────────────────────────────
analyzer  = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ── Regex pre-pass (catches patterns Presidio may miss in statement text) ────
REGEX_PATTERNS = {
    "ACCOUNT_NUMBER": re.compile(r"\b\d{8,17}\b"),
    "ROUTING_NUMBER": re.compile(r"\b\d{9}\b"),
    "SSN":            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE":          re.compile(r"\b(\+1[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
    "EMAIL":          re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE),
}

# ── Token vault: real_value → token  (in-memory for the session) ─────────────
_token_vault: dict[str, str] = {}


def _tokenize(value: str, label: str) -> str:
    """Replace a real value with a stable deterministic token."""
    if value not in _token_vault:
        digest = hashlib.sha256(value.encode()).hexdigest()[:8].upper()
        _token_vault[value] = f"[{label}_{digest}]"
    return _token_vault[value]


def _regex_redact(text: str) -> tuple[str, list[dict]]:
    """First pass: regex substitution with tokenization."""
    findings = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            token = _tokenize(match.group(), label)
            findings.append({
                "type": label,
                "original_hash": hashlib.sha256(match.group().encode()).hexdigest(),
                "token": token,
                "position": match.span(),
            })
        text = pattern.sub(lambda m: _tokenize(m.group(), label), text)
    return text, findings


def _presidio_redact(text: str) -> tuple[str, list[dict]]:
    """Second pass: Presidio NER-based redaction."""
    results = analyzer.analyze(text=text, language="en")
    findings = []
    for r in results:
        original = text[r.start:r.end]
        token = _tokenize(original, r.entity_type)
        findings.append({
            "type": r.entity_type,
            "score": round(r.score, 3),
            "token": token,
        })
    # Use Presidio anonymizer to replace spans
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, findings


def redact_pii(text: str) -> dict:
    """
    Full PII redaction pipeline:
      1. Regex pass
      2. Presidio NER pass
      3. Audit log
    Returns: {redacted_text, findings, timestamp}
    """
    text, regex_findings = _regex_redact(text)
    text, presidio_findings = _presidio_redact(text)

    all_findings = regex_findings + presidio_findings
    audit_log.info(
        "PII redaction complete | items_found=%d | timestamp=%s",
        len(all_findings),
        datetime.utcnow().isoformat(),
    )

    return {
        "redacted_text": text,
        "findings": all_findings,
        "timestamp": datetime.utcnow().isoformat(),
    }