"""Deterministic lexical analysis and retrieval for materials-science BM25.

The analyzer deliberately has no optional NLP dependency.  This keeps offline
index builds reproducible while covering Chinese text, English terminology,
Greek phase names and chemical formulae.
"""

from __future__ import annotations

import re
import unicodedata


_GREEK_TRANSLATION = str.maketrans(
    {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "ε": " epsilon ",
        "ζ": " zeta ",
        "η": " eta ",
        "θ": " theta ",
        "ι": " iota ",
        "κ": " kappa ",
        "λ": " lambda ",
        "μ": " mu ",
        "ν": " nu ",
        "ξ": " xi ",
        "ο": " omicron ",
        "π": " pi ",
        "ρ": " rho ",
        "σ": " sigma ",
        "τ": " tau ",
        "υ": " upsilon ",
        "φ": " phi ",
        "χ": " chi ",
        "ψ": " psi ",
        "ω": " omega ",
    }
)
_DASH_TRANSLATION = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
_CJK_OR_MATERIAL_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+(?:[._/+:-][a-z0-9]+)*"
)
_MATERIAL_PART = re.compile(r"[a-z0-9]+")


def material_tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/materials text without external dictionaries.

    Chinese spans use character bigrams and trigrams, which allows a query to
    match a phrase embedded in a longer unsegmented sentence.  Latin terms and
    normalized chemical formulae are kept intact; separator-delimited terms
    additionally expose their components (``beta-ga2o3`` -> ``beta``,
    ``ga2o3``).
    """

    if not text:
        return []

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.translate(_DASH_TRANSLATION).translate(_GREEK_TRANSLATION)
    tokens: list[str] = []

    for match in _CJK_OR_MATERIAL_TOKEN.finditer(normalized):
        token = match.group(0)
        if "\u3400" <= token[0] <= "\u9fff":
            if len(token) == 1:
                tokens.append(token)
                continue
            for size in (2, 3):
                if len(token) >= size:
                    tokens.extend(token[index : index + size] for index in range(len(token) - size + 1))
            continue

        tokens.append(token)
        parts = _MATERIAL_PART.findall(token)
        if len(parts) > 1:
            tokens.extend(parts)

    return tokens


__all__ = ["material_tokenize"]
