"""
Grounding rules for skincare recommendation.

Each rule maps skin score conditions to a named treatment profile and its
required ingredients. Multiple rules can match — their ingredients are
unioned into the final spec passed to the LLM pipeline.

Ingredient names are aligned to the `filters_json` tags in skincat.csv
where possible (noted inline), so they can be used directly for catalog
re-filtering between Pass 1 and Pass 2.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each rule has:
#   name        — treatment profile label (human-readable, passed to LLM)
#   conditions  — dict of score/type checks; all must be True for the rule
#                 to fire. Supported operators: lt, gt, lte, gte, eq
#   ingredients — list of ingredients required when this rule fires
#                 (mapped to filters_json tags where they exist)
# ---------------------------------------------------------------------------

PROFILE_RULES: list[dict] = [
    {
        # Dry skin with very low hydration AND high sebum signals a
        # compromised moisture barrier — prioritise humectants.
        "name": "Barrier Boost",
        "conditions": {
            "skin_type": {"eq": "Dry"},
            "hydration": {"lt": 30},
            "sebum":     {"gt": 70},
        },
        "ingredients": [
            "Hyaluronic Acid",   # filters_json tag: "Hyaluronic Acid"
        ],
    },
    {
        # Low lines score (high visible lines) triggers anti-ageing actives.
        # Note: "Squalane" is the correct INCI name (user spec: Squalene).
        # "Retinoid" covers retinol/retinal variants.
        "name": "Firm Restore",
        "conditions": {
            "lines": {"lt": 70},
        },
        "ingredients": [
            "Peptides",          # filters_json tag: "Peptides"
            "Squalane",          # filters_json tag: "Squalane"  (spec: Squalene)
            "Retinoid",          # no direct tag — used in LLM prompt rationale
            "Collagen",          # filters_json tag: "Collagen"
        ],
    },
    {
        # Low pores score (enlarged / congested pores) triggers exfoliants
        # and pore-minimising actives.
        "name": "Blemish Control",
        "conditions": {
            "pores": {"lt": 70},
        },
        "ingredients": [
            "BHA/Salicylic Acid",  # no direct tag — mapped to AHAs in prompt
            "Niacinamide",         # filters_json tag: "Niacinamide"
        ],
    },
    {
        # Oily skin with moderate sebum — keep blemish actives without
        # the full Barrier Boost stack.
        "name": "Blemish Control",
        "conditions": {
            "skin_type": {"eq": "Oily"},
            "sebum":     {"lt": 70},
        },
        "ingredients": [
            "BHA/Salicylic Acid",
            "Niacinamide",
        ],
    },
    {
        # Dry skin with moderate-to-low hydration and some sebum activity
        # benefits from gentle exfoliation + brightening + barrier repair.
        "name": "Radiance Reset",
        "conditions": {
            "skin_type": {"eq": "Dry"},
            "hydration": {"lt": 70},
            "sebum":     {"gt": 30},
        },
        "ingredients": [
            "AHA/Glycolic Acid",  # filters_json tag: "AHAs/Glycolic Acid"
            "Vitamin C",          # filters_json tag: "Vitamins"
            "Ceramides",          # filters_json tag: "Ceramide"
        ],
    },
]


# ---------------------------------------------------------------------------
# Rule evaluator
# ---------------------------------------------------------------------------

def _check_condition(value: float | str, operator_dict: dict) -> bool:
    """Evaluate a single condition operand against a score or type value."""
    for op, threshold in operator_dict.items():
        if op == "lt"  and not (value <  threshold): return False
        if op == "gt"  and not (value >  threshold): return False
        if op == "lte" and not (value <= threshold): return False
        if op == "gte" and not (value >= threshold): return False
        if op == "eq"  and not (str(value).lower() == str(threshold).lower()): return False
    return True


def _rule_matches(rule: dict, scores: dict) -> bool:
    """Return True if every condition in the rule is satisfied by scores."""
    for field, operator_dict in rule["conditions"].items():
        if field not in scores:
            return False
        if not _check_condition(scores[field], operator_dict):
            return False
    return True


def derive_ingredient_spec(scores: dict) -> dict:
    """
    Evaluate all PROFILE_RULES against the provided skin scores and return
    a structured ingredient specification.

    Parameters
    ----------
    scores : dict
        Must include at minimum: skin_type, hydration, sebum, pores, lines.
        Optional: age, pregnancy.

    Returns
    -------
    dict with keys:
        profiles            — list of matched rule names (e.g. ["Barrier Boost"])
        required_ingredients — deduplicated union of all matched ingredients
        skin_concerns       — human-readable concern labels for LLM context
        notes               — any safety flags (e.g. pregnancy contraindications)
    """
    matched_profiles: list[str] = []
    required: list[str] = []

    for rule in PROFILE_RULES:
        if _rule_matches(rule, scores):
            matched_profiles.append(rule["name"])
            required.extend(rule["ingredients"])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_required: list[str] = []
    for ing in required:
        if ing not in seen:
            seen.add(ing)
            unique_required.append(ing)

    # Derive concern labels from matched profile names
    concern_map = {
        "Barrier Boost":   "Dryness",
        "Firm Restore":    "Fine Lines & Wrinkles",
        "Blemish Control": "Pores",
        "Radiance Reset":  "Dullness",
    }
    skin_concerns = list(dict.fromkeys(
        concern_map[p] for p in matched_profiles if p in concern_map
    ))

    # Safety notes — not part of ingredient filtering but passed to LLM
    notes: list[str] = []
    if str(scores.get("pregnancy", "No")).lower() == "yes":
        notes.append("Avoid Retinoids and high-dose Salicylic Acid — pregnancy contraindicated.")
        if "Retinoid" in unique_required:
            unique_required.remove("Retinoid")

    return {
        "profiles":             matched_profiles,
        "required_ingredients": unique_required,
        "skin_concerns":        skin_concerns,
        "notes":                notes,
    }
