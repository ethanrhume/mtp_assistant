"""
Trigger phrase definitions and rule mappings for the rules layer.

Detection design (two steps):
  Step A — Embedding similarity (this file + embed_triggers.py):
    Phrases in TRIGGER_PHRASES, HOUSING_LEVEL_PHRASES, FOOD_LEVEL_PHRASES, and
    FOOD_MODIFIER_PHRASES are pre-embedded and stored in data/trigger_embeddings.json.
    At runtime the clarification/rules layer embeds transcript sentences and computes
    cosine similarity. Candidates above threshold are flagged.

  Step B — LLM clarification (clarifications.py + apply_rules.py):
    Ambiguous candidates (0.50–0.80 confidence) get a clarification question surfaced
    to the CHW. Safety candidates go through an LLM disambiguation call that determines
    whether a disclosure is current/ongoing (→ alert) or historical/past (→ logged only).
    HISTORICAL_DISCLOSURE_INDICATORS are the reference text for that LLM prompt —
    they are NOT embedded.

Embedding coverage:
  Embedded:     TRIGGER_PHRASES, HOUSING_LEVEL_PHRASES, FOOD_LEVEL_PHRASES,
                FOOD_MODIFIER_PHRASES
  Not embedded: HISTORICAL_DISCLOSURE_INDICATORS, all *_DESCRIPTIONS, all *_RULES,
                CONDITION_RESOLUTION_PHRASES
"""

# ---------------------------------------------------------------------------
# Safety trigger phrases — embedded, used for candidate detection
# ---------------------------------------------------------------------------

TRIGGER_PHRASES: dict[str, list[str]] = {
    "SI": [
        "client wants to die",
        "client expresses suicidal ideation",
        "client does not want to be alive",
        "client mentions ending their life",
        "client feels hopeless and sees no way forward",
        "client has a plan to hurt themselves",
    ],
    "DV": [
        "client reports partner violence",
        "client discloses domestic abuse",
        "client is afraid of someone at home",
        "client mentions being hit or threatened",
        "client reports unsafe situation at home",
    ],
    "CHILD_ABUSE": [
        "client reports abuse of a child in the home",
        "client reports unsafe situation for a child in the home",
        "client reports violence against a child in the home",
        "client mentions a child being hit or threatened",
        "client reports a child is afraid of someone at home",
    ],
    "RED_FLAG_CLINICAL": [
        "client reports chest pain",
        "client experiencing difficulty breathing",
        "client describes sudden severe headache",
        "client shows signs of stroke",
        "client reports altered mental status",
        "client mentions sudden vision loss",
        "client reports one-sided weakness or numbness",
    ],
    "MED_ACCESS": [
        "can't afford medication",
        "insurance won't cover",
        "medication not covered",
        "can't get my prescription",
        "pharmacy denied",
        "insurance denied the procedure",
        "prior authorization denied",
        "medication too expensive",
    ],
    "HYPERTENSION": [
        "high blood pressure",
        "hypertension",
        "blood pressure is high",
        "BP is elevated",
        "taking blood pressure medication",
    ],
    "DIABETES": [
        "diabetes",
        "diabetic",
        "blood sugar",
        "A1C",
        "insulin",
        "metformin",
        "sugar is high",
    ],
    "HEART_FAILURE": [
        "heart failure",
        "congestive heart failure",
        "CHF",
        "weak heart",
        "heart is not pumping well",
    ],
    "ASTHMA_COPD": [
        "asthma",
        "COPD",
        "inhaler",
        "can't breathe chronically",
        "chronic lung disease",
        "emphysema",
    ],
    "BH_SUD": [
        "depression",
        "anxiety",
        "mental health concern",
        "substance use",
        "alcohol problem",
        "drug use",
        "opioid",
        "in recovery",
        "psychiatric medication",
        "mood disorder",
        "PTSD",
        "trauma history",
    ],
}

# ---------------------------------------------------------------------------
# Housing acuity — embedded, keyed by level int
# ---------------------------------------------------------------------------

HOUSING_LEVEL_PHRASES: dict[int, list[str]] = {
    1: ["sleeping outside", "on the street", "no shelter", "living outside",
        "homeless on the street"],
    2: ["sleeping in my car", "living in my vehicle", "car is my home"],
    3: ["couch surfing", "staying with friends", "no stable place",
        "moving around", "mixture of street and couch"],
    4: ["hotel voucher", "motel voucher", "staying in a motel on a voucher"],
    5: ["eviction notice from sheriff", "law enforcement eviction",
        "marshal came", "locked out by landlord"],
    6: ["shelter", "less than three months", "shelter stay ending soon"],
    7: ["pay or vacate", "notice from landlord", "eviction letter",
        "14 day notice", "30 day notice"],
    8: ["shelter long term", "shelter permanent", "been in shelter a long time"],
    9: ["halfway house", "group home", "transitional housing program"],
    10: ["behind on rent", "owe back rent", "no eviction notice yet"],
    11: ["hard time paying rent", "rent is a struggle", "barely making rent"],
    12: ["housing is stable", "no housing issues", "rent is fine"],
}

HOUSING_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "client is unhoused and sleeping outside",
    2: "client is unhoused and sleeping in a vehicle",
    3: "client is couch surfing or in unstable temporary housing",
    4: "client is in a hotel or motel on a voucher program",
    5: "client has received a law enforcement eviction notice",
    6: "client is in a shelter with less than three months remaining",
    7: "client has received a pay or vacate notice from their landlord",
    8: "client is in a shelter in a longer-term or semi-permanent stay",
    9: "client is in halfway or group housing",
    10: "client is housed but behind on rent with no notice yet",
    11: "client is housed but struggling to pay rent",
    12: "client is stably housed",
}

# Maps housing level tuples → required resource subdomains for validate_resources
HOUSING_RESOURCE_RULES: dict[tuple, list[str]] = {
    (1, 2): ["emergency_shelter", "coordinated_entry"],
    (3, 4, 6, 8, 9): ["subsidized_housing", "affordable_housing"],
    (5, 7): ["legal_aid", "cash_assistance", "eviction_prevention"],
    (10, 11): ["cash_assistance"],
    (12,): [],
}

# ---------------------------------------------------------------------------
# Food acuity — embedded, keyed by level int
# ---------------------------------------------------------------------------

FOOD_LEVEL_PHRASES: dict[int, list[str]] = {
    1: ["no food", "ran out of food", "nothing to eat", "food insecure daily",
        "don't know where next meal is coming from"],
    2: ["hard time affording food", "EBT runs out", "food insecure regularly",
        "struggle with food most months"],
    3: ["tight but manage", "food is okay mostly", "at risk for food insecurity"],
    6: ["food is fine", "no food issues", "eating well"],
}

FOOD_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "client is experiencing acute food insecurity with little or no food available",
    2: "client regularly struggles to afford food but is not in daily crisis",
    3: "client is at risk for food insecurity but currently managing",
    6: "client has stable food access",
}

FOOD_RESOURCE_RULES: dict[int, list[str]] = {
    1: ["food_pantry", "ebt_application", "farmers_market_ebt"],
    2: ["food_pantry", "ebt_application", "farmers_market_ebt"],
    3: ["ebt_application", "financial_assistance", "farmers_market_ebt"],
    6: [],
}

# ---------------------------------------------------------------------------
# Food modifiers — embedded, keyed by modifier name
# ---------------------------------------------------------------------------

FOOD_MODIFIER_PHRASES: dict[str, list[str]] = {
    "infants_toddlers": ["infant", "toddler", "baby", "under five",
                         "young children at home"],
    "school_age": ["school age", "kids in school", "elementary school",
                   "middle school", "high school age kids"],
}

# Maps modifier keys → additional subdomains appended to FOOD_RESOURCE_RULES output
FOOD_MODIFIERS: dict[str, list[str]] = {
    "infants_toddlers": ["tanf"],
    "school_age": ["backpack_program"],
}

# ---------------------------------------------------------------------------
# Chronic condition resolution exclusion phrases — NOT embedded
# If matched above threshold, cancels the corresponding condition rule.
# ---------------------------------------------------------------------------

CONDITION_RESOLUTION_PHRASES: dict[str, list[str]] = {
    "DIABETES": ["diabetes resolved", "no longer diabetic",
                 "gestational diabetes that resolved", "pre-diabetes reversed"],
    "ASTHMA_COPD": ["asthma resolved", "outgrew asthma", "asthma cleared"],
    "HYPERTENSION": ["blood pressure normalized", "hypertension resolved"],
    "HEART_FAILURE": ["heart failure resolved", "heart function restored"],
}

# ---------------------------------------------------------------------------
# Historical disclosure reference phrases — NOT embedded
# Used in the LLM prompt that disambiguates current vs. historical disclosures.
# ---------------------------------------------------------------------------

HISTORICAL_DISCLOSURE_INDICATORS: dict[str, list[str]] = {
    "CHILD_ABUSE": [
        "client discloses they were abused as a child",
        "client reports childhood abuse",
        "client experienced abuse growing up",
        "client shares history of childhood trauma",
        "client was abused when they were young",
        "client describes abuse they experienced as a child",
    ],
    "DV": [
        "client describes past relationship violence",
        "client discloses they were abused by a former partner",
        "client shares history of domestic violence",
        "client experienced abuse in a previous relationship",
    ],
}
