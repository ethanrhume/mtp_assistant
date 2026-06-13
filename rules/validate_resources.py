"""
validate_resources(candidates, note, chw_clarifications, rules_output) -> dict

Post-retrieval validation layer. Filters and re-ranks retrieved resource candidates
against the resolved housing/food levels and geographic constraints.
Returns validated resource list plus any unresolved warnings.
"""


def _geo_ok(resource: dict, client_zip: str | None) -> bool:
    zip_codes = resource.get("zip_codes", [])
    if not zip_codes:
        return True
    if client_zip is None:
        return True
    return client_zip in zip_codes


def validate_resources(
    candidates: list[dict],
    note: dict,
    chw_clarifications: dict,
    rules_output: dict,
) -> dict:
    """
    Filter candidates by:
      1. Geographic constraints (backstop — retrieve_resources already filters,
         but this enforces the invariant in case candidates come from other sources)
      2. Housing subdomain match (if housing level resolved)
      3. Food subdomain match + modifier subdomains (if food level resolved)

    Returns dict with:
      "resources": validated list
      "unresolved_warnings": list of warning strings
    """
    client_zip = note.get("extracted", {}).get("client_zip")
    required_housing = set(rules_output.get("required_housing_subdomains", []))
    required_food    = set(rules_output.get("required_food_subdomains", []))
    unresolved_warnings = list(rules_output.get("unresolved_warnings", []))

    validated: list[dict] = []

    for resource in candidates:
        # Backstop geographic filter
        if not _geo_ok(resource, client_zip):
            continue

        domain = resource.get("domain", "")
        subdomain = resource.get("subdomain", "")

        if domain == "housing" and required_housing:
            if subdomain not in required_housing:
                continue

        if domain == "food" and required_food:
            if subdomain not in required_food:
                continue

        validated.append(resource)

    # Sequencing checks
    housing_level = (chw_clarifications.get("housing") or {}).get("level")
    if housing_level in (1, 2):
        has_shelter = any(r.get("subdomain") == "emergency_shelter" for r in validated)
        if not has_shelter:
            unresolved_warnings.append(
                "Emergency shelter resource not found — client may need immediate housing. "
                "Review resource database."
            )

    food_level = (chw_clarifications.get("food") or {}).get("level")
    if food_level == 1:
        has_pantry = any(r.get("subdomain") == "food_pantry" for r in validated)
        if not has_pantry:
            unresolved_warnings.append(
                "Food pantry resource not found — client may need immediate food access. "
                "Review resource database."
            )

    print(f"[validate_resources] {len(candidates)} candidates → {len(validated)} validated, "
          f"{len(unresolved_warnings)} warnings")

    return {
        "resources": validated,
        "unresolved_warnings": unresolved_warnings,
    }
