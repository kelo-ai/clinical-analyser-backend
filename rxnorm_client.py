"""
RxNorm integration for polypharmacy detection. RxNorm's REST API is free,
requires no API key, and is maintained by the US National Library of
Medicine (NLM). It's used here as the factual source of truth for drug
interactions - the LLM is only used to extract medication names from text,
never to invent interaction facts itself.

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""

import requests

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


def get_rxcui(drug_name: str) -> str | None:
    """
    Looks up the RxCUI (RxNorm Concept Unique Identifier) for a drug name.
    Returns None if the drug name isn't recognized - callers should treat
    this as "could not verify this medication," not a hard failure.
    """
    response = requests.get(
        f"{RXNORM_BASE}/rxcui.json",
        params={"name": drug_name, "search": 1},  # search=1 allows approximate matching
        timeout=10,
    )

    if response.status_code != 200:
        return None

    data = response.json()
    ids = data.get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


def check_interactions(medication_names: list[str]) -> dict:
    """
    Given a list of medication names, normalizes each to an RxCUI, then
    checks for known drug-drug interactions between them.

    Returns:
    {
        "resolved_medications": [{"name": ..., "rxcui": ...}, ...],
        "unresolved_medications": ["name that couldn't be matched", ...],
        "interactions": [
            {"drug1": ..., "drug2": ..., "description": ..., "severity": ...},
            ...
        ]
    }
    """
    resolved = []
    unresolved = []

    for name in medication_names:
        rxcui = get_rxcui(name)
        if rxcui:
            resolved.append({"name": name, "rxcui": rxcui})
        else:
            unresolved.append(name)

    interactions = []

    if len(resolved) >= 2:
        rxcui_list = "+".join(m["rxcui"] for m in resolved)

        response = requests.get(
            f"{RXNORM_BASE}/interaction/list.json",
            params={"rxcuis": rxcui_list},
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()
            interaction_groups = data.get("fullInteractionTypeGroup", [])

            for group in interaction_groups:
                for interaction_type in group.get("fullInteractionType", []):
                    pair = interaction_type.get("minConcept", [])
                    pair_names = [p.get("name", "unknown") for p in pair]

                    for pair_interaction in interaction_type.get("interactionPair", []):
                        interactions.append({
                            "drug1": pair_names[0] if len(pair_names) > 0 else "unknown",
                            "drug2": pair_names[1] if len(pair_names) > 1 else "unknown",
                            "description": pair_interaction.get("description", ""),
                            "severity": pair_interaction.get("severity", "not specified"),
                        })

    return {
        "resolved_medications": resolved,
        "unresolved_medications": unresolved,
        "interactions": interactions,
    }
