from __future__ import annotations
from collections import Counter
from typing import Iterable

INTERESTS = {
    "food": ["food", "cuisine", "street food", "coffee", "restaurant", "market", "vegan"],
    "history": ["history", "historic", "museum", "heritage", "architecture", "fort"],
    "nature": ["nature", "beach", "garden", "bird", "hike", "sunset", "park"],
    "adventure": ["adventure", "cycling", "kayak", "surf", "climb", "walk", "sailing"],
    "art": ["art", "gallery", "craft", "music", "culture", "theatre", "handmade"],
    "relaxation": ["relax", "quiet", "spa", "slow", "wellness", "peaceful"],
    "family": ["family", "children", "kid", "accessible", "easy walk"],
}

def normalize_interests(interests: Iterable[str] | str) -> list[str]:
    if isinstance(interests, str):
        interests = [x.strip() for x in interests.split(",")]
    return [x.lower().strip() for x in interests if x and x.strip()]

def score_example(example: dict, interests: list[str]) -> int:
    haystack = " ".join([example.get("tags", ""), example.get("title", ""),
                         example.get("recommendation", ""), example.get("area", "")]).lower()
    score = sum(2 for interest in interests if interest in haystack)
    for interest in interests:
        score += sum(1 for word in INTERESTS.get(interest, []) if word in haystack)
    return score

def baseline_guide(destination: str, interests: Iterable[str] | str, examples: list[dict], days: int = 2) -> str:
    wanted = normalize_interests(interests)
    ranked = sorted(examples, key=lambda x: score_example(x, wanted), reverse=True)
    selected = [x for x in ranked if score_example(x, wanted) > 0][:max(3, min(days * 2, 6))]
    if not selected:
        selected = ranked[:3]
    lines = [f"# {destination}: your {days}-day local mini-guide",
             "A practical starting point based on your interests: " + ", ".join(wanted or ["local highlights"]) + "."]
    for i, item in enumerate(selected, 1):
        lines += [f"\n## {i}. {item['title']} — {item['area']}",
                  item["recommendation"],
                  f"**Why it fits:** {item['why_it_fits']}",
                  f"**Local tip:** {item['tip']}"]
    lines += ["\n**Planning note:** This is a fictional demonstration destination. Confirm opening hours, prices, transport and accessibility before travel."]
    return "\n".join(lines)

def evaluate_overlap(output: str, interests: Iterable[str] | str) -> dict:
    wanted = normalize_interests(interests)
    text = output.lower()
    hits = [x for x in wanted if x in text]
    return {"requested_interests": wanted, "matched_interests": hits,
            "coverage": round(len(hits) / len(wanted), 2) if wanted else 1.0}
