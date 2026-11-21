from app.core import baseline_guide, evaluate_overlap, normalize_interests
from app.storage import load_examples

def test_dataset_is_structured_and_nonempty():
    rows = load_examples()
    assert len(rows) >= 10
    assert all({"title", "tags", "area", "recommendation", "why_it_fits", "tip"} <= row.keys() for row in rows)

def test_interests_normalize():
    assert normalize_interests("Food, Nature") == ["food", "nature"]

def test_guide_uses_requested_interest():
    guide = baseline_guide("Aurelia Bay", ["nature"], load_examples(), 2)
    assert "Aurelia Bay" in guide
    assert "nature" in guide.lower()
    assert "Tideglass" in guide or "Cedar Hill" in guide

def test_overlap_metric():
    result = evaluate_overlap("A guide for food and nature lovers", ["food", "nature"])
    assert result["coverage"] == 1.0
