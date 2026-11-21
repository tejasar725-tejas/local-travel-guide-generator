from app.core import baseline_guide
from app.storage import load_examples

def test_end_to_end_guide_has_recommendations():
    output = baseline_guide("Aurelia Bay", ["food", "art"], load_examples(), days=2)
    assert output.count("## ") >= 3
    assert "Local tip" in output
