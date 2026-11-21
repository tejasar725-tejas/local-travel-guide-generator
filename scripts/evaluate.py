"""Small reproducible relevance check for the baseline and optional merged model."""
from app.core import baseline_guide, evaluate_overlap
from app.storage import load_examples

CASES = [
    ("Aurelia Bay", ["food", "history"]),
    ("Aurelia Bay", ["nature", "adventure"]),
    ("Aurelia Bay", ["art", "family"]),
]
def main():
    examples = load_examples()
    print("CASE | BASELINE INTEREST COVERAGE")
    for destination, interests in CASES:
        guide = baseline_guide(destination, interests, examples)
        result = evaluate_overlap(guide, interests)
        print(f"{interests} | {result['coverage']:.0%} | {result['matched_interests']}")
        print("--- sample output ---")
        print(guide[:650], "\n")
if __name__ == "__main__": main()
