import re

# Test the hallucination risk patterns
text1 = "i'm not sure about this."
text2 = "i'm not certain about this."
text3 = "i think this is correct"

patterns = [
    r"\bi('m| am) not sure\b",
    r"\bi('m| am) not certain\b",
    r"\bprobably\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
]

for p in patterns:
    print(f"Pattern: {p}")
    for t in [text1, text2, text3]:
        m = bool(re.search(p, t.lower()))
        print(f"  {m}: '{t}'")
    print()
