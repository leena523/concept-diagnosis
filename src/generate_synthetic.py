"""Synthetic misconception data generator.

Run from the project root:  python src/generate_synthetic.py
Output: data/synthetic/synthetic_raw.csv  (columns: response_text, misconception_label)

NOTE: These are template-based seeds. Expect near-duplicates. Use them as a base,
then paraphrase with an LLM and hand-check a sample for label accuracy.
Do NOT copy sentences from data/raw/handwritten_test.csv into this file.
"""
import random
import pandas as pd

random.seed(42)


def pick(lst):
    return random.choice(lst)


def join(starter, text):
    """Prefix a starter and lowercase a leading ordinary word (not a formula)."""
    if starter and len(text) > 1 and text[0].isupper() and text[1].islower():
        text = text[0].lower() + text[1:]
    return starter + text


STARTERS = ["", "", "", "I think ", "So ", "Basically ", "My answer is that ",
            "Well ", "I remember that ", "In my notes it says "]

# ---------------------------------------------------------------- chemistry
STRONG = ["HCl", "HBr", "HI", "HNO3", "H2SO4", "HClO4"]
WEAK = ["HF", "CH3COOH", "HCN", "H2CO3", "HNO2", "H3PO4", "HCOOH", "HClO"]

# correct conjugate bases / acids
CONJ_BASE = {"HCl": "Cl-", "HF": "F-", "HNO3": "NO3-", "H2O": "OH-",
             "NH4+": "NH3", "CH3COOH": "CH3COO-", "H2SO4": "HSO4-",
             "HSO4-": "SO4 2-", "HCN": "CN-", "H2CO3": "HCO3-",
             "HCO3-": "CO3 2-", "H3O+": "H2O", "HBr": "Br-", "HNO2": "NO2-"}
CONJ_ACID = {"NH3": "NH4+", "OH-": "H2O", "H2O": "H3O+", "Cl-": "HCl",
             "F-": "HF", "CO3 2-": "HCO3-", "CH3COO-": "CH3COOH",
             "CN-": "HCN", "SO4 2-": "HSO4-", "NO3-": "HNO3"}

# plausible WRONG answers students give (not equal to the correct one)
WRONG_CONJ_BASE = {
    "HCl": ["H+", "OH-", "Cl+", "H2Cl+"], "HF": ["H+", "F+", "OH-", "H2F+"],
    "HNO3": ["H+", "NO3+", "OH-", "H2NO3+"], "H2O": ["H3O+", "H2", "O2-", "H+"],
    "NH4+": ["NH2-", "NH4-", "OH-", "NH5+"], "CH3COOH": ["H+", "CH3COOH2+", "OH-"],
    "H2SO4": ["SO4 2-", "H+", "OH-", "H3SO4+"], "HSO4-": ["H2SO4", "H+", "SO4+"],
    "HCN": ["CN+", "H+", "OH-"], "H3O+": ["OH-", "H2", "H4O 2+"],
}
WRONG_CONJ_ACID = {
    "NH3": ["NH2-", "NH3+", "OH-"], "OH-": ["O2-", "H3O+", "OH+"],
    "H2O": ["OH-", "H2", "H3O-"], "Cl-": ["HClO", "Cl2", "H+"],
    "F-": ["F2", "H+", "HFO"], "CO3 2-": ["H2CO3", "CO2", "CO3-"],
    "CH3COO-": ["CH3COO+", "CH3COOH2+", "H+"],
}

STRONG_WEAK_WRONG_ABOUT_STRONG = [   # calls a strong acid weak
    "{a} is a weak acid because it only partly dissociates",
    "{a} is weak because it has a low concentration",
    "{a} is a weak acid since it is not very reactive",
    "{a} is weak because the solution I made was dilute",
    "{a} must be a weak acid because it has a high pH in diluted solution",
    "{a} is a weak acid since its bond to hydrogen is too big to break",
]
STRONG_WEAK_WRONG_ABOUT_WEAK = [     # calls a weak acid strong
    "{a} is a strong acid because it has a very polar bond",
    "{a} completely dissociates in water so it is strong",
    "{a} is a strong acid since it is dangerous and corrosive",
    "{a} is strong because the solution is concentrated",
    "{a} is a strong acid because it has a lot of hydrogens",
    "{a} is strong because it has a very low pH in my lab sample",
]
STRONG_WEAK_GENERAL = [              # concept-level mistakes
    "A strong acid is just an acid with a high concentration",
    "weak acids do not dissociate at all in water",
    "Strong acids have a high pH and weak acids have a low pH",
    "A dilute acid is the same as a weak acid",
    "Acid strength depends on how much acid you add to the water",
    "Strong acid means it can burn through anything",
]


def gen_strong_weak():
    r = random.random()
    if r < 0.4:
        t = pick(STRONG_WEAK_WRONG_ABOUT_WEAK).format(a=pick(WEAK))
    elif r < 0.8:
        t = pick(STRONG_WEAK_WRONG_ABOUT_STRONG).format(a=pick(STRONG))
    else:
        t = pick(STRONG_WEAK_GENERAL)
    return join(pick(STARTERS), t)


def gen_conjugate_wrong():
    if random.random() < 0.7:
        s = pick(list(WRONG_CONJ_BASE))
        w = pick(WRONG_CONJ_BASE[s])
        t = pick(["The conjugate base of {s} is {w}",
                  "{w} is the conjugate base of {s}",
                  "when {s} donates a proton you get {w} as the conjugate base",
                  "conjugate base of {s} = {w}"]).format(s=s, w=w)
    else:
        s = pick(list(WRONG_CONJ_ACID))
        w = pick(WRONG_CONJ_ACID[s])
        t = pick(["The conjugate acid of {s} is {w}",
                  "{w} is the conjugate acid of {s}",
                  "when {s} accepts a proton you get {w} as the conjugate acid"]
                 ).format(s=s, w=w)
    return join(pick(STARTERS), t)


def gen_conjugate_correct():
    if random.random() < 0.6:
        s = pick(list(CONJ_BASE))
        t = pick(["The conjugate base of {s} is {c}",
                  "{c} is the conjugate base of {s}",
                  "after {s} loses a proton you get {c}",
                  "conjugate base of {s} = {c}"]).format(s=s, c=CONJ_BASE[s])
    else:
        s = pick(list(CONJ_ACID))
        t = pick(["The conjugate acid of {s} is {c}",
                  "{c} is the conjugate acid of {s}",
                  "after {s} picks up a proton you get {c}"]
                 ).format(s=s, c=CONJ_ACID[s])
    return join(pick(STARTERS), t)


# --------------------------------------------------------------------- math
def gen_function_notation():
    n = random.randint(2, 9)
    k = random.randint(2, 5)
    fn = pick(["f", "g", "h"])
    return pick(STARTERS) + pick([
        f"{fn}(x+{n}) means {fn}(x) + {n}",
        f"{fn}({n}) means {fn} times {n}",
        f"{fn}(x) means {fn} multiplied by x",
        f"{fn}(x+{n}) is the same as {fn}(x) + {fn}({n}) for any function",
        f"{fn}({k}x) is just {k} times {fn}(x) for every function",
        f"if {fn}(x) = x^2 then {fn}(x+{n}) = x^2 + {n}",
        f"if {fn}(x) = {k}x then {fn}(x+1) = {k}x + 1",
        f"the parentheses in {fn}(x) mean multiply {fn} and x",
        f"{fn}(a+b) = {fn}(a) + {fn}(b) always",
        f"{fn}^-1(x) is the same as 1/{fn}(x)",
        f"{fn}({fn}(x)) is just {fn}(x) squared",
        f"{fn}(x) and {fn} are the same thing",
        f"the notation {fn}({n}) tells you to multiply the function by {n}",
        f"{fn}(x) = {k}x + 1 so {fn}({n}) = {k}x + {n}",
    ])


def gen_prob_rule():
    a = random.choice([0.6, 0.7, 0.8])   # a + b > 1 so the 'add them' mistake is visibly wrong
    b = random.choice([0.5, 0.6, 0.7, 0.9])
    n = random.randint(2, 6)
    t = pick([
        "P(A or B) = P(A) + P(B) no matter what the events are",
        "P(A and B) = P(A) + P(B)",
        "the probability of A and B is always P(A) times P(B)",
        "P(A given B) is the same as P(B given A)",
        "P(A|B) = P(B|A) because conditioning is symmetric",
        f"if P(A) = {a} and P(B) = {b} then P(A or B) = {a} + {b} = {round(a + b, 1)} even if they overlap",
        f"if P(A) = {a} and P(B) = {b} then P(A and B) = {round(a + b, 1)}",
        f"P(A) = {a} and P(B) = {b} so P(A or B) = {round(a * b, 2)}",
        "to get P(A or B) you multiply P(A) and P(B)",
        "P(not A) = 1/P(A)",
        "P(A|B) = P(A and B) / P(A)",
        "P(B|A) = P(A and B) / P(B)",
        "P(A and B) = P(A|B) times P(B|A)",
        "P(A and B) = P(A) divided by P(B)",
        "when drawing without replacement you still multiply the original probabilities",
        "P(A or B) = P(A and B) + P(A)",
        f"P(A) = {round(1 + a, 1)} is possible if the event is very likely",
        f"rolling a die {n} times the chance of at least one six is {n}/6",
        f"flipping {n} coins the chance of all heads is {n} times 1/2",
        "P(not A) = P(A) - 1",
        "the probability of two events happening together is found by adding their probabilities",
        "P(A or B) = 1 - P(A) - P(B)",
    ])
    return join(pick(STARTERS), t)


EVENT_PAIRS = [("A", "B"), ("X", "Y"), ("E", "F"), ("rain", "sunshine"),
               ("rolling a 2", "rolling a 5"), ("getting heads", "getting tails"),
               ("drawing a king", "drawing a queen"), ("passing math", "failing math")]


def gen_independence():
    a, b = pick(EVENT_PAIRS)
    t = pick([
        "mutually exclusive events are independent",
        "if {a} and {b} are independent then they cannot happen together",
        "independent means P({a} and {b}) = 0",
        "disjoint events are always independent of each other",
        "{a} and {b} cannot both occur so they have no effect on each other and are independent",
        "independent and mutually exclusive mean the same thing",
        "if P({a} and {b}) = 0 then {a} and {b} are independent",
        "independent events have an empty intersection",
        "if {a} happens and {b} cannot then {a} and {b} are independent",
        "independent events are ones where one event prevents the other",
        "if {a} and {b} can happen together they must be dependent",
        "{a} and {b} are independent because both cannot happen on the same try",
        "since {a} rules out {b} the two events are independent",
        "the circles for independent events in a Venn diagram do not overlap",
        "independence is when the events exclude each other",
    ]).format(a=a, b=b)
    return join(pick(STARTERS), t)


def gen_correct():
    n = random.randint(2, 9)
    fn = pick(["f", "g", "h"])
    r = random.random()
    if r < 0.30:
        return gen_conjugate_correct()
    if r < 0.45:
        return join(pick(STARTERS), f"{pick(STRONG)} is a strong acid because it dissociates completely in water")
    if r < 0.60:
        return join(pick(STARTERS), f"{pick(WEAK)} is a weak acid because it only partially dissociates")
    return pick(STARTERS) + pick([
        f"{fn}(x+{n}) means we plug x+{n} into the function {fn}",
        f"{fn}({n}) is the output of {fn} when the input is {n}",
        f"{fn}(x) = x + {n} means the output is the input plus {n}",
        f"{fn}(x+{n}) is not equal to {fn}(x) + {n} unless {fn} is linear with slope 1",
        "P(A or B) = P(A) + P(B) - P(A and B)",
        "P(A or B) = P(A) + P(B) only when A and B are mutually exclusive",
        "for independent events P(A and B) = P(A) times P(B)",
        "P(A|B) = P(A and B) / P(B)",
        "P(not A) = 1 - P(A)",
        "mutually exclusive events cannot happen at the same time",
        "if A and B are independent then knowing A happened does not change P(B)",
        "two mutually exclusive events with nonzero probability are always dependent",
    ])


GENERATORS = {
    "correct": gen_correct,
    "strong_weak_confusion": gen_strong_weak,
    "conjugate_confusion": gen_conjugate_wrong,
    "function_notation": gen_function_notation,
    "probability_rule_misuse": gen_prob_rule,
    "independence_confusion": gen_independence,
}

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    rows = []
    for label, fn in GENERATORS.items():
        for _ in range(800):
            rows.append({"response_text": fn(), "misconception_label": label})
    df = pd.DataFrame(rows).drop_duplicates()

    out = ROOT / "data" / "synthetic" / "synthetic_raw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df["misconception_label"].value_counts())
    print("Total after dedupe:", len(df))
    print("Saved to", out)
