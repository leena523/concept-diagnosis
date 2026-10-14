"""Expanded synthetic dataset generator (v2).

Run from the project root:   python src/expand_dataset.py
Output: data/synthetic/expanded.csv   (response_text, misconception_label, group_id)

How it works
------------
* Each label has many hand-written TEMPLATES (different phrasings of the same idea).
* A template is filled with random "slot" values (acid names, function letters, numbers, event names)
  and then given a casual student style (starter phrase, contractions, small typos).
* Every template gets its own group_id. preprocess.py splits by group_id, so the validation set
  contains phrasings the model never saw in training. That makes validation much closer to the
  handwritten test set than a random split would be.
* The handwritten test set is NOT used here. Any exact overlap is removed in preprocess.py.
"""
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(42)

STARTERS = ["", "", "", "", "I think ", "So ", "Basically ", "My answer is ", "Well ",
            "I remember that ", "Pretty sure ", "I believe ", "Okay so ", "Honestly "]

LABEL_ORDER = ["correct", "strong_weak_confusion", "conjugate_confusion",
               "function_notation", "probability_rule_misuse", "independence_confusion"]
SHORT = {"correct": "cor", "strong_weak_confusion": "sw", "conjugate_confusion": "cj",
         "function_notation": "fn", "probability_rule_misuse": "pr",
         "independence_confusion": "ind"}
TARGET = {"correct": 520, "strong_weak_confusion": 400, "conjugate_confusion": 400,
          "function_notation": 400, "probability_rule_misuse": 400,
          "independence_confusion": 400}


def norm(text):
    """Normalized key used for deduplication and leakage checks (also used by preprocess.py)."""
    t = re.sub(r"\s+", " ", str(text).lower()).strip()
    starts = sorted({s.strip().lower() for s in STARTERS if s.strip()}, key=len, reverse=True)
    for s in starts:
        if t.startswith(s + " "):
            t = t[len(s) + 1:]
            break
    return t.rstrip(".!? ")


# ------------------------------------------------------------------ slot data
STRONG = ["HCl", "HBr", "HI", "HNO3", "H2SO4", "HClO4"]
WEAK = ["HF", "CH3COOH", "HCN", "H2CO3", "HNO2", "H3PO4", "HCOOH", "HClO"]
NAME = {"HCl": "hydrochloric acid", "HBr": "hydrobromic acid", "HI": "hydroiodic acid",
        "HNO3": "nitric acid", "H2SO4": "sulfuric acid", "HClO4": "perchloric acid",
        "HF": "hydrofluoric acid", "CH3COOH": "acetic acid", "HCN": "hydrocyanic acid",
        "H2CO3": "carbonic acid", "HNO2": "nitrous acid", "H3PO4": "phosphoric acid",
        "HCOOH": "formic acid", "HClO": "hypochlorous acid"}

# (species, correct conjugate BASE, [plausible wrong answers])
BASE_FRAMES = [
    ("HCl", "Cl-", ["H+", "OH-", "Cl+", "H2Cl+"]),
    ("HF", "F-", ["H+", "F+", "OH-", "H2F+"]),
    ("HNO3", "NO3-", ["H+", "NO3+", "OH-", "H2NO3+"]),
    ("H2O", "OH-", ["H3O+", "H2", "O2-", "H+"]),
    ("NH4+", "NH3", ["NH2-", "NH4-", "OH-", "NH5+"]),
    ("CH3COOH", "CH3COO-", ["H+", "CH3COOH2+", "OH-", "CH3COO+"]),
    ("H2SO4", "HSO4-", ["SO4 2-", "H+", "OH-", "H3SO4+"]),
    ("HSO4-", "SO4 2-", ["H2SO4", "H+", "SO4+"]),
    ("HCN", "CN-", ["CN+", "H+", "OH-", "H2CN+"]),
    ("H2CO3", "HCO3-", ["CO3 2-", "H+", "CO2", "H3CO3+"]),
    ("HCO3-", "CO3 2-", ["H2CO3", "CO2", "CO3-"]),
    ("H3O+", "H2O", ["OH-", "H2", "H4O 2+"]),
    ("HBr", "Br-", ["H+", "Br+", "OH-"]),
    ("HNO2", "NO2-", ["H+", "NO2+", "OH-"]),
    ("HClO", "ClO-", ["Cl-", "H+", "ClO+"]),
    ("H3PO4", "H2PO4-", ["HPO4 2-", "H+", "PO4 3-"]),
]
# (species, correct conjugate ACID, [plausible wrong answers])
ACID_FRAMES = [
    ("NH3", "NH4+", ["NH2-", "NH3+", "OH-", "NH2+"]),
    ("OH-", "H2O", ["O2-", "H3O+", "OH+", "H2"]),
    ("H2O", "H3O+", ["OH-", "H2", "H3O-"]),
    ("Cl-", "HCl", ["HClO", "Cl2", "H+"]),
    ("F-", "HF", ["F2", "H+", "HFO"]),
    ("CO3 2-", "HCO3-", ["H2CO3", "CO2", "CO3-"]),
    ("CH3COO-", "CH3COOH", ["CH3COO+", "CH3COOH2+", "H+"]),
    ("CN-", "HCN", ["CN+", "H+", "OH-"]),
    ("SO4 2-", "HSO4-", ["H2SO4", "H+", "SO4-"]),
    ("NO3-", "HNO3", ["H+", "NO3+", "OH-"]),
    ("HCO3-", "H2CO3", ["CO3 2-", "CO2", "HCO3+"]),
]
SYM_PAIRS = [("A", "B"), ("E", "F"), ("X", "Y"), ("A", "C")]
EVENT_PAIRS = SYM_PAIRS[:3] + [
    ("rolling a 2", "rolling a 5"), ("getting heads", "getting tails"),
    ("drawing a king", "drawing a queen"), ("passing the quiz", "failing the quiz"),
    ("a red card", "a black card")]
OVERLAPS = [(0.6, 0.7, 0.4, 0.9), (0.5, 0.6, 0.2, 0.9), (0.7, 0.8, 0.6, 0.9),
            (0.4, 0.5, 0.1, 0.8), (0.3, 0.6, 0.2, 0.7), (0.7, 0.5, 0.3, 0.9)]


def s_base():
    s, r, ws = rng.choice(BASE_FRAMES)
    return dict(s=s, r=r, w=rng.choice(ws))


def s_acid():
    s, r, ws = rng.choice(ACID_FRAMES)
    return dict(s=s, r=r, w=rng.choice(ws))


def s_strong():
    a = rng.choice(STRONG)
    return dict(a=a, an=NAME[a])


def s_weak():
    a = rng.choice(WEAK)
    return dict(a=a, an=NAME[a])


def s_func():
    f = rng.choice("fgh")
    g = rng.choice([c for c in "fgh" if c != f])
    n, k = rng.randint(2, 9), rng.randint(2, 9)
    return dict(f=f, g=g, n=n, k=k, v=2 * k + n, kn=k * n, n2=2 * n, nn=n * n)


def s_prob():
    A, B = rng.choice(SYM_PAIRS)
    a, b = rng.choice([0.6, 0.7, 0.8]), rng.choice([0.5, 0.6, 0.7, 0.9])
    pa, pb, pc, pu = rng.choice(OVERLAPS)
    return dict(A=A, B=B, a=a, b=b, s=round(a + b, 1), p=round(a * b, 2),
                big=rng.choice([1.2, 1.3, 1.5]), n=rng.randint(2, 6),
                pa=pa, pb=pb, pc=pc, pu=pu)


def s_indep():
    A, B = rng.choice(EVENT_PAIRS)
    return dict(A=A, B=B)


SAMPLERS = {"base": s_base, "acid": s_acid, "strong": s_strong, "weak": s_weak,
            "func": s_func, "prob": s_prob, "indep": s_indep, "plain": lambda: {}}

# ------------------------------------------------------------------ templates
BASE_T = [
    "The conjugate base of {s} is {w}",
    "{w} is the conjugate base of {s}",
    "when {s} donates a proton you get {w}, which is its conjugate base",
    "conjugate base of {s} = {w}",
    "{s} loses a proton and becomes {w}, its conjugate base",
    "after {s} gives up an H+ the leftover species {w} is the conjugate base",
    "the conj. base of {s} is {w}",
    "I put {w} for the conjugate base of {s}",
    "removing a proton from {s} leaves {w} as the conjugate base",
    "the answer I got for the conjugate base of {s} was {w}",
]
ACID_T = [
    "The conjugate acid of {s} is {w}",
    "{w} is the conjugate acid of {s}",
    "when {s} accepts a proton you get {w}, which is its conjugate acid",
    "conjugate acid of {s} = {w}",
    "{s} picks up a proton and becomes {w}, the conjugate acid",
    "the conj. acid of {s} is {w}",
    "adding an H+ to {s} gives {w} as the conjugate acid",
    "I put {w} for the conjugate acid of {s}",
]


def T(domain, texts):
    return [(domain, t) for t in texts]


WRONG = {}

WRONG["conjugate_confusion"] = (
    T("base", BASE_T) + T("acid", ACID_T) + T("plain", [
        "the conjugate base of an acid is what you get when it gains a proton",
        "a conjugate acid-base pair differs by one electron, not a proton",
        "a conjugate base is any base that reacts with the acid",
        "a conjugate acid is formed when a base loses a proton",
        "the conjugate base has more hydrogens than the acid it came from",
        "in a conjugate pair the base is the one with the extra proton",
        "the conjugate base of a strong acid is a strong base",
        "conjugate pairs are two acids that react with each other",
    ]))

WRONG["strong_weak_confusion"] = (
    T("strong", [
        "{a} is a weak acid because it only partly dissociates",
        "{a} is weak because the solution I made was dilute",
        "{a} is a weak acid since it is not very reactive",
        "{a} must be weak because its pH was not very low in my lab",
        "{an} is a weak acid since only a few of its molecules ionize",
        "{a} is weak because a drop of it barely stings on skin",
        "{an} is weak since it has fewer hydrogens than the others",
        "{a} is a weak acid because the bond to hydrogen is too strong to break",
    ]) + T("weak", [
        "{a} is a strong acid because it has a very polar bond",
        "{a} completely dissociates in water so it is strong",
        "{a} is a strong acid since it is dangerous and corrosive",
        "{a} is strong because the solution is concentrated",
        "{an} is a strong acid because it has a very low pH in my lab sample",
        "{a} is a strong acid because it has a lot of hydrogens",
        "{a} is a strong acid because it reacts with metals",
        "{an} is strong because it is poisonous",
        "{a} is strong since it contains a very electronegative atom",
    ]) + T("plain", [
        "a strong acid is one with a high concentration",
        "weak acids do not dissociate at all in water",
        "strong acids have a high pH and weak acids have a low pH",
        "a dilute acid is the same as a weak acid",
        "acid strength depends on how much acid you add to the water",
        "strong acid means it can burn through anything",
        "a concentrated acid is always a strong acid",
        "weak acids have a pH of 7",
        "strong acids only dissociate partially",
        "the strength of an acid is measured by its molarity",
        "a weak acid is just a strong acid that has been diluted",
        "if the pH is low then the acid is strong no matter the concentration",
    ]))

WRONG["function_notation"] = T("func", [
    "{f}(x+{n}) means {f}(x) plus {n}",
    "{f}({n}) means {f} times {n}",
    "{f}(x) means {f} multiplied by x",
    "{f}(x+{n}) is the same as {f}(x) + {f}({n}) for any function",
    "{f}({k}x) is just {k} times {f}(x) for every function",
    "if {f}(x) = x^2 then {f}(x+{n}) = x^2 + {n}",
    "if {f}(x) = {k}x then {f}(x+1) = {k}x + 1",
    "the parentheses in {f}(x) mean multiply {f} and x",
    "{f}(a+b) = {f}(a) + {f}(b) always",
    "{f}^-1(x) is the same as 1/{f}(x)",
    "{f}({f}(x)) is just {f}(x) squared",
    "{f}(x) and {f} are the same thing",
    "the notation {f}({n}) tells you to multiply the function by {n}",
    "{f}(x) = {k}x + 1 so {f}({n}) = {k}x + {n}",
    "if {f}(x) = x^2 then {f}({k}x) = {k}x^2",
    "{f}(-x) is the same as -{f}(x) for all functions",
    "{f}(x^2) just means you square {f}(x)",
    "the {f} distributes over the parentheses so {f}(a+b) = {f}(a) + {f}(b)",
    "if {f}(x) = {k}x + {n} then {f}(2) = {k}x + 2",
    "{f}({g}(x)) is the same as {f}(x) times {g}(x)",
    "to find {f}({n}) you multiply {n} by the function {f}",
    "if {f}(x) = x + {n} then {f}(x+{n}) = x + {n}",
])

WRONG["probability_rule_misuse"] = T("prob", [
    "P({A} or {B}) = P({A}) + P({B}) no matter what the events are",
    "P({A} and {B}) = P({A}) + P({B})",
    "the probability of {A} and {B} is always P({A}) times P({B})",
    "P({A} given {B}) is the same as P({B} given {A})",
    "P({A}|{B}) = P({B}|{A}) because conditioning is symmetric",
    "if P({A}) = {a} and P({B}) = {b} then P({A} or {B}) = {a} + {b} = {s} even if they overlap",
    "P({A}) = {a} and P({B}) = {b} so P({A} and {B}) = {s}",
    "P({A}) = {a} and P({B}) = {b} so P({A} or {B}) = {p}",
    "to get P({A} or {B}) you multiply P({A}) and P({B})",
    "P(not {A}) = 1/P({A})",
    "P({A}|{B}) = P({A} and {B}) / P({A})",
    "P({B}|{A}) = P({A} and {B}) / P({B})",
    "P({A} and {B}) = P({A}|{B}) times P({B}|{A})",
    "P({A} and {B}) = P({A}) divided by P({B})",
    "when drawing without replacement you still multiply the original probabilities",
    "P({A} or {B}) = P({A} and {B}) + P({A})",
    "P({A}) = {big} is possible if the event is very likely",
    "rolling a die {n} times the chance of at least one six is {n}/6",
    "flipping {n} coins the chance of all heads is {n} times 1/2",
    "P(not {A}) = P({A}) - 1",
    "the probability of two events happening together is found by adding their probabilities",
    "P({A} or {B}) = 1 - P({A}) - P({B})",
    "a coin flip landing heads or tails has probability 1/2 times 1/2 = 1/4",
    "drawing two aces in a row without replacement is 4/52 times 4/52",
    "P({A}|{B}) means P({A}) divided by P({B})",
    "P(not {A}) = 1 + P({A})",
    "P({A} and {B}) = P({A}) - P({B})",
    "the chance that at least one of {A} or {B} happens is P({A}) + P({B})",
])

WRONG["independence_confusion"] = T("indep", [
    "mutually exclusive events are independent",
    "if {A} and {B} are independent then they cannot happen together",
    "independent means P({A} and {B}) = 0",
    "disjoint events are always independent of each other",
    "{A} and {B} cannot both occur so they have no effect on each other and are independent",
    "independent and mutually exclusive mean the same thing",
    "if P({A} and {B}) = 0 then {A} and {B} are independent",
    "independent events have an empty intersection",
    "if {A} happens and {B} cannot then {A} and {B} are independent",
    "independent events are ones where one event prevents the other",
    "if {A} and {B} can happen together they must be dependent",
    "{A} and {B} are independent because both cannot happen on the same try",
    "since {A} rules out {B} the two events are independent",
    "the circles for independent events in a Venn diagram do not overlap",
    "independence is when the events exclude each other",
    "events that never overlap are independent",
    "if knowing {A} happened tells you {B} did not, then {A} and {B} are independent",
    "two independent events cannot both be true at the same time",
    "mutually exclusive means independent because one event does not affect the other",
    "if events cant occur together they are independent",
    "to check independence you just see whether the events can occur together",
    "independent events have probability zero of happening together",
    "independent = disjoint",
    "mutually exclusive events satisfy P({A} and {B}) = P({A}) times P({B}) so they are independent",
])

WRONG["correct"] = (
    T("base", [t.replace("{w}", "{r}") for t in BASE_T[:7]])
    + T("acid", [t.replace("{w}", "{r}") for t in ACID_T[:5]])
    + T("strong", [
        "{a} is a strong acid because it dissociates completely in water",
        "{a} is a strong acid, so it ionizes essentially 100% in water",
        "a dilute solution of {a} can still be a strong acid because strength is about dissociation",
        "{an} is a strong acid, which means nearly all of its molecules ionize in water",
    ]) + T("weak", [
        "{a} is a weak acid because it only partially dissociates",
        "{a} is a weak acid, so most of its molecules stay intact in water",
        "{a} is still a weak acid when it is concentrated, since strength is not the same as concentration",
        "{an} is a weak acid, which means only a small fraction of it ionizes",
    ]) + T("plain", [
        "concentration tells you how much acid is dissolved but strength tells you how much of it ionizes",
        "the conjugate base of a strong acid is a very weak base",
        "the conjugate base of a weak acid is a weak base",
        "the conjugate acid of a weak base is a weak acid",
        "at the same concentration a strong acid has a lower pH than a weak acid",
        "a weak acid has a small Ka because only a little of it dissociates",
        "a conjugate acid-base pair differs by exactly one proton",
        "the conjugate acid has one more proton than its base",
        "a stronger acid has a weaker conjugate base",
    ]) + T("func", [
        "{f}(x+{n}) means we plug x+{n} into the function {f}",
        "{f}({n}) is the output of {f} when the input is {n}",
        "{f}(x) = x + {n} means the output is the input plus {n}",
        "{f}(x+{n}) is not equal to {f}(x) + {n} in general",
        "if {f}(x) = {k}x then {f}(x+1) = {k}x + {k}",
        "if {f}(x) = x^2 then {f}(x+{n}) = x^2 + {n2}x + {nn}",
        "{f}(x) is read as {f} of x and it is the output for the input x",
        "the parentheses in {f}(x) hold the input of the function and do not mean multiplication",
        "{f}^-1 is the inverse function of {f}, not 1 over {f}(x)",
        "{f}({g}(x)) means apply {g} first and then apply {f}",
        "if {f}(x) = {k}x + {n} then {f}(2) = {v}",
        "{f}({k}x) is generally not {k} times {f}(x)",
        "if {f}(x) = {k}x then {f}({n}) = {kn}",
        "{f}(a+b) is generally not the same as {f}(a) + {f}(b)",
    ]) + T("prob", [
        "P({A} or {B}) = P({A}) + P({B}) - P({A} and {B})",
        "P({A} or {B}) = P({A}) + P({B}) only when {A} and {B} are mutually exclusive",
        "for independent events P({A} and {B}) = P({A}) times P({B})",
        "P({A}|{B}) = P({A} and {B}) / P({B})",
        "P(not {A}) = 1 - P({A})",
        "P({A}|{B}) and P({B}|{A}) are usually different",
        "a probability is always between 0 and 1",
        "when you draw without replacement the second probability changes",
        "the chance of at least one six in {n} rolls is 1 - (5/6)^{n}",
        "if P({A}) = {pa} and P({B}) = {pb} and P({A} and {B}) = {pc} then P({A} or {B}) = {pu}",
        "a single coin flip lands heads or tails with probability 1",
        "P({A} and {B}) = P({A}) times P({B}|{A})",
    ]) + T("indep", [
        "mutually exclusive events cannot happen at the same time",
        "if {A} and {B} are independent then knowing {A} happened does not change P({B})",
        "two mutually exclusive events with nonzero probability are always dependent",
        "independent events can happen together and P({A} and {B}) = P({A}) times P({B})",
        "{A} and {B} are independent when P({A}|{B}) = P({A})",
        "mutually exclusive events have P({A} and {B}) = 0 but that does not make them independent",
        "independent and mutually exclusive are two different ideas",
        "if {A} happens then {B} cannot, so {A} and {B} are dependent",
        "two coin flips are independent even though both can happen together",
        "drawing with replacement gives independent draws",
    ])
)

# ------------------------------------------------------------------ rendering
CONTRACTIONS = [(r"\bcannot\b", "cant"), (r"\bcannot\b", "can't"), (r"\bdoes not\b", "doesnt"),
                (r"\bdoes not\b", "doesn't"), (r"\bdo not\b", "dont"), (r"\bdo not\b", "don't"),
                (r"\bis not\b", "isn't"), (r"\bit is\b", "it's")]


def noise(text):
    if rng.random() < 0.35:
        for pat, rep in rng.sample(CONTRACTIONS, len(CONTRACTIONS)):
            if re.search(pat, text):
                text = re.sub(pat, rep, text, count=1)
                break
    if rng.random() < 0.25:
        words = re.findall(r"\b[a-z]{6,}\b", text)
        if words:
            w = rng.choice(words)
            i = rng.randint(1, len(w) - 3)
            text = text.replace(w, w[:i] + w[i + 1] + w[i] + w[i + 2:], 1)
    if rng.random() < 0.3 and not text.endswith((".", "!", "?")):
        text += "."
    return text


def render(domain, tmpl):
    starter = rng.choice(STARTERS)
    lead = tmpl
    # lowercase the first literal letter only (never a slot value like "Cl-")
    if tmpl[0].isupper() and tmpl[1].islower() and (starter or rng.random() < 0.3):
        lead = tmpl[0].lower() + tmpl[1:]
    return noise(starter + lead.format(**SAMPLERS[domain]()))


def generate(label, target):
    templates = WRONG[label]
    caps = [10 if "{" not in t else 30 for _, t in templates]
    counts = [0] * len(templates)
    rows, seen = [], set()
    while len(rows) < target:
        progressed = False
        for i, (dom, t) in enumerate(templates):
            if len(rows) >= target:
                break
            if counts[i] >= caps[i]:
                continue
            progressed = True
            for _ in range(25):
                text = render(dom, t)
                key = norm(text)
                if key not in seen:
                    seen.add(key)
                    rows.append({"response_text": text, "misconception_label": label,
                                 "group_id": f"{SHORT[label]}_{i:02d}"})
                    counts[i] += 1
                    break
            else:
                counts[i] = caps[i]
        if not progressed:
            break
    return rows


if __name__ == "__main__":
    rows = []
    for label in LABEL_ORDER:
        rows += generate(label, TARGET[label])
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "synthetic" / "expanded.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df["misconception_label"].value_counts())
    print("Total rows:", len(df), "| template groups:", df["group_id"].nunique())
    print("Saved to", out)
