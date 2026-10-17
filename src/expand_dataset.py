"""Expanded synthetic dataset generator (v3: matched correct/wrong pairs).

Run from the project root:   python src/expand_dataset.py
Output: data/synthetic/expanded.csv   (response_text, misconception_label, group_id)

What changed from v2
--------------------
v2 used label-specific templates, so some phrasings appeared mostly with wrong answers. The model
learned "this wording means a misconception" instead of checking the facts. v3 fixes that with
MATCHED PAIRS: each phrasing is generated with a correct fill AND a wrong fill, in equal numbers,
and both versions share one group_id. preprocess.py splits by group_id, so a held-out phrasing
appears in validation with BOTH its correct and wrong versions. That makes validation a real test of
whether the model learned the facts and not the wording.

Other pieces: strong/weak acid statements are built from a claim plus an optional reason, and the
label is decided by whether the claim and reason are right. Unpaired templates from v2 fill out
the rest. The handwritten test set is NOT used here (exact overlaps are removed in preprocess.py).
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
WRONG_TARGET = 400     # total rows per misconception label (pairs + unpaired fill)
CORRECT_EXTRA = 250    # extra unpaired correct rows on top of the paired right-hand sides
PAIR_N_SLOTTED = 14    # variants per side for templates with slots
PAIR_N_PLAIN = 6       # variants per side for templates without slots
SW_PER_SIDE = 40       # variants per side for each strong/weak frame


def norm(text):
    """Normalized key for deduplication and leakage checks (also used by preprocess.py)."""
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
                pa=pa, pb=pb, pc=pc, pu=pu, ps=round(pa + pb, 1))


def s_indep():
    A, B = rng.choice(EVENT_PAIRS)
    return dict(A=A, B=B)


SAMPLERS = {"base": s_base, "acid": s_acid, "strong": s_strong, "weak": s_weak,
            "func": s_func, "prob": s_prob, "indep": s_indep, "plain": lambda: {}}

# ------------------------------------------------------------------ styling
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


def decorate(text):
    """Starter phrase + noise for text that is already fully built."""
    return noise(rng.choice(STARTERS) + text)


def render(domain, tmpl):
    starter = rng.choice(STARTERS)
    lead = tmpl
    # lowercase the first literal letter only (never a slot value like "Cl-")
    if tmpl[0].isupper() and tmpl[1].islower() and (starter or rng.random() < 0.3):
        lead = tmpl[0].lower() + tmpl[1:]
    return noise(starter + lead.format(**SAMPLERS[domain]()))


# ------------------------------------------------------------------ MATCHED PAIRS
# pair(label, domain, RIGHT template, WRONG template). Both are generated in equal numbers
# and share one group_id.
PAIRS = []


def pair(label, domain, right, wrong):
    PAIRS.append((label, domain, right, wrong))


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
for t in BASE_T:
    pair("conjugate_confusion", "base", t.replace("{w}", "{r}"), t)
for t in ACID_T:
    pair("conjugate_confusion", "acid", t.replace("{w}", "{r}"), t)
for r, w in [
    ("a conjugate acid-base pair differs by exactly one proton",
     "a conjugate acid-base pair differs by one electron, not a proton"),
    ("the conjugate base of a strong acid is a very weak base",
     "the conjugate base of a strong acid is a strong base"),
    ("the conjugate acid has one more proton than its base",
     "the conjugate acid has one less proton than its base"),
    ("a conjugate base is formed when an acid loses a proton",
     "a conjugate base is formed when an acid gains a proton"),
    ("a stronger acid has a weaker conjugate base",
     "a stronger acid has a stronger conjugate base"),
    ("in a conjugate pair the acid is the one with the extra proton",
     "in a conjugate pair the base is the one with the extra proton"),
    ("a conjugate acid is formed when a base gains a proton",
     "a conjugate acid is formed when a base loses a proton"),
]:
    pair("conjugate_confusion", "plain", r, w)

for r, w in [
    ("strength is about how much an acid dissociates, not how concentrated it is",
     "a strong acid is just an acid with a high concentration"),
    ("a weak acid only partly dissociates in water", "weak acids do not dissociate at all in water"),
    ("a dilute solution of a strong acid is still a strong acid", "a dilute acid is the same as a weak acid"),
    ("at the same concentration strong acids have a lower pH than weak acids",
     "strong acids have a high pH and weak acids have a low pH"),
    ("a concentrated weak acid is still a weak acid", "a concentrated acid is always a strong acid"),
    ("a strong acid dissociates completely in water", "strong acids only dissociate partially"),
]:
    pair("strong_weak_confusion", "plain", r, w)

for r, w in [
    ("{f}(x+{n}) means we plug x+{n} into the function {f}", "{f}(x+{n}) means {f}(x) plus {n}"),
    ("{f}({n}) is the output of {f} when the input is {n}", "{f}({n}) means {f} times {n}"),
    ("{f}(a+b) is generally not the same as {f}(a) + {f}(b)", "{f}(a+b) = {f}(a) + {f}(b) always"),
    ("if {f}(x) = {k}x then {f}(x+1) = {k}x + {k}", "if {f}(x) = {k}x then {f}(x+1) = {k}x + 1"),
    ("if {f}(x) = x^2 then {f}(x+{n}) = x^2 + {n2}x + {nn}", "if {f}(x) = x^2 then {f}(x+{n}) = x^2 + {n}"),
    ("{f}^-1 is the inverse function of {f}, not 1 over {f}(x)", "{f}^-1(x) is the same as 1/{f}(x)"),
    ("{f}({g}(x)) means apply {g} first and then apply {f}", "{f}({g}(x)) is the same as {f}(x) times {g}(x)"),
    ("if {f}(x) = {k}x + {n} then {f}(2) = {v}", "if {f}(x) = {k}x + {n} then {f}(2) = {k}x + 2"),
    ("{f}({k}x) is generally not {k} times {f}(x)", "{f}({k}x) is just {k} times {f}(x) for every function"),
    ("the parentheses in {f}(x) hold the input and do not mean multiplication",
     "the parentheses in {f}(x) mean multiply {f} and x"),
    ("{f}(x) is read as {f} of x and it is the output for the input x", "{f}(x) means {f} multiplied by x"),
    ("if {f}(x) = {k}x then {f}({n}) = {kn}", "if {f}(x) = {k}x then {f}({n}) = {k}x + {n}"),
]:
    pair("function_notation", "func", r, w)

for r, w in [
    ("P({A} or {B}) = P({A}) + P({B}) - P({A} and {B})", "P({A} or {B}) = P({A}) + P({B}) no matter what the events are"),
    ("P({A}|{B}) = P({A} and {B}) / P({B})", "P({A}|{B}) = P({A} and {B}) / P({A})"),
    ("P(not {A}) = 1 - P({A})", "P(not {A}) = P({A}) - 1"),
    ("P({A}|{B}) and P({B}|{A}) are usually different", "P({A}|{B}) and P({B}|{A}) are always equal"),
    ("for independent events P({A} and {B}) = P({A}) times P({B})",
     "for any two events P({A} and {B}) = P({A}) times P({B})"),
    ("a probability can never be bigger than 1", "P({A}) = {big} is possible if the event is very likely"),
    ("P({A} or {B}) = P({A}) + P({B}) only when {A} and {B} are mutually exclusive",
     "P({A} or {B}) = P({A}) + P({B}) when {A} and {B} are independent"),
    ("without replacement the probability of the second draw changes",
     "without replacement you still multiply the original probabilities"),
    ("if P({A}) = {pa} and P({B}) = {pb} and P({A} and {B}) = {pc} then P({A} or {B}) = {pu}",
     "if P({A}) = {pa} and P({B}) = {pb} and P({A} and {B}) = {pc} then P({A} or {B}) = {pa} + {pb} = {ps}"),
    ("P({A} and {B}) = P({A}) times P({B}|{A})", "P({A} and {B}) = P({A}) times P({A}|{B})"),
    ("the chance of at least one six in {n} rolls is 1 - (5/6)^{n}",
     "the chance of at least one six in {n} rolls is {n}/6"),
    ("P({A}|{B}) means the probability of {A} given that {B} happened", "P({A}|{B}) means P({A}) divided by P({B})"),
]:
    pair("probability_rule_misuse", "prob", r, w)

for r, w in [
    ("independent and mutually exclusive are two different ideas", "independent and mutually exclusive mean the same thing"),
    ("mutually exclusive events with nonzero probability are dependent", "mutually exclusive events are independent"),
    ("independent events can happen together", "independent events cannot happen together"),
    ("if {A} and {B} are independent then P({A} and {B}) = P({A}) times P({B})",
     "if {A} and {B} are independent then P({A} and {B}) = 0"),
    ("if P({A} and {B}) = 0 then {A} and {B} are mutually exclusive",
     "if P({A} and {B}) = 0 then {A} and {B} are independent"),
    ("{A} and {B} are independent when knowing {A} happened does not change P({B})",
     "{A} and {B} are independent when {A} happening means {B} cannot happen"),
    ("disjoint events with nonzero probability are dependent", "disjoint events are always independent of each other"),
    ("events that never overlap are mutually exclusive, not independent", "events that never overlap are independent"),
    ("the Venn diagram circles of mutually exclusive events do not overlap",
     "the Venn diagram circles of independent events do not overlap"),
    ("if {A} and {B} can both happen at once they are not mutually exclusive",
     "if {A} and {B} can both happen at once they must be dependent"),
    ("drawing two cards with replacement gives independent draws",
     "drawing a heart and a spade on one draw are independent since both cannot happen"),
    ("two coin flips are independent because one flip does not affect the other",
     "heads and tails on one flip are independent because both cannot happen"),
]:
    pair("independence_confusion", "indep", r, w)


def gen_pairs(seen):
    rows, counters = [], {}
    for label, domain, right, wrong in PAIRS:
        k = counters.get(label, 0)
        counters[label] = k + 1
        gid = f"{SHORT[label]}_p{k:02d}"
        n = PAIR_N_SLOTTED if ("{" in right or "{" in wrong) else PAIR_N_PLAIN
        for lab, tmpl in (("correct", right), (label, wrong)):
            made = tries = 0
            while made < n and tries < 300:
                tries += 1
                text = render(domain, tmpl)
                key = norm(text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"response_text": text, "misconception_label": lab, "group_id": gid})
                made += 1
    return rows


# ---- strong/weak: claim + optional reason, labeled by whether claim and reason are right
SW_FRAMES = ["{a} is a {c} acid{b}", "{an} is a {c} acid{b}", "{a} is {c}{b}",
             "I would call {a} a {c} acid{b}", "{an} is {c}{b}", "{a} counts as a {c} acid{b}"]
RIGHT_REASON = {
    "strong": ["it dissociates completely in water", "it ionizes essentially 100% in water",
               "nearly all of its molecules ionize in water"],
    "weak": ["it only partially dissociates", "only a small fraction of its molecules ionize",
             "most of its molecules stay intact in water"],
}
BAD_REASONS = ["the solution is concentrated", "the solution I made was dilute", "it is dangerous and corrosive",
               "it has a low pH in my lab sample", "it has a lot of hydrogens", "it is not very reactive",
               "it reacts with metals", "it is poisonous", "it contains a very electronegative atom",
               "a drop of it barely stings on skin"]


def sw_text(frame, want_correct):
    a = rng.choice(STRONG + WEAK)
    truth = "strong" if a in STRONG else "weak"
    other = "weak" if truth == "strong" else "strong"
    if want_correct:
        claim, reason = truth, rng.choice(RIGHT_REASON[truth] + [None, None])
    else:
        kind = rng.choice(["wrong_class_consistent", "wrong_class_bad", "right_class_bad", "right_class_contradict"])
        if kind == "wrong_class_consistent":
            claim, reason = other, rng.choice(RIGHT_REASON[other])
        elif kind == "wrong_class_bad":
            claim, reason = other, rng.choice(BAD_REASONS)
        elif kind == "right_class_bad":
            claim, reason = truth, rng.choice(BAD_REASONS)
        else:
            claim, reason = truth, rng.choice(RIGHT_REASON[other])
    b = "" if reason is None else rng.choice([" because ", ", since "]) + reason
    return decorate(frame.format(a=a, an=NAME[a], c=claim, b=b))


def gen_sw_pairs(seen):
    rows = []
    for i, frame in enumerate(SW_FRAMES):
        for lab, want in (("correct", True), ("strong_weak_confusion", False)):
            made = tries = 0
            while made < SW_PER_SIDE and tries < 600:
                tries += 1
                text = sw_text(frame, want)
                key = norm(text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"response_text": text, "misconception_label": lab, "group_id": f"sw_f{i:02d}"})
                made += 1
    return rows


# ------------------------------------------------------------------ UNPAIRED fill (from v2)
def T(domain, texts):
    return [(domain, t) for t in texts]


UNPAIRED = {}
UNPAIRED["conjugate_confusion"] = T("base", BASE_T) + T("acid", ACID_T) + T("plain", [
    "a conjugate base is any base that reacts with the acid",
    "the conjugate base has more hydrogens than the acid it came from",
    "conjugate pairs are two acids that react with each other",
])
UNPAIRED["strong_weak_confusion"] = T("plain", [
    "acid strength depends on how much acid you add to the water",
    "strong acid means it can burn through anything",
    "weak acids have a pH of 7",
    "the strength of an acid is measured by its molarity",
    "a weak acid is just a strong acid that has been diluted",
    "if the pH is low then the acid is strong no matter the concentration",
]) + T("weak", ["{a} is a strong acid because it has a very polar bond",
                "{a} completely dissociates in water so it is strong"]) \
  + T("strong", ["{a} is a weak acid because it only partly dissociates"])
UNPAIRED["function_notation"] = T("func", [
    "{f}(x+{n}) is the same as {f}(x) + {f}({n}) for any function",
    "{f}({f}(x)) is just {f}(x) squared",
    "{f}(x) and {f} are the same thing",
    "the notation {f}({n}) tells you to multiply the function by {n}",
    "if {f}(x) = x^2 then {f}({k}x) = {k}x^2",
    "{f}(-x) is the same as -{f}(x) for all functions",
    "{f}(x^2) just means you square {f}(x)",
    "the {f} distributes over the parentheses so {f}(a+b) = {f}(a) + {f}(b)",
    "to find {f}({n}) you multiply {n} by the function {f}",
    "if {f}(x) = x + {n} then {f}(x+{n}) = x + {n}",
])
UNPAIRED["probability_rule_misuse"] = T("prob", [
    "the probability of {A} and {B} is always P({A}) times P({B})",
    "P({A}) = {a} and P({B}) = {b} so P({A} and {B}) = {s}",
    "P({A}) = {a} and P({B}) = {b} so P({A} or {B}) = {p}",
    "to get P({A} or {B}) you multiply P({A}) and P({B})",
    "P(not {A}) = 1/P({A})",
    "P({A} and {B}) = P({A}) divided by P({B})",
    "rolling a die {n} times the chance of at least one six is {n}/6",
    "flipping {n} coins the chance of all heads is {n} times 1/2",
    "the probability of two events happening together is found by adding their probabilities",
    "a coin flip landing heads or tails has probability 1/2 times 1/2 = 1/4",
    "drawing two aces in a row without replacement is 4/52 times 4/52",
    "P({A} and {B}) = P({A}) - P({B})",
])
UNPAIRED["independence_confusion"] = T("indep", [
    "{A} and {B} cannot both occur so they have no effect on each other and are independent",
    "independent events have an empty intersection",
    "independent events are ones where one event prevents the other",
    "independence is when the events exclude each other",
    "if knowing {A} happened tells you {B} did not, then {A} and {B} are independent",
    "two independent events cannot both be true at the same time",
    "mutually exclusive means independent because one event does not affect the other",
    "independent events have probability zero of happening together",
    "independent = disjoint",
    "mutually exclusive events satisfy P({A} and {B}) = P({A}) times P({B}) so they are independent",
])
UNPAIRED["correct"] = T("strong", [
    "{a} is a strong acid, so it ionizes essentially 100% in water",
    "{an} is a strong acid, which means nearly all of its molecules ionize in water",
]) + T("weak", [
    "{a} is a weak acid, so most of its molecules stay intact in water",
    "{an} is a weak acid, which means only a small fraction of it ionizes",
]) + T("plain", [
    "concentration tells you how much acid is dissolved but strength tells you how much of it ionizes",
    "the conjugate base of a weak acid is a weak base",
    "the conjugate acid of a weak base is a weak acid",
    "a weak acid has a small Ka because only a little of it dissociates",
    "a probability is always between 0 and 1",
    "mutually exclusive events cannot happen at the same time",
    "a single coin flip lands heads or tails with probability 1",
]) + T("func", [
    "{f}(x) = x + {n} means the output is the input plus {n}",
    "{f}(x+{n}) is not equal to {f}(x) + {n} in general",
]) + T("prob", [
    "when you draw without replacement the second probability changes",
    "P({A}|{B}) and P({B}|{A}) are usually different",
]) + T("indep", [
    "if {A} and {B} are independent then knowing {A} happened does not change P({B})",
    "{A} and {B} are independent when P({A}|{B}) = P({A})",
    "two mutually exclusive events with nonzero probability are always dependent",
])


def gen_unpaired(label, need, seen):
    templates = UNPAIRED[label]
    caps = [10 if "{" not in t else 30 for _, t in templates]
    counts = [0] * len(templates)
    rows = []
    while len(rows) < need:
        progressed = False
        for i, (dom, t) in enumerate(templates):
            if len(rows) >= need:
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
    seen = set()
    rows = gen_pairs(seen) + gen_sw_pairs(seen)
    df = pd.DataFrame(rows)
    for label in LABEL_ORDER:
        have = int((df["misconception_label"] == label).sum())
        need = CORRECT_EXTRA if label == "correct" else max(0, WRONG_TARGET - have)
        rows += gen_unpaired(label, need, seen)
    df = pd.DataFrame(rows)

    out = ROOT / "data" / "synthetic" / "expanded.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df["misconception_label"].value_counts())
    paired = df[df["group_id"].str.contains("_p") | df["group_id"].str.contains("_f")]
    by = paired.groupby("group_id")["misconception_label"].apply(lambda s: (s == "correct").mean())
    print("Total rows:", len(df), "| groups:", df["group_id"].nunique(),
          "| paired groups:", len(by), "| share of correct inside paired groups (should be near 0.5):",
          round(float(by.mean()), 3))
    print("Saved to", out)
