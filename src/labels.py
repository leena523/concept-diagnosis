LABELS = [
    "correct",
    "strong_weak_confusion",
    "conjugate_confusion",
    "function_notation",
    "probability_rule_misuse",
    "independence_confusion",
]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in id2label.items()}