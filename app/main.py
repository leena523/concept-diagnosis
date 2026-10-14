"""Concept Diagnosis web app: FastAPI backend.

Run from the project root:
    python -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000  (interactive API docs at /docs)

Needs the fine-tuned model in models/distilbert-misconception/ (created by src/train.py).
"""
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilbert-misconception"
STATIC_DIR = Path(__file__).resolve().parent / "static"

PROMPT_PREFIX = "Student response: "      # must match fmt() in src/train.py
MAX_LEN = 64                              # must match src/train.py
CONFIDENCE_THRESHOLD = 0.5                # below this the app says "not sure"
SUPERSCRIPTS = str.maketrans("⁺⁻₀₁₂₃₄₅₆₇₈₉", "+-0123456789")

DISPLAY_NAMES = {
    "correct": "No misconception detected",
    "strong_weak_confusion": "Strong vs. weak acid confusion",
    "conjugate_confusion": "Conjugate acid/base confusion",
    "function_notation": "Function notation misunderstanding",
    "probability_rule_misuse": "Probability rule misuse",
    "independence_confusion": "Independent vs. mutually exclusive confusion",
}

FEEDBACK = {
    "correct": (
        "Your response looks consistent with the concept. This tool can make mistakes, so "
        "check it against your notes if you are unsure."
    ),
    "strong_weak_confusion": (
        "Acid strength is about how completely an acid dissociates in water. It is not about "
        "concentration, the pH of one particular sample, or how dangerous the acid is. The common "
        "strong acids are HCl, HBr, HI, HNO3, H2SO4, and HClO4; most other acids you meet in "
        "introductory chemistry are weak."
    ),
    "conjugate_confusion": (
        "A conjugate base is what is left after an acid donates one proton (H+). A conjugate acid "
        "is what you get when a base accepts one proton. A conjugate pair differs by exactly one "
        "proton, so check both the formula and the charge of your answer."
    ),
    "function_notation": (
        "In f(x), the parentheses hold the input, and f(x) does not mean f times x. To evaluate "
        "f(x+2), substitute x+2 for x everywhere in the formula for f. In general f(a+b) is not "
        "f(a) + f(b), and f(2x) is not 2f(x)."
    ),
    "probability_rule_misuse": (
        "Check which rule applies. P(A or B) = P(A) + P(B) - P(A and B). P(A and B) = P(A) x P(B|A), "
        "which becomes P(A) x P(B) only for independent events. P(not A) = 1 - P(A). "
        "P(A|B) = P(A and B) / P(B). Every probability stays between 0 and 1."
    ),
    "independence_confusion": (
        "Independent and mutually exclusive are different ideas. Independent events: knowing one "
        "happened does not change the probability of the other, so P(A and B) = P(A) x P(B). "
        "Mutually exclusive events cannot both happen, so P(A and B) = 0. Two mutually exclusive "
        "events with nonzero probabilities are actually dependent."
    ),
}

LOW_CONFIDENCE_FEEDBACK = (
    "The model is not sure about this response. Try rewording it as a single clear statement, "
    "or check with your instructor or tutor."
)


class Classifier:
    """Loads the fine-tuned DistilBERT model once and scores responses."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.labels = []

    def load(self):
        # Imported here so this file can be imported (and tested) without torch installed.
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_dir.exists():
            raise RuntimeError(
                f"Model folder not found: {self.model_dir}. Run `python src/train.py` first.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]

    def scores(self, prepared_text):
        import torch
        enc = self.tokenizer(prepared_text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()
        return dict(zip(self.labels, probs))


classifier = Classifier(MODEL_DIR)


@asynccontextmanager
async def lifespan(app):
    classifier.load()
    yield


app = FastAPI(title="Concept Diagnosis", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=500)


def prepare(text):
    """Same cleaning and prompt format the model was trained on."""
    text = text.translate(SUPERSCRIPTS)
    text = re.sub(r"\s+", " ", text).strip()
    return PROMPT_PREFIX + text


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter a student response.")

    scores = classifier.scores(prepare(text))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    label, confidence = ranked[0]
    confident = confidence >= CONFIDENCE_THRESHOLD

    return {
        "label": label,
        "display_name": DISPLAY_NAMES.get(label, label),
        "confidence": round(confidence, 3),
        "is_confident": confident,
        "feedback": FEEDBACK.get(label, "") if confident else LOW_CONFIDENCE_FEEDBACK,
        "runner_up": {"label": ranked[1][0], "display_name": DISPLAY_NAMES.get(ranked[1][0], ranked[1][0]),
                      "confidence": round(ranked[1][1], 3)},
        "scores": [{"label": k, "display_name": DISPLAY_NAMES.get(k, k), "confidence": round(v, 3)}
                   for k, v in ranked],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier.model is not None}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
