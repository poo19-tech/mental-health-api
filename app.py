from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# ── App ───────────────────────────────────────────────
app = FastAPI()

# ── Load Model ────────────────────────────────────────
print("Loading model...")
MODEL_PATH = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
model      = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels              = 2,
    ignore_mismatched_sizes = True
)
model.eval()
print("✅ Model ready!")

# ── Word Lists ────────────────────────────────────────
PAST_TENSE_WORDS = {
    "was","were","had","did","went","said","got","made",
    "came","saw","knew","thought","told","felt","left",
    "kept","put","ran","fell","lost","found","gave","took",
    "broke","chose","spoke","woke","wore","wrote","began",
    "drank","ate","grew","threw","slept","wept","heard","held"
}
FUTURE_TENSE_WORDS = {
    "will","shall","would","gonna","plan","plans","planning",
    "intend","tomorrow","soon","eventually","someday","later",
    "future","next","hope","hopes","expect","expects"
}
FIRST_PERSON_WORDS = {
    "i","me","my","myself","mine","i'm","i've","i'll","i'd"
}
NEGATION_WORDS = {
    "not","never","no","neither","nobody","nothing","nowhere",
    "nor","cannot","can't","won't","don't","doesn't","didn't",
    "isn't","aren't","wasn't","weren't","hasn't","haven't",
    "hadn't","shouldn't","wouldn't","couldn't","without"
}
HOPELESSNESS_WORDS = {
    "hopeless","hopelessness","worthless","worthlessness",
    "pointless","useless","meaningless","purposeless",
    "empty","emptiness","hollow","void","numb","numbness",
    "despair","desperate","helpless","helplessness",
    "burden","unwanted","unloved","doomed","trapped","stuck"
}

# ── Request Body ──────────────────────────────────────
class TextInput(BaseModel):
    text: str

# ── analyze_text Function ─────────────────────────────
def analyze_text(text: str) -> dict:
    if not isinstance(text, str) or text.strip() == "":
        return {
            "depression_probability": 0.0,
            "past_tense_ratio"      : 0.0,
            "future_tense_ratio"    : 0.0,
            "hopelessness_score"    : 0.0,
            "detected_patterns"     : []
        }

    words    = text.split()[:200]
    text_cut = " ".join(words)

    inputs = tokenizer(
        text_cut,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 128,
        padding        = "max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        depression_prob = round(probs[0][1].item(), 4)

    tokens = re.findall(r"[a-z']+", text.lower())
    n      = max(len(tokens), 1)

    past_count   = sum(1 for t in tokens if t in PAST_TENSE_WORDS or
                      (t.endswith("ed") and len(t) > 4))
    future_count = sum(1 for t in tokens if t in FUTURE_TENSE_WORDS)
    hop_count    = sum(1 for t in tokens if t in HOPELESSNESS_WORDS)
    neg_count    = sum(1 for t in tokens if t in NEGATION_WORDS)
    fp_count     = sum(1 for t in tokens if t in FIRST_PERSON_WORDS)

    past_ratio   = round(past_count  / n, 4)
    future_ratio = round(future_count/ n, 4)
    hop_score    = round(hop_count   / n, 4)

    patterns = []
    if past_ratio > 0.15:
        patterns.append("high past tense — rumination detected")
    if future_ratio < 0.02 and depression_prob > 0.5:
        patterns.append("low future tense — lack of hope")
    if hop_count > 0:
        patterns.append(f"hopelessness words detected: {hop_count}")
    if neg_count > 2:
        patterns.append("high negation — negative thinking")
    if fp_count / n > 0.15:
        patterns.append("high self focus — first person overuse")
    if depression_prob > 0.7:
        patterns.append("high depression probability by BERT")

    return {
        "depression_probability": depression_prob,
        "past_tense_ratio"      : past_ratio,
        "future_tense_ratio"    : future_ratio,
        "hopelessness_score"    : hop_score,
        "detected_patterns"     : patterns
    }

# ── API Routes ────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/analyze")
def analyze(input: TextInput):
    result = analyze_text(input.text)
    return result