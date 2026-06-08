"""
Path 1 workflow: 
1. Load the model and filter the dataset 
2. Generate repeated sampled batches for each reformulation
3. Normalise answers to short-answer form
4. Perform semantic clustering
5. Compute summary quantities
6. Save outputs to JSON
"""

import os
import re
import json
import time
import random
import string
import math
from collections import defaultdict, Counter
from itertools import combinations

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------
# Configuration
# ----------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "ibm-research/popqa-tp"

MAX_NEW_TOKENS = 8
N_QUESTIONS = 12
N_PARAPHRASES_TOTAL = 4

SELECTION_SEED = 100
GENERATION_SEEDS = [256, 3, 9, 27, 81]

SAMPLES_PER_BATCH = 4
N_BATCHES_PER_TRANSFORM = 4
TEMPERATURE = 0.7
TOP_P = 0.95

OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "popqa_tp_path1.json")

# Model loading
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.config.use_cache = True
print("Model loaded.")

# Send inputs to the device where the model is actually placed
first_device = next(
    (d for d in model.hf_device_map.values() if d not in ["cpu", "disk"]),
    "cpu"
)

if isinstance(first_device, int):
    INPUT_DEVICE = f"cuda:{first_device}"
else:
    INPUT_DEVICE = first_device


# Derive batch seed to ensure reproducible
def derived_generation_seed(
    base_seed: int,
    question_id: int,
    reformulation_index: int,
    batch_index: int,
) -> int:
    seed = base_seed
    seed = (seed * 1000003 + question_id + 1) % (2**31 - 1)
    seed = (seed * 1000003 + reformulation_index + 1) % (2**31 - 1)
    seed = (seed * 1000003 + batch_index + 1) % (2**31 - 1)
    return seed


# Time Limit
class QuestionTimeout(Exception):
    pass


def check_deadline(deadline: float, stage: str = "") -> None:
    if time.perf_counter() > deadline:
        msg = "Question exceeded time limit"
        if stage:
            msg += f" during {stage}"
        raise QuestionTimeout(msg)


# -----------
# Text normalization
# -----------
_ARTICLES = {"a", "an", "the"}

# Lightweight normalisation for exact matches
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    toks = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(toks)


# Remove clauses, explanations and prefixes
def extract_short_answer(text: str) -> str:
    text = text.strip()

    text = text.splitlines()[0].strip()

    text = re.split(r"[.;:]", text)[0].strip()

    text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the answer is\s*)", "", text, flags=re.IGNORECASE)

    return text.strip()


# Check if answer belong to the dataset's list of acceptalbe forms
def is_correct_prediction(pred: str, gold_answers) -> bool:
    pred = extract_short_answer(pred)
    pred_norm = normalize_text(pred)

    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    gold_norms = {
        normalize_text(ans)
        for ans in gold_answers
        if isinstance(ans, str) and ans.strip()
    }
    return pred_norm in gold_norms


# Filter out answers that belongs to special answer buckets
def special_bucket(pred: str) -> str | None:
    pred = extract_short_answer(pred)

    if not pred.strip():
        return "__invalid__"

    lower = pred.lower()
    abstain_markers = [
        "i don't know",
        "unknown",
        "not sure",
        "cannot determine",
        "no information",
        "not provided",
    ]
    if any(m in lower for m in abstain_markers):
        return "__abstain__"

    denial_markers = [
        "is not a capital",
        "not a capital",
        "not a recognized capital",
        "not a country",
        "not a city",
        "not a sport",
        "not a recognized",
        "not a real place",
        "fictional",
        "not found",
        "there is no such",
    ]
    if any(m in lower for m in denial_markers):
        return "__denial__"

    return None


# Make YES/NO judgement for a semantic-equivalence prompt
def judge_yes_no(prompt: str) -> bool:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(INPUT_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
    return text.startswith("yes")


# Cache set for revisited pairs
_semantic_eq_cache = {}

# Pairwise semantic equivalence test used to build answer buckets
def same_meaning_under_question(question: str, ans_a: str, ans_b: str) -> bool:
    a = extract_short_answer(ans_a)
    b = extract_short_answer(ans_b)

    if normalize_text(a) == normalize_text(b):
        return True

    key = (question.strip().lower(), normalize_text(a), normalize_text(b))
    if key in _semantic_eq_cache:
        return _semantic_eq_cache[key]

    prompt_ab = f"""Question: {question}

        Answer A: {a}
        Answer B: {b}

        Do Answer A and Answer B express the same factual answer to the question?
        Reply with YES or NO only."""
    prompt_ba = f"""Question: {question}

        Answer A: {b}
        Answer B: {a}

        Do Answer A and Answer B express the same factual answer to the question?
        Reply with YES or NO only."""

    ab = judge_yes_no(prompt_ab)
    ba = judge_yes_no(prompt_ba)
    out = ab and ba
    _semantic_eq_cache[key] = out
    _semantic_eq_cache[(key[0], key[2], key[1])] = out
    return out


# Union-find structure for pairwise semantic equivalent links
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# Build semantic buckets for ordinary answers with fixed prompt
def build_semantic_cluster_map(
    question: str,
    answers: list[str],
    deadline: float | None = None,
) -> dict[str, str]:
    uniq_answers: list[str] = []
    seen = set()

    # Remove trivial identical strings
    for a in answers:
        if deadline is not None:
            check_deadline(deadline, "semantic cluster preprocessing")

        na = normalize_text(a)
        if na and na not in seen:
            seen.add(na)
            uniq_answers.append(a)

    if not uniq_answers:
        return {}

    # Merge answers if equivalent
    dsu = DSU(len(uniq_answers))

    for i in range(len(uniq_answers)):
        if deadline is not None:
            check_deadline(deadline, f"semantic cluster outer loop i={i}")

        for j in range(i + 1, len(uniq_answers)):
            if deadline is not None:
                check_deadline(deadline, f"semantic cluster pair ({i},{j})")

            if same_meaning_under_question(question, uniq_answers[i], uniq_answers[j]):
                dsu.union(i, j)

    root_to_label: dict[int, str] = {}
    cluster_map: dict[str, str] = {}
    next_id = 0

    for i, a in enumerate(uniq_answers):
        if deadline is not None:
            check_deadline(deadline, "semantic cluster label assignment")

        root = dsu.find(i)
        if root not in root_to_label:
            root_to_label[root] = f"__cluster_{next_id}__"
            next_id += 1
        cluster_map[normalize_text(a)] = root_to_label[root]

    return cluster_map


# ----------
# Prompt construction
# ----------
def build_standalone_prompt(question: str) -> str:
    return (
        "Answer the factual question.\n\n"
        f"Question: {question}\n\n"
        "Return only a short factual answer.\n"
        "No explanation.\n"
        "No full sentence."
    )


# ----------
# Generation
# ----------
# Sample n short answers for every prompt
def generate_many(
    question: str,
    n: int,
    generation_seed: int | None = None,
) -> list[str]:
    prompt = build_standalone_prompt(question)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(INPUT_DEVICE)

    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=n,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    texts = []
    for i in range(outputs.shape[0]):
        generated_ids = outputs[i][prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        texts.append(text)

    return texts


# ----------
# Dataset loading/grouping
# ----------
# Group questions in datasets together with three paraphrases
def load_popqa_tp_records():
    ds_obj = load_dataset(DATASET_NAME)
    if isinstance(ds_obj, dict):
        split_name = next(iter(ds_obj.keys()))
        ds = ds_obj[split_name]
    else:
        ds = ds_obj

    by_id = defaultdict(list)
    for row in ds:
        by_id[int(row["id"])].append(row)

    grouped = []
    for qid, rows in by_id.items():
        rows = sorted(rows, key=lambda r: int(r["template_id"]))

        originals = [r for r in rows if int(r["template_id"]) == 0]
        if not originals:
            continue

        original = originals[0]["paraphrase"]
        gold_answers_raw = originals[0]["possible_answers"]

        if isinstance(gold_answers_raw, str):
            try:
                gold_answers = json.loads(gold_answers_raw)
            except json.JSONDecodeError:
                gold_answers = [gold_answers_raw]
        else:
            gold_answers = gold_answers_raw

        paraphrases = [r["paraphrase"] for r in rows if int(r["template_id"]) != 0]
        if len(paraphrases) < (N_PARAPHRASES_TOTAL - 1):
            continue

        questions = [original] + paraphrases[: N_PARAPHRASES_TOTAL - 1]

        grouped.append({
            "id": qid,
            "questions": questions,
            "possible_answers": gold_answers,
        })

    grouped = sorted(grouped, key=lambda x: x["id"])
    return grouped


# Strict setting filters out categories that behaved unreliably
def is_clean_question(question: str) -> bool:
    q = question.strip().lower()

    keep = (
        ("capital of" in q) or
        ("capital city of" in q) or
        ("in what city was" in q) or
        ("what city was" in q) or
        ("what city is" in q) or
        ("hometown" in q) or
        ("what country is" in q) or
        ("in what country is" in q) or
        ("located in" in q) or
        ("where is " in q) or
        ("what sport does" in q) or
        ("plays what sport" in q)
    )
    if not keep:
        return False

    banned_substrings = [
        "composer of",
        "composed ",
        "director of",
        "directed ",
        "producer of",
        "produced ",
        "author of",
        "wrote ",
        "screenwriter",
        "father of",
        "dad of",
        "child of",
        "genre is",
        "genre does",
        "type of work",
        "fans of what genre",
    ]
    if any(b in q for b in banned_substrings):
        return False

    return True


# Filter out corrupted text/encoding artefacts
MOJIBAKE_MARKERS = [
    "�", "√", "∂", "Ã", "Â", "â€™", "â€œ", "â€", "â€“", "â€”", "¤"
]

def has_mojibake(text: str) -> bool:
    s = text.strip()
    if not s:
        return False

    if any(mark in s for mark in MOJIBAKE_MARKERS):
        return True

    weird = sum(ord(ch) > 127 for ch in s)
    if weird >= 2 and any(ch in s for ch in ["√", "∂", "Ã", "Â", "�"]):
        return True

    return False


# Filter out reformulations that were empirically not meaning-preserving
def is_clean_group(questions: list[str]) -> bool:
    lowers = [q.strip().lower() for q in questions]

    if any(q.startswith("where is ") for q in lowers):
        return False
    if any(re.match(r"what does .* play\??$", q) for q in lowers):
        return False

    if any(has_mojibake(q) for q in questions):
        return False

    return True


# ----------
# Experiment
# ----------
def run_one_example(
    example: dict,
    base_generation_seed: int,
    timeout_seconds: float = 480.0,
) -> dict:
    start_time = time.perf_counter()
    deadline = start_time + timeout_seconds

    qid = example["id"]
    questions = example["questions"]
    gold_answers = example["possible_answers"]

    transform_results = []
    all_ordinary_answers = []

    # Store raw outputs, postpone semantic bucketing until full answer pool generated
    for j, question in enumerate(questions, start=1):
        check_deadline(deadline, f"reformulation {j} start")
        batch_results = []

        for r in range(N_BATCHES_PER_TRANSFORM):
            check_deadline(deadline, f"reformulation {j}, batch {r+1} before generation")

            batch_seed = derived_generation_seed(
                base_seed=base_generation_seed,
                question_id=qid,
                reformulation_index=j,
                batch_index=r + 1,
            )

            preds = generate_many(
                question,
                SAMPLES_PER_BATCH,
                generation_seed=batch_seed,
            )

            check_deadline(deadline, f"reformulation {j}, batch {r+1} after generation")

            extracted = [extract_short_answer(p) for p in preds]
            specials = [special_bucket(x) for x in extracted]

            for x, sb in zip(extracted, specials):
                if sb is None:
                    all_ordinary_answers.append(x)

            batch_results.append({
                "batch_index": r + 1,
                "predictions": preds,
                "extracted_predictions": extracted,
                "special_buckets": specials,
            })

        transform_results.append({
            "reformulation_index": j,
            "question": question,
            "batch_results": batch_results,
        })

    check_deadline(deadline, "before semantic clustering")

    # Cluster all ordinary answers
    reference_question = questions[0]
    cluster_map = build_semantic_cluster_map(
        reference_question,
        all_ordinary_answers,
        deadline=deadline,
    )

    # Convert stored answers into semantic buckets
    for t_idx, t in enumerate(transform_results, start=1):
        check_deadline(deadline, f"second pass reformulation {t_idx} start")

        batch_dists = []
        all_correct_flags = []

        for b_idx, b in enumerate(t["batch_results"], start=1):
            check_deadline(deadline, f"second pass reformulation {t_idx}, batch {b_idx}")

            final_buckets = []

            for extracted_pred, sb in zip(b["extracted_predictions"], b["special_buckets"]):
                if sb is not None:
                    bucket = sb
                else:
                    norm = normalize_text(extracted_pred)
                    bucket = cluster_map.get(norm, f"__unclustered__::{norm}")

                final_buckets.append(bucket)
                all_correct_flags.append(is_correct_prediction(extracted_pred, gold_answers))

            # Compute reformulation empirical distribution
            dist = empirical_distribution(final_buckets)

            b["buckets"] = final_buckets
            b["distribution"] = dist
            del b["special_buckets"]

            batch_dists.append(dist)

        mean_dist = average_distributions(batch_dists)
        within_var = mean_pairwise_js(batch_dists)
        transform_entropy = entropy_of_dist(mean_dist)
        mean_correctness = sum(all_correct_flags) / len(all_correct_flags)

        t["mean_distribution"] = mean_dist
        t["within_variability"] = within_var
        t["semantic_entropy"] = transform_entropy
        t["mean_correctness_diagnostic"] = mean_correctness

    check_deadline(deadline, "final aggregation")

    mean_dists = [t["mean_distribution"] for t in transform_results]
    within_vals = [t["within_variability"] for t in transform_results]

    W_q = sum(within_vals) / len(within_vals) if within_vals else 0.0

    # B_q, W_q and E_q
    between_vals = []
    for i in range(len(transform_results)):
        check_deadline(deadline, f"between-transform outer loop i={i}")

        for j in range(i + 1, len(transform_results)):
            check_deadline(deadline, f"between-transform pair ({i},{j})")

            di = transform_results[i]["batch_results"]
            dj = transform_results[j]["batch_results"]

            cross = []
            for bi in di:
                for bj in dj:
                    cross.append(js_divergence(bi["distribution"], bj["distribution"]))

            if cross:
                between_vals.append(sum(cross) / len(cross))

    B_q = sum(between_vals) / len(between_vals) if between_vals else 0.0
    E_q = B_q - W_q

    pairwise_instability = mean_pairwise_js(mean_dists)
    aggregated_dist = average_distributions(mean_dists)
    aggregated_entropy = entropy_of_dist(aggregated_dist)

    mean_correctness_overall = (
        sum(t["mean_correctness_diagnostic"] for t in transform_results) / len(transform_results)
        if transform_results else 0.0
    )

    elapsed = time.perf_counter() - start_time

    return {
        "id": qid,
        "questions": questions,
        "possible_answers": gold_answers,
        "semantic_cluster_map": cluster_map,
        "transform_results": transform_results,
        "within_variability": W_q,
        "between_variability": B_q,
        "excess_transformation_instability": E_q,
        "pairwise_instability": pairwise_instability,
        "aggregated_distribution": aggregated_dist,
        "aggregated_entropy": aggregated_entropy,
        "mean_correctness_diagnostic": mean_correctness_overall,
        "elapsed_seconds": elapsed,
    }


# Convert semantic buckets to empirical distribution
def empirical_distribution(buckets: list[str]) -> dict[str, float]:
    counter = Counter(buckets)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


# Average sparse distribution
def average_distributions(dists: list[dict[str, float]]) -> dict[str, float]:
    keys = set()
    for d in dists:
        keys.update(d.keys())

    out = {}
    for k in keys:
        out[k] = sum(d.get(k, 0.0) for d in dists) / len(dists)
    return out


def entropy_of_dist(dist: dict[str, float]) -> float:
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())

    def kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0 and bk > 0:
                s += ak * math.log(ak / bk)
        return s

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_pairwise_js(dists: list[dict[str, float]]) -> float:
    if len(dists) < 2:
        return 0.0
    vals = [js_divergence(a, b) for a, b in combinations(dists, 2)]
    return sum(vals) / len(vals)


# Avoid unstable sign changes from floating point noise
def eq_sign_bucket(e: float, eps: float = 1e-12) -> str:
    if e > eps:
        return "positive"
    if e < -eps:
        return "negative"
    return "zero"


def correctness_bucket(x: float, eps: float = 1e-12) -> str:
    if x >= 1.0 - eps:
        return "all_correct"
    if x <= eps:
        return "all_incorrect"
    return "mixed"


def build_eq_correctness_table(results: list[dict]) -> dict:
    table = {
        "positive": {"all_correct": 0, "mixed": 0, "all_incorrect": 0},
        "zero": {"all_correct": 0, "mixed": 0, "all_incorrect": 0},
        "negative": {"all_correct": 0, "mixed": 0, "all_incorrect": 0},
    }

    for r in results:
        e_bucket = eq_sign_bucket(r["excess_transformation_instability"])
        c_bucket = correctness_bucket(r["mean_correctness_diagnostic"])
        table[e_bucket][c_bucket] += 1

    return table


# Run one experiment for a selection seed
def run_single_generation_seed(
    filtered: list[dict],
    generation_seed: int,
) -> dict:
    selection_rng = random.Random(SELECTION_SEED)

    candidate_pool = filtered[:]
    selection_rng.shuffle(candidate_pool)

    queue = candidate_pool[:N_QUESTIONS]
    next_pool_idx = N_QUESTIONS

    results = []
    skipped_timeout = []
    attempted = 0

    print("\n===================================")
    print(
        f"Run with selection_seed={SELECTION_SEED}, "
        f"generation_seed={generation_seed}"
    )
    print("===================================\n")

    while len(results) < N_QUESTIONS:
        if not queue:
            raise RuntimeError(
                "Queue became empty before enough completed questions were collected."
            )

        example = queue.pop(0)
        attempted += 1

        qid = example["id"]
        qtext = example["questions"][0]
        print(
            f"[Attempt {attempted}] Running id={qid} | "
            f"completed={len(results)}/{N_QUESTIONS}"
        )
        print(f"Question: {qtext}")

        try:
            res = run_one_example(
                example,
                base_generation_seed=generation_seed,
                timeout_seconds=480.0,
            )
            results.append(res)
            print(
                f"  -> finished in {res['elapsed_seconds']:.2f}s | "
                f"E_q={res['excess_transformation_instability']:.3f} | "
                f"correct_diag={res['mean_correctness_diagnostic']:.3f}"
            )

        # If times out, replace with next question
        except QuestionTimeout as e:
            print(f"  -> TIMEOUT, skipped for now: {e}")
            skipped_timeout.append({
                "id": qid,
                "question": qtext,
                "reason": str(e),
            })

            if next_pool_idx >= len(candidate_pool):
                raise RuntimeError(
                    "Ran out of backup examples after timeouts. "
                    "Increase the pool size or reduce timeout pressure."
                )

            replacement = candidate_pool[next_pool_idx]
            next_pool_idx += 1
            queue.append(replacement)

            print(f"  -> appended replacement id={replacement['id']} to queue tail")

        except Exception as e:
            print(f"  -> ERROR on id={qid}: {e}")
            raise

        print()

    avg_elapsed = sum(r["elapsed_seconds"] for r in results) / len(results)
    total_elapsed = sum(r["elapsed_seconds"] for r in results)

    avg_within_variability = sum(r["within_variability"] for r in results) / len(results)
    avg_between_variability = sum(r["between_variability"] for r in results) / len(results)
    avg_excess_instability = sum(r["excess_transformation_instability"] for r in results) / len(results)
    avg_pairwise_instability = sum(r["pairwise_instability"] for r in results) / len(results)
    avg_aggregated_entropy = sum(r["aggregated_entropy"] for r in results) / len(results)
    mean_correctness_diagnostic = sum(r["mean_correctness_diagnostic"] for r in results) / len(results)

    n_positive_excess = sum(
        r["excess_transformation_instability"] > 1e-12 for r in results
    )
    n_zero_excess = sum(
        abs(r["excess_transformation_instability"]) <= 1e-12 for r in results
    )
    n_negative_excess = len(results) - n_positive_excess - n_zero_excess

    frac_positive_excess = n_positive_excess / len(results)
    frac_zero_excess = n_zero_excess / len(results)
    frac_negative_excess = n_negative_excess / len(results)

    # Small representative list for quick inspection
    eq_correctness_table = build_eq_correctness_table(results)

    top_excess_examples = sorted(
        [
            {
                "id": r["id"],
                "question": r["questions"][0],
                "excess_transformation_instability": r["excess_transformation_instability"],
                "mean_correctness_diagnostic": r["mean_correctness_diagnostic"],
            }
            for r in results
        ],
        key=lambda x: x["excess_transformation_instability"],
        reverse=True,
    )[:5]

    return {
        "selection_seed": SELECTION_SEED,
        "generation_seed": generation_seed,
        "n_questions": len(results),
        "n_paraphrases_total": N_PARAPHRASES_TOTAL,
        "samples_per_batch": SAMPLES_PER_BATCH,
        "n_batches_per_transform": N_BATCHES_PER_TRANSFORM,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timeout_seconds_per_question": 480.0,
        "n_timeout_skips": len(skipped_timeout),
        "timeout_skips": skipped_timeout,
        "avg_within_variability": avg_within_variability,
        "avg_between_variability": avg_between_variability,
        "avg_excess_instability": avg_excess_instability,
        "avg_pairwise_instability": avg_pairwise_instability,
        "avg_aggregated_entropy": avg_aggregated_entropy,
        "mean_correctness_diagnostic": mean_correctness_diagnostic,
        "n_positive_excess": n_positive_excess,
        "n_zero_excess": n_zero_excess,
        "n_negative_excess": n_negative_excess,
        "frac_positive_excess": frac_positive_excess,
        "frac_zero_excess": frac_zero_excess,
        "frac_negative_excess": frac_negative_excess,
        "eq_correctness_table": eq_correctness_table,
        "top_excess_examples": top_excess_examples,
        "avg_elapsed_seconds": avg_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "results": results,
    }


# -----------
# Main
# ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grouped = load_popqa_tp_records()

    filtered = [
        g for g in grouped
        if is_clean_question(g["questions"][0]) and is_clean_group(g["questions"])
    ]

    print(f"Usable groups after filtering: {len(filtered)}")

    if len(filtered) < N_QUESTIONS:
        raise ValueError(
            f"Need at least {N_QUESTIONS} usable examples, but only found {len(filtered)}."
        )

    all_runs = []
    print("Beginning Path 1 multi-seed run...\n")

    for generation_seed in GENERATION_SEEDS:
        run_summary = run_single_generation_seed(
            filtered,
            generation_seed=generation_seed,
        )
        all_runs.append(run_summary)

    summary = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "selection_seed": SELECTION_SEED,
        "generation_seeds": GENERATION_SEEDS,
        "n_runs": len(all_runs),
        "n_questions_target": N_QUESTIONS,
        "n_paraphrases_total": N_PARAPHRASES_TOTAL,
        "samples_per_batch": SAMPLES_PER_BATCH,
        "n_batches_per_transform": N_BATCHES_PER_TRANSFORM,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "runs": all_runs,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===================================")
    print("Finished PopQA Path 1 multi-seed instability experiment.")
    print("===================================")


if __name__ == "__main__":
    main()