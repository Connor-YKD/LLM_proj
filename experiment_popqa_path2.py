"""
1. Load the model and filter the dataset 
2. Construct finite reformulation trees
3. At each node:
   1) replay the prefix
   2) sample answers under the current prompt history
   3) choose and write a representative answer back into history
4. Compute summary quantities
5. Save outputs to JSON
"""

import os
import re
import json
import time
import random
import string
import traceback
from itertools import permutations, product
from collections import defaultdict, Counter

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
SAMPLES_PER_NODE = 4
TEMPERATURE = 0.7
TOP_P = 0.95

SELECTION_SEEDS = [100]
BASE_GENERATION_SEEDS = [256, 3, 9, 27, 81]

OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "popqa_tp_path2.json")


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

    # Convert connected components into cluster labels
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
# Store history record for prompt history
def make_history_record(step_idx: int, question: str, answer: str) -> dict:
    return {
        "step": step_idx,
        "question": question,
        "answer": extract_short_answer(answer),
    }


# Build current prompt from prior history record with next reformulation
def build_path2_prompt(history_records: list[dict], current_question: str) -> str:
    if history_records:
        history_json = json.dumps(history_records, ensure_ascii=False, indent=2)
    else:
        history_json = "[]"

    return (
        "You are answering repeated meaning-preserving reformulations of the same factual question.\n"
        "Use the previous question-answer pairs as context, but answer the current question directly.\n\n"
        "Previous question-answer pairs:\n"
        f"{history_json}\n\n"
        "Current question:\n"
        f"{current_question}\n\n"
        "Return only a short factual answer.\n"
        "No explanation.\n"
        "No full sentence."
    )


# ----------
# Generation
# ----------
# Sample n short answers from current prompt using given seed
def sample_many(prompt: str, n: int, generation_seed: int | None = None) -> list[str]:
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

    # Draw multiple stochastic completion from current prompt
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
    predictions = []
    for i in range(outputs.shape[0]):
        generated_ids = outputs[i][prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(text)

    return predictions


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
# Tree utilities
# ----------
# Choose representative answer written back into prompt history based on largest semantic bucket
def choose_history_answer(
    question: str,
    extracted_predictions: list[str],
    deadline: float | None = None,
) -> str:
    cleaned = [extract_short_answer(x) for x in extracted_predictions]
    cleaned = [x for x in cleaned if x.strip()]

    if not cleaned:
        return ""

    # Separate ordinary answer from special buckets
    ordinary_answers = []
    special_examples = defaultdict(list)

    for ans in cleaned:
        sb = special_bucket(ans)
        if sb is None:
            ordinary_answers.append(ans)
        else:
            special_examples[sb].append(ans)

    candidate_counts = Counter()
    representative = {}

    for sb, vals in special_examples.items():
        candidate_counts[sb] = len(vals)
        if sb == "__abstain__":
            representative[sb] = "I don't know"
        elif sb == "__invalid__":
            representative[sb] = ""
        else:
            representative[sb] = vals[0]

    # For ordinary answers, choose representative of largest semantic clusters
    if ordinary_answers:
        unique_norms = {normalize_text(a) for a in ordinary_answers if normalize_text(a)}
        if len(unique_norms) == 1:
            only_ans = ordinary_answers[0]
            candidate_counts["__ordinary_single_cluster__"] = len(ordinary_answers)
            representative["__ordinary_single_cluster__"] = only_ans
        else:
            cluster_map = build_semantic_cluster_map(
                question,
                ordinary_answers,
                deadline=deadline,
            )

            cluster_members = defaultdict(list)
            for ans in ordinary_answers:
                norm = normalize_text(ans)
                label = cluster_map.get(norm, f"__unclustered__::{norm}")
                cluster_members[label].append(ans)

            for label, vals in cluster_members.items():
                candidate_counts[label] = len(vals)
                representative[label] = vals[0]

    if not candidate_counts:
        return ""

    # Write back the representative to stablize history
    best_label, _ = max(candidate_counts.items(), key=lambda kv: kv[1])
    return representative[best_label]


# Seed for current tree prefix
def prefix_generation_seed(base_seed: int, prefix: tuple[int, ...]) -> int:
    seed = base_seed
    for i, idx in enumerate(prefix, start=1):
        seed = (seed * 1000003 + 97 * i + idx + 1) % (2**31 - 1)
    return seed


# Enumerate all admissible reformulation prefixes
def build_tree_nodes(n_questions: int) -> list[tuple[int, ...]]:
    assert n_questions >= 2
    remaining = list(range(1, n_questions))

    nodes = {(0,)}
    for depth in range(1, len(remaining) + 1):
        for perm in permutations(remaining, depth):
            nodes.add((0,) + perm)

    nodes = sorted(nodes, key=lambda x: (len(x), x))
    return nodes


# All extensions with length one step without resuing reformulations
def child_nodes(node: tuple[int, ...], n_questions: int) -> list[tuple[int, ...]]:
    used = set(node)
    remaining = [i for i in range(1, n_questions) if i not in used]
    return [node + (r,) for r in remaining]


def node_questions(node: tuple[int, ...], questions: list[str]) -> list[str]:
    return [questions[i] for i in node]


# Replay one tree node and estimate the score
def estimate_score_for_node(
    node: tuple[int, ...],
    questions: list[str],
    gold_answers,
    base_generation_seed: int,
    deadline: float | None = None,
) -> dict:
    qs = node_questions(node, questions)

    # Replay prefixes sequentially
    history_records = []
    step_trace = []

    final_prompt = None
    final_predictions = None
    final_extracted = None
    final_correct_flags = None
    final_score = None

    for local_step, q_idx in enumerate(node, start=1):
        if deadline is not None:
            check_deadline(deadline, f"node {node}, local_step {local_step} start")

        current_question = questions[q_idx]
        # Build history-conditioned prompt
        prompt = build_path2_prompt(history_records, current_question)

        if deadline is not None:
            check_deadline(deadline, f"node {node}, local_step {local_step} before sampling")

        gen_seed = prefix_generation_seed(base_generation_seed, node[:local_step])
        predictions = sample_many(
            prompt,
            SAMPLES_PER_NODE,
            generation_seed=gen_seed,
        )

        if deadline is not None:
            check_deadline(deadline, f"node {node}, local_step {local_step} after sampling")

        extracted = [extract_short_answer(p) for p in predictions]
        correct_flags = [is_correct_prediction(p, gold_answers) for p in extracted]
        score = sum(correct_flags) / len(correct_flags)

        step_trace.append({
            "local_step": local_step,
            "question_index": q_idx,
            "current_question": current_question,
            "history_before_step": history_records.copy(),
            "prompt": prompt,
            "predictions": predictions,
            "extracted_predictions": extracted,
            "correct_flags": correct_flags,
            "score": score,
            "generation_seed": gen_seed,
        })

        # For nonterminal steps, write answer back to history
        if local_step < len(node):
            if deadline is not None:
                check_deadline(deadline, f"node {node}, local_step {local_step} before history update")

            history_answer = choose_history_answer(
                current_question,
                extracted,
                deadline=deadline,
            )
            history_records.append(
                make_history_record(
                    step_idx=local_step,
                    question=current_question,
                    answer=history_answer,
                )
            )
        else:
            final_prompt = prompt
            final_predictions = predictions
            final_extracted = extracted
            final_correct_flags = correct_flags
            final_score = score

    return {
        "node": list(node),
        "depth": len(node),
        "questions_so_far": qs,
        "history_records_used": history_records,
        "step_trace": step_trace,
        "prompt": final_prompt,
        "predictions": final_predictions,
        "extracted_predictions": final_extracted,
        "correct_flags": final_correct_flags,
        "score": final_score,
    }


# ----------
# Per-example experiment
# ----------
# Run full tree for a question group
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

    nodes = build_tree_nodes(len(questions))

    node_results = {}
    for node in nodes:
        check_deadline(deadline, f"before node {node}")

        node_results[node] = estimate_score_for_node(
            node,
            questions,
            gold_answers,
            base_generation_seed=base_generation_seed,
            deadline=deadline,
        )

        check_deadline(deadline, f"after node {node}")

    # Compare nonterminal node with average of its children
    comparisons = []
    for node in nodes:
        children = child_nodes(node, len(questions))
        if not children:
            continue

        current_score = node_results[node]["score"]
        child_scores = [node_results[ch]["score"] for ch in children]
        child_avg = sum(child_scores) / len(child_scores)
        delta = child_avg - current_score

        comparisons.append({
            "node": list(node),
            "depth": len(node),
            "current_score": current_score,
            "child_nodes": [list(ch) for ch in children],
            "child_scores": child_scores,
            "child_average_score": child_avg,
            "delta": delta,
            "nonnegative": delta >= 0,
        })

    # Nodewise score movement
    avg_delta = sum(c["delta"] for c in comparisons) / len(comparisons)
    frac_nonnegative = sum(c["nonnegative"] for c in comparisons) / len(comparisons)

    depth_summary = {}
    for d in sorted(set(c["depth"] for c in comparisons)):
        comps_d = [c for c in comparisons if c["depth"] == d]
        depth_summary[d] = {
            "n_nodes": len(comps_d),
            "avg_delta": sum(c["delta"] for c in comps_d) / len(comps_d),
            "frac_nonnegative": sum(c["nonnegative"] for c in comps_d) / len(comps_d),
        }

    full_nodes = [node for node in nodes if len(node) == len(questions)]
    initial_score = node_results[(0,)]["score"]
    final_scores = [node_results[node]["score"] for node in full_nodes]
    avg_final_score = sum(final_scores) / len(final_scores)
    elapsed = time.perf_counter() - start_time

    return {
        "id": qid,
        "questions": questions,
        "possible_answers": gold_answers,
        "node_results": {str(k): v for k, v in node_results.items()},
        "comparisons": comparisons,
        "avg_delta": avg_delta,
        "frac_nonnegative": frac_nonnegative,
        "depth_summary": depth_summary,
        "initial_score": initial_score,
        "avg_final_score": avg_final_score,
        "avg_path_gain": avg_final_score - initial_score,
        "elapsed_seconds": elapsed,
    }


# -----------
# Main
# -----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grouped = load_popqa_tp_records()

    dedup = {}
    for item in grouped:
        key = item["questions"][0].strip().lower()
        if key not in dedup:
            dedup[key] = item
    grouped = list(dedup.values())

    filtered = [
        g for g in grouped
        if is_clean_question(g["questions"][0]) and is_clean_group(g["questions"])
    ]
    print(f"After filtering: {len(filtered)} grouped questions.")

    if len(filtered) < N_QUESTIONS:
        raise ValueError(
            f"Need at least {N_QUESTIONS} usable examples, but only found {len(filtered)}."
        )

    all_runs = []

    # Loop over selection seed with generation seed
    for selection_seed, base_generation_seed in product(SELECTION_SEEDS, BASE_GENERATION_SEEDS):
        print("\n===================================")
        print(
            f"Run with selection_seed={selection_seed}, "
            f"base_generation_seed={base_generation_seed}"
        )
        print("===================================")

        selection_rng = random.Random(selection_seed)
        selected_grouped = selection_rng.sample(
            filtered,
            k=min(N_QUESTIONS, len(filtered))
        )

        skipped_timeout = []
        results = []

        for i, ex in enumerate(selected_grouped, start=1):
            print(f"[{i}/{len(selected_grouped)}] id={ex['id']}", flush=True)
            try:
                result = run_one_example(
                    ex,
                    base_generation_seed=base_generation_seed,
                    timeout_seconds=480.0,
                )
                results.append(result)
                print(f"   finished in {result['elapsed_seconds']:.2f}s", flush=True)

            except QuestionTimeout as e:
                print(f"   TIMEOUT on id={ex['id']}: {e}", flush=True)
                skipped_timeout.append({
                    "id": ex["id"],
                    "question": ex["questions"][0],
                    "reason": str(e),
                })
                continue

            except Exception:
                print(f">> Python exception on id={ex['id']}", flush=True)
                traceback.print_exc()
                raise

        if not results:
            print("No completed results in this run.")
            continue

        overall_avg_delta = sum(r["avg_delta"] for r in results) / len(results)
        overall_frac_nonnegative = sum(r["frac_nonnegative"] for r in results) / len(results)
        overall_initial_score = sum(r["initial_score"] for r in results) / len(results)
        overall_avg_final_score = sum(r["avg_final_score"] for r in results) / len(results)
        overall_avg_path_gain = sum(r["avg_path_gain"] for r in results) / len(results)

        total_elapsed = sum(r["elapsed_seconds"] for r in results)
        avg_elapsed = total_elapsed / len(results)

        pooled_depth = defaultdict(list)
        for r in results:
            for d, stats in r["depth_summary"].items():
                pooled_depth[int(d)].append(stats)

        pooled_depth_summary = {}
        for d, stats_list in pooled_depth.items():
            pooled_depth_summary[d] = {
                "avg_delta": sum(s["avg_delta"] for s in stats_list) / len(stats_list),
                "frac_nonnegative": sum(s["frac_nonnegative"] for s in stats_list) / len(stats_list),
            }

        run_summary = {
            "selection_seed": selection_seed,
            "base_generation_seed": base_generation_seed,
            "n_questions": len(results),
            "n_timeout_skips": len(skipped_timeout),
            "timeout_skips": skipped_timeout,
            "overall_initial_score": overall_initial_score,
            "overall_avg_final_score": overall_avg_final_score,
            "overall_avg_path_gain": overall_avg_path_gain,
            "overall_avg_delta": overall_avg_delta,
            "overall_frac_nonnegative": overall_frac_nonnegative,
            "pooled_depth_summary": pooled_depth_summary,
            "avg_elapsed_seconds": avg_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "results": results,
        }

        all_runs.append(run_summary)

        print("Run summary:")
        print(f"  overall initial score:    {overall_initial_score:.3f}")
        print(f"  overall avg final score:  {overall_avg_final_score:.3f}")
        print(f"  overall avg path gain:    {overall_avg_path_gain:.3f}")
        print(f"  overall avg delta:        {overall_avg_delta:.3f}")
        print(f"  frac nonnegative:         {overall_frac_nonnegative:.3f}")
        print(f"  avg time/question:        {avg_elapsed:.2f}s")

    # Final multi-seed experiment summary
    summary = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "n_runs": len(all_runs),
        "selection_seeds": SELECTION_SEEDS,
        "base_generation_seeds": BASE_GENERATION_SEEDS,
        "n_paraphrases_total": N_PARAPHRASES_TOTAL,
        "samples_per_node": SAMPLES_PER_NODE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "runs": all_runs,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===================================")
    print("Finished all multi-seed tree-based runs.")
    print("===================================")


if __name__ == "__main__":
    main()
