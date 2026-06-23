#!/usr/bin/env python3
"""Fixed greedy generation evaluation for Phase-2 soft-prefix checkpoints.

This script is intentionally schema-aware for the FG26/MUSE non-3D tasks.
It reports JSON parse rate separately from semantic task metrics, so a model
cannot look good only because teacher-forcing loss decreased.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_phase2_connector_only_generate import EvalDataset, collate  # noqa: E402
from train_phase2_qformer_sft_ddp import Phase2QFormerConnector, prune_qformer_text_modules  # noqa: E402
from train_phase2_qwen_softprefix_smoke import try_import_torch_npu  # noqa: E402


try:
    from rdkit import Chem  # type: ignore
except Exception:  # noqa: BLE001
    Chem = None


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extract_json_obj(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty"
    candidates = [stripped]
    if stripped.startswith("```"):
        unfenced = stripped.strip("`").strip()
        if unfenced.lower().startswith("json"):
            unfenced = unfenced[4:].strip()
        candidates.insert(0, unfenced)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, stripped[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        if isinstance(obj, dict):
            return obj, None
        last_error = f"json root is {type(obj).__name__}"
    return None, locals().get("last_error", "no json object")


def norm_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def norm_id(value: Any) -> str | None:
    text = norm_text(value)
    return text.upper() if text and len(text) == 1 else text


def first_value(obj: Any, keys: list[str]) -> Any:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMBER_RE.search(value)
        if match:
            return float(match.group(0))
    return None


def as_number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        parsed = as_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def canonical_smiles(smiles: str | None) -> str | None:
    if smiles is None or Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def rmse(errors: list[float]) -> float:
    if not errors:
        return math.nan
    return math.sqrt(sum(x * x for x in errors) / len(errors))


def score_prediction(task: str, pred: dict[str, Any] | None, gold: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {"parse_ok": pred is not None}
    if pred is None or gold is None:
        metrics["semantic_ok"] = False
        return metrics

    if "formula" in task and "scalar" not in task and "rank" not in task and "verify" not in task:
        pred_formula = norm_text(first_value(pred, ["formula", "molecular_formula", "answer"]))
        gold_formula = norm_text(first_value(gold, ["formula", "molecular_formula", "answer"]))
        metrics["formula_exact"] = pred_formula == gold_formula
        metrics["semantic_ok"] = bool(metrics["formula_exact"])
        return metrics

    if "smiles" in task and "rank" not in task and "verify" not in task:
        pred_smiles = norm_text(first_value(pred, ["smiles", "canonical_smiles", "answer"]))
        gold_smiles = norm_text(first_value(gold, ["smiles", "canonical_smiles", "answer"]))
        metrics["smiles_raw_exact"] = pred_smiles == gold_smiles
        pred_can = canonical_smiles(pred_smiles)
        gold_can = canonical_smiles(gold_smiles)
        if pred_can is not None and gold_can is not None:
            metrics["smiles_canonical_exact"] = pred_can == gold_can
            metrics["semantic_ok"] = bool(metrics["smiles_canonical_exact"])
        else:
            metrics["smiles_canonical_exact"] = None
            metrics["semantic_ok"] = bool(metrics["smiles_raw_exact"])
        return metrics

    if "scalar" in task:
        pred_value = as_float(first_value(pred, ["value", "answer"]))
        gold_value = as_float(first_value(gold, ["value", "answer"]))
        pred_property = norm_text(first_value(pred, ["property"]))
        gold_property = norm_text(first_value(gold, ["property"]))
        pred_unit = norm_text(first_value(pred, ["unit"]))
        gold_unit = norm_text(first_value(gold, ["unit"]))
        if pred_value is not None and gold_value is not None:
            err = abs(pred_value - gold_value)
            metrics["abs_error"] = err
            metrics["squared_error"] = err * err
        metrics["property_exact"] = pred_property == gold_property
        metrics["unit_exact"] = pred_unit == gold_unit
        metrics["semantic_ok"] = pred_value is not None and metrics["property_exact"] and metrics["unit_exact"]
        return metrics

    if task in {"spectrum_to_ir_peaks", "spectrum_to_raman_peaks", "spectrum_to_uv_peaks"}:
        key = "peaks_eV" if task.endswith("uv_peaks") else "peaks_cm-1"
        tol = 0.20 if task.endswith("uv_peaks") else 40.0
        pred_peaks = sorted(as_number_list(pred.get(key)))
        gold_peaks = sorted(as_number_list(gold.get(key)))
        pred_n = as_float(pred.get("n_peaks"))
        gold_n = as_float(gold.get("n_peaks"))
        metrics["modality_exact"] = norm_text(pred.get("modality")) == norm_text(gold.get("modality"))
        metrics["unit_exact"] = norm_text(pred.get("unit")) == norm_text(gold.get("unit"))
        metrics["count_exact"] = int(pred_n) == int(gold_n) if pred_n is not None and gold_n is not None else False
        paired = list(zip(pred_peaks, gold_peaks))
        errors = [abs(a - b) for a, b in paired]
        if errors:
            metrics["peak_mae"] = sum(errors) / len(errors)
            metrics["peak_rmse"] = rmse(errors)
            metrics["peak_within_tol_rate"] = sum(err <= tol for err in errors) / len(errors)
        metrics["semantic_ok"] = (
            metrics["modality_exact"]
            and metrics["unit_exact"]
            and metrics["count_exact"]
            and bool(errors)
            and all(err <= tol for err in errors)
        )
        return metrics

    if task.startswith("spec_mol_verify"):
        pred_status = norm_text(first_value(pred, ["status", "answer"]))
        gold_status = norm_text(first_value(gold, ["status", "answer"]))
        metrics["status_exact"] = pred_status == gold_status
        metrics["semantic_ok"] = bool(metrics["status_exact"])
        return metrics

    if task == "spec_mol_rank_k4":
        pred_selected = norm_id(first_value(pred, ["selected_id", "selected", "answer"]))
        gold_selected = norm_id(first_value(gold, ["selected_id", "selected", "answer"]))
        pred_ranked = [norm_id(x) for x in pred.get("ranked_ids", [])] if isinstance(pred.get("ranked_ids"), list) else []
        gold_ranked = [norm_id(x) for x in gold.get("ranked_ids", [])] if isinstance(gold.get("ranked_ids"), list) else []
        metrics["selected_id_exact"] = pred_selected == gold_selected
        metrics["ranked_ids_exact"] = pred_ranked == gold_ranked
        metrics["semantic_ok"] = bool(metrics["selected_id_exact"])
        return metrics

    if task == "relative_lower_electronic_energy_same_formula":
        pred_selected = norm_id(first_value(pred, ["selected_id", "winner_id", "answer"]))
        gold_selected = norm_id(first_value(gold, ["selected_id", "winner_id", "answer"]))
        pred_property = norm_text(first_value(pred, ["property"]))
        gold_property = norm_text(first_value(gold, ["property"]))
        metrics["selected_id_exact"] = pred_selected == gold_selected
        metrics["property_exact"] = pred_property == gold_property
        metrics["semantic_ok"] = bool(metrics["selected_id_exact"])
        return metrics

    metrics["semantic_ok"] = pred == gold
    metrics["json_exact"] = pred == gold
    return metrics


def update_summary(acc: dict[str, Any], metrics: dict[str, Any]) -> None:
    acc["n"] += 1
    acc["parse_ok"] += int(bool(metrics.get("parse_ok")))
    acc["semantic_ok"] += int(bool(metrics.get("semantic_ok")))
    for key, value in metrics.items():
        if isinstance(value, bool):
            acc[f"{key}_n"] += 1
            acc[f"{key}_ok"] += int(value)
    for key in ("abs_error", "squared_error", "peak_mae", "peak_rmse", "peak_within_tol_rate"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            acc[f"{key}_sum"] += float(value)
            acc[f"{key}_n"] += 1


def finalize_summary(raw: dict[str, Any]) -> dict[str, Any]:
    n = max(1, raw.get("n", 0))
    out: dict[str, Any] = {
        "n": raw.get("n", 0),
        "parse_rate": raw.get("parse_ok", 0) / n,
        "semantic_acc": raw.get("semantic_ok", 0) / n,
    }
    for key, value in sorted(raw.items()):
        if key.endswith("_n") and value:
            base = key[:-2]
            ok_key = f"{base}_ok"
            sum_key = f"{base}_sum"
            if ok_key in raw:
                out[f"{base}_rate"] = raw[ok_key] / value
            if sum_key in raw:
                out[f"{base}_mean"] = raw[sum_key] / value
    if raw.get("squared_error_n"):
        out["rmse"] = math.sqrt(raw["squared_error_sum"] / raw["squared_error_n"])
    return out


def build_model_and_connector(args: argparse.Namespace) -> tuple[Any, Any, Phase2QFormerConnector]:
    try_import_torch_npu()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = None
    if args.device_map == "auto":
        max_memory = {i: args.max_npu_memory for i in range(torch.npu.device_count())}
        max_memory["cpu"] = args.max_cpu_memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        max_memory=max_memory,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    embed_device = model.get_input_embeddings().weight.device
    hidden_size = model.get_input_embeddings().weight.shape[-1]
    connector = Phase2QFormerConnector(
        bert_path=args.bert_path,
        matterchat_root=args.matterchat_root,
        z_dim=args.bridge_dim,
        llm_hidden_size=hidden_size,
        num_query_tokens=args.num_queries,
        dropout=0.0,
        num_prefix_types=args.num_prefix_types,
    ).to(device=embed_device, dtype=torch.bfloat16)
    prune_qformer_text_modules(connector)
    state = torch.load(args.connector_ckpt, map_location="cpu")
    connector.load_state_dict(state, strict=False)
    connector.eval()
    return tokenizer, model, connector


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    tokenizer, model, connector = build_model_and_connector(args)
    dataset = EvalDataset(
        jsonl=Path(args.eval_jsonl),
        bridge_root=Path(args.bridge_root),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
        num_workers=0,
    )

    embed_device = model.get_input_embeddings().weight.device
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = defaultdict(float)
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    bad_parse: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as out_handle:
        for batch in loader:
            input_ids = batch["input_ids"].to(embed_device)
            text_mask = batch["attention_mask"].to(embed_device)
            z_tokens = batch["prefix_tokens"].to(embed_device, dtype=torch.bfloat16)
            z_mask = batch["prefix_mask"].to(embed_device)
            prefix_types = batch["prefix_types"].to(embed_device)

            text_embeds = model.get_input_embeddings()(input_ids)
            soft_prefix = connector(z_tokens, z_mask, prefix_types)
            inputs_embeds = torch.cat([soft_prefix, text_embeds], dim=1)
            soft_mask = torch.ones(
                (input_ids.shape[0], soft_prefix.shape[1]),
                device=embed_device,
                dtype=text_mask.dtype,
            )
            attention_mask = torch.cat([soft_mask, text_mask], dim=1)
            generated = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for row, gold_text, pred_text in zip(batch["rows"], batch["answers"], pred_texts):
                task = row.get("task_type", "")
                pred_obj, parse_error = extract_json_obj(pred_text)
                gold_obj, _ = extract_json_obj(gold_text)
                metrics = score_prediction(task, pred_obj, gold_obj)
                if parse_error:
                    bad_parse[parse_error.split("\n", 1)[0][:120]] += 1
                update_summary(global_summary, metrics)
                update_summary(by_task[task], metrics)
                out_handle.write(
                    json.dumps(
                        {
                            "task_type": task,
                            "prompt": row["messages"][0]["content"],
                            "gold": gold_text,
                            "pred": pred_text,
                            "pred_json": pred_obj,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "decode": {
            "do_sample": False,
            "forced_json_prefix": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "total": finalize_summary(global_summary),
        "by_task": {task: finalize_summary(stats) for task, stats in sorted(by_task.items())},
        "parse_errors_top": bad_parse.most_common(10),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--bert-path", default="models/bert-base-uncased")
    parser.add_argument("--matterchat-root", default="third_party/matterchat_code/code/MatterChat_code")
    parser.add_argument("--connector-ckpt", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bridge-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--num-prefix-types", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-npu-memory", default="16GiB")
    parser.add_argument("--max-cpu-memory", default="160GiB")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
