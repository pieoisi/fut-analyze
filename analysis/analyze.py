#!/usr/bin/env python3
"""Extract trial-level eGeMAPSv02 features and evaluate small-sample models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import opensmile
import soundfile as sf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SAMPLE_RATE = 16_000
EXPECTED_FEATURES = {
    "ease": "言葉の出やすさ (1-5)",
    "valence": "気分 valence (-1/0/+1)",
    "arousal": "気分 arousal (-1/0/+1)",
    "affect_9class": "気分 9カテゴリ",
    "stuck": "自由回答中の詰まりボタン有無",
}
AFFECT_LABELS = {
    (-1, 1): "tense_unpleasant",
    (0, 1): "activated_neutral",
    (1, 1): "excited_pleasant",
    (-1, 0): "unpleasant",
    (0, 0): "neutral",
    (1, 0): "pleasant",
    (-1, -1): "low_unpleasant",
    (0, -1): "calm_neutral",
    (1, -1): "calm_pleasant",
}
BASE_COLUMNS = [
    "session",
    "trial",
    "participant",
    "condition",
    "rep",
    "recording_start_sample",
    "recording_end_sample",
    "sample_rate_hz",
    "feature_window_type",
    "label_window_type",
    "ease",
    "affect_valence",
    "affect_arousal",
    "affect_label",
    "stuck_count",
    "stuck",
]


def parse_int(value: str | None, field: str) -> int:
    if value is None or value.strip() == "":
        raise ValueError(f"{field} is missing")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer: {value!r}") from exc


def read_session(csv_path: Path) -> tuple[list[dict[str, Any]], str, bool]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"event", "trial", "participant", "condition"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing CSV columns: {', '.join(sorted(missing))}")
        events = list(reader)

    participant_ids = {row["participant"].strip() for row in events if row.get("participant", "").strip()}
    if len(participant_ids) != 1:
        raise ValueError(f"{csv_path}: expected one participant ID, found {len(participant_ids)}")
    participant = participant_ids.pop()

    starts = [row for row in events if row.get("event") == "session_start"]
    if len(starts) != 1 or starts[0].get("audio_saved", "").strip().lower() not in {"true", "false"}:
        raise ValueError(f"{csv_path}: expected one session_start with audio_saved=true/false")
    audio_saved = starts[0]["audio_saved"].strip().lower() == "true"

    trials: dict[str, dict[str, Any]] = {}
    for row in events:
        trial_id = row.get("trial", "").strip()
        if not trial_id:
            continue
        trial = trials.setdefault(
            trial_id,
            {
                "session": csv_path.stem,
                "trial": trial_id,
                "participant": participant,
                "condition": "",
                "rep": "",
                "rec_start": [],
                "rec_end": [],
                "rating": None,
                "affect_rating": None,
                "stuck_count": 0,
            },
        )
        condition = row.get("condition", "").strip()
        if condition:
            if condition not in {"silence", "read", "free"}:
                raise ValueError(f"{csv_path}: trial {trial_id} has unknown condition {condition!r}")
            if trial["condition"] and trial["condition"] != condition:
                raise ValueError(f"{csv_path}: trial {trial_id} has conflicting conditions")
            trial["condition"] = condition
        if row.get("rep", "").strip():
            trial["rep"] = row["rep"].strip()

        event = row.get("event", "").strip()
        if event in {"rec_start", "rec_end"}:
            sample = parse_int(row.get("audio_sample_index"), f"{event}.audio_sample_index")
            rate = parse_int(row.get("audio_sample_rate_hz"), f"{event}.audio_sample_rate_hz")
            if rate != SAMPLE_RATE:
                raise ValueError(f"{csv_path}: trial {trial_id} has marker rate {rate}, expected {SAMPLE_RATE}")
            trial[event].append(sample)
        elif event == "rating":
            # The UI logs every selection change; the last event is the final answer.
            trial["rating"] = row
        elif event == "affect_rating":
            trial["affect_rating"] = row
        elif event == "stuck_btn" and row.get("phase", "").strip() == "rec":
            trial["stuck_count"] += 1

    for trial_id, trial in trials.items():
        if not trial["condition"]:
            raise ValueError(f"{csv_path}: trial {trial_id} is missing its condition")
    return list(trials.values()), participant, audio_saved


def parse_labels(trial: dict[str, Any]) -> dict[str, Any]:
    rating = trial["rating"]
    ease = None
    if rating is not None and rating.get("note", "").strip():
        ease = parse_int(rating["note"], "rating.note")
        if ease not in range(1, 6):
            raise ValueError(f"trial {trial['trial']}: rating must be between 1 and 5")

    affect = trial["affect_rating"]
    valence = arousal = affect_label = None
    if affect is not None:
        raw_valence = affect.get("affect_valence", "").strip()
        raw_arousal = affect.get("affect_arousal", "").strip()
        supplied_label = affect.get("affect_label", "").strip()
        if raw_valence or raw_arousal:
            valence = parse_int(raw_valence, "affect_valence")
            arousal = parse_int(raw_arousal, "affect_arousal")
            if valence not in {-1, 0, 1} or arousal not in {-1, 0, 1}:
                raise ValueError(f"trial {trial['trial']}: affect values must be -1, 0, or 1")
            affect_label = AFFECT_LABELS[(valence, arousal)]
            if supplied_label and supplied_label != affect_label:
                raise ValueError(
                    f"trial {trial['trial']}: affect_label {supplied_label!r} "
                    f"does not match valence/arousal ({affect_label})"
                )
        elif supplied_label and supplied_label != "unknown":
            raise ValueError(f"trial {trial['trial']}: missing affect values for {supplied_label!r}")

    is_free = trial["condition"] == "free"
    return {
        "ease": ease,
        "affect_valence": valence,
        "affect_arousal": arousal,
        "affect_label": affect_label,
        "stuck_count": trial["stuck_count"],
        "stuck": int(trial["stuck_count"] > 0) if is_free else None,
    }


def load_audio(wav_path: Path) -> np.ndarray:
    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{wav_path}: sample rate is {info.samplerate}, expected {SAMPLE_RATE}")
    if info.channels != 1:
        raise ValueError(f"{wav_path}: expected mono audio, found {info.channels} channels")
    if info.subtype != "PCM_16":
        raise ValueError(f"{wav_path}: expected 16-bit PCM, found {info.subtype}")
    audio, rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"{wav_path}: audio does not match the required 16 kHz mono format")
    return audio


def extract_features(input_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise ValueError(f"No session CSV files found in {input_dir}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=SAMPLE_RATE,
        num_workers=1,
        verbose=False,
    )
    feature_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    for csv_path in csv_paths:
        trials, _, audio_saved = read_session(csv_path)
        wav_path = csv_path.with_suffix(".wav")
        audio: np.ndarray | None = None
        audio_issue = ""
        if not audio_saved:
            audio_issue = "audio_saved=false"
        elif not wav_path.is_file():
            audio_issue = "matching WAV is missing"
        else:
            audio = load_audio(wav_path)

        for trial in trials:
            try:
                labels = parse_labels(trial)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: {exc}") from exc
            start_markers, end_markers = trial["rec_start"], trial["rec_end"]
            if audio_issue:
                exclusions.append({"session": trial["session"], "trial": trial["trial"], "reason": audio_issue})
                continue
            if len(start_markers) != 1 or len(end_markers) != 1:
                exclusions.append({
                    "session": trial["session"],
                    "trial": trial["trial"],
                    "reason": "incomplete or duplicate recording boundary",
                })
                continue

            start, end = start_markers[0], end_markers[0]
            assert audio is not None
            if start < 0 or end <= start or end > len(audio):
                raise ValueError(
                    f"{csv_path}: trial {trial['trial']} has invalid sample interval "
                    f"[{start}, {end}) for {len(audio)} WAV frames"
                )
            signal = audio[start:end]
            if signal.size == 0:
                raise ValueError(f"{csv_path}: trial {trial['trial']} has an empty audio interval")

            result = smile.process_signal(signal, SAMPLE_RATE)
            if len(result.index) != 1:
                raise ValueError(f"{csv_path}: expected one functionals row for trial {trial['trial']}")
            features = result.iloc[0].to_dict()
            if not features or not all(math.isfinite(float(value)) for value in features.values()):
                raise ValueError(f"{csv_path}: non-finite or empty openSMILE features in trial {trial['trial']}")

            row: dict[str, Any] = {
                "session": trial["session"],
                "trial": trial["trial"],
                "participant": trial["participant"],
                "condition": trial["condition"],
                "rep": trial["rep"],
                "recording_start_sample": start,
                "recording_end_sample": end,
                "sample_rate_hz": SAMPLE_RATE,
                "feature_window_type": "whole_trial",
                "label_window_type": "whole_trial",
                **labels,
                **{f"opensmile_{name}": float(value) for name, value in features.items()},
            }
            feature_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trial_features.csv", feature_rows)
    write_csv(output_dir / "exclusions.csv", exclusions, ["session", "trial", "reason"])
    return feature_rows, exclusions


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fields = list(BASE_COLUMNS)
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def task_labels(row: dict[str, Any], task: str) -> Any:
    if task == "ease":
        return row.get("ease")
    if task == "valence":
        return row.get("affect_valence")
    if task == "arousal":
        return row.get("affect_arousal")
    if task == "affect_9class":
        return row.get("affect_label")
    if task == "stuck":
        return row.get("stuck")
    raise ValueError(f"Unknown task: {task}")


class CumulativeOrdinalLogistic(ClassifierMixin, BaseEstimator):
    """Regularized logistic classifiers for each ordered-label threshold."""

    def __init__(self, c: float = 0.1):
        self.c = c

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.sort(np.unique(y))
        if len(self.classes_) < 2:
            raise ValueError("Ordinal classification requires at least two observed classes")
        self.models_ = []
        self.constant_probabilities_ = []
        for threshold in self.classes_[:-1]:
            above_threshold = (y > threshold).astype(int)
            if len(np.unique(above_threshold)) == 1:
                self.models_.append(None)
                self.constant_probabilities_.append(float(above_threshold[0]))
                continue
            model = LogisticRegression(C=self.c, class_weight="balanced", max_iter=5000)
            model.fit(x, above_threshold)
            self.models_.append(model)
            self.constant_probabilities_.append(None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        cumulative = []
        for model, constant in zip(self.models_, self.constant_probabilities_, strict=True):
            if model is None:
                cumulative.append(np.full(x.shape[0], constant))
            else:
                positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
                cumulative.append(model.predict_proba(x)[:, positive_column])

        probability_above = np.minimum.accumulate(np.column_stack(cumulative), axis=1)
        probabilities = np.empty((x.shape[0], len(self.classes_)))
        probabilities[:, 0] = 1 - probability_above[:, 0]
        for index in range(1, len(self.classes_) - 1):
            probabilities[:, index] = probability_above[:, index - 1] - probability_above[:, index]
        probabilities[:, -1] = probability_above[:, -1]
        return self.classes_[np.argmax(probabilities, axis=1)]


def make_model(task: str, feature_count: int):
    classifier = (
        CumulativeOrdinalLogistic(c=0.1)
        if task in {"ease", "valence", "arousal"}
        else LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=42)
    )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=min(10, feature_count)),
        classifier,
    )


def evaluate_task(
    rows: list[dict[str, Any]], feature_names: list[str], task: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usable = [(row, task_labels(row, task)) for row in rows if task_labels(row, task) is not None]
    counts = Counter(str(label) for _, label in usable)
    participants = {row["participant"] for row, _ in usable if row.get("participant")}
    participant_counts = {
        str(label): len({row["participant"] for row, value in usable if value == label and row.get("participant")})
        for label in {value for _, value in usable}
    }
    summary: dict[str, Any] = {
        "description": EXPECTED_FEATURES[task],
        "sample_count": len(usable),
        "participant_count": len(participants),
        "class_counts": dict(sorted(counts.items())),
        "class_participant_counts": dict(sorted(participant_counts.items())),
        "status": "not_evaluable",
    }
    if len(usable) < 2 or len(participants) < 2:
        summary["reason"] = "At least two labeled participants are required for speaker-independent evaluation."
        return summary, []
    if any(not row.get("participant") for row, _ in usable):
        summary["reason"] = "Participant ID is missing; speaker-independent evaluation is not possible."
        return summary, []
    if len(counts) < 2:
        summary["reason"] = "Only one label class is present."
        return summary, []
    unsupported_classes = [label for label, count in participant_counts.items() if count < 2]
    if unsupported_classes:
        summary["reason"] = (
            "Each class must occur in at least two participants for leave-one-participant-out evaluation; "
            f"unsupported classes: {', '.join(unsupported_classes)}."
        )
        return summary, []

    x = np.asarray([[float(row[name]) for name in feature_names] for row, _ in usable], dtype=np.float64)
    y = np.asarray([label for _, label in usable])
    groups = np.asarray([row["participant"] for row, _ in usable])
    class_labels = sorted(set(y.tolist()), key=str)
    splitter = LeaveOneGroupOut()
    predictions: list[dict[str, Any]] = []
    skipped_folds = 0

    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        if len(set(y[train].tolist())) < 2:
            skipped_folds += 1
            continue
        model = make_model(task, len(feature_names))
        model.fit(x[train], y[train])
        predicted = model.predict(x[test])
        for index, value in zip(test, predicted, strict=True):
            row = usable[index][0]
            prediction = int(round(float(value))) if task in {"ease", "valence", "arousal"} else value
            predictions.append({
                "task": task,
                "session": row["session"],
                "trial": row["trial"],
                "participant": row["participant"],
                "fold": fold,
                "actual": y[index],
                "predicted": prediction,
            })

    summary["skipped_folds"] = skipped_folds
    summary["evaluated_sample_count"] = len(predictions)
    if not predictions:
        summary["reason"] = "No held-out participant fold had at least two training classes."
        return summary, predictions

    actual = np.asarray([item["actual"] for item in predictions])
    predicted = np.asarray([item["predicted"] for item in predictions])
    summary.update({
        "status": "exploratory",
        "coverage": len(predictions) / len(usable),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=class_labels, average="macro", zero_division=0)),
        "confusion_matrix_labels": [str(label) for label in class_labels],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=class_labels).tolist(),
    })
    recalls = recall_score(actual, predicted, labels=class_labels, average=None, zero_division=0)
    summary["class_recall"] = {str(label): float(value) for label, value in zip(class_labels, recalls, strict=True)}
    if task in {"ease", "valence", "arousal"}:
        kappa = cohen_kappa_score(actual, predicted, labels=class_labels, weights="quadratic")
        summary["mae"] = float(mean_absolute_error(actual.astype(float), predicted.astype(float)))
        summary["quadratic_weighted_kappa"] = float(kappa) if math.isfinite(float(kappa)) else None
    summary["caution"] = (
        "Exploratory only: small datasets and sparse classes cannot establish reliable "
        "speaker-independent performance."
    )
    return summary, predictions


def evaluate_models(
    rows: list[dict[str, Any]], output_dir: Path, targets: list[str]
) -> dict[str, Any]:
    environment = {
        "python": platform.python_version(),
        "opensmile": version("opensmile"),
        "numpy": version("numpy"),
        "scikit_learn": version("scikit-learn"),
        "soundfile": version("soundfile"),
    }
    if not rows:
        summary = {
            "status": "not_evaluable",
            "reason": "No trials had extractable audio features.",
            "environment": environment,
            "tasks": {},
        }
        (output_dir / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(output_dir / "predictions.csv", [], ["task", "session", "trial", "participant", "fold", "actual", "predicted"])
        return summary

    feature_names = [name for name in rows[0] if name.startswith("opensmile_")]
    task_summaries: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for task in targets:
        task_summary, predictions = evaluate_task(rows, feature_names, task)
        task_summaries[task] = task_summary
        all_predictions.extend(predictions)

    summary = {
        "status": "complete",
        "feature_set": "eGeMAPSv02",
        "feature_level": "Functionals",
        "feature_count": len(feature_names),
        "environment": environment,
        "evaluation": "Leave-one-participant-out; preprocessing and feature selection are fitted inside each training fold.",
        "model": {
            "ordinal_targets": "CumulativeOrdinalLogistic (balanced logistic thresholds, C=0.1)",
            "nominal_targets": "LinearSVC (C=0.1, class_weight=balanced)",
            "feature_selection": "SelectKBest(f_classif, k=min(10, feature_count))",
            "random_state": 42,
            "hyperparameter_tuning": "none",
        },
        "tasks": task_summaries,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "predictions.csv",
        all_predictions,
        ["task", "session", "trial", "participant", "fold", "actual", "predicted"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing paired session CSV and WAV files")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Output folder (defaults to <input_dir>/analysis_output)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(EXPECTED_FEATURES),
        default=list(EXPECTED_FEATURES),
        help="Independent labels to evaluate",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "analysis_output").resolve()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    if output_dir == input_dir:
        parser.error("Output folder must differ from the input folder")

    rows, exclusions = extract_features(input_dir, output_dir)
    summary = evaluate_models(rows, output_dir, args.targets)
    summary["extracted_trial_count"] = len(rows)
    summary["excluded_trial_count"] = len(exclusions)
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "extracted_trial_count": len(rows),
        "excluded_trial_count": len(exclusions),
        "evaluation_file": str(output_dir / "evaluation.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
