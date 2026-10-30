#!/usr/bin/env python3
"""Validate the Cortex Training model catalog under model-catalog/models/.

Each model folder holds a model.json and job files named
``<hw>-<kind>-<mode>.json`` under ``standard/`` and ``long/``. For every model
folder this checks that:

- file names match ``{h200,b200,b300}-{training,inference}-{lora,full}.json``,
  and each file is a valid job definition;
- the folder slug derives from the model ID in model.json, and no two models
  share a slug or ID;
- each sub-job's model name matches model.json, the job ``hardware`` matches the
  file's prefix, every ``lora`` file carries a valid LoRA ``peft_config``
  object, and no ``full`` file has one;
- MoE models record their ``moe`` geometry in model.json;
- each sub-job uses 1, 2, 4, or a multiple of 8 GPUs, and tensor- and
  sequence-parallel sizes divide ``n_gpus``;
- the model's long-context files have a larger ``max_seq_len`` than its
  standard ones;
- model.json's ``last_validated`` dates name exactly the model's files;
- ``make_config(..., rl=True)`` builds a valid job for each context, hardware
  and mode that has both a training and an inference file, except a pair whose
  sampling ``max_seq_len`` exceeds the training window (that rollout cannot be
  collated, so the files stay as separate training and inference jobs);
- ``make_distillation_config`` builds a valid student job (training + sampling)
  and a sampling-only teacher job for the default distillation selection, with
  the teacher's ``max_seq_len`` above the student's.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = "model.json"
MODEL_KEYS = {"$schema", "model_id", "license", "architecture", "moe", "last_validated"}
REQUIRED_MODEL_KEYS = {"model_id", "license", "architecture", "last_validated"}
ARCHITECTURES = ("dense", "moe")
MOE_KEYS = ("num_moe_layers", "topk", "num_experts")
JOB_KEYS = {"hardware", "sub_job_configs"}
# SubJobConfig fields, REST API section 8.1.
SUB_JOB_KEYS = {
    "job_type",
    "model_name",
    "training_config",
    "inference_config",
    "global_batch_size",
    "dtype",
    "seed",
    "model_post_init",
    "source_checkpoint_info",
}


class CatalogValidationError(ValueError):
    """Raised with every problem found in the catalog, one per line."""


def examples_utils():
    """Import the repository's recipes.utils, never another ``recipes`` package."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import recipes
    from recipes import utils

    location = Path(recipes.__file__).resolve().parent
    if location != REPO_ROOT / "recipes":
        raise ImportError(
            f"`recipes` resolves to {location}, not this repository's recipes/; "
            f"put {REPO_ROOT} first on sys.path"
        )
    return utils


@dataclass(frozen=True)
class CatalogFile:
    """One job file of a model: ``<context>/<hw>-<kind>-<mode>.json``."""

    path: Path
    context: str
    hardware: str
    kind: str
    mode: str

    @property
    def key(self) -> str:
        return f"{self.context}/{self.path.stem}"


@dataclass
class CatalogModel:
    folder: Path
    info: dict[str, Any] | None = None
    files: list[CatalogFile] = field(default_factory=list)


def _relative(path: Path, catalog_dir: Path) -> str:
    try:
        return path.relative_to(catalog_dir).as_posix()
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogValidationError("must be a JSON object")
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _scan_model_folder(folder: Path, catalog_dir: Path, errors: list[str]) -> CatalogModel:
    utils = examples_utils()
    model = CatalogModel(folder=folder)
    for entry in sorted(folder.iterdir()):
        location = _relative(entry, catalog_dir)
        if entry.name == MODEL_FILE and entry.is_file():
            continue
        if entry.name not in utils.CATALOG_CONTEXTS or not entry.is_dir():
            errors.append(
                f"{location}: unexpected entry; a model folder holds {MODEL_FILE}, "
                f"{'/ and '.join(utils.CATALOG_CONTEXTS)}/"
            )
            continue
        for path in sorted(entry.iterdir()):
            match = utils.CATALOG_FILE_PATTERN.match(path.name)
            if match is None or not path.is_file():
                errors.append(
                    f"{_relative(path, catalog_dir)}: file names must match "
                    "{h200,b200,b300}-{training,inference}-{lora,full}.json"
                )
                continue
            hardware, kind, mode = match.groups()
            model.files.append(CatalogFile(path, entry.name, hardware, kind, mode))
    path = folder / MODEL_FILE
    if not path.is_file():
        errors.append(f"{_relative(path, catalog_dir)}: missing")
        return model
    try:
        model.info = _load_json_object(path)
    except CatalogValidationError as exc:
        errors.append(f"{_relative(path, catalog_dir)}: {exc}")
    return model


def _model_info_errors(model: CatalogModel) -> list[str]:
    utils = examples_utils()
    info = model.info or {}
    errors = []
    unexpected = sorted(set(info) - MODEL_KEYS)
    if unexpected:
        errors.append(f"unexpected keys {unexpected}")
    missing = sorted(REQUIRED_MODEL_KEYS - set(info))
    if missing:
        errors.append(f"missing keys {missing}")
    model_id = info.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append("model_id must be a non-empty string")
    elif utils.model_slug(model_id) != model.folder.name:
        errors.append(
            f"folder name must be {utils.model_slug(model_id)!r}, the lowercased last "
            f"segment of model_id {model_id!r}"
        )
    license_info = info.get("license")
    if not isinstance(license_info, dict) or set(license_info) != {"name", "url"}:
        errors.append("license must be an object with a name and a url")
    else:
        if not isinstance(license_info["name"], str) or not license_info["name"]:
            errors.append("license.name must be a non-empty string")
        if not isinstance(license_info["url"], str) or not license_info["url"].startswith("https://"):
            errors.append("license.url must be an https URL")
    architecture = info.get("architecture")
    if architecture not in ARCHITECTURES:
        errors.append(f"architecture must be one of {list(ARCHITECTURES)}, got {architecture!r}")
    moe = info.get("moe")
    if architecture == "moe":
        if not isinstance(moe, dict) or set(moe) != set(MOE_KEYS):
            errors.append(f"MoE models need moe geometry with exactly {list(MOE_KEYS)}")
        elif not all(_positive_int(moe[key]) for key in MOE_KEYS):
            errors.append(f"moe geometry values must be positive integers, got {moe}")
        else:
            try:
                utils.moe_geometry_from_config(
                    None,
                    num_moe_layers_override=moe["num_moe_layers"],
                    topk_override=moe["topk"],
                    num_experts_override=moe["num_experts"],
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"moe geometry is invalid: {exc}")
    elif moe is not None:
        errors.append("only MoE models record moe geometry")
    dates = info.get("last_validated")
    keys = {catalog_file.key for catalog_file in model.files}
    if not isinstance(dates, dict):
        errors.append("last_validated must map each file of the model to a date or null")
    else:
        for key in sorted(set(dates) - keys):
            errors.append(f"last_validated names {key}, which has no file")
        for key in sorted(keys - set(dates)):
            errors.append(f"last_validated is missing {key} (use null until a run validates it)")
        for key, value in sorted(dates.items()):
            if value is None:
                continue
            try:
                date.fromisoformat(value)
            except (TypeError, ValueError):
                errors.append(f"last_validated[{key!r}] must be an ISO date or null, got {value!r}")
    return errors


def _sub_job_errors(sub_job: dict[str, Any]) -> list[str]:
    """Schema checks from REST API sections 8.1 to 8.3 for one sub-job."""
    from cortex_training.client.optimizer import normalize_optimizer_config

    errors = []
    unexpected = sorted(set(sub_job) - SUB_JOB_KEYS)
    if unexpected:
        errors.append(f"unknown sub-job keys {unexpected}")
    if not isinstance(sub_job.get("model_name"), str) or not sub_job["model_name"].strip():
        errors.append("model_name must be a non-empty string")
    job_type = sub_job.get("job_type")
    config_key = {"training": "training_config", "sampling": "inference_config"}.get(job_type)
    if config_key is None:
        errors.append(f"job_type must be 'training' or 'sampling', got {job_type!r}")
        return errors
    other_key = "inference_config" if config_key == "training_config" else "training_config"
    if other_key in sub_job:
        errors.append(f"a {job_type} sub-job cannot carry {other_key}")
    config = sub_job.get(config_key)
    if not isinstance(config, dict):
        errors.append(f"a {job_type} sub-job needs a {config_key} object")
        return errors
    required = ("max_seq_len", "n_gpus") + (("train_batch_size",) if job_type == "training" else ())
    for name in required:
        if not _positive_int(config.get(name)):
            errors.append(f"{config_key}.{name} must be a positive integer, got {config.get(name)!r}")
    if job_type == "training":
        try:
            optimizer, hoisted_clipping = normalize_optimizer_config(
                config.get("optimizer"), location=f"{config_key}.optimizer"
            )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if not optimizer and config.get("gradient_clipping", hoisted_clipping) is None:
                errors.append(f"{config_key}.optimizer must carry at least one setting")
    return errors


def job_errors(body: dict[str, Any]) -> list[str]:
    """Why a create-job body is not a valid job definition (empty when it is).

    It must pass the checks ``cortex-training submit --dry-run`` runs on a body
    (without reading any connection settings), set ``hardware`` to a supported
    value, use no job-level key other than ``sub_job_configs``, and give each
    sub-job the fields REST API sections 8.1 to 8.3 require.
    """
    from cortex_training.cli.commands import _validate_create_job_body
    from cortex_training.client import Hardware

    errors = []
    unexpected = sorted(set(body) - JOB_KEYS)
    if unexpected:
        errors.append(f"job-level keys other than {sorted(JOB_KEYS)} are not catalog fields: {unexpected}")
    hardware = body.get("hardware")
    if hardware not in {member.value for member in Hardware}:
        errors.append(f"hardware must be one of {[member.value for member in Hardware]}, got {hardware!r}")
    try:
        _validate_create_job_body(copy.deepcopy(body))
    except ValueError as exc:
        errors.append(f"cortex-training submit --dry-run fails: {exc}")
    for index, sub_job in enumerate(body.get("sub_job_configs") or ()):
        if isinstance(sub_job, dict):
            errors.extend(f"sub_job_configs[{index}]: {error}" for error in _sub_job_errors(sub_job))
    return errors


def _parallel_errors(config: dict[str, Any]) -> list[str]:
    errors = []
    n_gpus = config.get("n_gpus")
    if not _positive_int(n_gpus):
        return errors
    if n_gpus not in (1, 2, 4) and n_gpus % 8:
        errors.append(f"n_gpus {n_gpus} must be 1, 2, 4, or a multiple of 8")
    sizes = {
        "sp_size": config.get("sp_size"),
        "vllm_config.tensor_parallel_size": (config.get("vllm_config") or {}).get("tensor_parallel_size"),
    }
    for name, size in sizes.items():
        if size is None:
            continue
        if not _positive_int(size) or n_gpus % size:
            errors.append(f"{name} {size} must divide n_gpus {n_gpus}")
    return errors


def _file_errors(catalog_file: CatalogFile, model_id: str | None) -> tuple[list[str], list[int]]:
    utils = examples_utils()
    try:
        body = _load_json_object(catalog_file.path)
    except CatalogValidationError as exc:
        return [str(exc)], []
    errors = job_errors(body)
    expected_hardware = catalog_file.hardware.upper()
    if body.get("hardware") != expected_hardware:
        errors.append(
            f"hardware must be {expected_hardware!r} to match the {catalog_file.hardware}- prefix, "
            f"got {body.get('hardware')!r}"
        )
    job_type = utils.CATALOG_JOB_TYPES[catalog_file.kind]
    config_key = utils.CATALOG_CONFIG_KEYS[catalog_file.kind]
    sub_jobs = body.get("sub_job_configs")
    if (
        not isinstance(sub_jobs, list)
        or len(sub_jobs) != 1
        or not isinstance(sub_jobs[0], dict)
        or sub_jobs[0].get("job_type") != job_type
        or not isinstance(sub_jobs[0].get(config_key), dict)
    ):
        errors.append(f"a {catalog_file.kind} file holds exactly one {job_type} sub-job with a {config_key}")
        return errors, []
    sub_job = sub_jobs[0]
    config = sub_job[config_key]
    if model_id is not None and sub_job.get("model_name") != model_id:
        errors.append(f"model_name {sub_job.get('model_name')!r} does not match model.json model_id {model_id!r}")
    # A LoRA adapter must be an object; job_errors' dry-run checks then validate
    # it with the client's PEFT validator. The dry run skips a null one.
    if catalog_file.mode == "lora" and config.get("peft_config") is None:
        errors.append(f"lora files carry {config_key}.peft_config, a LoRA adapter object")
    if catalog_file.mode == "full" and "peft_config" in config:
        errors.append(f"full files must not carry {config_key}.peft_config")
    errors.extend(_parallel_errors(config))
    max_seq_len = config.get("max_seq_len")
    return errors, [max_seq_len] if _positive_int(max_seq_len) else []


def _context_errors(lengths: dict[str, list[int]]) -> list[str]:
    standard, long = lengths.get("standard"), lengths.get("long")
    if standard and long and min(long) <= max(standard):
        return [
            f"long-context files need a larger max_seq_len than standard ones "
            f"(long {min(long)} <= standard {max(standard)})"
        ]
    return []


def _file_max_seq_len(catalog_file: CatalogFile) -> int | None:
    try:
        body = _load_json_object(catalog_file.path)
    except CatalogValidationError:
        return None
    sub_jobs = body.get("sub_job_configs")
    if not isinstance(sub_jobs, list) or not sub_jobs or not isinstance(sub_jobs[0], dict):
        return None
    config = sub_jobs[0].get("training_config") or sub_jobs[0].get("inference_config")
    if not isinstance(config, dict):
        return None
    value = config.get("max_seq_len")
    return value if _positive_int(value) else None


def _sampling_window_exceeds_training(files: list[CatalogFile], context: str, hardware: str, mode: str) -> bool:
    """True when this tier's inference file samples past the training window.

    ``make_config(..., rl=True)`` rejects that pair: the runner collates each
    rollout at the training ``max_seq_len``. The files remain valid on their
    own, so the catalog keeps them and does not build an RL job.
    """
    windows: dict[str, int] = {}
    for catalog_file in files:
        if (catalog_file.context, catalog_file.hardware, catalog_file.mode) != (context, hardware, mode):
            continue
        length = _file_max_seq_len(catalog_file)
        if length is not None:
            windows[catalog_file.kind] = length
    training, inference = windows.get("training"), windows.get("inference")
    return training is not None and inference is not None and inference > training


def _rl_errors(model_id: str, files: list[CatalogFile], catalog_dir: Path) -> tuple[list[str], int]:
    """Build every RL job the model's files allow with make_config and check it."""
    utils = examples_utils()
    kinds: dict[tuple[str, str, str], set[str]] = {}
    for catalog_file in files:
        kinds.setdefault((catalog_file.context, catalog_file.hardware, catalog_file.mode), set()).add(catalog_file.kind)
    errors = []
    combinations = []
    for context, hardware, mode in sorted(key for key, found in kinds.items() if found == set(utils.CATALOG_KINDS)):
        if _sampling_window_exceeds_training(files, context, hardware, mode):
            continue
        combinations.append((context, hardware, mode))
    for context, hardware, mode in combinations:
        label = f"make_config(model={model_id!r}, context={context!r}, hardware={hardware!r}, lora={mode == 'lora'}, rl=True)"
        try:
            body = utils.make_config(
                model_id,
                context,
                hardware=hardware,
                lora=mode == "lora",
                rl=True,
                catalog_dir=catalog_dir,
            )
        except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
            errors.append(f"{label} fails: {exc}")
            continue
        errors.extend(f"{label} builds an invalid job: {error}" for error in job_errors(body))
    return errors, len(combinations)


def _distillation_errors(catalog_dir: Path) -> tuple[list[str], int]:
    """Build the shipped on-policy distillation jobs with make_distillation_config and check them.

    This validates the pair the runner actually submits: the student (training +
    sampling) and the sampling-only teacher from the default distillation
    selection, sized to the small default windows, with the example's committed
    ``overrides.json`` applied to the student (so n_gpus/train_batch_size are the
    shipped small footprint, not the catalog's full-size training defaults). Both
    default models must be present -- a missing one is an error, not a silent
    skip. Each body must pass the same in-process job validation as an RL job, the
    teacher must be sampling-only, and its window must exceed the student's.
    """
    utils = examples_utils()
    student = utils.DEFAULT_DISTILLATION_STUDENT
    teacher = utils.DEFAULT_DISTILLATION_TEACHER
    label = f"make_distillation_config(student={student!r}, teacher={teacher!r})"
    missing = [model_id for model_id in (student, teacher) if not (catalog_dir / utils.model_slug(model_id)).is_dir()]
    if missing:
        return [f"{label}: default distillation model(s) missing from the catalog: {', '.join(missing)}"], 0
    overrides_path = REPO_ROOT / "recipes" / "distillation" / "overrides.json"
    if not overrides_path.is_file():
        return [f"{label}: the example's shipped overrides.json is missing at {overrides_path}"], 0
    try:
        student_body, teacher_body = utils.make_distillation_config(
            student,
            teacher,
            utils.DEFAULT_DISTILLATION_CONTEXT,
            student_max_seq_len=utils.DEFAULT_DISTILLATION_STUDENT_MAX_SEQ_LEN,
            teacher_max_seq_len=utils.DEFAULT_DISTILLATION_TEACHER_MAX_SEQ_LEN,
            overrides=[overrides_path],
            catalog_dir=catalog_dir,
        )
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        return [f"{label} fails: {exc}"], 0
    errors = [f"{label} builds an invalid student job: {error}" for error in job_errors(student_body)]
    errors.extend(f"{label} builds an invalid teacher job: {error}" for error in job_errors(teacher_body))
    teacher_job_types = [sub.get("job_type") for sub in teacher_body.get("sub_job_configs") or ()]
    if teacher_job_types != ["sampling"]:
        errors.append(f"{label}: the teacher job must be sampling-only, got {teacher_job_types}")
    student_window = _sub_job_max_seq_len(student_body, "training")
    teacher_window = _sub_job_max_seq_len(teacher_body, "sampling")
    if student_window is not None and teacher_window is not None and teacher_window <= student_window:
        errors.append(
            f"{label}: teacher max_seq_len {teacher_window} must exceed the student's {student_window}"
        )
    return errors, 1


def _sub_job_max_seq_len(body: dict[str, Any], job_type: str) -> int | None:
    config_key = {"training": "training_config", "sampling": "inference_config"}[job_type]
    for sub_job in body.get("sub_job_configs") or ():
        if isinstance(sub_job, dict) and sub_job.get("job_type") == job_type:
            value = (sub_job.get(config_key) or {}).get("max_seq_len")
            return value if _positive_int(value) else None
    return None


def validate_catalog(catalog_dir: Path | None = None) -> dict[str, Any]:
    """Validate every model folder; return counts, or raise with every problem."""
    catalog_dir = Path(catalog_dir or examples_utils().CATALOG_DIR).resolve()
    errors: list[str] = []
    if not catalog_dir.is_dir():
        raise CatalogValidationError(f"{catalog_dir} is not a directory")
    models = []
    for folder in sorted(catalog_dir.iterdir()):
        if not folder.is_dir():
            errors.append(f"{folder.name}: only model folders belong in {catalog_dir.name}/")
            continue
        models.append(_scan_model_folder(folder, catalog_dir, errors))
    if not models:
        errors.append(f"{catalog_dir} has no model folders")

    seen_ids: dict[str, str] = {}
    seen_slugs: dict[str, str] = {}
    counts = {"models": len(models), "files": 0, "training": 0, "inference": 0, "rl": 0, "distillation": 0}
    for model in models:
        name = model.folder.name
        info = model.info or {}
        errors_before = len(errors)
        if model.info is not None:
            errors.extend(f"{name}/{MODEL_FILE}: {error}" for error in _model_info_errors(model))
        model_id = info.get("model_id") if isinstance(info.get("model_id"), str) else None
        if model_id is not None:
            if model_id in seen_ids:
                errors.append(f"{name}/{MODEL_FILE}: model_id {model_id!r} is also used by {seen_ids[model_id]}")
            seen_ids.setdefault(model_id, name)
            slug = examples_utils().model_slug(model_id)
            if slug in seen_slugs:
                errors.append(f"{name}/{MODEL_FILE}: slug {slug!r} is also used by {seen_slugs[slug]}")
            seen_slugs.setdefault(slug, name)
        lengths: dict[str, list[int]] = {}
        for catalog_file in model.files:
            counts["files"] += 1
            counts[catalog_file.kind] += 1
            file_errors, file_lengths = _file_errors(catalog_file, model_id)
            location = _relative(catalog_file.path, catalog_dir)
            errors.extend(f"{location}: {error}" for error in file_errors)
            lengths.setdefault(catalog_file.context, []).extend(file_lengths)
        errors.extend(f"{name}: {error}" for error in _context_errors(lengths))
        if model_id is not None and len(errors) == errors_before:
            rl_errors, rl_jobs = _rl_errors(model_id, model.files, catalog_dir)
            errors.extend(f"{name}: {error}" for error in rl_errors)
            counts["rl"] += rl_jobs

    # The on-policy distillation jobs span two models, so check them once the
    # per-model files are known to be valid.
    if not errors:
        distillation_errors, distillation_jobs = _distillation_errors(catalog_dir)
        errors.extend(distillation_errors)
        counts["distillation"] += distillation_jobs

    if errors:
        raise CatalogValidationError("\n".join(errors))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Model folders to validate (default: model-catalog/models).",
    )
    args = parser.parse_args()
    try:
        summary = validate_catalog(args.catalog_dir)
    except CatalogValidationError as exc:
        print(f"Cortex Training catalog is invalid:\n{exc}", file=sys.stderr)
        return 1
    print(
        "Cortex Training catalog valid: "
        f"{summary['models']} models, {summary['files']} job files "
        f"({summary['training']} training, {summary['inference']} inference), "
        f"{summary['rl']} RL jobs built with make_config, "
        f"{summary['distillation']} distillation job pairs built with make_distillation_config"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
