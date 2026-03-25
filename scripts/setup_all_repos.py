#!/usr/bin/env python3
"""Clone all upstream repositories required by the multimodal pipeline.

The audio stack already ships a repo manager under ``audio_models``.  This
utility orchestrates it alongside a curated set of computer-vision and NLP
repositories so a single command prepares everything under ``external/``.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

try:  # Prefer the authoritative audio registry when available
    from audio_models.external.repo_manager import (
        AUDIO_REPOS as AUDIO_REPO_REGISTRY,
        ensure_repo as ensure_audio_repo,
    )
except Exception:  # pragma: no cover - fallback when package not installed
    AUDIO_REPO_REGISTRY = {}

    def ensure_audio_repo(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(
            "audio_models package is not importable. Run this script from the project root "
            "or install the audio_models package first."
        )


try:  # Mirror the audio manager for the CV stack when available
    from cv_models.external.repo_manager import (
        CV_REPOS as CV_REPO_REGISTRY,
        ensure_repo as ensure_cv_repo,
    )
except Exception:  # pragma: no cover - fallback when package not installed
    CV_REPO_REGISTRY = {}

    def ensure_cv_repo(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(
            "cv_models package is not importable. Run this script from the project root "
            "or install the cv_models package first."
        )


@dataclass(frozen=True)
class RepoSpec:
    """Specification for repositories managed outside the audio package."""

    key: str
    name: str
    url: str
    category: str
    commit: Optional[str] = None
    env_var: Optional[str] = None
    subdirectory: Optional[str] = None
    init_submodules: bool = False

    def destination(self, root: Path) -> Path:
        return root / self.category / self.name

    def resolved_path(self, root: Path) -> Path:
        dest = self.destination(root)
        return dest / self.subdirectory if self.subdirectory else dest


EXTRA_REPOS: Dict[str, RepoSpec] = {
    # Vision repositories live in cv_models.external.repo_manager so analyzers
    # and setup tooling share a single source of truth.  The "intelligent-video-"
    # extractor listed in requirements.csv remains manual due to its private origin.

    # Multimodal
    "meld": RepoSpec(
        key="meld",
        name="MELD",
        url="https://github.com/declare-lab/MELD.git",
        category="multimodal",
    ),

    # Audio (non-audio_models registry)
    "opensmile": RepoSpec(
        key="opensmile",
        name="opensmile",
        url="https://github.com/audeering/opensmile.git",
        category="audio",
    ),
    "fairseq": RepoSpec(
        key="fairseq",
        name="fairseq",
        url="https://github.com/facebookresearch/fairseq.git",
        category="audio",
        init_submodules=True,
    ),
    "audiostretchy": RepoSpec(
        key="audiostretchy",
        name="audiostretchy",
        url="https://github.com/twardoch/audiostretchy.git",
        category="audio",
    ),

    # NLP
    "heinsen_routing": RepoSpec(
        key="heinsen_routing",
        name="heinsen_routing",
        url="https://github.com/glassroom/heinsen_routing.git",
        category="nlp",
    ),
    "simcse": RepoSpec(
        key="simcse",
        name="SimCSE",
        url="https://github.com/princeton-nlp/SimCSE.git",
        category="nlp",
    ),
    "deberta": RepoSpec(
        key="deberta",
        name="DeBERTa",
        url="https://github.com/microsoft/DeBERTa.git",
        category="nlp",
    ),
    "albert": RepoSpec(
        key="albert",
        name="ALBERT",
        url="https://github.com/google-research/ALBERT.git",
        category="nlp",
    ),
    "sentence_transformers": RepoSpec(
        key="sentence_transformers",
        name="sentence-transformers",
        url="https://github.com/UKPLab/sentence-transformers.git",
        category="nlp",
    ),
    "allennlp_models": RepoSpec(
        key="allennlp_models",
        name="allennlp-models",
        url="https://github.com/allenai/allennlp-models.git",
        category="nlp",
    ),
}


def run_git(args: Iterable[str], *, cwd: Optional[Path] = None) -> None:
    command = ["git", *args]
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def clone_extra_repo(spec: RepoSpec, *, root: Path, force_update: bool = False) -> Path:
    if spec.env_var:
        override = os.getenv(spec.env_var)
        if override:
            override_path = Path(override).expanduser()
            if override_path.exists():
                return override_path if not spec.subdirectory else override_path / spec.subdirectory
            logging.warning("%s points to missing path %s", spec.env_var, override_path)

    dest = spec.destination(root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if force_update:
            logging.info("Updating %s", spec.key)
            run_git(["fetch", "--all", "--tags"], cwd=dest)
            if spec.commit:
                run_git(["reset", "--hard", spec.commit], cwd=dest)
            else:
                run_git(["pull", "--ff-only"], cwd=dest)
    else:
        logging.info("Cloning %s into %s", spec.url, dest)
        run_git(["clone", spec.url, str(dest)])
        if spec.commit:
            run_git(["checkout", spec.commit], cwd=dest)

    if spec.init_submodules:
        run_git(["submodule", "update", "--init", "--recursive"], cwd=dest)

    resolved = spec.resolved_path(root)
    if not resolved.exists():
        raise FileNotFoundError(f"Expected {resolved} after cloning {spec.key}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone required external model repositories")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Subset of repository keys to clone",
    )

    category_choices = {"audio"}
    if CV_REPO_REGISTRY:
        category_choices.add("vision")
    category_choices.update(spec.category for spec in EXTRA_REPOS.values())
    parser.add_argument(
        "--category",
        choices=sorted(category_choices),
        help="Clone only repositories within the selected category",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Fetch and reset repositories even if they already exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known repositories and exit",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external",
        help="Override the root directory that stores cloned repositories",
    )
    return parser.parse_args()


def list_repositories() -> None:
    entries = []
    for key, repo in AUDIO_REPO_REGISTRY.items():
        entries.append((key, "audio", repo.url))
    for key, repo in CV_REPO_REGISTRY.items():
        entries.append((key, "vision", repo.url))
    for key, spec in EXTRA_REPOS.items():
        entries.append((key, spec.category, spec.url))
    width = max(len(key) for key, _, _ in entries) if entries else 0
    for key, category, url in sorted(entries):
        print(f"{key.ljust(width)}  {category:<10}  {url}")


def main() -> None:
    args = parse_args()

    if args.list:
        list_repositories()
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    external_root = args.external_root.resolve()
    external_root.mkdir(parents=True, exist_ok=True)

    all_known_keys = (
        set(AUDIO_REPO_REGISTRY.keys())
        | set(CV_REPO_REGISTRY.keys())
        | set(EXTRA_REPOS.keys())
    )
    selected_keys = set(all_known_keys)
    if args.only:
        invalid = sorted(set(args.only) - all_known_keys)
        if invalid:
            raise SystemExit(f"Unknown repository keys: {', '.join(invalid)}")
        selected_keys = set(args.only)

    def _category_for_key(key: str) -> str:
        if key in AUDIO_REPO_REGISTRY:
            return "audio"
        if key in CV_REPO_REGISTRY:
            return "vision"
        if key in EXTRA_REPOS:
            return EXTRA_REPOS[key].category
        raise KeyError(key)

    if args.category:
        selected_keys = {key for key in selected_keys if _category_for_key(key) == args.category}
    if not selected_keys:
        logging.warning("No repositories matched the requested filters")
        return

    for key in sorted(selected_keys):
        if key in AUDIO_REPO_REGISTRY:
            logging.info("Preparing audio repository %s", key)
            path = ensure_audio_repo(key, force_update=args.force_update)
        elif key in CV_REPO_REGISTRY:
            logging.info("Preparing vision repository %s", key)
            path = ensure_cv_repo(
                key,
                base_dir=external_root / "vision",
                force_update=args.force_update,
            )
        else:
            spec = EXTRA_REPOS[key]
            logging.info("Preparing %s repository %s", spec.category, key)
            path = clone_extra_repo(spec, root=external_root, force_update=args.force_update)
        logging.info("Repository %s ready at %s", key, path)


if __name__ == "__main__":
    main()
