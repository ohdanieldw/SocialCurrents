"""Clone all upstream repositories required by the audio pipelines.

Run this helper before executing any of the heavy audio analysers so the
original research code bases are present locally.  The script is intentionally
simple and only relies on the ``repo_manager`` utility inside the
``audio_models`` package, meaning it can be executed as
``python -m audio_models.scripts.setup_audio_repos`` once the poetry
environment is active.
"""
from __future__ import annotations

import argparse
import logging

from audio_models.external.repo_manager import AUDIO_REPOS, ensure_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone audio model repositories")
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Fetch and reset repositories even if they already exist",
    )
    parser.add_argument(
        "--only",
        choices=sorted(AUDIO_REPOS.keys()),
        help="Clone a single repository by key instead of all",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    targets = [args.only] if args.only else sorted(AUDIO_REPOS.keys())

    for key in targets:
        repo = AUDIO_REPOS[key]
        logging.info("Preparing repository %s (%s)", repo.name, repo.url)
        path = ensure_repo(key, force_update=args.force_update)
        logging.info("Repository ready at %s", path)


if __name__ == "__main__":
    main()
