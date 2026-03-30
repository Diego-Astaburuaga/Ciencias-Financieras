from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.paths import INTERIM_DAILY_RV_DIR, INTERIM_EVENT_5_MIN_DIR


EVENT_RENAMES = {
    "Event_down": "mark_t",
    "lambda_hawkes": "lambda_t",
}

RV_RENAMES = {
    "perc_event_down": "ratio_event",
    "lambda_hawkes": "lambda_t",
}


@dataclass
class MigrationResult:
    path: Path
    changed: bool
    renamed: dict[str, str]


def _apply_renames(df: pd.DataFrame, rename_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    applied: dict[str, str] = {}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            applied[old] = new
    if not applied:
        return df, applied
    return df.rename(columns=applied), applied


def _migrate_file(path: Path, rename_map: dict[str, str], *, index: bool) -> MigrationResult:
    df = pd.read_parquet(path)
    updated, applied = _apply_renames(df, rename_map)
    changed = bool(applied)
    if changed:
        updated.to_parquet(path, index=index)
    return MigrationResult(path=path, changed=changed, renamed=applied)


def migrate_event_5_min(event_dir: Path = INTERIM_EVENT_5_MIN_DIR) -> list[MigrationResult]:
    event_dir = Path(event_dir)
    results: list[MigrationResult] = []
    for path in sorted(event_dir.glob("*_5m_events.parquet")):
        results.append(_migrate_file(path, EVENT_RENAMES, index=False))
    return results


def migrate_daily_rv(rv_dir: Path = INTERIM_DAILY_RV_DIR) -> list[MigrationResult]:
    rv_dir = Path(rv_dir)
    results: list[MigrationResult] = []
    for path in sorted(rv_dir.glob("*_rv.parquet")):
        results.append(_migrate_file(path, RV_RENAMES, index=True))
    return results


def _print_summary(label: str, results: list[MigrationResult]) -> None:
    changed = [r for r in results if r.changed]
    print(f"{label}: {len(changed)}/{len(results)} files updated")
    for result in changed:
        rename_str = ", ".join(f"{old}->{new}" for old, new in result.renamed.items())
        print(f"  - {result.path.name}: {rename_str}")


def main() -> None:
    event_results = migrate_event_5_min()
    rv_results = migrate_daily_rv()

    _print_summary("event_5_min", event_results)
    _print_summary("daily_rv", rv_results)


if __name__ == "__main__":
    main()
