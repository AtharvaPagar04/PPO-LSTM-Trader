import csv
from pathlib import Path


def write_trace_csv(path: Path, field_name: str, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", field_name])
        writer.writeheader()
        for step, value in enumerate(values):
            writer.writerow({"step": step, field_name: value})
