import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_metrics(tripinfo_path, summary_path):
    trip_root = ET.parse(tripinfo_path).getroot()
    trips = trip_root.findall("tripinfo")

    mean_tt = 0.0
    mean_delay = ""
    if trips:
        mean_tt = sum(float(t.attrib.get("duration", "0")) for t in trips) / len(trips)
        if all("timeLoss" in t.attrib for t in trips):
            mean_delay = sum(float(t.attrib["timeLoss"]) for t in trips) / len(trips)

    sum_root = ET.parse(summary_path).getroot()
    last = sum_root.findall("step")[-1]
    loaded = int(float(last.attrib.get("loaded", "0")))
    departed = int(float(last.attrib.get("inserted", "0")))
    arrived = int(float(last.attrib.get("ended", "0")))
    teleports = int(float(last.attrib.get("teleports", "0")))

    return {
        "loaded": loaded,
        "departed": departed,
        "arrived": arrived,
        "mean_travel_time_s": round(mean_tt, 2),
        "mean_delay_s": round(mean_delay, 2) if mean_delay != "" else "",
        "throughput": arrived,
        "teleports": teleports,
    }


def write_row(out_csv, row):
    fieldnames = [
        "scenario",
        "seed",
        "controller",
        "loaded",
        "departed",
        "arrived",
        "mean_travel_time_s",
        "mean_delay_s",
        "throughput",
        "teleports",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one SUMO run from tripinfo + summary.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller", choices=["fixed", "actuated"], required=True)
    parser.add_argument("--tripinfo", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    tripinfo_path = Path(args.tripinfo)
    summary_path = Path(args.summary)
    out_csv = Path(args.out_csv)

    metrics = parse_metrics(tripinfo_path, summary_path)
    row = {
        "scenario": args.scenario,
        "seed": args.seed,
        "controller": args.controller,
        **metrics,
    }
    write_row(out_csv, row)

    print(",".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
