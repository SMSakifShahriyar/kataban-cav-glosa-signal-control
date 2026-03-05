import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMO_HOME = Path(r"C:\Program Files (x86)\Eclipse\Sumo")
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = str(SUMO_HOME)
TOOLS = Path(os.environ["SUMO_HOME"]) / "tools"
if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

import traci


def edge_heading(edge_elem):
    lanes = edge_elem.findall("lane")
    if not lanes:
        return 0.0, 0.0
    shape = lanes[0].attrib.get("shape", "")
    pts = [tuple(map(float, p.split(","))) for p in shape.split()] if shape else []
    if len(pts) < 2:
        return 0.0, 0.0
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    return x2 - x1, y2 - y1


def edge_axis(edge_elem):
    dx, dy = edge_heading(edge_elem)
    return "NS" if abs(dy) >= abs(dx) else "EW"


def extract_tl_structure(net_file):
    root = ET.parse(net_file).getroot()
    tl = root.find("tlLogic")
    if tl is None:
        raise RuntimeError("No tlLogic found in network")
    tl_id = tl.attrib["id"]
    phases = [p.attrib["state"] for p in tl.findall("phase")]
    phase_durs = [int(float(p.attrib["duration"])) for p in tl.findall("phase")]

    edges = {e.attrib["id"]: e for e in root.findall("edge") if e.attrib.get("function") != "internal"}
    conns = [c for c in root.findall("connection") if c.attrib.get("tl") == tl_id]
    link_count = len(phases[0])


    main_phase_idx = []
    for i, st in enumerate(phases):
        if "y" in st or "Y" in st:
            continue
        if any(ch in ("G", "g") for ch in st):
            main_phase_idx.append(i)
    if len(main_phase_idx) < 2:
        raise RuntimeError("Could not find two main green phases")


    phase_role = {}
    for pi in main_phase_idx[:2]:
        st = phases[pi]
        ns = 0
        ew = 0
        for c in conns:
            li = int(c.attrib["linkIndex"])
            if li >= len(st) or st[li] not in ("G", "g"):
                continue
            frm = c.attrib["from"]
            axis = edge_axis(edges[frm]) if frm in edges else "EW"
            if axis == "NS":
                ns += 1
            else:
                ew += 1
        phase_role[pi] = "NS" if ns >= ew else "EW"

    ns_phase = next((i for i, role in phase_role.items() if role == "NS"), main_phase_idx[0])
    ew_phase = next((i for i, role in phase_role.items() if role == "EW"), [i for i in main_phase_idx if i != ns_phase][0])


    ns_green_idx = {k for k, ch in enumerate(phases[ns_phase]) if ch in ("G", "g")}
    ew_green_idx = {k for k, ch in enumerate(phases[ew_phase]) if ch in ("G", "g")}
    ns_yellow = None
    ew_yellow = None
    for i, st in enumerate(phases):
        y_idx = {k for k, ch in enumerate(st) if ch in ("y", "Y")}
        if not y_idx:
            continue
        if y_idx.issubset(ns_green_idx):
            ns_yellow = i
        if y_idx.issubset(ew_green_idx):
            ew_yellow = i

    if ns_yellow is None or ew_yellow is None:
        raise RuntimeError("Could not infer yellow transition phases")


    group_lanes = {"NS": set(), "EW": set()}
    group_edges = {"NS": set(), "EW": set()}
    approach_lanes = {}
    for c in conns:
        frm = c.attrib["from"]
        frm_lane = c.attrib["fromLane"]
        lane_id = f"{frm}_{frm_lane}"
        axis = edge_axis(edges[frm]) if frm in edges else "EW"
        group_lanes[axis].add(lane_id)
        group_edges[axis].add(frm)
        approach_lanes.setdefault(frm, set()).add(lane_id)

    return {
        "tl_id": tl_id,
        "link_count": link_count,
        "phase_states": phases,
        "phase_durs": phase_durs,
        "ns_phase": ns_phase,
        "ew_phase": ew_phase,
        "ns_yellow": ns_yellow,
        "ew_yellow": ew_yellow,
        "group_lanes": {k: sorted(v) for k, v in group_lanes.items()},
        "group_edges": {k: sorted(v) for k, v in group_edges.items()},
        "approach_lanes": {k: sorted(v) for k, v in approach_lanes.items()},
    }


def queue_for_group(group_lanes):
    return sum(traci.lane.getLastStepHaltingNumber(lid) for lid in group_lanes if lid in traci.lane.getIDList())


def queue_for_approach(lanes):
    return sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lanes if lid in traci.lane.getIDList())


def run_controller(
    scenario,
    seed,
    min_green,
    max_green,
    check_interval,
    pressure_delta,
    starvation_red,
    pressure_hold_checks,
    route_file=None,
    output_root=None,
    scenario_label=None,
):
    cfg = ROOT / f"kataban_{scenario}_actuated.sumocfg"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config: {cfg}")

    net_file = ROOT / "kataban_joined_nomicros.net.xml"
    info = extract_tl_structure(net_file)
    tl_id = info["tl_id"]

    out_scenario = scenario_label if scenario_label else scenario
    if output_root:
        out_dir = Path(output_root) / out_scenario / f"seed_{seed}"
    else:
        out_dir = ROOT / "outputs" / "baselines" / "actuated" / scenario / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = out_dir / "tripinfo.xml"
    summary = out_dir / "summary.xml"
    queue = out_dir / "queue.xml"
    log = out_dir / "sumo.log"
    decision_log = out_dir / "decision_trace.csv"
    switch_log = out_dir / "switch_trace.csv"

    sumo_cmd = [
        str(SUMO_HOME / "bin" / "sumo.exe"),
        "-c",
        str(cfg),
        "--seed",
        str(seed),
        "--tripinfo-output",
        str(tripinfo),
        "--summary-output",
        str(summary),
        "--queue-output",
        str(queue),
        "--duration-log.statistics",
        "true",
        "--no-step-log",
        "true",
        "--log",
        str(log),
    ]
    if route_file:
        sumo_cmd.extend(["--route-files", str(route_file)])
    traci.start(sumo_cmd)

    phases = info["phase_states"]
    ns_green = phases[info["ns_phase"]]
    ew_green = phases[info["ew_phase"]]
    ns_yellow = phases[info["ns_yellow"]]
    ew_yellow = phases[info["ew_yellow"]]
    all_red = "r" * info["link_count"]

    ns_lanes = info["group_lanes"]["NS"]
    ew_lanes = info["group_lanes"]["EW"]
    ns_edges = info["group_edges"]["NS"]
    ew_edges = info["group_edges"]["EW"]
    approach_lanes = info["approach_lanes"]


    current_group = "NS"
    mode = "green"
    green_elapsed = 0
    mode_elapsed = 0
    red_elapsed = {"NS": 0, "EW": 0}
    pressure_hold = 0
    traci.trafficlight.setRedYellowGreenState(tl_id, ns_green)

    decision_rows = []
    switch_rows = []

    def group_queue(name):
        return queue_for_group(ns_lanes if name == "NS" else ew_lanes)

    def opposite(name):
        return "EW" if name == "NS" else "NS"

    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        sim_t = int(traci.simulation.getTime())
        mode_elapsed += 1


        if mode == "green":
            green_elapsed += 1
            red_elapsed[current_group] = 0
            red_elapsed[opposite(current_group)] += 1

        if mode == "green":
            if green_elapsed < min_green:
                continue
            if green_elapsed % check_interval != 0 and green_elapsed < max_green:
                continue

            cur_q = group_queue(current_group)
            opp_group = opposite(current_group)
            opp_q = group_queue(opp_group)
            pressure = opp_q - cur_q


            app_q = {e: queue_for_approach(approach_lanes[e]) for e in sorted(approach_lanes)}
            lane_q = {}
            for e in sorted(approach_lanes):
                lane_q[e] = {l: traci.lane.getLastStepHaltingNumber(l) for l in approach_lanes[e] if l in traci.lane.getIDList()}

            rule = "hold"
            switch = False
            if green_elapsed >= max_green:
                rule = "max_green"
                switch = True
            elif red_elapsed[opp_group] >= starvation_red:
                rule = "starvation_guard"
                switch = True
            else:
                if pressure >= pressure_delta:
                    pressure_hold += 1
                else:
                    pressure_hold = 0
                if pressure_hold >= pressure_hold_checks:
                    rule = "pressure_hysteresis"
                    switch = True

            decision_rows.append(
                {
                    "time": sim_t,
                    "mode": mode,
                    "active_group": current_group,
                    "green_elapsed": green_elapsed,
                    "ns_group_queue": group_queue("NS"),
                    "ew_group_queue": group_queue("EW"),
                    "cur_group_queue": cur_q,
                    "opp_group_queue": opp_q,
                    "pressure_opp_minus_cur": pressure,
                    "red_elapsed_ns": red_elapsed["NS"],
                    "red_elapsed_ew": red_elapsed["EW"],
                    "pressure_hold": pressure_hold,
                    "rule": rule,
                    "switch": int(switch),
                    "approach_queues": ";".join(f"{k}:{app_q[k]}" for k in sorted(app_q)),
                    "lane_queues": ";".join(
                        f"{e}[{','.join(f'{lid}:{lane_q[e][lid]}' for lid in sorted(lane_q[e]))}]"
                        for e in sorted(lane_q)
                    ),
                }
            )

            if switch:
                switch_rows.append(
                    {
                        "time": sim_t,
                        "from_group": current_group,
                        "to_group": opp_group,
                        "green_elapsed": green_elapsed,
                        "rule": rule,
                        "ns_group_queue": group_queue("NS"),
                        "ew_group_queue": group_queue("EW"),
                        "red_elapsed_ns": red_elapsed["NS"],
                        "red_elapsed_ew": red_elapsed["EW"],
                    }
                )
                mode = "yellow"
                mode_elapsed = 0
                pressure_hold = 0
                traci.trafficlight.setRedYellowGreenState(tl_id, ns_yellow if current_group == "NS" else ew_yellow)
                continue

        if mode == "yellow":
            if mode_elapsed >= 3:
                mode = "allred"
                mode_elapsed = 0
                traci.trafficlight.setRedYellowGreenState(tl_id, all_red)
                continue

        if mode == "allred":
            if mode_elapsed >= 2:
                mode = "green"
                mode_elapsed = 0
                green_elapsed = 0
                current_group = opposite(current_group)
                traci.trafficlight.setRedYellowGreenState(tl_id, ns_green if current_group == "NS" else ew_green)

    traci.close()


    with decision_log.open("w", newline="", encoding="utf-8") as f:
        if decision_rows:
            w = csv.DictWriter(f, fieldnames=list(decision_rows[0].keys()))
            w.writeheader()
            w.writerows(decision_rows)
    with switch_log.open("w", newline="", encoding="utf-8") as f:
        if switch_rows:
            w = csv.DictWriter(f, fieldnames=list(switch_rows[0].keys()))
            w.writeheader()
            w.writerows(switch_rows)

    print(f"scenario={out_scenario} seed={seed} tl={tl_id}")
    print(f"outputs={out_dir}")
    print(
        f"config: minGreen={min_green}, maxGreen={max_green}, checkInterval={check_interval}, "
        f"pressureDelta={pressure_delta}, starvationRed={starvation_red}, holdChecks={pressure_hold_checks}"
    )


def main():
    parser = argparse.ArgumentParser(description="Pressure-based TraCI actuated baseline controller.")
    parser.add_argument("--scenario", choices=["offpeak", "peak", "oversat"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-green", type=int, default=35)
    parser.add_argument("--max-green", type=int, default=60)
    parser.add_argument("--check-interval", type=int, default=5)
    parser.add_argument("--pressure-delta", type=int, default=20)
    parser.add_argument("--starvation-red", type=int, default=90)
    parser.add_argument("--pressure-hold-checks", type=int, default=2)
    parser.add_argument("--route-file", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--scenario-label", type=str, default=None)
    args = parser.parse_args()

    run_controller(
        scenario=args.scenario,
        seed=args.seed,
        min_green=args.min_green,
        max_green=args.max_green,
        check_interval=args.check_interval,
        pressure_delta=args.pressure_delta,
        starvation_red=args.starvation_red,
        pressure_hold_checks=args.pressure_hold_checks,
        route_file=args.route_file,
        output_root=args.output_root,
        scenario_label=args.scenario_label,
    )


if __name__ == "__main__":
    main()
