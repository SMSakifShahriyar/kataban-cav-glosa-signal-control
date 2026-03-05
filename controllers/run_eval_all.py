import csv
import os
import re
import statistics as stats
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
SUMO_HOME_DEFAULT = Path(r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_EXE = SUMO_HOME_DEFAULT / "bin" / "sumo.exe"
PYTHON_EXE = sys.executable

SCENARIOS = {
    "offpeak": {
        "route_file": ROOT / "demand" / "routes_offpeak.rou.xml",
        "actuated_cfg_scenario": "offpeak",
    },
    "peak": {
        "route_file": ROOT / "demand" / "routes_peak.rou.xml",
        "actuated_cfg_scenario": "peak",
    },
    "oversat_910": {
        "route_file": ROOT / "demand" / "routes_oversat_910.rou.xml",
        "actuated_cfg_scenario": "oversat",
    },
}
SEEDS = [1, 2, 3, 4, 5]
CONTROLLERS = ["rl", "fixed", "actuated"]

NET_FILE = ROOT / "kataban_joined_nomicros.net.xml"
FIXED_ADD_FILE = ROOT / "tls_fixed.add.xml"
RL_MODEL = ROOT / "outputs" / "rl" / "tune1_run" / "models" / "ppo_peak_final.zip"

TIME_TO_TELEPORT = 300
SIM_BEGIN = 0
SIM_END = 3600


RL_DELTA_TIME = 5
RL_YELLOW = 3
RL_ALL_RED = 2
RL_MIN_GREEN = 30
RL_MAX_GREEN = 60
RL_REWARD_NAME = "queue_switch_penalty"
RL_SWITCH_PENALTY = 10.0

RUNS_CSV = ROOT / "results" / "rl_vs_baselines_all_runs.csv"
AGG_CSV = ROOT / "results" / "rl_vs_baselines_all_aggregate.csv"
OUTPUT_ROOT = ROOT / "outputs" / "eval_all"


def ensure_sumo():
    sumo_home = Path(os.environ.get("SUMO_HOME", str(SUMO_HOME_DEFAULT)))
    if not sumo_home.exists():
        raise FileNotFoundError(f"SUMO_HOME not found: {sumo_home}")
    os.environ["SUMO_HOME"] = str(sumo_home)
    tools = sumo_home / "tools"
    if str(tools) not in sys.path:
        sys.path.append(str(tools))
    if not (sumo_home / "bin" / "sumo.exe").exists():
        raise FileNotFoundError(f"sumo.exe not found under: {sumo_home / 'bin'}")


def patch_all_red_traffic_signal(all_red_time: int):
    from sumo_rl.environment import env as sumo_env_mod
    from sumo_rl.environment.traffic_signal import TrafficSignal

    class AllRedTrafficSignal(TrafficSignal):
        ALL_RED_TIME = all_red_time

        def _build_phases(self):
            super()._build_phases()
            self.pending_green_phase = self.green_phase
            self.in_all_red = False
            all_red_state = "r" * len(self.all_phases[0].state)
            self.all_red_phase_idx = len(self.all_phases)
            self.all_phases.append(self.sumo.trafficlight.Phase(self.ALL_RED_TIME, all_red_state))

            programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
            logic = programs[0]
            logic.type = 0
            logic.phases = self.all_phases
            self.sumo.trafficlight.setProgramLogic(self.id, logic)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)

        def set_next_phase(self, new_phase: int):
            new_phase = int(new_phase)
            force_switch = self.time_since_last_phase_change >= self.max_green
            if force_switch and new_phase == self.green_phase:
                new_phase = (self.green_phase + 1) % self.num_green_phases

            min_block = self.yellow_time + self.ALL_RED_TIME + self.min_green
            if (self.green_phase == new_phase and not force_switch) or (
                self.time_since_last_phase_change < min_block and not force_switch
            ):
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                self.next_action_time = self.env.sim_step + self.delta_time
                return

            yellow_idx = self.yellow_dict[(self.green_phase, new_phase)]
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[yellow_idx].state)
            self.pending_green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.in_all_red = False
            self.time_since_last_phase_change = 0

        def update(self):
            self.time_since_last_phase_change += 1
            if self.is_yellow and not self.in_all_red and self.time_since_last_phase_change == self.yellow_time:
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.all_red_phase_idx].state)
                self.in_all_red = True
            elif (
                self.is_yellow
                and self.in_all_red
                and self.time_since_last_phase_change == self.yellow_time + self.ALL_RED_TIME
            ):
                self.green_phase = self.pending_green_phase
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                self.is_yellow = False
                self.in_all_red = False

    sumo_env_mod.TrafficSignal = AllRedTrafficSignal


def build_reward_fn():
    if RL_REWARD_NAME != "queue_switch_penalty":
        return RL_REWARD_NAME
    last_phase = {}

    def reward_fn(ts):
        queue_term = -float(ts.get_total_queued())
        prev = last_phase.get(ts.id, ts.green_phase)
        switched = 1.0 if ts.green_phase != prev else 0.0
        last_phase[ts.id] = ts.green_phase
        return queue_term - RL_SWITCH_PENALTY * switched

    return reward_fn


def parse_metrics(tripinfo_path: Path, summary_path: Path):
    trip_root = ET.parse(tripinfo_path).getroot()
    trips = trip_root.findall("tripinfo")
    mean_tt = 0.0
    mean_delay = ""
    if trips:
        mean_tt = sum(float(t.attrib.get("duration", "0")) for t in trips) / len(trips)
        if all("timeLoss" in t.attrib for t in trips):
            mean_delay = sum(float(t.attrib["timeLoss"]) for t in trips) / len(trips)

    sum_root = ET.parse(summary_path).getroot()
    steps = sum_root.findall("step")
    if not steps:
        raise RuntimeError(f"No <step> entries in summary: {summary_path}")
    last = steps[-1]
    departed = int(float(last.attrib.get("inserted", "0")))
    arrived = int(float(last.attrib.get("ended", "0")))
    teleports_total = int(float(last.attrib.get("teleports", "0")))
    return {
        "departed": departed,
        "arrived": arrived,
        "teleports_total": teleports_total,
        "mean_travel_time_s": round(mean_tt, 2),
        "mean_delay_s": round(mean_delay, 2) if mean_delay != "" else "",
        "throughput": arrived,
    }


def parse_teleports_by_reason(log_path: Path):
    jam = 0
    yield_ = 0
    pattern = re.compile(r"Teleporting vehicle .*?\(([^)]+)\)")
    if not log_path.exists():
        return jam, yield_
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            reason = m.group(1).strip().lower()
            if "jam" in reason:
                jam += 1
            elif "yield" in reason:
                yield_ += 1
    return jam, yield_


def run_fixed(scenario: str, route_file: Path, seed: int):
    out_dir = OUTPUT_ROOT / "fixed" / scenario / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tripinfo = out_dir / "tripinfo.xml"
    summary = out_dir / "summary.xml"
    queue = out_dir / "queue.xml"
    log = out_dir / "sumo.log"

    cmd = [
        str(SUMO_EXE),
        "--net-file",
        str(NET_FILE),
        "--route-files",
        str(route_file),
        "--additional-files",
        str(FIXED_ADD_FILE),
        "--begin",
        str(SIM_BEGIN),
        "--end",
        str(SIM_END),
        "--time-to-teleport",
        str(TIME_TO_TELEPORT),
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
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Fixed run failed: scenario={scenario}, seed={seed}\n{proc.stdout}")
    return tripinfo, summary, log


def run_actuated(scenario: str, route_file: Path, cfg_scenario: str, seed: int):
    out_root = OUTPUT_ROOT / "actuated"
    out_dir = out_root / scenario / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        str(ROOT / "actuated_controller.py"),
        "--scenario",
        cfg_scenario,
        "--scenario-label",
        scenario,
        "--seed",
        str(seed),
        "--min-green",
        "35",
        "--max-green",
        "60",
        "--check-interval",
        "5",
        "--pressure-delta",
        "20",
        "--starvation-red",
        "90",
        "--pressure-hold-checks",
        "2",
        "--route-file",
        str(route_file),
        "--output-root",
        str(out_root),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Actuated run failed: scenario={scenario}, seed={seed}\n{proc.stdout}")

    tripinfo = out_dir / "tripinfo.xml"
    summary = out_dir / "summary.xml"
    log = out_dir / "sumo.log"
    return tripinfo, summary, log


def make_rl_env(route_file: Path, seed: int, out_dir: Path):
    from sumo_rl import SumoEnvironment

    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = out_dir / "tripinfo.xml"
    summary = out_dir / "summary.xml"
    queue = out_dir / "queue.xml"
    log = out_dir / "sumo.log"

    extra_cmd = (
        f"--tripinfo-output {tripinfo} "
        f"--summary-output {summary} "
        f"--queue-output {queue} "
        f"--duration-log.statistics true "
        f"--no-step-log true "
        f"--log {log}"
    )
    env = SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(route_file),
        use_gui=False,
        begin_time=SIM_BEGIN,
        num_seconds=SIM_END,
        time_to_teleport=TIME_TO_TELEPORT,
        delta_time=RL_DELTA_TIME,
        yellow_time=RL_YELLOW,
        min_green=RL_MIN_GREEN,
        max_green=RL_MAX_GREEN,
        single_agent=True,
        reward_fn=build_reward_fn(),
        sumo_seed=seed,
        fixed_ts=False,
        sumo_warnings=False,
        additional_sumo_cmd=extra_cmd,
    )
    return env, tripinfo, summary, log


def run_rl(model, scenario: str, route_file: Path, seed: int):
    out_dir = OUTPUT_ROOT / "rl" / scenario / f"seed_{seed}"
    env, tripinfo, summary, log = make_rl_env(route_file, seed, out_dir)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
    env.close()
    return tripinfo, summary, log


def format_pm(mean_value, std_value):
    return f"{mean_value:.2f} +- {std_value:.2f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]):
    grouped = {}
    for r in rows:
        key = (r["controller"], r["scenario"])
        grouped.setdefault(key, []).append(r)

    out = []
    for (controller, scenario), group in sorted(grouped.items()):
        tt = [float(x["mean_travel_time_s"]) for x in group]
        delay = [float(x["mean_delay_s"]) if x["mean_delay_s"] != "" else 0.0 for x in group]
        th = [float(x["throughput"]) for x in group]
        tele = [float(x["teleports_total"]) for x in group]

        tt_mean = stats.mean(tt)
        tt_std = stats.stdev(tt) if len(tt) > 1 else 0.0
        delay_mean = stats.mean(delay)
        delay_std = stats.stdev(delay) if len(delay) > 1 else 0.0
        th_mean = stats.mean(th)
        th_std = stats.stdev(th) if len(th) > 1 else 0.0
        tele_mean = stats.mean(tele)
        tele_std = stats.stdev(tele) if len(tele) > 1 else 0.0

        out.append(
            {
                "controller": controller,
                "scenario": scenario,
                "tt_mean_s": round(tt_mean, 2),
                "tt_std_s": round(tt_std, 2),
                "tt_mean_pm_std": format_pm(tt_mean, tt_std),
                "delay_mean_s": round(delay_mean, 2),
                "delay_std_s": round(delay_std, 2),
                "delay_mean_pm_std": format_pm(delay_mean, delay_std),
                "throughput_mean": round(th_mean, 2),
                "throughput_std": round(th_std, 2),
                "throughput_mean_pm_std": format_pm(th_mean, th_std),
                "teleports_mean": round(tele_mean, 2),
                "teleports_std": round(tele_std, 2),
                "teleports_mean_pm_std": format_pm(tele_mean, tele_std),
            }
        )
    return out


def print_summary_markdown(agg_rows: list[dict]):
    print("\ncontroller | scenario | TT_mean+-std | throughput_mean+-std | teleports_mean+-std")
    print("--- | --- | --- | --- | ---")
    for r in agg_rows:
        print(
            f"{r['controller']} | {r['scenario']} | {r['tt_mean_pm_std']} | "
            f"{r['throughput_mean_pm_std']} | {r['teleports_mean_pm_std']}"
        )


def run_all():
    ensure_sumo()
    patch_all_red_traffic_signal(RL_ALL_RED)

    if not NET_FILE.exists():
        raise FileNotFoundError(f"Missing network: {NET_FILE}")
    if not FIXED_ADD_FILE.exists():
        raise FileNotFoundError(f"Missing fixed additional file: {FIXED_ADD_FILE}")
    if not RL_MODEL.exists():
        raise FileNotFoundError(f"Missing RL model: {RL_MODEL}")
    for s in SCENARIOS.values():
        if not s["route_file"].exists():
            raise FileNotFoundError(f"Missing route file: {s['route_file']}")

    model = PPO.load(str(RL_MODEL))
    rows = []
    total = len(CONTROLLERS) * len(SCENARIOS) * len(SEEDS)
    done = 0

    for controller in CONTROLLERS:
        for scenario, spec in SCENARIOS.items():
            route_file = spec["route_file"]
            cfg_scenario = spec["actuated_cfg_scenario"]
            for seed in SEEDS:
                done += 1
                print(f"[{done}/{total}] controller={controller} scenario={scenario} seed={seed}")
                if controller == "fixed":
                    tripinfo, summary, log = run_fixed(scenario, route_file, seed)
                elif controller == "actuated":
                    tripinfo, summary, log = run_actuated(scenario, route_file, cfg_scenario, seed)
                else:
                    tripinfo, summary, log = run_rl(model, scenario, route_file, seed)

                metrics = parse_metrics(tripinfo, summary)
                tele_jam, tele_yield = parse_teleports_by_reason(log)
                rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "seed": seed,
                        "departed": metrics["departed"],
                        "arrived": metrics["arrived"],
                        "teleports_total": metrics["teleports_total"],
                        "teleports_jam": tele_jam,
                        "teleports_yield": tele_yield,
                        "mean_travel_time_s": metrics["mean_travel_time_s"],
                        "mean_delay_s": metrics["mean_delay_s"],
                        "throughput": metrics["throughput"],
                    }
                )

    write_csv(
        RUNS_CSV,
        rows,
        [
            "controller",
            "scenario",
            "seed",
            "departed",
            "arrived",
            "teleports_total",
            "teleports_jam",
            "teleports_yield",
            "mean_travel_time_s",
            "mean_delay_s",
            "throughput",
        ],
    )
    agg_rows = aggregate_rows(rows)
    write_csv(
        AGG_CSV,
        agg_rows,
        [
            "controller",
            "scenario",
            "tt_mean_s",
            "tt_std_s",
            "tt_mean_pm_std",
            "delay_mean_s",
            "delay_std_s",
            "delay_mean_pm_std",
            "throughput_mean",
            "throughput_std",
            "throughput_mean_pm_std",
            "teleports_mean",
            "teleports_std",
            "teleports_mean_pm_std",
        ],
    )

    print("\nSanity checks:")
    bad_tp = [r for r in rows if int(r["teleports_total"]) > 0]
    if bad_tp:
        print("Teleport violations found:")
        for r in bad_tp:
            print(
                f"  controller={r['controller']} scenario={r['scenario']} "
                f"seed={r['seed']} teleports_total={r['teleports_total']} "
                f"(jam={r['teleports_jam']} yield={r['teleports_yield']})"
            )
    else:
        print("Teleports are zero across all 45 runs.")

    zero_arrived = [r for r in rows if int(r["arrived"]) <= 0]
    if zero_arrived:
        print("Arrivals <= 0 found:")
        for r in zero_arrived:
            print(f"  controller={r['controller']} scenario={r['scenario']} seed={r['seed']} arrived={r['arrived']}")
    else:
        print("All runs have arrivals > 0.")

    very_low = [r for r in rows if int(r["arrived"]) < 100]
    if very_low:
        print("Very low arrivals (<100) found:")
        for r in very_low:
            print(f"  controller={r['controller']} scenario={r['scenario']} seed={r['seed']} arrived={r['arrived']}")
    else:
        print("No very low-arrival runs (<100).")

    print_summary_markdown(agg_rows)
    print(f"\nWrote per-run CSV: {RUNS_CSV}")
    print(f"Wrote aggregate CSV: {AGG_CSV}")


if __name__ == "__main__":
    run_all()
