import argparse
import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMO_HOME = Path(r"C:\Program Files (x86)\Eclipse\Sumo")


def ensure_sumo_home():
    if "SUMO_HOME" not in os.environ:
        os.environ["SUMO_HOME"] = str(DEFAULT_SUMO_HOME)


def load_cfg(cfg_path: Path):
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    last = sum_root.findall("step")[-1]
    departed = int(float(last.attrib.get("inserted", "0")))
    arrived = int(float(last.attrib.get("ended", "0")))
    teleports = int(float(last.attrib.get("teleports", "0")))

    return {
        "departed": departed,
        "arrived": arrived,
        "mean_travel_time_s": round(mean_tt, 2),
        "mean_delay_s": round(mean_delay, 2) if mean_delay != "" else "",
        "throughput": arrived,
        "teleports": teleports,
    }


def build_reward_fn(env_cfg):
    reward_name = env_cfg.get("reward_fn", "queue")
    if reward_name != "queue_switch_penalty":
        return reward_name

    switch_penalty = float(env_cfg.get("switch_penalty", 0.0))
    last_phase = {}

    def reward_fn(ts):
        queue_term = -float(ts.get_total_queued())
        prev = last_phase.get(ts.id, ts.green_phase)
        switched = 1.0 if ts.green_phase != prev else 0.0
        last_phase[ts.id] = ts.green_phase
        return queue_term - switch_penalty * switched

    return reward_fn


def make_eval_env(cfg, seed: int, out_dir: Path):
    from sumo_rl import SumoEnvironment

    net_file = ROOT / cfg["network"]["net_file"]
    route_file = ROOT / cfg["routes"]["peak"]
    sim = cfg["simulation"]
    env_cfg = cfg["env"]
    reward_fn = build_reward_fn(env_cfg)

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
        net_file=str(net_file),
        route_file=str(route_file),
        use_gui=False,
        begin_time=sim["begin_time"],
        num_seconds=sim["num_seconds"],
        time_to_teleport=sim["time_to_teleport"],
        delta_time=env_cfg["delta_time"],
        yellow_time=env_cfg["yellow_time"],
        min_green=env_cfg["min_green"],
        max_green=env_cfg["max_green"],
        single_agent=env_cfg["single_agent"],
        reward_fn=reward_fn,
        sumo_seed=seed,
        fixed_ts=False,
        sumo_warnings=False,
        additional_sumo_cmd=extra_cmd,
    )
    return env, tripinfo, summary


def collect_baseline_row(controller: str, seed: int):
    base_dir = ROOT / "outputs" / "baselines" / controller / "peak" / f"seed_{seed}"
    tripinfo = base_dir / "tripinfo.xml"
    summary = base_dir / "summary.xml"
    m = parse_metrics(tripinfo, summary)
    return {"scenario": "peak", "seed": seed, "controller": controller, **m}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on peak seeds and compare baselines.")
    parser.add_argument("--config", default="configs/rl_peak.yaml")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    ensure_sumo_home()
    cfg = load_cfg(ROOT / args.config)
    patch_all_red_traffic_signal(int(cfg["env"]["all_red_time"]))

    default_model = ROOT / cfg["paths"]["model_dir"] / "ppo_peak_final.zip"
    model_path = Path(args.model_path) if args.model_path else default_model
    model = PPO.load(str(model_path))

    eval_root = ROOT / cfg["paths"]["eval_output_dir"]
    seeds = [int(s) for s in cfg["evaluation"]["seeds"]]
    rows = []

    for seed in seeds:
        out_dir = eval_root / f"seed_{seed}"
        env, tripinfo, summary = make_eval_env(cfg, seed, out_dir)
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
        env.close()

        rl_metrics = parse_metrics(tripinfo, summary)
        rows.append({"scenario": "peak", "seed": seed, "controller": "rl_ppo", **rl_metrics})
        rows.append(collect_baseline_row("fixed", seed))
        rows.append(collect_baseline_row("actuated", seed))

    out_csv = ROOT / cfg["paths"]["eval_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "seed",
        "controller",
        "departed",
        "arrived",
        "mean_travel_time_s",
        "mean_delay_s",
        "throughput",
        "teleports",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Evaluation complete: {out_csv}")
    for r in rows:
        print(",".join(f"{k}={r[k]}" for k in fieldnames))


if __name__ == "__main__":
    main()
