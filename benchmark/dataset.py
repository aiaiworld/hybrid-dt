from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


FEATURES = [
    "cpu",
    "memory",
    "request_rate",
    "sessions",
    "throughput",
    "latency",
    "loss",
    "utilization",
    "arrival_rate",
    "service_rate",
    "sla_latency",
]

SLICES = ["eMBB", "URLLC", "mMTC"]
NFS = ["AMF", "SMF", "UPF1", "UPF2", "PCF", "NSSF", "NRF", "UDM"]
UE_GROUPS = ["UE_eMBB", "UE_URLLC", "UE_mMTC"]
NODES = NFS + SLICES + UE_GROUPS


@dataclass(frozen=True)
class DatasetBundle:
    x: np.ndarray
    y_latency: np.ndarray
    y_violation: np.ndarray
    graph: Dict[str, np.ndarray]
    metadata: Dict[str, object]


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = adj.copy().astype(float)
    adj += np.eye(adj.shape[0])
    degree = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    return adj / degree


def build_5gc_graph() -> Dict[str, np.ndarray]:
    n = len(NODES)
    idx = {name: i for i, name in enumerate(NODES)}

    control = np.zeros((n, n))
    user = np.zeros((n, n))
    slice_dep = np.zeros((n, n))

    control_edges = [
        ("AMF", "SMF"),
        ("AMF", "UDM"),
        ("AMF", "NRF"),
        ("AMF", "NSSF"),
        ("SMF", "PCF"),
        ("SMF", "NRF"),
        ("PCF", "UDM"),
        ("NSSF", "NRF"),
    ]
    user_edges = [
        ("SMF", "UPF1"),
        ("SMF", "UPF2"),
        ("UPF1", "UE_eMBB"),
        ("UPF1", "UE_URLLC"),
        ("UPF2", "UE_eMBB"),
        ("UPF2", "UE_mMTC"),
    ]
    slice_edges = [
        ("eMBB", "AMF"),
        ("eMBB", "SMF"),
        ("eMBB", "UPF1"),
        ("eMBB", "UPF2"),
        ("eMBB", "UE_eMBB"),
        ("URLLC", "AMF"),
        ("URLLC", "SMF"),
        ("URLLC", "UPF1"),
        ("URLLC", "UE_URLLC"),
        ("mMTC", "AMF"),
        ("mMTC", "SMF"),
        ("mMTC", "UPF2"),
        ("mMTC", "UE_mMTC"),
    ]

    for edges, mat in [(control_edges, control), (user_edges, user), (slice_edges, slice_dep)]:
        for src, dst in edges:
            mat[idx[src], idx[dst]] = 1.0
            mat[idx[dst], idx[src]] = 1.0

    return {
        "control": _normalize_adjacency(control),
        "user": _normalize_adjacency(user),
        "slice": _normalize_adjacency(slice_dep),
        "all": _normalize_adjacency(control + user + slice_dep),
    }


def make_windows(
    series: np.ndarray,
    slice_latency: np.ndarray,
    slice_violation: np.ndarray,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, yl, yv = [], [], []
    last_start = len(series) - window - horizon + 1
    for start in range(last_start):
        end = start + window
        target_t = end + horizon - 1
        xs.append(series[start:end])
        yl.append(slice_latency[target_t])
        yv.append(slice_violation[target_t])
    return np.asarray(xs), np.asarray(yl), np.asarray(yv)


def generate_synthetic_5gc(
    timesteps: int = 1800,
    window: int = 12,
    horizon: int = 3,
    seed: int = 7,
    noise: float = 0.035,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_nodes, n_features = len(NODES), len(FEATURES)
    idx = {name: i for i, name in enumerate(NODES)}
    f = {name: i for i, name in enumerate(FEATURES)}

    graph = build_5gc_graph()
    series = np.zeros((timesteps, n_nodes, n_features), dtype=float)
    slice_latency = np.zeros((timesteps, len(SLICES)), dtype=float)
    slice_violation = np.zeros((timesteps, len(SLICES)), dtype=float)

    base_service = {
        "AMF": 145.0,
        "SMF": 115.0,
        "UPF1": 210.0,
        "UPF2": 170.0,
        "PCF": 90.0,
        "NSSF": 75.0,
        "NRF": 95.0,
        "UDM": 88.0,
    }
    sla = np.array([45.0, 12.0, 80.0])
    slice_mix = np.array([0.58, 0.12, 0.30])

    bursts = np.zeros(timesteps)
    for center in rng.choice(np.arange(80, timesteps - 80), size=9, replace=False):
        width = rng.integers(12, 46)
        amp = rng.uniform(0.25, 0.75)
        pulses = amp * np.exp(-0.5 * ((np.arange(timesteps) - center) / width) ** 2)
        bursts += pulses

    smoothed_load = np.zeros(3)
    for t in range(timesteps):
        daily = 0.5 + 0.5 * np.sin(2 * np.pi * t / 288.0 - 0.8)
        fast = 0.5 + 0.5 * np.sin(2 * np.pi * t / 53.0)
        total = 65.0 + 95.0 * daily + 22.0 * fast + 130.0 * bursts[t]
        raw_slice_load = total * slice_mix * rng.normal(1.0, 0.06, size=3)
        raw_slice_load[1] += 12.0 * bursts[t] + 6.0 * fast
        raw_slice_load[2] += 18.0 * (daily > 0.82)
        smoothed_load = 0.82 * smoothed_load + 0.18 * raw_slice_load

        embb, urllc, mmtc = smoothed_load
        control_arrival = 0.34 * embb + 0.95 * urllc + 0.48 * mmtc
        session_arrival = 0.45 * embb + 0.40 * urllc + 0.22 * mmtc
        upf1_arrival = 0.62 * embb + 1.25 * urllc
        upf2_arrival = 0.38 * embb + 0.95 * mmtc

        arrivals = {
            "AMF": control_arrival,
            "SMF": session_arrival,
            "UPF1": upf1_arrival,
            "UPF2": upf2_arrival,
            "PCF": 0.34 * session_arrival,
            "NSSF": 0.16 * control_arrival,
            "NRF": 0.21 * control_arrival,
            "UDM": 0.28 * control_arrival,
        }

        # Service capacity drifts as pods autoscale or contend for CPU.
        for nf in NFS:
            node = idx[nf]
            service = base_service[nf] * (1.0 + 0.08 * np.sin(2 * np.pi * t / 417.0 + node))
            service *= rng.normal(1.0, 0.018)
            arrival = arrivals[nf]
            util = np.clip(arrival / max(service, 1.0), 0.02, 0.985)
            queue_delay = 1.0 / max(service - arrival, 3.0)
            latency = 3.0 + 35.0 * queue_delay + 18.0 * util**2
            cpu = np.clip(0.08 + 0.82 * util + rng.normal(0, noise), 0, 1)
            memory = np.clip(0.18 + 0.48 * util + 0.05 * bursts[t] + rng.normal(0, noise), 0, 1)
            loss = np.clip((util - 0.78) * 0.12 + rng.normal(0, noise / 4), 0, 0.2)

            series[t, node, f["cpu"]] = cpu
            series[t, node, f["memory"]] = memory
            series[t, node, f["request_rate"]] = arrival
            series[t, node, f["sessions"]] = 1.7 * session_arrival if nf in {"SMF", "UPF1", "UPF2"} else arrival
            series[t, node, f["throughput"]] = arrival * (3.8 if nf.startswith("UPF") else 0.4)
            series[t, node, f["latency"]] = latency
            series[t, node, f["loss"]] = loss
            series[t, node, f["utilization"]] = util
            series[t, node, f["arrival_rate"]] = arrival
            series[t, node, f["service_rate"]] = service
            series[t, node, f["sla_latency"]] = 0.0

        nf_lat = {nf: series[t, idx[nf], f["latency"]] for nf in NFS}
        nf_loss = {nf: series[t, idx[nf], f["loss"]] for nf in NFS}
        latencies = np.array(
            [
                0.28 * nf_lat["AMF"] + 0.34 * nf_lat["SMF"] + 0.52 * min(nf_lat["UPF1"], nf_lat["UPF2"]) + 7.0 * nf_loss["UPF1"],
                0.42 * nf_lat["AMF"] + 0.36 * nf_lat["SMF"] + 0.72 * nf_lat["UPF1"] + 13.0 * nf_loss["UPF1"],
                0.46 * nf_lat["AMF"] + 0.22 * nf_lat["SMF"] + 0.44 * nf_lat["UPF2"] + 5.5 * nf_loss["UPF2"],
            ]
        )
        latencies += rng.normal(0, [1.0, 0.45, 1.4])
        slice_latency[t] = np.maximum(latencies, 0.1)
        slice_violation[t] = (slice_latency[t] > sla).astype(float)

        for s, name in enumerate(SLICES):
            node = idx[name]
            series[t, node, f["request_rate"]] = smoothed_load[s]
            series[t, node, f["sessions"]] = smoothed_load[s] * (1.4 if name != "mMTC" else 3.8)
            series[t, node, f["throughput"]] = smoothed_load[s] * [4.2, 1.1, 0.45][s]
            series[t, node, f["latency"]] = slice_latency[t, s]
            series[t, node, f["loss"]] = [nf_loss["UPF1"], nf_loss["UPF1"], nf_loss["UPF2"]][s]
            series[t, node, f["utilization"]] = slice_latency[t, s] / sla[s]
            series[t, node, f["arrival_rate"]] = smoothed_load[s]
            series[t, node, f["service_rate"]] = sla[s]
            series[t, node, f["sla_latency"]] = sla[s]
            series[t, node, f["cpu"]] = np.mean([series[t, idx["AMF"], f["cpu"]], series[t, idx["SMF"], f["cpu"]]])
            series[t, node, f["memory"]] = np.mean([series[t, idx["AMF"], f["memory"]], series[t, idx["SMF"], f["memory"]]])

        for s, name in enumerate(UE_GROUPS):
            node = idx[name]
            series[t, node, f["request_rate"]] = smoothed_load[s]
            series[t, node, f["sessions"]] = smoothed_load[s] * [1.2, 0.9, 4.5][s]
            series[t, node, f["throughput"]] = smoothed_load[s] * [4.0, 0.9, 0.35][s]
            series[t, node, f["latency"]] = slice_latency[t, s]
            series[t, node, f["loss"]] = slice_violation[t, s] * 0.04
            series[t, node, f["utilization"]] = min(slice_latency[t, s] / sla[s], 1.5)
            series[t, node, f["arrival_rate"]] = smoothed_load[s]
            series[t, node, f["service_rate"]] = sla[s]
            series[t, node, f["sla_latency"]] = sla[s]

    x, y_latency, y_violation = make_windows(series, slice_latency, slice_violation, window, horizon)
    metadata = {
        "nodes": NODES,
        "features": FEATURES,
        "slices": SLICES,
        "window": window,
        "horizon": horizon,
        "sla_latency_ms": dict(zip(SLICES, sla.tolist())),
        "seed": seed,
    }
    return DatasetBundle(x=x, y_latency=y_latency, y_violation=y_violation, graph=graph, metadata=metadata)


def train_val_test_split(n: int, train: float = 0.65, val: float = 0.15) -> Tuple[slice, slice, slice]:
    n_train = int(n * train)
    n_val = int(n * val)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)
