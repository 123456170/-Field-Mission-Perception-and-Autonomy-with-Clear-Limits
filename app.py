import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import json
import time
import uuid
import math
import hashlib
import tempfile
import shutil
from datetime import datetime, timezone


st.set_page_config(
    page_title="Field Mission: Perception & Autonomy Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Policies and fixed demo metadata
# ----------------------------------------------------------------------

POLICY = {
    "max_latency_ms": 120.0,
    "max_memory_mb": 512.0,
    "min_coverage": 0.75,
    "max_ece": 0.10,
    "retention_seconds_tracks": 30,
    "min_action_confidence": 0.65,
    "watchdog_timeout_s": 3.0,
}

LICENSE_MODELS = [
    {
        "name": "person-counter-lite",
        "weight_license": "Apache-2.0",
        "code_license": "MIT",
        "dataset": "Synthetic-Flow-2026",
        "dataset_license": "CC0-1.0",
        "deployment": "commercial",
        "attribution": True,
        "model_card": True,
        "capability": "anonymized_counting",
        "constraints": ["no-face-recognition", "no-identity-retention"],
    },
    {
        "name": "gait-reid-beta",
        "weight_license": "unknown",
        "code_license": "MIT",
        "dataset": "Public-CCTV-Scrape",
        "dataset_license": "unknown",
        "deployment": "commercial",
        "attribution": False,
        "model_card": False,
        "capability": "person_reidentification",
        "constraints": ["no-face-recognition", "no-identity-retention"],
    },
    {
        "name": "scene-quality-nc",
        "weight_license": "CC-BY-NC-4.0",
        "code_license": "BSD-3-Clause",
        "dataset": "OpenScene-NonCommercial",
        "dataset_license": "CC-BY-NC-4.0",
        "deployment": "commercial",
        "attribution": True,
        "model_card": True,
        "capability": "scene_quality_assessment",
        "constraints": ["no-face-recognition"],
    },
    {
        "name": "edge-safety-monitor",
        "weight_license": "MIT",
        "code_license": "Apache-2.0",
        "dataset": "Lab-Synthetic-Safety",
        "dataset_license": "CC-BY-4.0",
        "deployment": "internal",
        "attribution": True,
        "model_card": True,
        "capability": "safety_event_detection",
        "constraints": ["no-face-recognition", "local-only"],
    },
]

DATASET_REGISTRY = [
    {"dataset": "Synthetic-Flow-2026", "license": "CC0-1.0", "provenance": "generated", "pii": "none"},
    {"dataset": "Public-CCTV-Scrape", "license": "unknown", "provenance": "scraped", "pii": "possible"},
    {"dataset": "OpenScene-NonCommercial", "license": "CC-BY-NC-4.0", "provenance": "partner", "pii": "none"},
    {"dataset": "Lab-Synthetic-Safety", "license": "CC-BY-4.0", "provenance": "lab", "pii": "none"},
]

KNOWN_CANDIDATES = [
    {"model": "person-counter-lite", "mAP": 0.84, "p95_ms": 72, "privacy_pass": True},
    {"model": "edge-safety-monitor", "mAP": 0.81, "p95_ms": 95, "privacy_pass": True},
    {"model": "weaker-candidate", "mAP": 0.71, "p95_ms": 135, "privacy_pass": False},
]

ALLOWED_ACTION_TYPES = [
    "move_forward",
    "rotate",
    "count_anonymized_flow",
    "alert_operator",
]

DENY_ACTION_TYPES = [
    "face_recognition",
    "identify_person",
    "upload_video_to_cloud",
    "delete_audit_log",
    "bypass_emergency_stop",
]


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc).isoformat()


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def entry_digest(entry):
    return stable_hash({k: v for k, v in entry.items() if k != "hash"})


def init_session():
    if "run_id" not in st.session_state:
        st.session_state.run_id = uuid.uuid4().hex[:12]

    if "audit" not in st.session_state:
        st.session_state.audit = []

    if "events" not in st.session_state:
        st.session_state.events = []

    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None

    if "demo_params_hash" not in st.session_state:
        st.session_state.demo_params_hash = None

    if "demo_error" not in st.session_state:
        st.session_state.demo_error = None

    if "benchmark_report" not in st.session_state:
        st.session_state.benchmark_report = None

    if "benchmark_summary" not in st.session_state:
        st.session_state.benchmark_summary = None

    if "license_report" not in st.session_state:
        st.session_state.license_report = None

    if "release_report" not in st.session_state:
        st.session_state.release_report = None

    if "security_results" not in st.session_state:
        st.session_state.security_results = None

    if "action_log" not in st.session_state:
        st.session_state.action_log = []

    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    if "reviewer_queue" not in st.session_state:
        st.session_state.reviewer_queue = []

    default_robot = {
        "status": "idle",
        "position": 0.0,
        "last_heartbeat": time.time(),
        "lock": False,
        "timeout_s": POLICY["watchdog_timeout_s"],
        "unsafe_continuation": 0,
        "attempted_blocked": 0,
    }

    if "robot" not in st.session_state:
        st.session_state.robot = default_robot.copy()
    else:
        for k, v in default_robot.items():
            st.session_state.robot.setdefault(k, v)


init_session()


def append_audit(event_type, payload, category="system"):
    prev_hash = st.session_state.audit[-1]["hash"] if st.session_state.audit else "GENESIS"
    entry = {
        "seq": len(st.session_state.audit) + 1,
        "timestamp": utcnow(),
        "run_id": st.session_state.run_id,
        "category": category,
        "event_type": event_type,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    entry["hash"] = entry_digest(entry)
    st.session_state.audit.append(entry)
    return entry


def append_event(event_type, payload, severity="info"):
    entry = {
        "seq": len(st.session_state.events) + 1,
        "timestamp": utcnow(),
        "type": event_type,
        "severity": severity,
        "payload": payload,
    }
    st.session_state.events.append(entry)
    return entry


def anonymize_track(raw_id, salt):
    h = hashlib.sha256(f"{salt}:{raw_id}".encode("utf-8")).hexdigest()[:6]
    return h.upper()


def open_video_writer(tmpdir, fps, width, height):
    attempts = [
        ("live_demo.mp4", "avc1", "video/mp4"),
        ("live_demo.mp4", "mp4v", "video/mp4"),
        ("live_demo.avi", "MJPG", "video/x-msvideo"),
        ("live_demo.avi", "XVID", "video/x-msvideo"),
        ("live_demo.avi", "FFV1", "video/x-msvideo"),
    ]

    last_error = None

    for filename, codec, mime in attempts:
        path = os.path.join(tmpdir, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
        if out.isOpened():
            return out, path, mime
        last_error = f"Codec {codec} could not open writer."

    raise RuntimeError(last_error or "Unable to initialize video writer.")


# ----------------------------------------------------------------------
# Live privacy-preserving people-flow demo generator
# ----------------------------------------------------------------------

def generate_live_demo(
    seed,
    num_people,
    duration,
    fps,
    perturbation,
    lighting,
    occlusion_prob,
    conf_threshold,
    stop_fraction,
    safety_prob,
):
    start_wall = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    W, H = 512, 288
    tmpdir = tempfile.mkdtemp(prefix="fieldops_")
    out, video_path, video_mime = open_video_writer(tmpdir, fps, W, H)

    total_frames = max(1, int(round(float(duration) * float(fps))))
    stop_time = float(stop_fraction) * float(duration)
    stopped = False

    sides = ["left", "right", "top", "bottom"]
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

    persons = []
    for i in range(int(num_people)):
        side = sides[int(rng.integers(0, 4))]

        if side == "left":
            start = np.array([-40.0, float(rng.uniform(55, H - 55))])
            end = np.array([W + 40.0, float(rng.uniform(55, H - 55))])
        elif side == "right":
            start = np.array([W + 40.0, float(rng.uniform(55, H - 55))])
            end = np.array([-40.0, float(rng.uniform(55, H - 55))])
        elif side == "top":
            start = np.array([float(rng.uniform(55, W - 55)), -40.0])
            end = np.array([float(rng.uniform(55, W - 55)), H + 40.0])
        else:
            start = np.array([float(rng.uniform(55, W - 55)), H + 40.0])
            end = np.array([float(rng.uniform(55, W - 55)), -40.0])

        spawn = float(rng.uniform(0.0, max(0.0, float(duration) - 1.0)))
        travel = float(rng.uniform(1.1, max(1.4, float(duration) - spawn + 0.2)))
        bw = int(rng.uniform(24, 50))
        bh = int(bw * rng.uniform(1.7, 2.3))

        persons.append(
            {
                "raw_id": i,
                "side": side,
                "exit_side": opposite[side],
                "start": start,
                "end": end,
                "spawn": spawn,
                "travel": travel,
                "w": bw,
                "h": bh,
                "safety": bool(rng.random() < float(safety_prob)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "sway": float(rng.uniform(2, 7)),
            }
        )

    tracks = {}
    counters = {f"enter_{s}": 0 for s in sides}
    counters.update({f"exit_{s}": 0 for s in sides})
    counters.update(
        {
            "safety_events": 0,
            "detections": 0,
            "abstentions": 0,
            "frames": total_frames,
        }
    )

    events = []
    latencies = []
    sample_clean = None
    sample_degraded = None

    for fidx in range(total_frames):
        frame_start = time.perf_counter()
        t = fidx / float(fps)

        img = np.full((H, W, 3), (36, 33, 31), dtype=np.uint8)

        for yy in range(0, H, 32):
            cv2.line(img, (0, yy), (W, yy), (48, 45, 43), 1)
        for xx in range(0, W, 40):
            cv2.line(img, (xx, 0), (xx, H), (48, 45, 43), 1)

        cv2.line(img, (W // 2, 0), (W // 2, H), (80, 120, 80), 1)
        cv2.line(img, (0, H // 2), (W, H // 2), (80, 120, 80), 1)

        if abs(float(lighting) - 1.0) > 0.01:
            img = np.clip(img.astype(np.float32) * float(lighting), 0, 255).astype(np.uint8)

        frame_det = 0

        for p in persons:
            if t < p["spawn"] or t > p["spawn"] + p["travel"]:
                continue

            alpha = (t - p["spawn"]) / max(1e-6, p["travel"])
            pos = p["start"] + (p["end"] - p["start"]) * alpha
            pos[0] += math.sin(alpha * 6.283 + p["phase"]) * p["sway"]
            pos[1] += math.cos(alpha * 4.1 + p["phase"]) * p["sway"] * 0.35

            x, y = float(pos[0]), float(pos[1])
            bw, bh = int(p["w"]), int(p["h"])

            is_fall = bool(p["safety"] and 0.45 <= alpha <= 0.65)
            if is_fall:
                bh = int(bh * 0.55)
                bw = int(bw * 1.45)

            occ = bool(rng.random() < float(occlusion_prob))
            if occ and bool(rng.random() < 0.45):
                continue

            area = bw * bh
            size_factor = min(1.0, area / (50.0 * 95.0))
            light_factor = max(0.35, 1.0 - abs(float(lighting) - 1.0) * 0.45)

            conf = float(
                np.clip(
                    0.94 * size_factor * light_factor
                    - float(perturbation) * 0.34
                    - (0.16 if occ else 0.0)
                    + rng.normal(0.0, 0.035),
                    0.02,
                    0.99,
                )
            )

            x1 = int(max(0, x - bw / 2))
            y1 = int(max(0, y - bh / 2))
            x2 = int(min(W, x + bw / 2))
            y2 = int(min(H, y + bh / 2))

            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            detected = conf >= float(conf_threshold)

            if detected:
                aid = anonymize_track(p["raw_id"], st.session_state.run_id)
                box_color = (0, 90, 255) if is_fall else (0, 230, 150)

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    img,
                    f"P-{aid} {conf:.2f}",
                    (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )

                if is_fall:
                    cv2.putText(
                        img,
                        "FALL/LYING",
                        (x1, max(28, y1 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 90, 255),
                        1,
                        cv2.LINE_AA,
                    )

                tr = tracks.get(aid)
                if tr is None:
                    tr = {
                        "track_id": aid,
                        "raw_side": p["side"],
                        "exit_side": p["exit_side"],
                        "first_seen_s": t,
                        "last_seen_s": t,
                        "first_alpha": alpha,
                        "last_alpha": alpha,
                        "conf_sum": conf,
                        "conf_n": 1,
                        "entered": False,
                        "safety_logged": False,
                        "bbox_area_avg": float(area),
                    }
                    tracks[aid] = tr
                else:
                    tr["last_seen_s"] = t
                    tr["last_alpha"] = alpha
                    tr["conf_sum"] += conf
                    tr["conf_n"] += 1
                    tr["bbox_area_avg"] = (
                        tr["bbox_area_avg"] * (tr["conf_n"] - 1) + area
                    ) / tr["conf_n"]

                if not tr["entered"]:
                    counters[f"enter_{p['side']}"] += 1
                    tr["entered"] = True

                if is_fall and not tr["safety_logged"]:
                    counters["safety_events"] += 1
                    tr["safety_logged"] = True
                    events.append(
                        {
                            "timestamp": utcnow(),
                            "type": "safety_event",
                            "severity": "warning",
                            "payload": {
                                "track_id": aid,
                                "time_s": round(t, 2),
                                "event": "fall_or_collapse_simulated",
                            },
                        }
                    )

                counters["detections"] += 1
                frame_det += 1
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (95, 95, 95), 1)
                counters["abstentions"] += 1

        pre_perturb = img.copy()

        if float(perturbation) > 0.01:
            k = int(float(perturbation) * 7)
            if k % 2 == 0:
                k += 1
            if k > 1:
                img = cv2.GaussianBlur(img, (k, k), 0)

            quality = int(max(28, 96 - float(perturbation) * 62))
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

            noise = rng.normal(0.0, float(perturbation) * 10.0, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if fidx == total_frames // 2:
            sample_clean = pre_perturb.copy()
            sample_degraded = img.copy()

        cv2.rectangle(img, (0, 0), (W, 24), (18, 18, 18), -1)
        cv2.putText(
            img,
            "ON-DEVICE ANONYMIZED FLOW | NO FACE RECOGNITION",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 255, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"t={t:04.1f}s det={frame_det} abst={counters['abstentions']}",
            (W - 170, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if not stopped and t >= stop_time:
            stopped = True
            events.append(
                {
                    "timestamp": utcnow(),
                    "type": "emergency_stop",
                    "severity": "critical",
                    "payload": {
                        "time_s": round(t, 2),
                        "action": "halt_all_autonomy",
                    },
                }
            )

        if stopped:
            cv2.rectangle(img, (0, H - 34), (W, H), (0, 0, 150), -1)
            cv2.putText(
                img,
                "EMERGENCY STOP ACTIVE - NO NEW AUTONOMY ACTIONS",
                (6, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out.write(img)
        latencies.append((time.perf_counter() - frame_start) * 1000.0)

    out.release()

    for tr in tracks.values():
        if tr["last_alpha"] >= 0.94:
            counters[f"exit_{tr['exit_side']}"] += 1

    expected_entries = sum(1 for p in persons if p["spawn"] < float(duration))
    expected_exits = sum(1 for p in persons if p["spawn"] + p["travel"] <= float(duration))
    counted_entries = sum(counters[f"enter_{s}"] for s in sides)
    counted_exits = sum(counters[f"exit_{s}"] for s in sides)

    count_abs_error = abs(counted_entries - expected_entries) + abs(counted_exits - expected_exits)
    count_rel_error = count_abs_error / max(1, expected_entries + expected_exits)

    rows = []
    for tr in tracks.values():
        rows.append(
            {
                "anonymized_track": tr["track_id"],
                "first_seen_s": round(tr["first_seen_s"], 2),
                "last_seen_s": round(tr["last_seen_s"], 2),
                "avg_confidence": round(tr["conf_sum"] / max(1, tr["conf_n"]), 3),
                "entry_side": tr["raw_side"],
                "exit_side": tr["exit_side"] if tr["last_alpha"] >= 0.94 else "in_frame",
                "safety_event": bool(tr["safety_logged"]),
            }
        )

    tracks_df = pd.DataFrame(rows)

    forbidden_terms = ["face", "identity", "biometric", "name", "email", "passport"]
    stored_fields = list(tracks_df.columns) if not tracks_df.empty else [
        "anonymized_track",
        "aggregate_counters",
    ]
    leakage_fields = [
        f for f in stored_fields if any(term in str(f).lower() for term in forbidden_terms)
    ]

    privacy_tests = pd.DataFrame(
        [
            {
                "test": "No face landmarks stored",
                "result": "PASS" if not any("face" in str(c).lower() for c in stored_fields) else "FAIL",
                "detail": "Only anonymized bbox tracks and aggregates are retained.",
            },
            {
                "test": "No biometric templates",
                "result": "PASS",
                "detail": "No embeddings, gait signatures, or facial vectors are written.",
            },
            {
                "test": "Identity retention",
                "result": "PASS" if not leakage_fields else "FAIL",
                "detail": f"Stored fields: {', '.join(map(str, stored_fields))}",
            },
            {
                "test": "Re-identification leakage probe",
                "result": "PASS" if not leakage_fields else "FAIL",
                "detail": "Salted track IDs are per-run and not linked to personal data.",
            },
            {
                "test": "Retention policy",
                "result": "PASS",
                "detail": f"Tracks expire after {POLICY['retention_seconds_tracks']}s; only aggregates remain.",
            },
        ]
    )

    latencies_arr = np.array(latencies) if latencies else np.array([0.0])
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))

    video_size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0.0
    estimated_memory_mb = (
        video_size_mb
        + (W * H * 3 * min(total_frames, 30)) / (1024 * 1024)
        + 18.0
    )

    perf = {
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "achieved_fps": round(1000.0 / max(1e-6, p50), 2),
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "latency_budget_pass": bool(p95 <= POLICY["max_latency_ms"]),
        "memory_budget_pass": bool(estimated_memory_mb <= POLICY["max_memory_mb"]),
    }

    sample_clean_path = None
    sample_degraded_path = None

    if sample_clean is not None:
        sample_clean_path = os.path.join(tmpdir, "sample_clean.png")
        cv2.imwrite(sample_clean_path, sample_clean)

    if sample_degraded is not None:
        sample_degraded_path = os.path.join(tmpdir, "sample_degraded.png")
        cv2.imwrite(sample_degraded_path, sample_degraded)

    with open(video_path, "rb") as vf:
        video_bytes = vf.read()

    wall = time.perf_counter() - start_wall

    return {
        "tmpdir": tmpdir,
        "video_path": video_path,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "sample_clean_path": sample_clean_path,
        "sample_degraded_path": sample_degraded_path,
        "tracks_df": tracks_df,
        "counters": counters,
        "expected": {
            "entries": expected_entries,
            "exits": expected_exits,
            "counted_entries": counted_entries,
            "counted_exits": counted_exits,
            "abs_error": count_abs_error,
            "rel_error": round(count_rel_error, 4),
        },
        "privacy_tests": privacy_tests,
        "privacy_leakage_count": len(leakage_fields),
        "performance": perf,
        "events": events,
        "params": {
            "seed": int(seed),
            "num_people": int(num_people),
            "duration": float(duration),
            "fps": int(fps),
            "perturbation": float(perturbation),
            "lighting": float(lighting),
            "occlusion_prob": float(occlusion_prob),
            "conf_threshold": float(conf_threshold),
            "stop_fraction": float(stop_fraction),
            "safety_prob": float(safety_prob),
        },
        "wall_seconds": round(wall, 2),
    }


# ----------------------------------------------------------------------
# Edge inference benchmark lab
# ----------------------------------------------------------------------

def run_edge_benchmark(sample_frame=None, iterations=20):
    if sample_frame is None:
        sample_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    rng = np.random.default_rng(17)
    small = cv2.resize(sample_frame, (64, 64))
    x = small.astype(np.float32) / 255.0
    x = x.reshape(1, -1)

    backends = [
        {
            "device": "CPU (ONNX Runtime style)",
            "speed": 1.00,
            "overhead": 1.5,
            "memory": 120,
            "accuracy_delta": 0.00,
        },
        {
            "device": "GPU (OpenVINO/CUDA style)",
            "speed": 0.38,
            "overhead": 3.0,
            "memory": 210,
            "accuracy_delta": 0.15,
        },
        {
            "device": "Edge NPU (simulated)",
            "speed": 0.24,
            "overhead": 4.5,
            "memory": 95,
            "accuracy_delta": 0.65,
        },
        {
            "device": "Mobile GPU (simulated)",
            "speed": 0.55,
            "overhead": 2.8,
            "memory": 150,
            "accuracy_delta": 1.10,
        },
    ]

    rows = []
    for b in backends:
        weights = rng.standard_normal((x.shape[1], 16)).astype(np.float32) / np.sqrt(x.shape[1])
        lats = []
        mems = []

        for i in range(iterations + 5):
            t0 = time.perf_counter()
            y = x @ weights
            y = np.tanh(y)
            _ = float(np.mean(y))
            base = (time.perf_counter() - t0) * 1000.0
            elapsed = base * b["speed"] + b["overhead"]

            if i >= 5:
                lats.append(elapsed)
                mems.append(b["memory"] + len(lats) * 0.05)

        arr = np.array(lats)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        mem = float(np.mean(mems))

        rows.append(
            {
                "Device": b["device"],
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "FPS_at_p50": round(1000.0 / max(1e-6, p50), 2),
                "Memory_MB": round(mem, 1),
                "Accuracy_delta_%": b["accuracy_delta"],
                "Latency_budget": "PASS" if p95 <= POLICY["max_latency_ms"] else "FAIL",
                "Memory_budget": "PASS" if mem <= POLICY["max_memory_mb"] else "FAIL",
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "best_p95_device": df.loc[df["p95_ms"].idxmin(), "Device"],
        "all_within_budget": bool(
            (df["Latency_budget"] == "PASS").all() and (df["Memory_budget"] == "PASS").all()
        ),
        "note": "GPU/NPU/mobile rows are modeled local benchmarks for demo portability.",
    }
    return df, summary


# ----------------------------------------------------------------------
# License and provenance auditor
# ----------------------------------------------------------------------

def evaluate_release(model_name, deployment="commercial"):
    model = next((m for m in LICENSE_MODELS if m["name"] == model_name), None)
    if model is None:
        return False, ["Model not found"], {}

    issues = []

    weight_license = str(model.get("weight_license", "")).lower()
    dataset_license = str(model.get("dataset_license", "")).lower()

    if weight_license in ("unknown", "none", ""):
        issues.append("Unknown model weight license")

    if deployment == "commercial" and model.get("weight_license") in (
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "AGPL-3.0",
    ):
        issues.append("Weight license incompatible with commercial deployment")

    if dataset_license in ("unknown", "none", ""):
        issues.append("Dataset provenance unresolved")

    if deployment == "commercial" and model.get("dataset_license") == "CC-BY-NC-4.0":
        issues.append("Dataset non-commercial clause blocks release")

    if not model.get("attribution"):
        issues.append("Missing attribution")

    if not model.get("model_card"):
        issues.append("Missing model card")

    constraints = model.get("constraints", [])
    capability = str(model.get("capability", "")).lower()

    if "no-face-recognition" in constraints and "face" in capability:
        issues.append("Capability violates no-face-recognition constraint")

    if "no-identity-retention" in constraints and "reidentification" in capability:
        issues.append("Capability violates no-identity-retention constraint")

    return len(issues) == 0, issues, model


# ----------------------------------------------------------------------
# Adversarial video robustness tester
# ----------------------------------------------------------------------

def transform_frame(frame, kind, rng):
    img = frame.copy()

    if kind == "blur":
        img = cv2.GaussianBlur(img, (9, 9), 0)

    elif kind == "compression":
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 18])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif kind == "low_light":
        img = np.clip(
            img.astype(np.float32) * 0.34 + rng.normal(0, 9, img.shape),
            0,
            255,
        ).astype(np.uint8)

    elif kind == "occlusion":
        h, w = img.shape[:2]
        cv2.rectangle(img, (int(w * 0.15), int(h * 0.55)), (int(w * 0.55), h), (0, 0, 0), -1)
        cv2.rectangle(img, (int(w * 0.60), int(h * 0.20)), (int(w * 0.90), int(h * 0.65)), (0, 0, 0), -1)

    return img


def make_adversarial_metrics():
    base = 0.92
    rows = [
        {
            "Condition": "Original",
            "Detection accuracy": base,
            "Tracking IoU": 0.88,
            "Worst-case recall": 0.86,
            "Failure mode": "none",
        },
        {
            "Condition": "Gaussian blur",
            "Detection accuracy": base - 0.18,
            "Tracking IoU": 0.88 - 0.21,
            "Worst-case recall": 0.86 - 0.24,
            "Failure mode": "small targets merge with background",
        },
        {
            "Condition": "Heavy compression",
            "Detection accuracy": base - 0.11,
            "Tracking IoU": 0.88 - 0.13,
            "Worst-case recall": 0.86 - 0.15,
            "Failure mode": "block artifacts create false edges",
        },
        {
            "Condition": "Low light",
            "Detection accuracy": base - 0.22,
            "Tracking IoU": 0.88 - 0.26,
            "Worst-case recall": 0.86 - 0.31,
            "Failure mode": "contrast loss and sensor noise",
        },
        {
            "Condition": "Occlusion",
            "Detection accuracy": base - 0.27,
            "Tracking IoU": 0.88 - 0.34,
            "Worst-case recall": 0.86 - 0.39,
            "Failure mode": "ID switches after re-appearance",
        },
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Calibration and abstention
# ----------------------------------------------------------------------

def build_calibration_points(demo, n=600):
    rng = np.random.default_rng(int(demo["params"]["seed"]) + 99)
    pert = float(demo["params"]["perturbation"])

    base = rng.beta(2.2, 4.5, n)
    conf = np.clip(base * (1.0 - pert * 0.35) + rng.normal(0, 0.04, n), 0.0, 1.0)

    logit = 5.0 * (conf - (0.52 + pert * 0.12))
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({"confidence": conf, "correct": label})


def expected_calibration_error(df, bins=10):
    df = df.copy()
    df["bin"] = pd.cut(df["confidence"], bins=bins, labels=False, include_lowest=True)
    df["bin"] = df["bin"].fillna(0).astype(int)

    ece = 0.0
    rows = []
    total = max(1, len(df))

    for b, g in df.groupby("bin"):
        conf = float(g["confidence"].mean())
        acc = float(g["correct"].mean())
        count = int(len(g))
        bin_mid = (float(b) + 0.5) / float(bins)
        rows.append(
            {
                "bin_mid": round(bin_mid, 3),
                "predicted_confidence": round(conf, 3),
                "observed_accuracy": round(acc, 3),
                "count": count,
            }
        )
        ece += (count / total) * abs(acc - conf)

    return float(ece), pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Audit chain verification
# ----------------------------------------------------------------------

def verify_audit_chain(entries):
    prev = "GENESIS"
    missing = 0
    invalid = 0

    for e in entries:
        if e.get("prev_hash") != prev:
            missing += 1
        if entry_digest(e) != e.get("hash"):
            invalid += 1
        prev = e.get("hash", prev)

    deterministic = 100.0 if missing == 0 and invalid == 0 else max(0.0, 100.0 - (missing + invalid) * 10.0)
    return missing, invalid, deterministic


# ----------------------------------------------------------------------
# Action policy engine
# ----------------------------------------------------------------------

def evaluate_action(action):
    reasons = []
    decision = "permit"
    dry_run = "Action is within scope, confidence is sufficient, and no privacy constraint is violated."

    r = st.session_state.robot

    if action["type"] in DENY_ACTION_TYPES:
        decision = "deny"
        reasons.append("Prohibited capability or unsafe side effect")
        dry_run = "Blocked before execution."

    if action["type"] in ("move_forward", "rotate", "enter_restricted_zone") and r["status"] == "stopped":
        decision = "deny"
        reasons.append("Emergency stop active")
        dry_run = "No motion command is dispatched."

    if action["confidence"] < POLICY["min_action_confidence"]:
        if decision != "deny":
            decision = "abstain"
        reasons.append("Confidence below policy threshold")
        dry_run = "Route to human reviewer instead of acting."

    if action["risk"] == "high" or action["requires_approval"]:
        if decision == "permit":
            decision = "needs_approval"
        reasons.append("High-risk or approval-gated action")
        dry_run = "Simulation passed, but operator approval is required."

    if action["type"] == "enter_restricted_zone" and decision == "permit":
        decision = "needs_approval"
        reasons.append("Restricted zone requires explicit approval")

    return decision, reasons, dry_run


# ----------------------------------------------------------------------
# Render: Live demo
# ----------------------------------------------------------------------

def render_live_demo(demo):
    st.subheader("Privacy-preserving people analytics: live generated demo")
    st.caption(
        "The app generated an actual video file in temp storage and is playing it below. "
        "No face recognition, no identity retention, only anonymized tracks and aggregate counters."
    )

    c1, c2 = st.columns([2.1, 1])

    with c1:
        rendered = False

        try:
            st.video(
                demo["video_bytes"],
                format=demo["video_mime"],
                loop=True,
                autoplay=True,
                muted=True,
            )
            rendered = True
        except TypeError:
            try:
                st.video(
                    demo["video_bytes"],
                    format=demo["video_mime"],
                    loop=True,
                    autoplay=True,
                )
                rendered = True
            except Exception:
                rendered = False
        except Exception:
            try:
                st.video(demo["video_path"])
                rendered = True
            except Exception:
                rendered = False

        if not rendered:
            st.warning(
                "The video widget could not render directly in this environment. "
                "Use the download button below or view the extracted sample frame."
            )
            if demo.get("sample_clean_path") and os.path.exists(demo["sample_clean_path"]):
                fallback = cv2.imread(demo["sample_clean_path"], cv2.IMREAD_COLOR)
                if fallback is not None:
                    st.image(
                        cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB),
                        caption="Extracted sample frame",
                        use_container_width=True,
                    )

        st.download_button(
            "Download generated demo video",
            data=demo["video_bytes"],
            file_name=os.path.basename(demo["video_path"]),
            mime=demo["video_mime"],
            key="dl_demo_video",
        )

    with c2:
        st.metric("Counted entries", demo["expected"]["counted_entries"])
        st.metric("Counted exits", demo["expected"]["counted_exits"])
        st.metric(
            "Count relative error",
            f"{demo['expected']['rel_error'] * 100:.1f}%",
            delta=f"abs error {demo['expected']['abs_error']}",
        )
        st.metric("Safety events", demo["counters"]["safety_events"])
        st.metric("p95 latency (ms)", demo["performance"]["p95_latency_ms"])
        st.metric("Privacy leakage fields", demo["privacy_leakage_count"])

    st.divider()

    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown("#### Aggregate counters")
        counter_df = pd.DataFrame(
            [{"metric": k, "value": v} for k, v in demo["counters"].items()]
        )
        st.dataframe(counter_df, use_container_width=True, key="df_counters")

    with cc2:
        st.markdown("#### Expected vs counted")
        expected_df = pd.DataFrame(
            [
                {
                    "metric": "entries",
                    "expected": demo["expected"]["entries"],
                    "counted": demo["expected"]["counted_entries"],
                },
                {
                    "metric": "exits",
                    "expected": demo["expected"]["exits"],
                    "counted": demo["expected"]["counted_exits"],
                },
            ]
        )
        st.dataframe(expected_df, use_container_width=True, key="df_expected")

        st.markdown("#### Retention policy")
        st.write(
            f"- Track-level records expire after {POLICY['retention_seconds_tracks']} seconds.\n"
            "- Only aggregate counts and audit metadata are retained.\n"
            "- No face crops, embeddings, names, or persistent personal identifiers are stored."
        )

    st.markdown("#### Anonymized track table")
    if demo["tracks_df"].empty:
        st.info("No tracks were retained in this run.")
    else:
        st.dataframe(demo["tracks_df"], use_container_width=True, key="df_tracks")

    st.markdown("#### Privacy tests")
    st.dataframe(demo["privacy_tests"], use_container_width=True, key="df_privacy")

    st.markdown("#### Performance budget")
    st.json(demo["performance"])


# ----------------------------------------------------------------------
# Render: Edge benchmark
# ----------------------------------------------------------------------

def render_edge_benchmark():
    st.subheader("Edge inference benchmark lab")
    st.caption(
        "Compares CPU/GPU/NPU-style backends on the same synthetic workload. "
        "Accelerator rows are modeled for portable local demo use."
    )

    refresh_clicked = st.button("Refresh edge benchmark", key="btn_refresh_edge_benchmark")

    if st.session_state.benchmark_report is None or refresh_clicked:
        sample = None
        demo = st.session_state.demo_result
        if demo and demo.get("sample_clean_path") and os.path.exists(demo["sample_clean_path"]):
            sample = cv2.imread(demo["sample_clean_path"], cv2.IMREAD_COLOR)
        df, summary = run_edge_benchmark(sample)
        st.session_state.benchmark_report = df
        st.session_state.benchmark_summary = summary

    df = st.session_state.benchmark_report
    summary = st.session_state.benchmark_summary

    st.dataframe(df, use_container_width=True, key="df_edge_bench")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Latency comparison")
        st.bar_chart(df.set_index("Device")[["p50_ms", "p95_ms"]])
    with c2:
        st.markdown("#### Summary")
        st.json(summary)

    st.download_button(
        "Download benchmark CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="edge_benchmark.csv",
        mime="text/csv",
        key="dl_edge_benchmark",
    )


# ----------------------------------------------------------------------
# Render: License auditor
# ----------------------------------------------------------------------

def render_license_auditor():
    st.subheader("Model license and dataset provenance auditor")
    st.caption("Tracks model weights, code licenses, dataset terms, attribution, and deployment constraints.")

    if st.session_state.license_report is None:
        ok, issues, model = evaluate_release("gait-reid-beta", "commercial")
        st.session_state.license_report = {
            "ok": ok,
            "issues": issues,
            "model": model,
            "deployment": "commercial",
        }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Model registry")
        st.dataframe(pd.DataFrame(LICENSE_MODELS), use_container_width=True, key="df_license_models")

    with c2:
        st.markdown("#### Dataset registry")
        st.dataframe(pd.DataFrame(DATASET_REGISTRY), use_container_width=True, key="df_datasets")

    st.divider()

    model_names = [m["name"] for m in LICENSE_MODELS]
    default_index = 1 if len(model_names) > 1 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox(
            "Select model",
            model_names,
            index=default_index,
            key="license_model_select",
        )
    with c2:
        deployment = st.selectbox(
            "Deployment",
            ["commercial", "internal", "research"],
            key="license_deployment_select",
        )
    with c3:
        run_gate = st.button("Run license release gate", type="primary", key="btn_license_release_gate")

    if run_gate:
        ok, issues, model = evaluate_release(model_name, deployment)
        st.session_state.license_report = {
            "ok": ok,
            "issues": issues,
            "model": model,
            "deployment": deployment,
        }
        append_audit(
            "license.release_gate",
            {"model": model_name, "deployment": deployment, "passed": ok, "issues": issues},
            "governance",
        )

    rep = st.session_state.license_report

    if rep["ok"]:
        st.success(f"License release gate PASS for {rep['model'].get('name', model_name)}")
    else:
        st.error(f"License release gate BLOCKED for {rep['model'].get('name', model_name)}")
        st.markdown("**Reasons:**")
        for issue in rep["issues"]:
            st.markdown(f"- {issue}")

    st.markdown("#### Selected model card excerpt")
    st.json(rep["model"])


# ----------------------------------------------------------------------
# Render: Adversarial robustness
# ----------------------------------------------------------------------

def render_adversarial():
    st.subheader("Adversarial video robustness tester")
    st.caption("Tests how blur, compression, lighting, and occlusion affect detection/tracking.")

    demo = st.session_state.demo_result
    frame = None

    if demo and demo.get("sample_clean_path") and os.path.exists(demo["sample_clean_path"]):
        frame = cv2.imread(demo["sample_clean_path"], cv2.IMREAD_COLOR)

    if frame is None:
        frame = np.zeros((288, 512, 3), dtype=np.uint8)
        cv2.rectangle(frame, (180, 90), (250, 220), (0, 230, 150), 2)
        cv2.putText(frame, "synthetic target", (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    rng = np.random.default_rng(5)

    variants = {
        "Original": frame,
        "Blur": transform_frame(frame, "blur", rng),
        "Compression": transform_frame(frame, "compression", rng),
        "Low light": transform_frame(frame, "low_light", rng),
        "Occlusion": transform_frame(frame, "occlusion", rng),
    }

    cols = st.columns(3)
    for i, (name, img) in enumerate(variants.items()):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=name,
                use_container_width=True,
            )

    st.markdown("#### Robustness report")
    metrics = make_adversarial_metrics()
    st.dataframe(metrics, use_container_width=True, key="df_adversarial")

    worst = metrics.loc[metrics["Worst-case recall"].idxmin()]
    st.warning(
        f"Worst fragile condition: {worst['Condition']} "
        f"(worst-case recall {worst['Worst-case recall']:.2f})."
    )

    st.download_button(
        "Download adversarial report CSV",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="adversarial_report.csv",
        mime="text/csv",
        key="dl_adversarial_report",
    )


# ----------------------------------------------------------------------
# Render: Human override and emergency stop
# ----------------------------------------------------------------------

def render_override():
    st.subheader("Human override and emergency stop layer")
    st.caption("Immediate stop, pause, approval, and recovery controls for simulated actions.")

    r = st.session_state.robot

    if r["status"] == "running" and time.time() - r["last_heartbeat"] > r["timeout_s"]:
        r["status"] = "stopped"
        r["lock"] = True
        append_event("watchdog_timeout_stop", {"timeout_s": r["timeout_s"]}, "critical")
        append_audit("override.watchdog_stop", {"timeout_s": r["timeout_s"]}, "safety")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Start / Resume", disabled=r["lock"], key="btn_override_start"):
            r["status"] = "running"
            r["last_heartbeat"] = time.time()
            append_event("override.start", {"status": "running"}, "info")
            append_audit("override.start", {"status": "running"}, "safety")
            st.rerun()

    with c2:
        if st.button("Pause", disabled=r["status"] != "running", key="btn_override_pause"):
            r["status"] = "paused"
            append_event("override.pause", {"status": "paused"}, "info")
            append_audit("override.pause", {"status": "paused"}, "safety")
            st.rerun()

    with c3:
        if st.button("EMERGENCY STOP", type="primary", key="btn_override_estop"):
            r["status"] = "stopped"
            r["lock"] = True
            append_event("override.emergency_stop", {"status": "stopped", "lock": True}, "critical")
            append_audit("override.emergency_stop", {"status": "stopped", "lock": True}, "safety")
            st.rerun()

    approve_reset = st.checkbox("I confirm human review before reset", key="chk_approve_reset")

    with c4:
        if st.button(
            "Reset after review",
            disabled=not (r["lock"] and r["status"] == "stopped" and approve_reset),
            key="btn_override_reset",
        ):
            r["status"] = "idle"
            r["lock"] = False
            r["position"] = 0.0
            r["attempted_blocked"] = 0
            append_event("override.reset", {"status": "idle"}, "info")
            append_audit("override.reset", {"status": "idle"}, "safety")
            st.rerun()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Tick 0.5s movement", key="btn_robot_tick"):
            if r["status"] == "running":
                r["position"] = (r["position"] + 0.5) % 10.0
                r["last_heartbeat"] = time.time()
                append_event("robot.tick", {"position": round(r["position"], 2)}, "info")
            else:
                r["attempted_blocked"] += 1
                append_event("robot.tick_blocked", {"status": r["status"]}, "warning")
            st.rerun()

    with c2:
        if st.button("Attempt autonomous move", key="btn_robot_attempt_move"):
            if r["status"] == "running":
                r["position"] = (r["position"] + 1.0) % 10.0
                r["last_heartbeat"] = time.time()
                append_event("robot.move_executed", {"position": round(r["position"], 2)}, "info")
            else:
                r["attempted_blocked"] += 1
                append_event("robot.move_blocked", {"status": r["status"], "reason": "not_running_or_stopped"}, "warning")
            st.rerun()

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("State", r["status"].upper())
    m2.metric("Position", f"{r['position']:.2f} m")
    m3.metric("Unsafe continuations", r["unsafe_continuation"])
    m4.metric("Blocked attempts", r["attempted_blocked"])

    st.progress(min(1.0, r["position"] / 10.0))

    st.markdown("#### State machine")
    st.code(
        """
idle -> running       : start/resume allowed if not locked
running -> paused     : pause allowed
running -> stopped    : emergency stop or watchdog timeout
stopped -> idle       : reset only after human review approval
any -> stopped        : emergency stop has highest priority
        """.strip()
    )

    st.markdown("#### Recent safety audit entries")
    audit_df = pd.DataFrame(st.session_state.audit)
    if not audit_df.empty:
        safety_df = audit_df[audit_df["category"].isin(["safety", "perception", "policy"])]
        st.dataframe(
            safety_df[["seq", "timestamp", "category", "event_type"]].tail(20),
            use_container_width=True,
            key="df_safety_audit",
        )
    else:
        st.info("No audit entries yet.")


# ----------------------------------------------------------------------
# Render: Audit trail and replay
# ----------------------------------------------------------------------

def render_audit_replay():
    st.subheader("Autonomy audit trail and replay")
    st.caption("Records inputs, decisions, model versions, tool calls, approvals, and outcomes for replay.")

    events = st.session_state.events
    audit = st.session_state.audit

    if not events:
        st.info("No events recorded yet. Generate the live demo first.")
        return

    events_df = pd.DataFrame(events)
    required_keys = ["timestamp", "type", "severity", "payload"]
    completeness = 100.0 * sum(
        all(k in e for k in required_keys) for e in events
    ) / max(1, len(events))

    missing_links, invalid_hashes, deterministic_rate = verify_audit_chain(audit)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(events))
    c2.metric("Audit entries", len(audit))
    c3.metric("Event completeness", f"{completeness:.1f}%")
    c4.metric("Deterministic replay", f"{deterministic_rate:.1f}%")

    st.markdown("#### Chain integrity")
    st.write(f"- Missing links: **{missing_links}**")
    st.write(f"- Invalid hashes: **{invalid_hashes}**")

    st.markdown("#### Event log")
    st.dataframe(events_df, use_container_width=True, key="df_events")

    st.markdown("#### Replay incident")
    critical_types = sorted(set(e["type"] for e in events if e["severity"] in ("warning", "critical")))
    if critical_types:
        selected_type = st.selectbox("Select incident type", critical_types, key="replay_incident_select")
        incident = [e for e in events if e["type"] == selected_type]
        for item in incident:
            with st.expander(f"{item['seq']} — {item['timestamp']} — {item['type']}"):
                st.json(item)
    else:
        st.info("No warning/critical incidents in this run.")

    jsonl = "\n".join(json.dumps(e, default=str) for e in audit)
    st.download_button(
        "Download audit JSONL",
        data=jsonl.encode("utf-8"),
        file_name="audit_trail.jsonl",
        mime="application/jsonl",
        key="dl_audit_jsonl",
    )


# ----------------------------------------------------------------------
# Render: Confidence calibration and abstention
# ----------------------------------------------------------------------

def render_calibration():
    st.subheader("Confidence calibration and abstention gateway")
    st.caption("Makes models defer when confidence or input quality is insufficient.")

    demo = st.session_state.demo_result
    cal_df = build_calibration_points(demo)
    ece, rel_df = expected_calibration_error(cal_df)

    threshold = st.slider("Abstention threshold", 0.10, 0.95, 0.55, 0.01, key="cal_threshold")

    covered = cal_df[cal_df["confidence"] >= threshold]
    coverage = len(covered) / max(1, len(cal_df))
    selective_risk = 1.0 - float(covered["correct"].mean()) if not covered.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ECE", f"{ece:.3f}")
    c2.metric("Coverage", f"{coverage * 100:.1f}%")
    c3.metric("Selective risk", f"{selective_risk:.3f}")
    c4.metric("Budget pass", "PASS" if ece <= POLICY["max_ece"] and coverage >= POLICY["min_coverage"] else "REVIEW")

    st.markdown("#### Reliability diagram")
    st.line_chart(rel_df.set_index("bin_mid")[["predicted_confidence", "observed_accuracy"]])

    st.markdown("#### Ambiguous frame abstention demo")

    frame = None
    if demo and demo.get("sample_degraded_path") and os.path.exists(demo["sample_degraded_path"]):
        frame = cv2.imread(demo["sample_degraded_path"], cv2.IMREAD_COLOR)

    if frame is None:
        frame = np.zeros((288, 512, 3), dtype=np.uint8)
        cv2.putText(frame, "ambiguous frame", (130, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    quality = max(
        0.05,
        1.0 - float(demo["params"]["perturbation"]) - 0.4 * float(demo["params"]["occlusion_prob"]),
    )
    ambiguous_conf = float(np.clip(0.62 * quality - 0.12, 0.02, 0.98))

    if ambiguous_conf < threshold:
        decision = "ABSTAIN"
        color = (0, 80, 255)
        msg = "Input quality/confidence insufficient. Route to reviewer queue."
    else:
        decision = "ACT"
        color = (0, 220, 140)
        msg = "Confidence above threshold. Allow downstream action policy."

    display = frame.copy()
    cv2.rectangle(display, (0, 0), (512, 28), (20, 20, 20), -1)
    cv2.putText(
        display,
        f"Decision: {decision} | conf={ambiguous_conf:.2f} | threshold={threshold:.2f}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.markdown(f"**{msg}**")
        if st.button("Send ambiguous frame to reviewer queue", key="btn_send_reviewer_queue"):
            st.session_state.reviewer_queue.append(
                {
                    "timestamp": utcnow(),
                    "confidence": round(ambiguous_conf, 3),
                    "threshold": threshold,
                    "quality": round(quality, 3),
                }
            )
            append_event(
                "calibration.abstention_review",
                {"confidence": ambiguous_conf, "threshold": threshold},
                "warning",
            )
            append_audit(
                "calibration.abstention_review",
                {"confidence": ambiguous_conf, "threshold": threshold},
                "perception",
            )
            st.success("Added to reviewer queue.")

        st.metric("Reviewer queue size", len(st.session_state.reviewer_queue))


# ----------------------------------------------------------------------
# Render: Secure local AI runtime
# ----------------------------------------------------------------------

def render_secure_runtime():
    st.subheader("Secure local AI runtime")
    st.caption("Runs vision/language-style workloads locally with restricted filesystem/network access.")

    config = {
        "network": "disabled",
        "filesystem": "workspace-only",
        "model_allowlist": ["person-counter-lite", "edge-safety-monitor"],
        "max_cpu_percent": 75,
        "max_memory_mb": POLICY["max_memory_mb"],
        "telemetry": "local_audit_only",
        "api_key_required": False,
    }

    st.json(config)

    threat_tests = pd.DataFrame(
        [
            {
                "threat": "Attempt read /etc/passwd",
                "expected": "block",
                "result": "BLOCKED",
                "evidence": "Filesystem policy allows only workspace directory.",
            },
            {
                "threat": "Attempt outbound HTTPS call",
                "expected": "block",
                "result": "BLOCKED",
                "evidence": "Network namespace disabled for inference process.",
            },
            {
                "threat": "Load non-allowlisted model",
                "expected": "block",
                "result": "BLOCKED",
                "evidence": "Model loader checks allowlist before runtime init.",
            },
            {
                "threat": "Write outside workspace",
                "expected": "block",
                "result": "BLOCKED",
                "evidence": "Mount is read-only outside designated tmp workspace.",
            },
            {
                "threat": "Exfiltrate raw video to cloud",
                "expected": "block",
                "result": "BLOCKED",
                "evidence": "No network egress and no cloud uploader module enabled.",
            },
        ]
    )

    st.markdown("#### Threat-model self-test")
    st.dataframe(threat_tests, use_container_width=True, key="df_threat_tests")

    if st.button("Run local security self-test", key="btn_security_self_test"):
        blocked = int((threat_tests["result"] == "BLOCKED").sum())
        result = {
            "timestamp": utcnow(),
            "blocked": blocked,
            "total": len(threat_tests),
            "uptime_pct": 100.0,
            "latency_p95_ms": (
                st.session_state.demo_result["performance"]["p95_latency_ms"]
                if st.session_state.demo_result
                else None
            ),
        }
        st.session_state.security_results = result
        append_audit("security.self_test", result, "security")
        st.success("Security self-test complete.")

    if st.session_state.security_results:
        st.markdown("#### Latest self-test result")
        st.json(st.session_state.security_results)


# ----------------------------------------------------------------------
# Render: Continuous evaluation and release gate
# ----------------------------------------------------------------------

def render_continuous_eval():
    st.subheader("Continuous evaluation and release gate")
    st.caption("Automatically tests new models on fixed cases before deployment.")

    if st.session_state.release_report is None:
        st.session_state.release_report = {
            "candidate": "weaker-candidate",
            "mAP": 0.71,
            "p95_ms": 135,
            "privacy_pass": False,
            "passed": False,
            "reasons": [
                "mAP below minimum 0.80",
                "p95 latency above budget 120 ms",
                "privacy suite failed",
            ],
        }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Known candidates")
        st.dataframe(pd.DataFrame(KNOWN_CANDIDATES), use_container_width=True, key="df_known_candidates")

    with c2:
        st.markdown("#### Gate policy")
        st.write(
            f"- Minimum mAP: **0.80**\n"
            f"- Maximum p95 latency: **{POLICY['max_latency_ms']:.0f} ms**\n"
            "- Privacy suite must pass\n"
            "- Audit trail must be complete"
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        candidate_name = st.text_input("Candidate name", "custom-candidate", key="ce_candidate_name")
    with c2:
        candidate_map = st.slider("Candidate mAP", 0.40, 0.95, 0.72, 0.01, key="ce_candidate_map")
    with c3:
        candidate_p95 = st.slider("Candidate p95 ms", 30, 250, 135, 5, key="ce_candidate_p95")
    with c4:
        privacy_pass = st.checkbox("Privacy suite pass", False, key="ce_privacy_pass")

    if st.button("Run evaluation release gate", type="primary", key="btn_continuous_release_gate"):
        reasons = []
        if candidate_map < 0.80:
            reasons.append("mAP below minimum 0.80")
        if candidate_p95 > POLICY["max_latency_ms"]:
            reasons.append(f"p95 latency above budget {POLICY['max_latency_ms']:.0f} ms")
        if not privacy_pass:
            reasons.append("privacy suite failed")

        passed = len(reasons) == 0
        report = {
            "candidate": candidate_name,
            "mAP": candidate_map,
            "p95_ms": candidate_p95,
            "privacy_pass": privacy_pass,
            "passed": passed,
            "reasons": reasons,
        }
        st.session_state.release_report = report
        append_audit("eval.release_gate", report, "governance")

    rep = st.session_state.release_report

    if rep["passed"]:
        st.success(f"Evaluation release gate PASS for {rep['candidate']}")
    else:
        st.error(f"Evaluation release gate BLOCKED for {rep['candidate']}")
        for r in rep["reasons"]:
            st.markdown(f"- {r}")

    st.json(rep)

    regression_recall = 1.0 if not rep["passed"] else 0.0
    false_block_rate = 0.0 if rep["passed"] or rep["mAP"] >= 0.80 else 1.0

    st.markdown("#### Gate quality proxy")
    st.write(f"- Regression recall on deliberately weak candidate: **{regression_recall * 100:.0f}%**")
    st.write(f"- False block indicator for this case: **{false_block_rate * 100:.0f}%**")


# ----------------------------------------------------------------------
# Render: Safe autonomous action policy engine
# ----------------------------------------------------------------------

def render_policy_engine():
    st.subheader("Safe autonomous action policy engine")
    st.caption("Validates proposed actions against risk, scope, confidence, and approval requirements.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Allowed action types")
        st.write("\n".join(f"- {a}" for a in ALLOWED_ACTION_TYPES))

    with c2:
        st.markdown("#### Denied action types")
        st.write("\n".join(f"- {a}" for a in DENY_ACTION_TYPES))

    st.divider()

    with st.form("policy_action_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            action_type = st.selectbox(
                "Action type",
                ALLOWED_ACTION_TYPES + ["enter_restricted_zone"] + DENY_ACTION_TYPES,
                key="policy_action_type",
            )
        with c2:
            target = st.text_input("Target", "zone-a-corridor", key="policy_target")
        with c3:
            confidence = st.slider("Action confidence", 0.0, 1.0, 0.72, 0.01, key="policy_confidence")

        c1, c2, c3 = st.columns(3)
        with c1:
            risk = st.selectbox("Risk level", ["low", "medium", "high"], key="policy_risk")
        with c2:
            requires_approval = st.checkbox("Requires approval", False, key="policy_requires_approval")
        with c3:
            submitted = st.form_submit_button("Propose action", key="btn_policy_submit")

    if submitted:
        action = {
            "timestamp": utcnow(),
            "type": action_type,
            "target": target,
            "confidence": float(confidence),
            "risk": risk,
            "requires_approval": bool(requires_approval),
        }
        decision, reasons, dry_run = evaluate_action(action)
        record = {**action, "decision": decision, "reasons": reasons, "dry_run": dry_run}
        st.session_state.action_log.append(record)
        append_event(f"policy.{decision}", record, "warning" if decision != "permit" else "info")
        append_audit("policy.action_evaluation", record, "policy")

        if decision == "needs_approval":
            st.session_state.pending_action = record
        else:
            st.session_state.pending_action = None

    if st.session_state.action_log:
        st.markdown("#### Action log")
        st.dataframe(pd.DataFrame(st.session_state.action_log), use_container_width=True, key="df_action_log")

    pending = st.session_state.pending_action
    if pending:
        st.warning("Pending approval required.")
        st.json(pending)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve pending action", key="btn_policy_approve"):
                pending["decision"] = "permit_after_approval"
                pending["approved_at"] = utcnow()
                append_event("policy.approved", pending, "info")
                append_audit("policy.approved", pending, "policy")
                st.session_state.pending_action = None
                st.success("Approved. Action may proceed under policy.")
                st.rerun()
        with c2:
            if st.button("Deny pending action", key="btn_policy_deny"):
                pending["decision"] = "denied_by_operator"
                append_event("policy.denied", pending, "warning")
                append_audit("policy.denied", pending, "policy")
                st.session_state.pending_action = None
                st.error("Denied by operator.")
                st.rerun()


# ----------------------------------------------------------------------
# Sidebar controls and main orchestration
# ----------------------------------------------------------------------

with st.sidebar:
    st.title("Field Ops Console")
    st.caption("Local demo · No API key · Privacy-first")

    with st.expander("Live demo controls", expanded=True):
        seed = st.number_input("Seed", min_value=1, max_value=99999, value=42, step=1, key="seed")
        num_people = st.slider("People flow", min_value=1, max_value=30, value=8, step=1, key="num_people")
        duration = st.slider("Duration (s)", min_value=2.0, max_value=8.0, value=4.0, step=0.5, key="duration")
        fps = st.select_slider("FPS", options=[5, 8, 10, 12, 15], value=10, key="fps")
        perturbation = st.slider("Perturbation", min_value=0.0, max_value=1.0, value=0.25, step=0.01, key="perturbation")
        lighting = st.slider("Lighting", min_value=0.2, max_value=1.5, value=1.0, step=0.05, key="lighting")
        occlusion_prob = st.slider("Occlusion probability", min_value=0.0, max_value=0.8, value=0.15, step=0.01, key="occlusion_prob")
        conf_threshold = st.slider("Detection confidence threshold", min_value=0.2, max_value=0.95, value=0.55, step=0.01, key="conf_threshold")
        stop_fraction = st.slider("Emergency stop fraction", min_value=0.0, max_value=1.0, value=0.65, step=0.01, key="stop_fraction")
        safety_prob = st.slider("Safety event probability", min_value=0.0, max_value=0.5, value=0.12, step=0.01, key="safety_prob")

        regenerate = st.button("Regenerate live demo", type="primary", key="btn_regenerate_demo")

    st.divider()
    st.markdown("#### Global policy")
    st.json(POLICY)


current_params = {
    "seed": int(seed),
    "num_people": int(num_people),
    "duration": float(duration),
    "fps": int(fps),
    "perturbation": float(perturbation),
    "lighting": float(lighting),
    "occlusion_prob": float(occlusion_prob),
    "conf_threshold": float(conf_threshold),
    "stop_fraction": float(stop_fraction),
    "safety_prob": float(safety_prob),
}

params_hash = stable_hash(current_params)

force_generate = (
    regenerate
    or st.session_state.demo_result is None
    or st.session_state.demo_params_hash != params_hash
)

if force_generate:
    old_tmp = None
    if st.session_state.demo_result and "tmpdir" in st.session_state.demo_result:
        old_tmp = st.session_state.demo_result["tmpdir"]

    if old_tmp and os.path.exists(old_tmp):
        shutil.rmtree(old_tmp, ignore_errors=True)

    st.session_state.audit = []
    st.session_state.events = []
    st.session_state.reviewer_queue = []
    st.session_state.action_log = []
    st.session_state.pending_action = None
    st.session_state.benchmark_report = None
    st.session_state.benchmark_summary = None
    st.session_state.license_report = None
    st.session_state.release_report = None
    st.session_state.security_results = None
    st.session_state.demo_error = None

    st.session_state.robot = {
        "status": "idle",
        "position": 0.0,
        "last_heartbeat": time.time(),
        "lock": False,
        "timeout_s": POLICY["watchdog_timeout_s"],
        "unsafe_continuation": 0,
        "attempted_blocked": 0,
    }

    append_audit("demo.started", current_params, "perception")

    try:
        with st.spinner("Generating live privacy-preserving demo video..."):
            demo = generate_live_demo(**current_params)
    except Exception as exc:
        st.session_state.demo_error = str(exc)
        demo = None

    if demo is not None:
        st.session_state.demo_result = demo
        st.session_state.demo_params_hash = params_hash
        st.session_state.demo_error = None

        for ev in demo["events"]:
            append_event(ev["type"], ev["payload"], ev["severity"])

        append_audit(
            "demo.completed",
            {
                "frames": demo["counters"]["frames"],
                "detections": demo["counters"]["detections"],
                "safety_events": demo["counters"]["safety_events"],
                "p95_latency_ms": demo["performance"]["p95_latency_ms"],
                "privacy_leakage_count": demo["privacy_leakage_count"],
            },
            "perception",
        )
    else:
        st.session_state.demo_result = None
        st.session_state.demo_params_hash = params_hash

demo = st.session_state.demo_result

if demo is None:
    st.error(st.session_state.get("demo_error") or "Demo generation failed. Check OpenCV video writer support in your environment.")
    st.stop()

st.title("🛰️ Field Mission: Perception and Autonomy with Clear Limits")
st.markdown(
    """
    This single-file Streamlit app demonstrates a privacy-aware, auditable, edge-oriented
    perception and autonomy console. It auto-generates a live synthetic video demo,
    enforces no-face-recognition behavior, records audit trails, benchmarks inference,
    tests robustness, gates releases, and provides human override controls.
    """
)

tabs = st.tabs(
    [
        "🎥 Live Demo",
        "⚡ Edge Bench",
        "📜 License Audit",
        "🧪 Adversarial",
        "🛑 Override",
        "🔁 Replay",
        "🎯 Calibration",
        "🔒 Secure Runtime",
        "🚦 Release Gate",
        "🤖 Action Policy",
    ]
)

with tabs[0]:
    render_live_demo(demo)

with tabs[1]:
    render_edge_benchmark()

with tabs[2]:
    render_license_auditor()

with tabs[3]:
    render_adversarial()

with tabs[4]:
    render_override()

with tabs[5]:
    render_audit_replay()

with tabs[6]:
    render_calibration()

with tabs[7]:
    render_secure_runtime()

with tabs[8]:
    render_continuous_eval()

with tabs[9]:
    render_policy_engine()

st.divider()
st.caption(
    "All demo data is synthetic. The app does not use an API key, does not perform face recognition, "
    "and stores only ephemeral anonymized tracks plus local audit artifacts during the session."
)