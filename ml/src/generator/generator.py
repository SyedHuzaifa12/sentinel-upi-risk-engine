"""Synthetic UPI event generator — simulation loop, fraud-budget planning,
determinism, and serialization.

Usage (see cli.py for the command-line wrapper):

    from ml.src.generator.generator import generate
    events, summary = generate(days=90, n_payers=5000, seed=42, fraud_rate=0.007)
"""
import csv
from datetime import datetime, timedelta, timezone

import numpy as np

from .population import (
    assign_recurring_payees,
    build_payees,
    build_payers,
)
from feature_lib.event import UPIEvent, make_ulid, new_event_dict
from .typologies import inject_ato_burst, inject_mule_fanin, inject_qr_swap, inject_scam_collect

# Fraud-budget weights (fraction of the target fraud-event count). MULE_FANIN
# is coarse-grained (30-200 events per operation) so it's planned first and
# capped; SCAM_COLLECT is single-event-per-instance so it's used last, as the
# fine adjustment that lands the final rate inside the configured band.
_TYPOLOGY_WEIGHTS = {"mule": 0.35, "ato": 0.20, "qr": 0.20, "scam": 0.25}
# Reduced from 30-200 (2026-08-16): a 30-200-sender fan-in in 1-6 hours put
# mule velocity orders of magnitude above any legitimate merchant, making
# payee_fanin_velocity a perfect (AUC 1.0) separator -- unrealistically easy.
# See PROGRESS.md "Bugs found and fixed".
_MULE_MIN_SENDERS = 8
_MULE_MAX_SENDERS = 40
_MULE_AVG_SENDERS = 24
_ATO_AVG_EVENTS = 5.5
_QR_AVG_EVENTS = 5.0

# Legit events that deliberately mimic fraud-shaped behavior (2026-08-16 fix,
# see PROGRESS.md "Bugs found and fixed"): without these, amount/collect-
# request/new-payee patterns only ever appeared in fraud, which is not
# realistic -- genuine one-off big purchases, friend-to-friend collect
# requests for real shared expenses, and occasional late-night activity all
# happen in real UPI traffic too.
LEGIT_BIG_ONEOFF_RATE = 0.03      # of new-payee txns, fraction that are a genuine large one-off purchase
LEGIT_BIG_ONEOFF_MULTIPLIER = (3.0, 15.0)
LEGIT_BIG_COLLECT_RATE = 0.15     # of COLLECT_REQUEST txns, fraction that are a real shared-expense request
LEGIT_BIG_COLLECT_MULTIPLIER = (2.0, 6.0)


def _simulate_legit(rng, payers, payees, sim_start, days):
    payee_lookup = {p.payee_id: p for p in payees}
    payees_by_day_cache = {}

    def payees_by_day(day):
        if day not in payees_by_day_cache:
            payees_by_day_cache[day] = [p for p in payees if p.created_day <= day]
        return payees_by_day_cache[day]

    events = []
    for day in range(days):
        day_events = []
        available = payees_by_day(day)

        for payer in payers:
            n_txns = int(rng.poisson(payer.daily_txn_rate))
            for _ in range(n_txns):
                is_new_payee_txn = None  # computed below, once we know the payee
                payee = _choose_payee(rng, payer, available, payee_lookup)
                if payee is None:
                    continue
                is_new_payee_txn = payee.payee_id not in payer.known_payee_ids

                if rng.random() < 0.9:
                    hour = int(rng.integers(payer.active_hour_start,
                                             max(payer.active_hour_start + 1, payer.active_hour_end)))
                else:
                    hour = int(rng.integers(0, 24))
                minute = float(rng.uniform(0, 59))
                ts = sim_start + timedelta(days=day, hours=hour, minutes=minute)

                amount = float(rng.lognormal(mean=payer.amount_mu, sigma=payer.amount_sigma))
                if is_new_payee_txn and rng.random() < LEGIT_BIG_ONEOFF_RATE:
                    # A genuine large one-off purchase to a new merchant/payee.
                    amount *= float(rng.uniform(*LEGIT_BIG_ONEOFF_MULTIPLIER))

                if payee.kind == "merchant":
                    txn_type = "P2M"
                    initiation_mode = ["SCAN_QR", "INTENT"][int(rng.integers(0, 2))]
                else:
                    txn_type = "P2P"
                    if rng.random() < payer.collect_request_rate:
                        initiation_mode = "COLLECT_REQUEST"
                        if rng.random() < LEGIT_BIG_COLLECT_RATE:
                            # A real shared-expense request between friends (rent split, group gift).
                            amount *= float(rng.uniform(*LEGIT_BIG_COLLECT_MULTIPLIER))
                    else:
                        initiation_mode = ["INTENT", "CONTACT"][int(rng.integers(0, 2))]

                day_events.append(new_event_dict(
                    timestamp=ts,
                    payer_vpa=payer.vpa,
                    payee_vpa=payee.vpa,
                    amount=amount,
                    txn_type=txn_type,
                    initiation_mode=initiation_mode,
                    device_id=payer.device_id,
                    payer_bank=payer.bank,
                    payee_bank=payee.bank,
                    payer_account_age_days=payer.account_age_days,
                    label_is_fraud=False,
                    label_typology="legit",
                ))
                payer.known_payee_ids.add(payee.payee_id)

        day_events.sort(key=lambda e: e["timestamp"])
        events.extend(day_events)

    return events, payees_by_day


def _choose_payee(rng, payer, available_payees, payee_lookup):
    r = rng.random()
    if payer.recurring_payee_ids and r < 0.6:
        payee_id = payer.recurring_payee_ids[int(rng.integers(0, len(payer.recurring_payee_ids)))]
        return payee_lookup[payee_id]

    if r < 0.6 + payer.new_payee_rate:
        candidates = [p for p in available_payees if p.payee_id not in payer.known_payee_ids]
        if candidates:
            return candidates[int(rng.integers(0, len(candidates)))]

    recurring_set = set(payer.recurring_payee_ids)
    known_non_recurring = [pid for pid in payer.known_payee_ids if pid not in recurring_set]
    if known_non_recurring:
        payee_id = known_non_recurring[int(rng.integers(0, len(known_non_recurring)))]
        return payee_lookup[payee_id]

    if available_payees:
        return available_payees[int(rng.integers(0, len(available_payees)))]
    return None


def _plan_and_inject_fraud(rng, payers, payees, sim_start, days, used_vpas,
                            payees_by_day, fraud_rate, legit_count):
    target_fraud_count = fraud_rate * legit_count / (1 - fraud_rate)

    mule_budget = target_fraud_count * _TYPOLOGY_WEIGHTS["mule"]
    ato_budget = target_fraud_count * _TYPOLOGY_WEIGHTS["ato"]
    qr_budget = target_fraud_count * _TYPOLOGY_WEIGHTS["qr"]

    used_device_ids = set()

    # MULE_FANIN — skip entirely at scales too small to fit even one
    # operation's minimum sender count without blowing the fraud-rate band.
    if mule_budget >= _MULE_MIN_SENDERS and len(payers) >= _MULE_MIN_SENDERS:
        n_mule_ops = max(1, round(mule_budget / _MULE_AVG_SENDERS))
        # Cap the per-operation sender range to the available budget so a
        # single operation can't blow past the target fraud rate on its own
        # at small test scales; at full scale this cap sits at/above
        # _MULE_MAX_SENDERS anyway and the real 8-40 range applies unmodified.
        per_op_budget = mule_budget / n_mule_ops
        max_senders = int(min(_MULE_MAX_SENDERS, max(_MULE_MIN_SENDERS, round(per_op_budget))))
        mule_instances = inject_mule_fanin(
            rng, payers, sim_start, days, used_vpas, used_device_ids,
            n_operations=n_mule_ops, min_senders=_MULE_MIN_SENDERS, max_senders=max_senders,
        )
    else:
        mule_instances = []

    n_ato_instances = max(1, round(ato_budget / _ATO_AVG_EVENTS)) if ato_budget >= 1 else 0
    ato_instances = inject_ato_burst(
        rng, payers, payees, sim_start, days, payees_by_day, used_device_ids, n_ato_instances,
    )

    regulars_by_payee = {}
    for payer in payers:
        for payee_id in payer.recurring_payee_ids:
            regulars_by_payee.setdefault(payee_id, []).append(payer.payer_id)

    n_qr_instances = max(1, round(qr_budget / _QR_AVG_EVENTS)) if qr_budget >= 1 else 0
    qr_instances = inject_qr_swap(
        rng, payers, payees, sim_start, days, used_vpas, regulars_by_payee, n_qr_instances,
    )

    def fraud_labeled_count(instances):
        return sum(1 for inst in instances for e in inst if e["label_is_fraud"])

    used_so_far = (
        fraud_labeled_count(mule_instances)
        + fraud_labeled_count(ato_instances)
        + fraud_labeled_count(qr_instances)
    )

    # Safety trim: if mule/ato/qr alone already overshoot the target, drop
    # whole instances (never split one) starting with the least essential
    # typology, until there's room left for scam_collect to fine-tune.
    for instances in (qr_instances, ato_instances):
        while used_so_far > target_fraud_count and instances:
            removed = instances.pop()
            used_so_far -= sum(1 for e in removed if e["label_is_fraud"])

    scam_target = max(0, round(target_fraud_count - used_so_far))
    scam_instances = inject_scam_collect(rng, payers, sim_start, days, payees_by_day, scam_target)

    all_instances = mule_instances + ato_instances + qr_instances + scam_instances
    fraud_events = [e for inst in all_instances for e in inst]
    return fraud_events


def _assign_ulids(rng, events):
    events.sort(key=lambda e: e["timestamp"])
    for event in events:
        event["txn_id"] = make_ulid(event["timestamp"], rng)
    return events


def generate(days=90, n_payers=5000, n_payees=2000, seed=42, fraud_rate=0.007, sim_start=None):
    """Runs the full simulation and returns (events, summary).

    `events` is a list of validated `UPIEvent` instances, in ascending
    timestamp order. Fully deterministic for a given `seed`.
    """
    rng = np.random.default_rng(seed)
    if sim_start is None:
        sim_start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    used_vpas = set()
    payers = build_payers(rng, n_payers, used_vpas)
    payees = build_payees(rng, n_payees, days, used_vpas)
    assign_recurring_payees(rng, payers, payees)

    legit_events, payees_by_day = _simulate_legit(rng, payers, payees, sim_start, days)

    fraud_events = _plan_and_inject_fraud(
        rng, payers, payees, sim_start, days, used_vpas, payees_by_day,
        fraud_rate, len(legit_events),
    )

    all_events = legit_events + fraud_events
    _assign_ulids(rng, all_events)

    validated = [UPIEvent(**e) for e in all_events]

    summary = _build_summary(validated, payers, payees)
    return validated, summary


def _build_summary(events, payers, payees):
    total = len(events)
    fraud_events = [e for e in events if e.label_is_fraud]
    n_fraud = len(fraud_events)

    per_typology = {}
    for e in events:
        key = e.label_typology or "legit"
        per_typology[key] = per_typology.get(key, 0) + 1

    unique_payers = len({e.payer_vpa for e in events})
    unique_payees = len({e.payee_vpa for e in events})
    timestamps = [e.timestamp for e in events]

    return {
        "event_count": total,
        "fraud_rate": (n_fraud / total) if total else 0.0,
        "fraud_event_count": n_fraud,
        "per_typology_counts": per_typology,
        "unique_payers": unique_payers,
        "unique_payees": unique_payees,
        "date_range": {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
        },
    }


def write_outputs(events, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "events.jsonl"
    csv_path = out_dir / "events.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(event.model_dump_json())
            f.write("\n")

    fieldnames = list(UPIEvent.model_fields.keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = event.model_dump()
            row["timestamp"] = event.timestamp.isoformat()
            writer.writerow(row)

    return jsonl_path, csv_path


def events_to_hash_input(events):
    """A stable byte representation of the full event list, used by the
    determinism test — same seed must produce an identical hash."""
    return "\n".join(e.model_dump_json() for e in events).encode("utf-8")
