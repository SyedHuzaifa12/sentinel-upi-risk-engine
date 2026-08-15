"""Fraud typology injectors.

Every typology here is encoded as a *relationship between multiple events*
(fan-in at a payee, a device switch, a burst pattern, a lookalike VPA) —
never as a single-row flag a naive classifier could key on directly. See
data/synthetic/DATA_CARD.md for the honesty note on QR_SWAP in particular.

Each injector returns a list of "instances," where an instance is itself a
list of event dicts (pre-ULID, pre-pydantic-validation) — this lets the
budget planner in generator.py trim whole instances if a typology
overshoots its share of the target fraud rate, without ever splitting an
instance in a way that would corrupt its behavioural pattern.
"""
from datetime import timedelta

from .population import BANK_HANDLES, make_disposable_vpa
from feature_lib.event import new_event_dict as _new_event

_LOOKALIKE_SWAPS = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def make_lookalike_vpa(rng, original_vpa, used_vpas):
    """A one-character mutation of a real merchant VPA: a swapped
    letter/digit, or a doubled character. Deliberately subtle."""
    local, _, bank_handle = original_vpa.partition("@")
    for _ in range(50):
        mutation = rng.integers(0, 3)
        pos = int(rng.integers(0, len(local)))
        if mutation == 0:
            # swap a letter for a visually similar digit, if one exists nearby
            candidates = [i for i, c in enumerate(local) if c in _LOOKALIKE_SWAPS]
            if not candidates:
                continue
            idx = candidates[rng.integers(0, len(candidates))]
            new_local = local[:idx] + _LOOKALIKE_SWAPS[local[idx]] + local[idx + 1:]
        elif mutation == 1:
            # duplicate a character
            new_local = local[:pos] + local[pos] + local[pos:]
        else:
            # insert a random digit
            digit = str(int(rng.integers(0, 10)))
            new_local = local[:pos] + digit + local[pos:]

        vpa = f"{new_local}@{bank_handle}"
        if vpa != original_vpa and vpa not in used_vpas:
            used_vpas.add(vpa)
            return vpa
    raise RuntimeError("could not build a unique lookalike VPA")


def inject_mule_fanin(rng, payers, sim_start, days, used_vpas, used_device_ids,
                       n_operations, min_senders=30, max_senders=200):
    """Each operation: a disposable VPA receives from many unrelated payers
    within a short window, then fans most of it back out and goes silent.
    Inbound legs are the fraud target (label_is_fraud=True); outbound legs
    are retained for fan-out features but NOT labeled fraud (the payer on
    those events is the mule itself, so payer-deviation features are
    meaningless there — see DATA_CARD.md)."""
    instances = []
    n_payers = len(payers)
    max_senders = max(min_senders, min(max_senders, n_payers))

    for _ in range(n_operations):
        mule_vpa, mule_bank = make_disposable_vpa(rng, used_vpas)
        mule_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        while mule_device in used_device_ids:
            mule_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        used_device_ids.add(mule_device)

        mule_day = int(rng.integers(10, max(11, days - 10)))
        day_start = sim_start + timedelta(days=mule_day)
        window_start = day_start + timedelta(hours=float(rng.uniform(0, 20)))
        window_hours = float(rng.uniform(1, 6))

        n_senders = int(rng.integers(min_senders, max_senders + 1))
        sender_ids = rng.choice(n_payers, size=n_senders, replace=False)

        instance_events = []
        latest_inbound = window_start
        for sender_id in sender_ids:
            payer = payers[int(sender_id)]
            offset_hours = float(rng.uniform(0, window_hours))
            ts = window_start + timedelta(hours=offset_hours)
            latest_inbound = max(latest_inbound, ts)

            amount = float(rng.lognormal(mean=payer.amount_mu, sigma=payer.amount_sigma))
            initiation_mode = ["INTENT", "SCAN_QR", "CONTACT"][int(rng.integers(0, 3))]

            instance_events.append(_new_event(
                timestamp=ts,
                payer_vpa=payer.vpa,
                payee_vpa=mule_vpa,
                amount=amount,
                txn_type="P2P",
                initiation_mode=initiation_mode,
                device_id=payer.device_id,
                payer_bank=payer.bank,
                payee_bank=mule_bank,
                payer_account_age_days=payer.account_age_days,
                label_is_fraud=True,
                label_typology="MULE_FANIN",
            ))

        # Fan-out: mule pays a handful of cash-out payees, then goes silent.
        n_outflows = int(rng.integers(2, 7))
        cashout_targets = rng.choice(n_payers, size=min(n_outflows, n_payers), replace=False)
        outbound_start = latest_inbound + timedelta(hours=float(rng.uniform(0.1, 1.0)))
        for i, target_id in enumerate(cashout_targets):
            target_payee = payers[int(target_id)]
            ts = outbound_start + timedelta(minutes=float(rng.uniform(0, 90)) * (i + 1) / len(cashout_targets))
            amount = float(rng.lognormal(mean=target_payee.amount_mu, sigma=target_payee.amount_sigma))
            instance_events.append(_new_event(
                timestamp=ts,
                payer_vpa=mule_vpa,
                payee_vpa=target_payee.vpa,
                amount=amount,
                txn_type="P2P",
                initiation_mode="INTENT",
                device_id=mule_device,
                payer_bank=mule_bank,
                payee_bank=target_payee.bank,
                payer_account_age_days=int(rng.integers(0, 3)),
                label_is_fraud=False,
                label_typology="MULE_OUTBOUND",
            ))

        instances.append(instance_events)

    return instances


def inject_scam_collect(rng, payers, sim_start, days, payees_by_day, n_instances):
    """A COLLECT_REQUEST to a payer who almost never uses that mode, for
    5-100x their personal average amount, to a payee brand new to them."""
    instances = []
    if n_instances <= 0 or not payers:
        return instances

    by_collect_rate = sorted(range(len(payers)), key=lambda i: payers[i].collect_request_rate)
    candidate_pool = by_collect_rate[: max(1, len(payers) // 2)]

    for _ in range(n_instances):
        payer = payers[candidate_pool[int(rng.integers(0, len(candidate_pool)))]]
        day = int(rng.integers(1, days))
        available_payees = payees_by_day(day)
        candidates = [p for p in available_payees if p.payee_id not in payer.known_payee_ids]
        if not candidates:
            continue
        payee = candidates[int(rng.integers(0, len(candidates)))]

        if rng.random() < 0.7:
            hour = _unusual_hour(rng, payer.active_hour_start, payer.active_hour_end)
        else:
            hour = int(rng.integers(payer.active_hour_start, max(payer.active_hour_start + 1, payer.active_hour_end)))
        ts = sim_start + timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 59)))

        median_amount = 2.71828 ** payer.amount_mu
        amount = median_amount * float(rng.uniform(5, 100))

        instances.append([_new_event(
            timestamp=ts,
            payer_vpa=payer.vpa,
            payee_vpa=payee.vpa,
            amount=amount,
            txn_type="P2P",
            initiation_mode="COLLECT_REQUEST",
            device_id=payer.device_id,
            payer_bank=payer.bank,
            payee_bank=payee.bank,
            payer_account_age_days=payer.account_age_days,
            label_is_fraud=True,
            label_typology="SCAM_COLLECT",
        )])

    return instances


def inject_ato_burst(rng, payers, sim_start, days, payees_by_day, used_device_ids, n_instances):
    """One device the payer doesn't normally use fires several escalating
    payments in a short window at an unusual hour, all to new payees."""
    instances = []
    if n_instances <= 0 or not payers:
        return instances

    for _ in range(n_instances):
        payer = payers[int(rng.integers(0, len(payers)))]
        day = int(rng.integers(1, days))

        attacker_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        while attacker_device in used_device_ids or attacker_device == payer.device_id:
            attacker_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        used_device_ids.add(attacker_device)

        hour = _unusual_hour(rng, payer.active_hour_start, payer.active_hour_end)
        start_ts = sim_start + timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 30)))

        available_payees = payees_by_day(day)
        candidates = [p for p in available_payees if p.payee_id not in payer.known_payee_ids]
        if len(candidates) < 3:
            continue

        n_bursts = min(int(rng.integers(3, 9)), len(candidates))
        chosen = rng.choice(len(candidates), size=n_bursts, replace=False)

        base_amount = 2.71828 ** payer.amount_mu * float(rng.uniform(0.8, 1.5))
        offsets = sorted(float(x) for x in rng.uniform(0, 15, size=n_bursts))

        instance_events = []
        amount = base_amount
        for i, payee_idx in enumerate(chosen):
            payee = candidates[int(payee_idx)]
            ts = start_ts + timedelta(minutes=offsets[i])
            amount *= float(rng.uniform(1.2, 2.0))
            instance_events.append(_new_event(
                timestamp=ts,
                payer_vpa=payer.vpa,
                payee_vpa=payee.vpa,
                amount=amount,
                txn_type="P2P",
                initiation_mode=["INTENT", "CONTACT"][int(rng.integers(0, 2))],
                device_id=attacker_device,
                payer_bank=payer.bank,
                payee_bank=payee.bank,
                payer_account_age_days=payer.account_age_days,
                label_is_fraud=True,
                label_typology="ATO_BURST",
            ))
        instances.append(instance_events)

    return instances


def inject_qr_swap(rng, payers, payees, sim_start, days, used_vpas,
                    regulars_by_payee, n_instances, min_regulars=3):
    """A merchant's regular payers pay a near-lookalike VPA at the
    merchant's normal hour/amount — deliberately hard to catch (see
    DATA_CARD.md): nothing about amount, timing, or relationship frequency
    looks anomalous, only the payee VPA string differs."""
    instances = []
    if n_instances <= 0:
        return instances

    payees_by_id = {p.payee_id: p for p in payees}
    qualifying = [pid for pid, regs in regulars_by_payee.items() if len(regs) >= min_regulars]
    if not qualifying:
        return instances

    for _ in range(n_instances):
        merchant_id = qualifying[int(rng.integers(0, len(qualifying)))]
        merchant = payees_by_id[merchant_id]
        regular_payer_ids = regulars_by_payee[merchant_id]

        lookalike_vpa = make_lookalike_vpa(rng, merchant.vpa, used_vpas)
        lookalike_bank = BANK_HANDLES[int(rng.integers(0, len(BANK_HANDLES)))]

        swap_start_day = int(rng.integers(1, max(2, days - 5)))
        swap_duration = int(rng.integers(3, 11))

        n_affected = int(rng.integers(3, min(9, len(regular_payer_ids) + 1)))
        affected_ids = rng.choice(regular_payer_ids, size=n_affected, replace=False)

        instance_events = []
        for payer_id in affected_ids:
            payer = payers[int(payer_id)]
            day = swap_start_day + int(rng.integers(0, swap_duration))
            hour = int(rng.integers(merchant.active_hour_start,
                                     max(merchant.active_hour_start + 1, merchant.active_hour_end)))
            ts = sim_start + timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 59)))
            amount = float(rng.lognormal(mean=merchant.amount_mu, sigma=merchant.amount_sigma))

            instance_events.append(_new_event(
                timestamp=ts,
                payer_vpa=payer.vpa,
                payee_vpa=lookalike_vpa,
                amount=amount,
                txn_type="P2M",
                initiation_mode="SCAN_QR",
                device_id=payer.device_id,
                payer_bank=payer.bank,
                payee_bank=lookalike_bank,
                payer_account_age_days=payer.account_age_days,
                label_is_fraud=True,
                label_typology="QR_SWAP",
            ))
        instances.append(instance_events)

    return instances


def _unusual_hour(rng, active_start, active_end):
    for _ in range(20):
        hour = int(rng.integers(0, 24))
        if not (active_start <= hour < active_end):
            return hour
    return (active_start - 3) % 24
