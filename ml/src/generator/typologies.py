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
                       n_operations, min_senders=8, max_senders=40):
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
        # Spread over 6-24h (was 1-6h): a fan-in this diffuse still stands
        # out against ordinary payees, but no longer at an order-of-
        # magnitude velocity no legitimate merchant could ever approach.
        window_hours = float(rng.uniform(6, 24))

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


SCAM_COLLECT_SLOPPY_RATE = 0.25


def inject_scam_collect(rng, payers, sim_start, days, payees_by_day, n_instances):
    """A COLLECT_REQUEST to a payer who almost never uses that mode, to a
    payee brand new to them. Amount is moderately elevated (2-15x personal
    average, most mass well within the overlap of the overall amount
    distribution) rather than a separate 5-100x range that never overlapped
    legitimate spending -- see PROGRESS.md "Bugs found and fixed" for why a
    non-overlapping range made amount alone a near-perfect (AUC ~0.997)
    separator. `SCAM_COLLECT_SLOPPY_RATE` of instances additionally use only
    a mild 1-3x multiplier, modeling a scammer who doesn't ask for an
    unusually large sum."""
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
        if rng.random() < SCAM_COLLECT_SLOPPY_RATE:
            amount = median_amount * float(rng.uniform(1.0, 3.0))
        else:
            amount = median_amount * float(rng.uniform(2.0, 15.0))

        # ~35% skip the collect-request altogether (a scammer just asks the
        # victim to "send it directly" instead) -- otherwise is_collect_request
        # was a near-perfect (AUC ~0.99) single-feature tell for this typology,
        # since almost no legit traffic uses COLLECT_REQUEST at all.
        if rng.random() < 0.35:
            initiation_mode = ["INTENT", "CONTACT"][int(rng.integers(0, 2))]
        else:
            initiation_mode = "COLLECT_REQUEST"

        instances.append([_new_event(
            timestamp=ts,
            payer_vpa=payer.vpa,
            payee_vpa=payee.vpa,
            amount=amount,
            txn_type="P2P",
            initiation_mode=initiation_mode,
            device_id=payer.device_id,
            payer_bank=payer.bank,
            payee_bank=payee.bank,
            payer_account_age_days=payer.account_age_days,
            label_is_fraud=True,
            label_typology="SCAM_COLLECT",
        )])

    return instances


ATO_BURST_SLOPPY_RATE = 0.25


def inject_ato_burst(rng, payers, payees, sim_start, days, payees_by_day, used_device_ids, n_instances,
                      established_rate=0.5):
    """One device the payer doesn't normally use fires several payments in
    a short window.

    Roughly `established_rate` of instances target payees the payer has
    ALREADY paid before, not just strangers -- a real account-takeover
    attacker often drains to accounts the victim has an existing
    relationship with. The rest target payees new to the payer, as before.
    This mix is deliberate: without it, every fraud typology shared the
    same "payee new to payer" tell, which made that property a near-perfect
    accidental proxy for the fraud label as a whole (see PROGRESS.md "Bugs
    found and fixed").

    `ATO_BURST_SLOPPY_RATE` of instances are additionally "sloppy": normal
    hours instead of unusual ones, and flat (non-escalating) amounts --
    real account takeovers aren't always textbook, and per-typology
    single-feature AUC on amount/hour was previously near 0.93 from every
    instance escalating and hitting an unusual hour without exception."""
    instances = []
    if n_instances <= 0 or not payers:
        return instances

    payees_by_id = {p.payee_id: p for p in payees}

    for _ in range(n_instances):
        payer = payers[int(rng.integers(0, len(payers)))]
        day = int(rng.integers(1, days))
        is_sloppy = rng.random() < ATO_BURST_SLOPPY_RATE

        attacker_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        while attacker_device in used_device_ids or attacker_device == payer.device_id:
            attacker_device = f"dev-{int(rng.integers(0, 10**9)):09d}"
        used_device_ids.add(attacker_device)

        if is_sloppy:
            hour = int(rng.integers(payer.active_hour_start,
                                     max(payer.active_hour_start + 1, payer.active_hour_end)))
        else:
            hour = _unusual_hour(rng, payer.active_hour_start, payer.active_hour_end)
        start_ts = sim_start + timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 30)))

        target_established = rng.random() < established_rate
        candidates = []
        if target_established:
            candidates = [payees_by_id[pid] for pid in payer.known_payee_ids if pid in payees_by_id]
            if len(candidates) < 3:
                target_established = False  # not enough history -- fall back to new-payee mode

        if not target_established:
            available_payees = payees_by_day(day)
            candidates = [p for p in available_payees if p.payee_id not in payer.known_payee_ids]

        if len(candidates) < 3:
            continue

        n_bursts = min(int(rng.integers(3, 9)), len(candidates))
        chosen = rng.choice(len(candidates), size=n_bursts, replace=False)

        base_amount = 2.71828 ** payer.amount_mu * float(rng.uniform(0.8, 1.5))
        # Spread wider than a strict 15-minute burst ~40% of the time (up to
        # 2 hours): a burst crammed into every single instance made
        # payer_txn_count_1h/device_txn_count_1h near-perfect (AUC ~0.90)
        # tells on their own -- a real ATO session doesn't always fire that
        # fast.
        burst_span_minutes = 15.0 if rng.random() >= 0.4 else float(rng.uniform(30, 120))
        offsets = sorted(float(x) for x in rng.uniform(0, burst_span_minutes, size=n_bursts))

        instance_events = []
        amount = base_amount
        for i, payee_idx in enumerate(chosen):
            payee = candidates[int(payee_idx)]
            ts = start_ts + timedelta(minutes=offsets[i])
            if is_sloppy:
                amount *= float(rng.uniform(0.9, 1.2))  # flat, not escalating
            else:
                # Milder than the original 1.2-2.0x/step (which could compound
                # to 250x+ by the 8th burst, far outside any legit amount
                # range): still a real escalation, but overlapping.
                amount *= float(rng.uniform(1.05, 1.35))
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
    looks anomalous, only the payee VPA string differs.

    The lookalike VPA also receives a handful of ordinary, legitimately-
    labeled background transactions from unrelated payers *before* the
    swap window -- attackers reusing a "seasoned" account with some
    innocuous history (rather than a virgin brand-new one) is realistic,
    and it's also a fix: without this, the lookalike's own
    payee_total_txn_count/payee_age_hours_in_system were a near-perfect
    (AUC ~0.97) giveaway, since a truly zero-history payee is otherwise
    only ever seen for genuinely brand-new legit payees or other fraud
    typologies (see PROGRESS.md "Bugs found and fixed")."""
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

        instance_events = []
        n_affected = int(rng.integers(3, min(9, len(regular_payer_ids) + 1)))
        affected_ids = rng.choice(regular_payer_ids, size=n_affected, replace=False)

        seed_window_start = max(0, swap_start_day - 60)
        if swap_start_day > seed_window_start:
            n_seed = int(rng.integers(20, 60))
            seed_payer_ids = list(rng.choice(len(payers), size=min(n_seed, len(payers)), replace=False))

            # Overlap a fraction of the affected regulars into the seed pool:
            # some of them already scanned the lookalike once before, weeks
            # earlier (a barely-noticed one-off), so their labeled QR_SWAP
            # event isn't necessarily their first-ever payment to it either.
            # Without this, payer_payee_txn_count == 0 was true for 100% of
            # QR_SWAP events and nothing else as consistently, making it a
            # near-perfect single-feature tell (see PROGRESS.md "Bugs found
            # and fixed") -- the swap is supposed to be hard to catch, not
            # trivially flagged by "payee new to this payer."
            overlap_fraction = float(rng.uniform(0.4, 0.6))
            n_overlap = int(round(len(affected_ids) * overlap_fraction))
            if n_overlap > 0:
                overlap_ids = rng.choice(affected_ids, size=n_overlap, replace=False)
                seed_payer_ids.extend(int(pid) for pid in overlap_ids)

            for seed_id in seed_payer_ids:
                seed_payer = payers[int(seed_id)]
                seed_day = int(rng.integers(seed_window_start, swap_start_day))
                seed_hour = int(rng.integers(seed_payer.active_hour_start,
                                              max(seed_payer.active_hour_start + 1, seed_payer.active_hour_end)))
                seed_ts = sim_start + timedelta(days=seed_day, hours=seed_hour, minutes=float(rng.uniform(0, 59)))
                seed_amount = float(rng.lognormal(mean=seed_payer.amount_mu, sigma=seed_payer.amount_sigma))
                instance_events.append(_new_event(
                    timestamp=seed_ts,
                    payer_vpa=seed_payer.vpa,
                    payee_vpa=lookalike_vpa,
                    amount=seed_amount,
                    txn_type="P2M",
                    initiation_mode="SCAN_QR",
                    device_id=seed_payer.device_id,
                    payer_bank=seed_payer.bank,
                    payee_bank=lookalike_bank,
                    payer_account_age_days=seed_payer.account_age_days,
                    label_is_fraud=False,
                    label_typology="legit",
                ))

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
