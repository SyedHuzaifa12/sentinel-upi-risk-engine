"""Payer/payee population for the synthetic UPI event generator.

Everything here is built once from a single seeded RNG, in a fixed call
order, so the whole population (and everything simulated from it) is
byte-identical for a given --seed.
"""
import math
from dataclasses import dataclass, field

BANK_HANDLES = ["okaxis", "ybl", "oksbi", "okhdfcbank", "ibl", "axl", "okicici", "paytm"]
BANK_NAMES = ["AXIS", "YESBANK", "SBI", "HDFC", "ICICI", "IDBI", "KOTAK", "PAYTM_BANK"]

FIRST_NAMES = [
    "raj", "priya", "amit", "sneha", "vikram", "anita", "rahul", "pooja", "arjun", "neha",
    "suresh", "kavita", "manoj", "divya", "sanjay", "meera", "ravi", "shreya", "vijay", "anjali",
    "karan", "swati", "deepak", "nisha", "ajay", "ritu", "vinod", "priyanka", "sunil", "asha",
]
LAST_NAMES = [
    "kumar", "sharma", "patel", "singh", "reddy", "gupta", "verma", "iyer", "nair", "rao",
    "mehta", "joshi", "shah", "desai", "chauhan", "malhotra", "pillai", "menon", "das", "bose",
]
MERCHANT_WORDS = [
    "kirana", "stores", "traders", "supermarket", "mobile", "medicals", "bakery", "electronics",
    "textiles", "hardware", "sweets", "cafe", "fashions", "opticals", "footwear",
]

# Crockford-alphabet-free plain digits/letters, used only for disposable-looking
# mule handles so they read as clearly distinct from human `firstname.lastname` VPAs.
_MULE_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# A large share of real Indian UPI IDs are phone numbers, not name.name@bank
# handles. Applied to payers and individual (non-merchant) payees only, at
# this rate, so a downstream digit-heavy-handle feature can't trivially use
# "digit ratio" to separate legitimate traffic from disposable mule handles.
PHONE_HANDLE_RATE = 0.375


@dataclass
class Payer:
    payer_id: int
    vpa: str
    bank: str
    device_id: str
    account_age_days: int
    amount_mu: float
    amount_sigma: float
    active_hour_start: int
    active_hour_end: int
    new_payee_rate: float
    recurring_payee_ids: list
    daily_txn_rate: float
    known_payee_ids: set = field(default_factory=set)
    collect_request_rate: float = 0.05


@dataclass
class Payee:
    payee_id: int
    vpa: str
    bank: str
    kind: str  # "merchant" or "individual"
    amount_mu: float
    amount_sigma: float
    active_hour_start: int
    active_hour_end: int
    created_day: int  # 0 for initial population, 1..days-1 for steady churn


def _unique_human_vpa(rng, bank_handle, used_vpas):
    for _ in range(50):
        first = FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]
        last = LAST_NAMES[rng.integers(0, len(LAST_NAMES))]
        vpa = f"{first}.{last}@{bank_handle}"
        if vpa not in used_vpas:
            used_vpas.add(vpa)
            return vpa
    # Exhausted first.last combos for this bank — disambiguate with a suffix,
    # still clearly human-shaped (unlike the mule handles' pure random digits).
    suffix = int(rng.integers(10, 99))
    vpa = f"{first}.{last}{suffix}@{bank_handle}"
    used_vpas.add(vpa)
    return vpa


def _unique_phone_vpa(rng, bank_handle, used_vpas):
    """A 10-digit Indian-mobile-style handle (first digit 6-9) — a
    legitimate, digit-heavy VPA shape, distinct from both the human
    firstname.lastname handles and the disposable mule handles."""
    for _ in range(50):
        first_digit = str(int(rng.integers(6, 10)))
        rest = "".join(str(int(rng.integers(0, 10))) for _ in range(9))
        vpa = f"{first_digit}{rest}@{bank_handle}"
        if vpa not in used_vpas:
            used_vpas.add(vpa)
            return vpa
    raise RuntimeError("could not allocate a unique phone-style VPA")


def _unique_person_vpa(rng, bank_handle, used_vpas):
    """A legitimate person-style handle: phone-number-shaped at
    PHONE_HANDLE_RATE, name.name-shaped otherwise."""
    if rng.random() < PHONE_HANDLE_RATE:
        return _unique_phone_vpa(rng, bank_handle, used_vpas)
    return _unique_human_vpa(rng, bank_handle, used_vpas)


def _unique_merchant_vpa(rng, bank_handle, used_vpas):
    for _ in range(50):
        name = FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]
        word = MERCHANT_WORDS[rng.integers(0, len(MERCHANT_WORDS))]
        vpa = f"{name}{word}@{bank_handle}"
        if vpa not in used_vpas:
            used_vpas.add(vpa)
            return vpa
    suffix = int(rng.integers(10, 99))
    vpa = f"{name}{word}{suffix}@{bank_handle}"
    used_vpas.add(vpa)
    return vpa


def make_disposable_vpa(rng, used_vpas):
    """A mule-style handle: random letters + random digits, no name structure
    at all — deliberately unlike any legitimate payer/payee VPA in this
    population."""
    for _ in range(50):
        letters = "".join(_MULE_LETTERS[rng.integers(0, len(_MULE_LETTERS))] for _ in range(3))
        digits = int(rng.integers(1000, 9999))
        bank_handle = BANK_HANDLES[rng.integers(0, len(BANK_HANDLES))]
        vpa = f"{letters}{digits}@{bank_handle}"
        if vpa not in used_vpas:
            used_vpas.add(vpa)
            return vpa, bank_handle
    raise RuntimeError("could not allocate a unique disposable VPA")


def build_payers(rng, n_payers, used_vpas):
    payers = []
    for payer_id in range(n_payers):
        bank_handle = BANK_HANDLES[rng.integers(0, len(BANK_HANDLES))]
        vpa = _unique_person_vpa(rng, bank_handle, used_vpas)
        device_id = f"dev-{int(rng.integers(0, 10**9)):09d}"
        account_age_days = int(min(4000, max(1, rng.gamma(shape=2.0, scale=250.0))))

        # Personal spending scale: log-normal params vary payer to payer.
        amount_mu = float(rng.uniform(5.5, 8.0))     # ~ INR 245 to 2980 median
        amount_sigma = float(rng.uniform(0.3, 0.9))

        active_hour_start = int(rng.integers(6, 11))
        active_duration = int(rng.integers(8, 15))
        active_hour_end = min(23, active_hour_start + active_duration)

        # Most people rarely try brand-new payees; a long tail is more adventurous.
        new_payee_rate = float(rng.beta(2.0, 20.0))
        collect_request_rate = float(rng.beta(1.5, 30.0))
        daily_txn_rate = float(rng.gamma(shape=2.0, scale=0.35))  # mean ~0.7 txns/day

        payers.append(Payer(
            payer_id=payer_id,
            vpa=vpa,
            bank=bank_handle,
            device_id=device_id,
            account_age_days=account_age_days,
            amount_mu=amount_mu,
            amount_sigma=amount_sigma,
            active_hour_start=active_hour_start,
            active_hour_end=active_hour_end,
            new_payee_rate=new_payee_rate,
            recurring_payee_ids=[],
            daily_txn_rate=daily_txn_rate,
            collect_request_rate=collect_request_rate,
        ))
    return payers


def build_payees(rng, n_payees, days, used_vpas, churn_fraction=0.20):
    """Builds the payee pool. `churn_fraction` of payees are NOT present on
    day 0 — they're introduced steadily across the whole simulation window
    (required so "payee first seen after day 0" can't become a perfect
    fraud separator; see DATA_CARD.md and ml/tests/test_generator.py).

    Coverage across the 90 days is stratified (a minimum count per rolling
    ~10-day window), not left to pure chance, so the churn guarantee holds
    deterministically for any seed rather than merely "usually."
    """
    n_churned = int(round(n_payees * churn_fraction))
    n_initial = n_payees - n_churned

    payees = []
    for payee_id in range(n_initial):
        payees.append(_make_payee(rng, payee_id, used_vpas, created_day=0))

    # Stratify the churned payees across 10-day windows spanning days 1..days-1.
    # Ceiling division (not floor) so the final, possibly-partial window at the
    # end of the run still gets its guaranteed minimum -- otherwise a `days`
    # value that isn't an exact multiple of 10 would leave its tail uncovered.
    window_size = 10
    n_windows = max(1, math.ceil((days - 1) / window_size))
    min_per_window = 2
    guaranteed = min(n_churned, min_per_window * n_windows)

    created_days = []
    for w in range(n_windows):
        window_start = 1 + w * window_size
        window_end = min(days - 1, window_start + window_size - 1)
        for _ in range(min_per_window):
            if len(created_days) >= guaranteed:
                break
            created_days.append(int(rng.integers(window_start, window_end + 1)))

    remaining = n_churned - len(created_days)
    if remaining > 0:
        created_days.extend(int(d) for d in rng.integers(1, days, size=remaining))

    for offset, created_day in enumerate(created_days):
        payee_id = n_initial + offset
        payees.append(_make_payee(rng, payee_id, used_vpas, created_day=created_day))

    return payees


def _make_payee(rng, payee_id, used_vpas, created_day):
    is_merchant = rng.random() < 0.35  # merchants are a minority but high fan-in
    bank_handle = BANK_HANDLES[rng.integers(0, len(BANK_HANDLES))]

    if is_merchant:
        vpa = _unique_merchant_vpa(rng, bank_handle, used_vpas)
        amount_mu = float(rng.uniform(5.0, 7.5))
        amount_sigma = float(rng.uniform(0.2, 0.5))  # merchants: tighter spread
        active_hour_start = int(rng.integers(8, 11))
        active_hour_end = min(22, active_hour_start + int(rng.integers(8, 12)))
        kind = "merchant"
    else:
        vpa = _unique_person_vpa(rng, bank_handle, used_vpas)
        amount_mu = float(rng.uniform(5.0, 8.0))
        amount_sigma = float(rng.uniform(0.3, 0.9))
        active_hour_start = int(rng.integers(6, 11))
        active_hour_end = min(23, active_hour_start + int(rng.integers(8, 15)))
        kind = "individual"

    return Payee(
        payee_id=payee_id,
        vpa=vpa,
        bank=bank_handle,
        kind=kind,
        amount_mu=amount_mu,
        amount_sigma=amount_sigma,
        active_hour_start=active_hour_start,
        active_hour_end=active_hour_end,
        created_day=created_day,
    )


def assign_recurring_payees(rng, payers, payees):
    """Each payer gets 3-8 persistent payees (rent/family/regular merchants),
    drawn only from payees available on day 0 -- payees introduced later via
    churn are only ever reachable as "new" payees, never as someone's
    long-standing recurring contact."""
    day0_merchants = [p.payee_id for p in payees if p.kind == "merchant" and p.created_day == 0]
    day0_individuals = [p.payee_id for p in payees if p.kind == "individual" and p.created_day == 0]

    for payer in payers:
        n_recurring = int(rng.integers(3, 9))
        recurring = []
        for _ in range(n_recurring):
            if day0_merchants and rng.random() < 0.6:
                recurring.append(int(day0_merchants[rng.integers(0, len(day0_merchants))]))
            elif day0_individuals:
                recurring.append(int(day0_individuals[rng.integers(0, len(day0_individuals))]))
        payer.recurring_payee_ids = list(dict.fromkeys(recurring))  # de-dup, keep order
        payer.known_payee_ids.update(payer.recurring_payee_ids)
