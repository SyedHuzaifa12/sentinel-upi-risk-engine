"""The canonical UPI event schema.

Exactly the fields a payment app knows at request time — no aggregates, no
counts, no chargeback-style fields, no country. Anything a real-time risk
engine has to compute from history (fan-in, payee age, velocity) belongs in
feature_lib's feature layer built on top of this, never in the raw event.

This module has no dependency on ml/ or backend/ — it is the shared contract
between the synthetic data generator (ml/src/generator), training, the
FastAPI serving layer, and the stream worker. Nothing in feature_lib may
import from ml/ or backend/; those import from here instead.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

# Crockford base32: excludes I, L, O, U to avoid visual ambiguity.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def make_ulid(timestamp: datetime, rng) -> str:
    """A minimal ULID: 48-bit millisecond timestamp + 80-bit randomness,
    Crockford base32 encoded (26 chars). Lexicographic sort matches
    chronological order as long as timestamps are non-decreasing when this
    is called, which the generator guarantees by assigning ULIDs only after
    final timeline ordering.

    `rng` is the shared seeded numpy Generator — no other randomness source
    is used here, so output is byte-identical for a given seed.
    """
    ts_ms = int(timestamp.timestamp() * 1000)
    randomness = int.from_bytes(rng.bytes(10), byteorder="big")  # 80 bits

    value = (ts_ms << 80) | randomness
    chars = []
    for _ in range(26):
        value, remainder = divmod(value, 32)
        chars.append(_ULID_ALPHABET[remainder])
    return "".join(reversed(chars))


def new_event_dict(timestamp, payer_vpa, payee_vpa, amount, txn_type, initiation_mode,
                    device_id, payer_bank, payee_bank, payer_account_age_days,
                    label_is_fraud, label_typology):
    """Shared factory for the pre-ULID event dict, used by both the legit
    simulation loop (ml/src/generator/generator.py) and the fraud typology
    injectors (ml/src/generator/typologies.py) so every event — fraud or
    not — has the same shape."""
    return {
        "timestamp": timestamp,
        "payer_vpa": payer_vpa,
        "payee_vpa": payee_vpa,
        "amount": round(float(amount), 2),
        "txn_type": txn_type,
        "initiation_mode": initiation_mode,
        "device_id": device_id,
        "payer_bank": payer_bank,
        "payee_bank": payee_bank,
        "payer_account_age_days": int(payer_account_age_days),
        "label_is_fraud": label_is_fraud,
        "label_typology": label_typology,
    }


class UPIEvent(BaseModel):
    txn_id: str
    timestamp: datetime
    payer_vpa: str
    payee_vpa: str
    amount: float
    txn_type: Literal["P2P", "P2M"]
    initiation_mode: Literal["SCAN_QR", "INTENT", "COLLECT_REQUEST", "CONTACT"]
    device_id: str
    payer_bank: str
    payee_bank: str
    payer_account_age_days: int
    label_is_fraud: bool
    label_typology: Optional[str] = None
