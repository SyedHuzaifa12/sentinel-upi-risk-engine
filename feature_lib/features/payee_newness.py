"""Group C -- payee newness features. Computable even when the payee is
nearly unknown -- this is the cold-start core, the one group that has to
say something useful about a payee with zero transaction history.

The dictionary-name reference list below is feature_lib's own, independent
of ml/src/generator's name pools -- feature_lib must never import from ml/,
and a real dictionary-name check wouldn't know how any particular training
set's VPAs were generated anyway.
"""
import math
from collections import Counter

_COMMON_NAMES = {
    "raj", "priya", "amit", "sneha", "vikram", "anita", "rahul", "pooja", "arjun", "neha",
    "suresh", "kavita", "manoj", "divya", "sanjay", "meera", "ravi", "shreya", "vijay", "anjali",
    "karan", "swati", "deepak", "nisha", "ajay", "ritu", "vinod", "priyanka", "sunil", "asha",
    "kumar", "sharma", "patel", "singh", "reddy", "gupta", "verma", "iyer", "nair", "rao",
    "mehta", "joshi", "shah", "desai", "chauhan", "malhotra", "pillai", "menon", "das", "bose",
    "rohit", "sana", "farhan", "zoya", "aditya", "ishita", "gaurav", "tanvi", "harsh", "simran",
}


def _local_part(vpa: str) -> str:
    return vpa.split("@", 1)[0]


def _digit_ratio(local: str) -> float:
    if not local:
        return 0.0
    digits = sum(1 for c in local if c.isdigit())
    return digits / len(local)


def _shannon_entropy(local: str) -> float:
    if not local:
        return 0.0
    counts = Counter(local)
    total = len(local)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _has_dictionary_name(local: str) -> bool:
    lowered = local.lower()
    tokens = lowered.replace(".", " ").replace("_", " ").replace("-", " ").split()
    if any(token in _COMMON_NAMES for token in tokens):
        return True
    # also catch names glued together without a separator, e.g. "priyastores"
    return any(name in lowered for name in _COMMON_NAMES)


def payee_age_hours_in_system(event, store):
    first_seen = store.payee_first_seen(event.payee_vpa, event.timestamp)
    if first_seen is None:
        return (0.0, False)  # spec-mandated: 0 if never seen, not missing
    return ((event.timestamp - first_seen).total_seconds() / 3600.0, False)


def payee_total_txn_count(event, store):
    return (store.payee_txn_count(event.payee_vpa, event.timestamp), False)


def payee_is_unseen(event, store):
    return (store.payee_txn_count(event.payee_vpa, event.timestamp) == 0, False)


def payee_handle_digit_ratio(event, store):
    return (_digit_ratio(_local_part(event.payee_vpa)), False)


def payee_handle_entropy(event, store):
    return (_shannon_entropy(_local_part(event.payee_vpa)), False)


def payee_handle_has_dictionary_name(event, store):
    return (_has_dictionary_name(_local_part(event.payee_vpa)), False)
