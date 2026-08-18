"""Indian digit grouping for INR amounts -- the last 3 digits group normally,
every pair before that groups in twos (15,29,999 not 1,529,999). Kept as a
plain string filter (not a wider locale/i18n switch) since every amount on
this site is already denominated in INR by construction.
"""
from django import template

register = template.Library()


def _group_indian(int_part: str) -> str:
    if len(int_part) <= 3:
        return int_part
    head, tail = int_part[:-3], int_part[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


@register.filter
def indian_number(value):
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return value
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    return f"{sign}{_group_indian(whole)}.{frac}"


@register.filter
def indian_compact(value):
    """Lakh/crore abbreviation for large aggregates, e.g. 1250000 -> '12.50 L'."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return value
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_00_00_000:
        return f"{sign}{amount / 1_00_00_000:.2f} Cr"
    if amount >= 1_00_000:
        return f"{sign}{amount / 1_00_000:.2f} L"
    return f"{sign}{indian_number(amount)}"
