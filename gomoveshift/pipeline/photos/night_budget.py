"""Shared explicit GiB input for immutable private project storage budgets."""

from decimal import Decimal, InvalidOperation


def budget_bytes(value: str | None) -> int | None:
    """Blank means uncapped; positive GiB is rounded down to whole bytes."""
    if value is None or not str(value).strip():
        return None
    error = "Storage budget must be finite positive GiB between 1 byte and 2^63-1 bytes"
    try:
        amount = Decimal(str(value).strip())
        if not amount.is_finite() or amount <= 0 or amount > Decimal(2**63 - 1) / 2**30:
            raise ValueError(error)
        result = int(amount * 2**30)
        if result < 1 or result > 2**63 - 1:
            raise ValueError(error)
        return result
    except InvalidOperation as exc:
        raise ValueError(error) from exc
