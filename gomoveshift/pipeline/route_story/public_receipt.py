"""Distribution-safe copy of a private ``film_receipt`` for public release.

``film.film_receipt`` is privacy-safe evidence for a completed PRIVATE film
export -- it still carries exact satellite raster bounds/sha256/dimensions
and every visible map label's name, which is fine for the operator's own
audit trail and never fine to publish. ``public_receipt`` derives a redacted
copy for that second, public artifact: the aggregate facts a viewer can
legitimately see (route summary, the applied privacy trim, output filenames,
the renderer used, imagery credits, and photo/label counts) survive
verbatim; anything that could re-identify a place at raster precision or
name a specific label does not.

Publishing a route with no explicit, nonzero privacy trim applied is refused
outright -- see :class:`PublicReceiptRefusedError` -- so a public artifact can
never exist for a route whose start/end (the operator's driveway) was never
deliberately trimmed.
"""

from __future__ import annotations

from .route_privacy import TrimResult


class PublicReceiptRefusedError(ValueError):
    """Raised when no explicit, nonzero route privacy trim was applied."""


def _credit_only(imagery: dict) -> dict:
    """A ``SatelliteRaster.receipt()`` block reduced to its credit alone.

    ``bounds``/``sha256``/``dimensions`` are exact-location and
    exact-file evidence, useful for the operator's private audit trail and
    never for public release; ``credit`` is the only field a public viewer
    needs.
    """
    return {"credit": imagery["credit"]}


def public_receipt(receipt: dict, trim_result: TrimResult | None) -> dict:
    """Derive a distribution-safe copy of ``receipt`` (a ``film_receipt`` dict).

    Refuses with :class:`PublicReceiptRefusedError` when ``trim_result`` is
    absent, ``receipt`` carries no ``route_trim`` block, or the applied trim
    removed nothing (``start_m == end_m == 0``) -- an explicit, nonzero
    privacy trim must have been applied before any public artifact exists,
    so a route can never be published by omission of a trim decision.

    Keeps ``route_trim``, ``route_summary``, ``outputs``, and
    ``map_renderer`` verbatim. Reduces ``national_imagery``/
    ``regional_imagery``/``local_imagery`` to their ``credit`` only (see
    :func:`_credit_only`). Reduces ``map_labels`` to its ``count`` -- the
    named labels themselves are dropped. Reduces ``photos`` to a count --
    no media ids, placements, or coordinates.
    """
    route_trim = receipt.get("route_trim")
    if (
        trim_result is None
        or route_trim is None
        or (trim_result.trim.start_m == 0.0 and trim_result.trim.end_m == 0.0)
    ):
        raise PublicReceiptRefusedError(
            "a public receipt requires an explicit, nonzero route privacy trim "
            "(route_trim absent or start_m/end_m both 0)"
        )
    return {
        "route_trim": route_trim,
        "route_summary": receipt["route_summary"],
        "outputs": receipt["outputs"],
        "map_renderer": receipt["map_renderer"],
        "national_imagery": _credit_only(receipt["national_imagery"]),
        "regional_imagery": [_credit_only(raster) for raster in receipt["regional_imagery"]],
        "local_imagery": _credit_only(receipt["local_imagery"]),
        "map_labels": {"count": receipt["map_labels"]["count"]},
        "photos": {"count": len(receipt["photos"])},
    }
