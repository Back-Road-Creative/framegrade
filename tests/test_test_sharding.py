"""CI splits the suite across hosted runners with FRAMEGRADE_TEST_SHARD=i/N.

Every test must land in exactly one shard, or a test silently stops running in
CI; a malformed spec must stop the run, not run everything or nothing.
"""

import os

import pytest

from conftest import parse_test_shard, shard_of  # type: ignore[import-not-found]

NODEIDS = [
    f"tests/test_mod_{m}.py::test_case_{c}[{p}]" for m in range(40) for c in range(25) for p in "ab"
]


@pytest.mark.parametrize("total", [1, 2, 3, 4, 7])
def test_every_test_lands_in_exactly_one_shard(total):
    shards = {i: {n for n in NODEIDS if shard_of(n, total) == i} for i in range(1, total + 1)}
    assert set().union(*shards.values()) == set(NODEIDS)
    assert sum(len(s) for s in shards.values()) == len(NODEIDS)


def test_shard_assignment_is_stable_and_spread():
    # crc32, not hash(): PYTHONHASHSEED differs per xdist worker and per runner.
    assert [shard_of(n, 4) for n in NODEIDS[:50]] == [shard_of(n, 4) for n in NODEIDS[:50]]
    sizes = [sum(shard_of(n, 4) == i for n in NODEIDS) for i in range(1, 5)]
    assert min(sizes) > len(NODEIDS) / 4 * 0.8, sizes


def test_unset_spec_selects_everything():
    assert parse_test_shard(None) is None
    assert parse_test_shard("") is None


def test_spec_is_index_over_total():
    assert parse_test_shard("2/4") == (2, 4)
    assert parse_test_shard("1/1") == (1, 1)


@pytest.mark.parametrize("spec", ["0/4", "5/4", "4", "a/b", "1/0", "-1/4", "2/4/1"])
def test_malformed_spec_stops_the_run(spec):
    with pytest.raises(pytest.UsageError, match="FRAMEGRADE_TEST_SHARD"):
        parse_test_shard(spec)


def test_spec_is_consumed_at_collection():
    # A pytest this suite starts in a subprocess must run all of its own tests.
    assert "FRAMEGRADE_TEST_SHARD" not in os.environ
