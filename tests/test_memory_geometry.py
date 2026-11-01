"""Tests for the text memory and for the geometry ARIA can check."""

from __future__ import annotations

import json
import math

import pytest

from aria.geometry import causal, shapes
from aria.llm import ledger as L
from aria.llm import memory as M


# ---------------------------------------------------------------------------
# Memory: remembering
# ---------------------------------------------------------------------------


def test_remembering_something_gives_it_an_id_and_a_time(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    entry = store.add("a cube has six faces", source="aria/geometry/shapes.py", confidence="verified")
    assert entry.id == 1
    assert entry.when
    assert store.entries() == [entry]


def test_identical_text_is_not_stored_twice(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.add("a cube has six faces")
    store.add("A cube has  SIX faces.")
    assert len(store.entries()) == 1


def test_an_empty_memory_is_refused(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    with pytest.raises(ValueError):
        store.add("   ")


def test_kind_and_confidence_are_validated(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    with pytest.raises(ValueError):
        store.add("something", kind="rumour")
    with pytest.raises(ValueError):
        store.add("something", confidence="certain")


def test_corrupt_lines_are_skipped(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = M.MemoryStore(path)
    store.add("a tetrahedron has four faces")
    with path.open("a") as handle:
        handle.write("{not json\n")
    assert len(store.entries()) == 1


def test_forget_removes_one_entry(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    first = store.add("a cube has six faces")
    store.add("a tetrahedron has four faces")
    assert store.forget(first.id) is True
    assert store.forget(first.id) is False
    assert len(store.entries()) == 1


def test_user_preferences_are_subjective_correctable_and_not_verifier_results(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    first = store.add(
        "I liked the cheesecake; offer a variation again",
        kind="preference",
        tags=("dessert",),
        source="user-stated",
        confidence="stated",
    )

    context = store.context("cheesecake variation", max_chars=1200)
    assert "(preference, stated)" in context
    assert "not a fact, consent, or verifier result" in context
    assert "no observation, not dislike or failure" in context

    corrected = store.correct(
        first.id,
        "I prefer lemon cheesecake, not the chocolate one",
        source="user correction",
    )
    assert corrected.kind == "preference"
    assert corrected.supersedes == first.id
    assert first in store.entries()  # correction history remains inspectable
    assert corrected in store.active_entries()
    assert first not in store.active_entries()


# ---------------------------------------------------------------------------
# Memory: looking things up
# ---------------------------------------------------------------------------


def _seeded(tmp_path) -> M.MemoryStore:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.add(
        "the Schwarzschild horizon is a null surface at r = 2M",
        tags=("penrose", "horizon"),
        source="aria/geometry/causal.py",
        confidence="verified",
    )
    store.add(
        "the Schwarzschild singularity at r = 0 is spacelike",
        tags=("penrose", "singularity"),
        source="aria/geometry/causal.py",
        confidence="verified",
    )
    store.add(
        "the Resonance Engine used a solar-system ontology",
        tags=("resonance",),
        source="Drive: inference_log.md",
        confidence="stated",
    )
    return store


def test_find_returns_the_relevant_entry_first(tmp_path) -> None:
    store = _seeded(tmp_path)
    hits = store.find("where is the horizon", k=2)
    assert "horizon" in hits[0].text


def test_find_returns_nothing_for_an_unrelated_query(tmp_path) -> None:
    store = _seeded(tmp_path)
    assert store.find("olive oil prices") == []


def test_context_is_ready_to_put_in_front_of_a_model(tmp_path) -> None:
    store = _seeded(tmp_path)
    block = store.context("penrose singularity")
    assert block.startswith("What ARIA remembers")
    assert "verified" in block
    assert "aria/geometry/causal.py" in block
    assert "unverified" in block  # the warning about confidence is in the block


def test_context_respects_the_character_budget(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    for index in range(40):
        store.add(f"fact number {index} about triangles and their angles", tags=("triangle",))
    block = store.context("triangles angles", k=40, max_chars=600)
    assert len(block.splitlines()) < 12
    assert len(block) <= 600


def test_memory_correction_is_append_only_and_retrieves_the_replacement(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    original = store.add(
        "a cube has six faces",
        tags=("geometry", "cube"),
        source="old notebook",
        confidence="stated",
    )
    correction = store.correct(
        original.id,
        "a cube has six faces and eight vertices",
        source="aria/geometry/shapes.py",
        confidence="verified",
    )

    assert correction.kind == "correction"
    assert correction.supersedes == original.id
    assert [entry.id for entry in store.entries()] == [original.id, correction.id]
    assert [entry.id for entry in store.active_entries()] == [correction.id]
    hits = store.find("cube six faces")
    assert hits and hits[0].id == correction.id
    assert "supersedes #1" in store.render_markdown()


def test_memory_correction_must_target_an_active_existing_entry(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    original = store.add("the answer is six", source="user", confidence="stated")
    replacement = store.correct(original.id, "the answer is seven")
    with pytest.raises(ValueError, match="already superseded"):
        store.correct(original.id, "the answer is eight")
    with pytest.raises(KeyError, match="unknown memory"):
        store.correct(999, "a new answer")
    assert replacement.id != original.id


def test_memory_context_manifest_can_recover_omitted_whole_entries(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    for index in range(12):
        store.add(
            f"triangle theorem {index:02d} says all interior angles add to one hundred eighty degrees",
            tags=("triangle",),
            source=f"source-{index}",
            confidence="tested",
        )
    pack = store.context_pack("triangle theorem", k=12, max_chars=600)
    assert pack.reduced
    assert pack.omitted
    assert len(pack.text) <= 600
    serialized = pack.as_dict()
    assert serialized["recoverable"]
    assert any(entry["id"] in serialized["omitted_ids"] for entry in serialized["source_entries"])
    restored = pack.rebuild(10_000)
    assert not restored.reduced
    assert len(restored.included) == 12
    assert {entry.id for entry in restored.included} == {entry.id for entry in pack.source_entries}


def test_memory_context_budget_error_keeps_rebuild_sources(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.add("one triangle fact", tags=("triangle",), source="test")
    with pytest.raises(M.MemoryContextBudgetError) as raised:
        store.context_pack("triangle", max_chars=1)
    error = raised.value
    assert error.as_dict()["recoverable"] is True
    assert error.as_dict()["source_entries"][0]["text"] == "one triangle fact"
    assert error.rebuild(1000).text


# ---------------------------------------------------------------------------
# Memory: honesty
# ---------------------------------------------------------------------------


def test_audit_flags_an_entry_with_no_source(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.add("a cube has six faces", source="aria/geometry/shapes.py", confidence="verified")
    store.add("a dodecahedron probably tiles the plane", confidence="stated")
    flagged = store.audit()
    assert len(flagged) == 1
    assert "tiles the plane" in flagged[0].text


def test_digest_counts_by_kind_and_confidence(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.add("one", kind="fact", source="x", confidence="verified", tags=["geometry"])
    store.add("two", kind="procedure", source="y", confidence="tested", tags=["geometry"])
    digest = store.digest()
    assert digest["entries"] == 2
    assert digest["by_kind"]["fact"] == 1
    assert digest["by_kind"]["procedure"] == 1
    assert digest["tags"]["geometry"] == 2
    assert digest["unsourced"] == []


def test_markdown_rendering_has_sections(tmp_path) -> None:
    store = _seeded(tmp_path)
    text = store.render_markdown()
    assert text.startswith("# What ARIA remembers")
    assert "## fact" in text
    assert "null surface" in text


# ---------------------------------------------------------------------------
# Memory: fed by the ledger
# ---------------------------------------------------------------------------


def _worked_ledger(tmp_path) -> L.Ledger:
    book = L.Ledger(tmp_path / "ledger.jsonl")
    for prompt in ("solve 9x - 3 = 24", "solve 2x + 7 = 15"):
        book.record(prompt, L.contract_of(prompt))
    book.record("solve 9/x - 3 = 24", "goal: solve\nvar: x\nexpr: 9/x - 3 = 24")
    return book


def test_proven_work_becomes_a_procedure(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.import_ledger(_worked_ledger(tmp_path))
    procedures = [entry for entry in store.entries() if entry.kind == "procedure"]
    assert len(procedures) == 2
    assert all(entry.confidence == "verified" for entry in procedures)
    assert all(entry.source.startswith("ledger#") for entry in procedures)


def test_a_broken_attempt_becomes_a_correction(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    store.import_ledger(_worked_ledger(tmp_path))
    corrections = [entry for entry in store.entries() if entry.kind == "correction"]
    assert len(corrections) == 1
    assert "broke at" in corrections[0].text
    assert corrections[0].confidence == "tested"


def test_importing_twice_does_not_duplicate(tmp_path) -> None:
    store = M.MemoryStore(tmp_path / "memory.jsonl")
    book = _worked_ledger(tmp_path)
    store.import_ledger(book)
    before = len(store.entries())
    store.import_ledger(book)
    assert len(store.entries()) == before


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_a_cube_is_a_cube() -> None:
    cube = shapes.describe("cube", size=2.0)
    assert (cube["vertices"], cube["edges"], cube["faces"]) == (8, 12, 6)
    assert cube["euler_characteristic"] == 2
    assert cube["surface_area"] == pytest.approx(24.0)
    assert cube["volume"] == pytest.approx(8.0)
    assert cube["dihedral_angle_deg"] == pytest.approx(90.0)
    assert cube["dual"] == "octahedron"
    assert cube["symmetry_group_order"] == 48


@pytest.mark.parametrize(
    "name, volume",
    [
        ("tetrahedron", 1.0 / (6.0 * math.sqrt(2.0))),
        ("octahedron", math.sqrt(2.0) / 3.0),
        ("dodecahedron", (15.0 + 7.0 * math.sqrt(5.0)) / 4.0),
        ("icosahedron", 5.0 * (3.0 + math.sqrt(5.0)) / 12.0),
    ],
)
def test_platonic_volumes_for_unit_edge(name: str, volume: float) -> None:
    # properties() rounds to six decimal places, so compare at that resolution
    assert shapes.describe(name)["volume"] == pytest.approx(volume, abs=1e-6)


def test_every_platonic_solid_has_euler_characteristic_two() -> None:
    for solid in shapes.PLATONIC:
        assert solid.euler_characteristic == 2


def test_regular_polygon_properties() -> None:
    hexagon = shapes.describe("polygon", size=1.0, sides=6)
    assert hexagon["interior_angle_deg"] == pytest.approx(120.0)
    assert hexagon["area"] == pytest.approx(3.0 * math.sqrt(3.0) / 2.0)
    assert hexagon["symmetry_order"] == 12


def test_curves() -> None:
    sphere = shapes.describe("sphere", size=2.0)
    assert sphere["surface_area"] == pytest.approx(16.0 * math.pi)
    assert sphere["volume"] == pytest.approx(32.0 * math.pi / 3.0)
    circle = shapes.describe("circle", size=3.0)
    assert circle["circumference"] == pytest.approx(6.0 * math.pi)


def test_an_unknown_shape_is_an_error() -> None:
    with pytest.raises(ValueError):
        shapes.describe("dodecaplex")
    with pytest.raises(ValueError):
        shapes.describe("polygon", sides=2)


def test_geometry_identities_all_pass() -> None:
    results = shapes.checks()
    assert results
    assert all(ok for _, ok in results), [claim for claim, ok in results if not ok]


# ---------------------------------------------------------------------------
# Penrose diagrams
# ---------------------------------------------------------------------------


def test_null_rays_are_at_forty_five_degrees() -> None:
    # right-moving: t = x + c  (u fixed)
    a = causal.minkowski_to_penrose(0.0, 0.0)
    b = causal.minkowski_to_penrose(2.0, 2.0)
    assert b[0] - a[0] == pytest.approx(b[1] - a[1])

    # left-moving: t = -x + c  (v fixed)
    c = causal.minkowski_to_penrose(0.0, 0.0)
    d = causal.minkowski_to_penrose(2.0, -2.0)
    assert d[0] - c[0] == pytest.approx(-(d[1] - c[1]))


def test_the_diamond_is_bounded() -> None:
    for t, x in ((0.0, 0.0), (100.0, -30.0), (-1e6, 5.0)):
        big_t, big_x = causal.minkowski_to_penrose(t, x)
        assert abs(big_t) < math.pi
        assert abs(big_x) < math.pi


def test_the_tortoise_coordinate_diverges_at_the_horizon() -> None:
    assert causal.tortoise(2.0 + 1e-9) < -10.0
    assert causal.tortoise(4.0) > 0.0
    assert causal.tortoise(10.0) > causal.tortoise(4.0)


def test_kruskal_identity_holds() -> None:
    for r in (2.5, 3.0, 7.0):
        big_u, big_v = causal.kruskal(r, 0.5)
        assert big_u * big_v == pytest.approx((1.0 - r / 2.0) * math.exp(r / 2.0), rel=1e-9)


def test_regions_are_labelled_by_signs() -> None:
    assert causal.region_of(-1.0, 1.0).startswith("I")
    assert causal.region_of(1.0, 1.0).startswith("II")
    assert causal.region_of(1.0, -1.0).startswith("III")
    assert causal.region_of(-1.0, -1.0).startswith("IV")
    assert "horizon" in causal.region_of(0.0, 1.0)


def test_the_singularity_is_spacelike_and_the_horizon_is_null() -> None:
    data = causal.structure("schwarzschild")
    by_name = {element["name"]: element for element in data["elements"]}
    singularity = by_name["r = 0"]
    assert singularity["kind"] == "singularity"
    assert "spacelike" in singularity["means"]
    assert any(element["kind"] == "horizon" for element in data["elements"])


def test_frw_conformal_time() -> None:
    assert causal.frw_conformal_time(1e-12, 2.0) < 1e-5
    assert causal.frw_conformal_time(4.0, 2.0) == pytest.approx(2.0)   # matter era
    assert causal.frw_conformal_time(4.0, 1.0) == pytest.approx(4.0)   # radiation era


def test_an_unknown_diagram_is_an_error() -> None:
    with pytest.raises(ValueError):
        causal.structure("kerr-newman")


def test_causal_identities_all_pass() -> None:
    results = causal.checks()
    assert results
    assert all(ok for _, ok in results), [claim for claim, ok in results if not ok]


def test_diagram_text_names_the_boundaries() -> None:
    text = causal.diagram_text("minkowski")
    for marker in ("i⁻", "i⁺", "i⁰", "ℐ⁻", "ℐ⁺"):
        assert marker in text
