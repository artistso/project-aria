"""Dimensional analysis and the physical law library."""

import pytest

from aria.science.laws import LAW_LIBRARY, check_law, derive_dimension, list_laws
from aria.science.units import DimensionError, check_equation, parse_unit


def test_base_and_derived_units():
    assert str(parse_unit("kg*m/s^2")) == "kg m / s^2"
    assert parse_unit("m/s^2") == parse_unit("m*s^-2")
    assert parse_unit("N") == parse_unit("kg*m/s^2")
    assert parse_unit("J") == parse_unit("N*m")


def test_dimension_algebra():
    force = parse_unit("kg*m/s^2")
    distance = parse_unit("m")
    assert force * distance == parse_unit("kg*m^2/s^2")
    assert (force ** 2) == parse_unit("kg^2*m^2/s^4")
    assert (force / force).is_dimensionless


def test_unknown_unit_is_rejected():
    with pytest.raises(DimensionError):
        parse_unit("furlong/fortnight")


@pytest.mark.parametrize("name", sorted(LAW_LIBRARY))
def test_library_laws_are_dimensionally_consistent(name):
    report = check_law(name)
    assert report.consistent, report.to_text()


def test_broken_law_is_inconsistent():
    report = check_equation("F = m a^2", {"F": "kg*m/s^2", "m": "kg", "a": "m/s^2"})
    assert not report.consistent


def test_inhomogeneous_sum_is_rejected():
    report = check_equation("x = y + z", {"x": "m", "y": "m", "z": "s"})
    assert not report.consistent
    assert "inhomogeneous" in report.detail.lower() or "cannot add" in report.detail.lower()


def test_dimension_inversion():
    # F = m a  =>  [a] = [F]/[m] = m/s^2
    derived = derive_dimension("F = m a", {"F": "kg*m/s^2", "m": "kg"}, "a")
    assert derived == parse_unit("m/s^2")


def test_dimension_inversion_for_newton_constant():
    derived = derive_dimension("F = G m1 m2 / r^2", {"F": "kg*m/s^2", "m1": "kg", "m2": "kg", "r": "m"}, "G")
    assert derived == parse_unit("m^3/(kg*s^2)")


def test_law_listing():
    names = [law.name for law in list_laws()]
    assert "newton_second" in names
    assert len(names) == len(set(names))
