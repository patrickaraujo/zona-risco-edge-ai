"""Testes da lógica de zonas — não dependem do modelo."""

import pytest

from app.detector import Detection
from app.zones import RiskLevel, ZoneConfigError, anchor_point, assess, parse_zones

W, H = 1000, 1000
ZONES = parse_zones([
    {"name": "amarela", "level": "amarela", "polygon": [[0.2, 0.5], [1, 0.5], [1, 1], [0.2, 1]]},
    {"name": "vermelha", "level": "vermelha", "polygon": [[0.6, 0.7], [1, 0.7], [1, 1], [0.6, 1]]},
])


def person(x1, y1, x2, y2):
    return Detection(0, "person", 0.9, (x1, y1, x2, y2))


def test_anchor_is_bottom_center():
    assert anchor_point(person(100, 100, 200, 400)) == (150, 400)


@pytest.mark.parametrize("box, expected", [
    ((0, 100, 100, 400), RiskLevel.SEGURO),      # pés fora de tudo
    ((300, 200, 400, 600), RiskLevel.ATENCAO),   # pés só na amarela
    ((700, 300, 800, 900), RiskLevel.PERIGO),    # pés na vermelha (e na amarela)
])
def test_risk_levels(box, expected):
    [a] = assess([person(*box)], ZONES, W, H)
    assert a.risk == expected


def test_red_has_precedence_over_yellow():
    [a] = assess([person(700, 300, 800, 900)], ZONES, W, H)
    assert set(a.zones) == {"amarela", "vermelha"}
    assert a.risk == RiskLevel.PERIGO


def test_torso_inside_but_feet_outside_is_safe():
    # Caixa invade a zona visualmente, mas os pés (y2=450) estão acima dela
    [a] = assess([person(700, 100, 900, 450)], ZONES, W, H)
    assert a.risk == RiskLevel.SEGURO


def test_zones_are_resolution_independent():
    [a] = assess([person(1400, 600, 1600, 1800)], ZONES, 2000, 2000)
    assert a.risk == RiskLevel.PERIGO


@pytest.mark.parametrize("bad", [
    [],
    [{"name": "x", "level": "azul", "polygon": [[0, 0], [1, 0], [1, 1]]}],
    [{"name": "x", "level": "amarela", "polygon": [[0, 0], [1, 0]]}],
    [{"name": "x", "level": "amarela", "polygon": [[0, 0], [2, 0], [1, 1]]}],
])
def test_invalid_config_raises(bad):
    with pytest.raises(ZoneConfigError):
        parse_zones(bad)
