from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]


def _safe_get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_rows(data: Any) -> Iterable:
    return data if data is not None else []


def parse_planets(rows: Any) -> list[PlanetState]:
    return [
        PlanetState(
            id=int(r[0]),
            owner=int(r[1]),
            x=float(r[2]),
            y=float(r[3]),
            radius=float(r[4]),
            ships=int(r[5]),
            production=int(r[6]),
        )
        for r in _as_rows(rows)
    ]


def parse_fleets(rows: Any) -> list[FleetState]:
    return [
        FleetState(
            id=int(r[0]),
            owner=int(r[1]),
            x=float(r[2]),
            y=float(r[3]),
            angle=float(r[4]),
            from_planet_id=int(r[5]),
            ships=int(r[6]),
        )
        for r in _as_rows(rows)
    ]


def parse_observation(observation: Any) -> GameState:
    planets_raw = _safe_get(observation, "planets", [])
    fleets_raw = _safe_get(observation, "fleets", [])

    return GameState(
        step=int(_safe_get(observation, "step", 0)),
        player=int(_safe_get(observation, "player", 0)),
        planets=parse_planets(planets_raw),
        fleets=parse_fleets(fleets_raw),
    )