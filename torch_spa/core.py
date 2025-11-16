"""
Solar Position Algorithm

This module provides a Python implementation of the Solar Position Algorithm (SPA).
The SPA is a widely used method for calculating the position of the sun in the sky.


"""

from __future__ import annotations

__all__ = ["solar_position", "solar_zenith", "solar_declination", "SolarPosition"]

from typing import TYPE_CHECKING, Annotated, Iterator, Literal, overload

import torch
from torch_time import julian_day_time  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from torch_time.core import DatetimeLike  # type: ignore[import-untyped]


def _atmospheric_refraction(solar_elevation_angle: torch.Tensor) -> torch.Tensor:
    shape = solar_elevation_angle.shape
    elevation = solar_elevation_angle.ravel()
    out = torch.zeros_like(elevation)

    dawn_dusk = (elevation > -0.575) & (elevation <= 5)  # below the horizon
    day_time = (elevation > 5) & (elevation <= 85)

    # dawn and dusk
    e = elevation[dawn_dusk]
    out[dawn_dusk] = 1735 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))

    # day
    e = elevation[day_time].deg2rad().tan()
    out[day_time] = 58.1 / e - 0.07 / (e**3) + 0.000086 / (e**5)

    return out.reshape(shape) / 3600


def _obliquity_of_the_ecliptic(JC: torch.Tensor) -> torch.Tensor:
    out = 23 + (26 + (21.448 - JC * (46.815 + JC * (0.00059 - JC * 0.001813))) / 60) / 60
    out += 0.00256 * ((125.04 - 1934.136 * JC).deg2rad()).cos()
    return out.deg2rad()


# =============================================================================================== #
# -- main -- #
# =============================================================================================== #
class SolarPosition:
    __slots__ = ("_which", "_azimuth", "_elevation")
    _which: Literal["degrees", "radians"]
    _azimuth: torch.Tensor
    _elevation: torch.Tensor

    def __init__(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        /,
        *,
        which: Literal["degrees", "radians"] = "degrees",
    ) -> None:
        super().__init__()
        self._azimuth = azimuth
        self._elevation = elevation
        self._which = which

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self._azimuth, self._elevation))

    def __getitem__(self, index: int | slice | tuple[int | slice, ...]) -> SolarPosition:
        a, e = self._azimuth[index], self._elevation[index]
        return SolarPosition(a, e, which=self._which)

    def __len__(self) -> int:
        return len(self._azimuth)

    @property
    def azimuth(self) -> torch.Tensor:
        return self._azimuth

    @property
    def elevation(self) -> torch.Tensor:
        return self._elevation

    @property
    def which(self) -> Literal["degrees", "radians"]:
        return self._which

    def squeeze(self) -> SolarPosition:
        a, e = self
        return SolarPosition(a.squeeze(), e.squeeze(), which=self._which)

    def deg2rad(self) -> SolarPosition:
        a, e = self
        if self._which == "degrees":
            a = a.deg2rad()
            e = e.deg2rad()
        return SolarPosition(a, e, which="radians")

    def rad2deg(self) -> SolarPosition:
        a, e = self
        if self._which == "radians":
            a = a.rad2deg()
            e = e.rad2deg()
        return SolarPosition(a, e, which="degrees")

    def norm(self) -> SolarPosition:
        a, e = self.deg2rad()
        a = (a - torch.pi) / torch.pi
        e = e / (torch.pi / 2)

        return SolarPosition(a, e)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        azimuth = self._azimuth
        elevation = self._elevation
        which = self._which
        s += f"({azimuth.shape=}, {elevation.shape=}, {which=})"
        return s


@overload
def spa(
    datetime: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    /,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | int | None = ...,
) -> SolarPosition: ...
@overload
def spa(
    datetime: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    /,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | int | None = ...,
    return_zenith: Literal[True] = ...,
) -> torch.Tensor: ...
@overload
def spa(
    datetime: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    /,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | int | None = ...,
    return_declination: Literal[True] = ...,
) -> torch.Tensor: ...


def spa(
    datetime: Annotated[DatetimeLike, "datetime64"],
    latitude: Annotated[ArrayLike, "degrees"],
    longitude: Annotated[ArrayLike, "degrees"],
    /,
    dtype: torch.dtype | None = None,
    device: torch.device | str | int | None = None,
    **kw: bool,
) -> SolarPosition | torch.Tensor:
    # NOTE: we make an assumption that we are working with a 3D array to support vectorization
    # of the time, latitude, and longitude (T, Y, X).

    # 1. Handle the input arguments
    # 1.1) handle the datetime argument
    # TODO: I need to update the torch_time package to support the dtype and device arguments
    ymd, hms = julian_day_time(datetime)
    ymd, hms = ymd.to(dtype=dtype, device=device), hms.to(dtype=dtype, device=device)
    JD = ymd + hms
    JD = JD.reshape(-1, 1, 1)
    JC = (JD - 2451545) / 36525

    # 1.2) handle the latitude and longitude arguments
    if not isinstance(latitude, torch.Tensor):
        latitude = torch.as_tensor(latitude, dtype=dtype, device=device)
    if not isinstance(longitude, torch.Tensor):
        longitude = torch.as_tensor(longitude, dtype=dtype, device=device)
    lat, lon = torch.atleast_1d(latitude, longitude)
    if lat.shape != lon.shape:
        lat, lon = torch.meshgrid(lat, lon, indexing="ij")

    # 1.3) input arguments are now handled
    phi = lat.deg2rad()  #  ϕ

    # 2. Calculate the solar position
    # 2.1) Geom Mean Anom Sun (rad)
    anomaly = (357.52911 + JC * (35999.05029 - 0.0001537 * JC)).deg2rad()

    # 2.2) Sun Eq of Ctr (deg)
    equation_of_center = (
        anomaly.sin() * (1.914602 - JC * (0.004817 + 0.000014 * JC))
        + (anomaly * 2).sin() * (0.019993 - 0.000101 * JC)
        + (anomaly * 3).sin() * 0.000289
    )

    # 2.3) Sun True Long (deg)
    lon_u = (280.46646 + JC * (36000.76983 + JC * 0.0003032)) % 360  # mean longitude
    lon_true = lon_u + equation_of_center
    lambda_u = lon_u.deg2rad()  # λ

    # 2.4) Sun App Long (rad)
    lambda_apparent = (
        lon_true + 0.005693 * (lon_true * 2).sin() - 0.004783 * (lon_true * 3).sin()
    ).deg2rad()

    obliquity = _obliquity_of_the_ecliptic(JC)

    # 2.5) Sun Decl (rad)
    declination = (obliquity.sin() * lambda_apparent.sin()).arcsin()
    if kw.get("return_declination") is True:
        return declination.rad2deg()

    # 2.6) Eq of Time (min)
    x = 0.016708634 - JC * (0.000042037 + 0.0000001267 * JC)  # orbital eccentricity
    y = (obliquity / 2).tan() * (obliquity / 2).tan()
    equation_of_time = (
        y * (2 * lambda_u).sin()
        - 2 * x * anomaly.sin()
        + 4 * x * y * anomaly.sin() * (2 * lambda_u).cos()
        - 0.5 * y * y * (4 * lambda_u).sin()
        - 1.25 * x * x * (anomaly * 2).sin()
    ).rad2deg() * 4

    hour_angle = ((hms - 0.5).reshape(-1, 1, 1) * 1440 + equation_of_time + 4 * lon) % 1440
    hour_angle /= 4
    hour_angle[m := hour_angle < 0] += 180
    hour_angle[~m] -= 180
    hour_angle.deg2rad_()

    # 2.7) Solar Zenith
    zenith = (
        phi.sin() * declination.sin() + phi.cos() * declination.cos() * hour_angle.cos()
    ).arccos()  # radians
    if kw.get("return_zenith") is True:
        return zenith.rad2deg()

    # 2.8) Solar Elevation Angle
    elevation = 90 - zenith.rad2deg()
    elevation += _atmospheric_refraction(elevation)

    # 2.9) Solar Azimuth
    azimuth = (
        (
            (((phi.sin() * zenith.cos()) - declination.sin()) / (phi.cos() * zenith.sin())).clip(
                -1, 1
            )
        )
        .arccos()
        .rad2deg()
    )
    m = hour_angle <= 0
    azimuth[~m] += 180
    azimuth[m] = 540 - azimuth[m]
    azimuth %= 360

    return SolarPosition(azimuth, elevation, which="degrees")


def solar_position(
    datetime: Annotated[ArrayLike, "datetime64"],
    latitude: Annotated[ArrayLike, "degrees"],
    longitude: Annotated[ArrayLike, "degrees"],
) -> SolarPosition:
    return spa(datetime, latitude, longitude)


def solar_zenith(
    datetime: Annotated[ArrayLike, "datetime64"],
    latitude: Annotated[ArrayLike, "degrees"],
    longitude: Annotated[ArrayLike, "degrees"],
) -> Annotated[torch.Tensor, "degrees[0-180]"]:
    return spa(datetime, latitude, longitude, return_zenith=True)


def solar_declination(
    datetime: Annotated[ArrayLike, "datetime64"],
    latitude: Annotated[ArrayLike, "degrees"],
    longitude: Annotated[ArrayLike, "degrees"],
) -> torch.Tensor:
    return spa(datetime, latitude, longitude, return_declination=True)
