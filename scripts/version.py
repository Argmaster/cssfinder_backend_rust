"""Script providing version info and allowing for version manipulation."""
import re
from pathlib import Path

import click
import tomlkit

PYPROJECT_TOML = Path("pyproject.toml")
CARGO_TOML = Path("Cargo.toml")
README_MD = Path("README.md")


@click.group()
def main() -> None:
    pass


@main.command()
def get() -> None:
    version = read_version()
    click.echo(version)


def read_version() -> str:
    pyproject = tomlkit.loads(PYPROJECT_TOML.read_text())
    cargo = tomlkit.loads(CARGO_TOML.read_text())

    poetry_package_version = pyproject["tool"]["poetry"]["version"]
    project_package_version = pyproject["project"]["version"]

    if poetry_package_version != project_package_version:
        msg = (
            f"Version mismatch within pyproject versions {poetry_package_version} != "
            f"{project_package_version}"
        )
        raise ValueError(msg)

    cargo_package_version = cargo["package"]["version"]

    if cargo_package_version != project_package_version:
        msg = (
            f"Version mismatch with cargo package version {cargo_package_version} != "
            f"{project_package_version}"
        )
        raise ValueError(msg)

    return cargo_package_version


@main.command()
@click.argument("version")
def set(version: str) -> None:
    set_version(version)


@main.command()
@click.argument("offset", type=int)
def set_dev(offset: int) -> None:
    v = read_version()
    find = re.match(
        r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)", v
    )
    if find is None:
        raise ValueError(f"Not matched {v!r}")
    major = find.groupdict()["major"]
    minor = find.groupdict()["minor"]
    patch = find.groupdict()["patch"]

    new_version = "{major}.{minor}.{patch}-dev.{offset}".format(
        major=major, minor=minor, patch=int(patch) + 1, offset=offset
    )
    set_version(new_version)


def set_version(version: str) -> None:
    pyproject = tomlkit.loads(PYPROJECT_TOML.read_text())
    cargo = tomlkit.loads(CARGO_TOML.read_text())

    pyproject["tool"]["poetry"]["version"] = version
    pyproject["project"]["version"] = version
    cargo["package"]["version"] = version

    PYPROJECT_TOML.write_text(tomlkit.dumps(pyproject))
    CARGO_TOML.write_text(tomlkit.dumps(cargo))


if __name__ == "__main__":
    main()
