import click
import tomlkit
import subprocess
import re
import logging
from pathlib import Path
from typing import Tuple

PYPROJECT_TOML = "pyproject.toml"
CARGO_TOML = "Cargo.toml"
README_MD = "README.md"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def read_version(file_path: str) -> str:
    with Path(file_path).open("r") as f:
        toml_content = tomlkit.parse(f.read())
        return toml_content["tool"]["poetry"]["version"]


def write_version(file_path: str, version: str) -> None:
    with Path(file_path).open("r") as f:
        toml_content = tomlkit.parse(f.read())
    with Path(file_path).open("w") as f:
        for item in toml_content.as_string().splitlines():
            if "version" in item:
                f.write(f'version = "{version}"\n')
            else:
                f.write(f"{item}\n")


def set_version(version: str) -> None:
    # Validate version string
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError("Invalid version string. Must be in SemVer format (e.g. '1.2.3').")

    # Check version in one of the config files
    py_version = read_version(PYPROJECT_TOML)
    if py_version != version:
        raise ValueError(f"Version string in {PYPROJECT_TOML} does not match provided version.")

    # Update version in config files
    write_version(PYPROJECT_TOML, version)
    write_version(CARGO_TOML, version)

    # Update version in README.md
    readme_path = Path(README_MD)
    readme_content = readme_path.read_text()
    readme_content = readme_content.replace(py_version, version)
    readme_path.write_text(readme_content)

    logger.info(f"Version set to: {version}")


@click.group()
def version() -> None:
    pass


@click.command()
def get() -> None:
    py_version = read_version(PYPROJECT_TOML)
    cargo_version = read_version(CARGO_TOML)
    if py_version != cargo_version:
        raise ValueError("Versions in config files do not match.")
    click.echo(f"Current version: {py_version}")


@click.command()
@click.argument("version")
def set(version: str) -> None:
    set_version(version)


version.add_command(get)
version.add_command(set)

if __name__ == "__main__":
    version()
