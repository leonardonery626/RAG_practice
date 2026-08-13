from pathlib import Path
import shutil

import nox
from nox import Session

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
nox.needs_version = ">=2025.5.1"
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = nox.project.python_versions(
    nox.project.load_toml("pyproject.toml"),
    max_version="3.12",
)

LINT_TARGETS = ("src", "noxfile.py")
TYPE_TARGETS = ("src",)
DOCKERFILES = tuple(Path(".").glob("**/Dockerfile*"))

CI_REPORTS_DIR = Path(".reports")
LINTER_DIR = CI_REPORTS_DIR / "linter"
PYTEST_DIR = CI_REPORTS_DIR / "pytest"
COVERAGE_DIR = CI_REPORTS_DIR / "coverage"


# -----------------------------------------------------------------------------
# Project bootstrap and maintenance
# -----------------------------------------------------------------------------
@nox.session(name="dev", python=False, venv_backend="none")
def dev(session: Session) -> None:
    """Initialize the local environment with all dependency groups."""
    session.run(
        "uv",
        "sync",
        "--all-extras",
        "--all-groups",
        f"--python={PYTHON_VERSIONS[-1]}",
        external=True,
    )


@nox.session(venv_backend="none")
def lock(session: Session) -> None:
    """Upgrade the lock file for the project."""
    session.run("uv", "lock", "--upgrade", external=True)


# -----------------------------------------------------------------------------
# Project scripts and workflows
# -----------------------------------------------------------------------------
@nox.session(name="etl_pipeline", python=PYTHON_VERSIONS[-1])
def etl_pipeline(session: Session) -> None:
    """Run chunking then embeddings as a single pipeline."""
    session.run("uv", "sync", "--locked", external=True)
    session.run("uv", "run", "chunk_pdf", external=True)
    session.run("uv", "run", "build_embeddings", external=True)


# -----------------------------------------------------------------------------
# CI automation sessions
# -----------------------------------------------------------------------------
@nox.session(python=PYTHON_VERSIONS[-1])
def format(session: Session) -> None:
    """Check import sorting and formatting with Ruff."""
    session.run("uv", "sync", "--active", "--locked", "--only-group=format", external=True)
    session.run("uv", "run", "ruff", "check", ".", "--select", "I", external=True)
    session.run("uv", "run", "ruff", "format", ".", "--check", external=True)


@nox.session(python=PYTHON_VERSIONS[-1])
def lint(session: Session) -> None:
    """Run lint checks and generate CI-friendly reports."""
    session.run("uv", "sync", "--active", "--locked", "--only-group=lint", external=True)

    shutil.rmtree(LINTER_DIR, ignore_errors=True)
    LINTER_DIR.mkdir(parents=True, exist_ok=True)

    with (LINTER_DIR / "ruff.xml").open(mode="w", encoding="utf-8") as report:
        session.run(
            "uv",
            "run",
            "ruff",
            "check",
            *LINT_TARGETS,
            "--output-format=junit",
            "--exit-zero",
            stdout=report,
            stderr=None,
            external=True,
        )

    session.run(
        "uv",
        "run",
        "junit2html",
        str(LINTER_DIR / "ruff.xml"),
        "--report-matrix",
        str(LINTER_DIR / "ruff.html"),
        external=True,
    )
    session.run("uv", "run", "ruff", "check", *LINT_TARGETS, external=True)


@nox.session(python=PYTHON_VERSIONS[-1])
def typing(session: Session) -> None:
    """Run static type checks."""
    session.run("uv", "sync", "--active", "--locked", "--only-group=typing", external=True)
    session.run("uv", "run", "mypy", *TYPE_TARGETS, external=True)


@nox.session(python=PYTHON_VERSIONS)
def test(session: Session) -> None:
    """Run tests with coverage and emit CI reports."""
    session.run("uv", "sync", "--active", "--locked", "--only-group=test", external=True)

    pytest_session_dir = PYTEST_DIR / f"python-{session.python}"
    coverage_session_dir = COVERAGE_DIR / f"python-{session.python}"
    shutil.rmtree(pytest_session_dir, ignore_errors=True)
    shutil.rmtree(coverage_session_dir, ignore_errors=True)
    pytest_session_dir.mkdir(parents=True, exist_ok=True)
    coverage_session_dir.mkdir(parents=True, exist_ok=True)

    session.run(
        "uv",
        "run",
        "coverage",
        "run",
        "--source",
        "src",
        "-m",
        "pytest",
        "src/tests",
        "--junitxml",
        f"{pytest_session_dir}/pytest.xml",
        "--html",
        f"{pytest_session_dir}/pytest.html",
        external=True,
    )
    session.run(
        "uv",
        "run",
        "coverage",
        "html",
        "-d",
        str(coverage_session_dir),
        success_codes=[0, 2],
        external=True,
    )
    session.run(
        "uv",
        "run",
        "coverage",
        "xml",
        "-o",
        str(coverage_session_dir / "coverage.xml"),
        success_codes=[0, 2],
        external=True,
    )
    session.run("uv", "run", "coverage", "report", external=True)
