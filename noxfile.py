import nox
from nox import Session

MODULE_NAME = "pilot_rag"

nox.needs_version = (
    ">=2025.5.1"  # No upper bound to keep compliance with future versions
)
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True
nox.options.default_venv_backend = (
    "uv"  # Use uv backend by default for all virtual environments
)

# After having added the PDF, you can run the all workflows just by using the "nox" command - local venv sync + chunking + embeddings generation
nox.options.sessions = ["dev", "chunking", "embeddings"]

PYTHON_VERSIONS = nox.project.python_versions(
    nox.project.load_toml("pyproject.toml"),
    max_version="3.12",
)


# Defing an auxiliary function to synchronize project dependencies and env before running any session.
def _sync_project(session: Session) -> None:
    session.run(
        "uv",
        "sync",
        "--all-extras",
        "--all-groups",
        f"--python={PYTHON_VERSIONS[-1]}",
        external=True,
    )


# Define an auxiliary function to run a project script with optional default arguments.
def _run_project_script(
    session: Session, script_name: str, default_args: list[str] | None = None
) -> None:
    _sync_project(session)
    args = list(session.posargs) if session.posargs else (default_args or [])
    session.run("uv", "run", script_name, *args, external=True)


#### INITIALIZE THE LOCAL DEVELOPMENT ENVIRONMENT ####
@nox.session(name="dev", python=False, venv_backend="none")
def dev(session: Session) -> None:
    """Initialize the local development environment with shared tooling."""
    _sync_project(session)


#### SCRIPTS FOR RUNNING THE PROJECT WORKFLOWS ####
@nox.session(name="chunking", python=PYTHON_VERSIONS[-1])
def chunking(session: Session) -> None:
    """Run the PDF chunking workflow from the project CLI script."""
    _run_project_script(session, "chunk_pdf")


@nox.session(name="embeddings", python=PYTHON_VERSIONS[-1])
def embeddings(session: Session) -> None:
    """Run the embedding workflow from the project CLI script."""
    session.run(
        "uv",
        "sync",
        "--all-extras",
        "--all-groups",
        f"--python={PYTHON_VERSIONS[-1]}",
        external=True,
    )
    session.run("uv", "run", "build_embeddings")
