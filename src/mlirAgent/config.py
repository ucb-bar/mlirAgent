import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _auto_detect_project_root():
    """Walk up from this file to find the mlirAgent project root."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _auto_detect_merlin_root():
    """Resolve the Merlin mono-repo root from MERLIN_ROOT env var or by walking up."""
    explicit = os.getenv("MERLIN_ROOT")
    if explicit:
        return explicit
    # Default: two levels above project root  (merlin/projects/mlirAgent -> merlin)
    return os.path.dirname(os.path.dirname(_auto_detect_project_root()))


_merlin_root = _auto_detect_merlin_root()
_default_build_dir = os.path.join(
    _merlin_root, "build", "vanilla", "host", "debug", "iree-spacemit-3.10.0.dev"
)


class Config:
    # --- Source Paths ---
    _MERLIN_ROOT = _merlin_root
    IREE_SRC_PATH = os.getenv(
        "IREE_SRC_PATH", os.path.join(_MERLIN_ROOT, "third_party", "iree_bar")
    )
    LLVM_SRC_PATH = os.getenv(
        "LLVM_SRC_PATH",
        os.path.join(IREE_SRC_PATH, "third_party", "llvm-project"),
    )

    # --- Build Paths ---
    BUILD_DIR = os.getenv("BUILD_DIR", _default_build_dir)
    INSTALL_DIR = os.path.join(BUILD_DIR, "install")

    # --- Binaries ---
    BUILD_LLVM_DIR = os.getenv(
        "BUILD_LLVM_DIR", os.path.join(BUILD_DIR, "llvm-project")
    )
    LLVM_LIT_PATH = os.getenv(
        "LLVM_LIT_PATH", os.path.join(BUILD_LLVM_DIR, "bin", "lit")
    )
    FILECHECK_PATH = os.getenv(
        "FILECHECK_PATH", os.path.join(BUILD_LLVM_DIR, "bin", "FileCheck")
    )

    BUILD_TOOLS_DIR = os.getenv(
        "BUILD_TOOLS_DIR", os.path.join(BUILD_DIR, "tools")
    )
    IREE_COMPILE_PATH = os.getenv(
        "IREE_COMPILE_PATH", os.path.join(BUILD_TOOLS_DIR, "iree-compile")
    )

    # --- Evolve Harness ---
    EVOLVE_CONFIGS_DIR = os.getenv(
        "EVOLVE_CONFIGS_DIR",
        os.path.join(_auto_detect_project_root(), "configs"),
    )
    OPENEVOLVE_PATH = os.getenv(
        "OPENEVOLVE_PATH",
        os.path.join(_auto_detect_project_root(), "third_party", "openevolve"),
    )

    # --- Agent Data ---
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
    RECIPES_DIR = PROJECT_ROOT / "data" / "cookbook" / "LLVM_recipes"

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")

    # LanceDB
    LANCEDB_DIR = PROJECT_ROOT / "data" / "lancedb"

    # --- .env Variables ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    @classmethod
    def validate(cls):
        """Ensures the environment is set up correctly."""
        os.makedirs(cls.ARTIFACTS_DIR, exist_ok=True)

        if not os.path.exists(cls.IREE_SRC_PATH):
            print(
                f"WARNING: Source path not found at {cls.IREE_SRC_PATH}. "
                "'reconfigure=True' will fail."
            )

        ninja_file = os.path.join(cls.BUILD_DIR, "build.ninja")
        if not os.path.exists(ninja_file):
            print("WARNING: No 'build.ninja' found. Agent must run `reconfigure=True`.")


# Run validation
Config.validate()
