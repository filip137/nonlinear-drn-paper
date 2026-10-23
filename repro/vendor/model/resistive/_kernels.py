"""Build and load the opt-in compiled Shockley kernels (diode_kernels.c).

The source is compiled with the system C compiler ($CC, else cc or gcc) on first use, into
$DRN_KERNEL_CACHE or ~/.cache/nonlinear-drn/kernels. The library name hashes the source, the
compiler identity and the flags, so an edit or another compiler builds a new library; builds
go to a temporary file and are renamed into place, so concurrent processes cannot collide.
-ffp-contract=off keeps multiply-adds unfused, as in the torch path.
"""

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).with_name("diode_kernels.c")
FLAGS = ("-O2", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math")

_library = None


class KernelUnavailable(RuntimeError):
    """The compiled kernels cannot be built or loaded on this machine."""


def _compiler():
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        raise KernelUnavailable("Compiled kernels need a C compiler: set $CC or install cc/gcc.")
    return compiler


def _declare(library):
    L, D, I, P = ctypes.c_long, ctypes.c_double, ctypes.c_int, ctypes.c_void_p
    library.double_diode_update.argtypes = [L, L, P, L, P, P, P, D, D, D, I, D, D, I, I, D, I, D, D]
    library.double_diode_update.restype = I
    library.single_diode_update.argtypes = [L, L, P, L, P, P, P, P, D, D, D, I, D, D, D, I, I, D, D, I, D]
    library.single_diode_update.restype = I
    return library


def load():
    """Return the loaded kernel library, building it on first use."""

    global _library
    if _library is not None:
        return _library
    compiler = _compiler()
    try:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KernelUnavailable(f"C compiler {compiler!r} is not usable: {exc}") from exc
    identity = repr((compiler, version.splitlines()[:1], FLAGS)).encode()
    digest = hashlib.sha256(SOURCE.read_bytes() + identity).hexdigest()[:16]
    cache = Path(os.environ.get("DRN_KERNEL_CACHE") or Path.home()/".cache"/"nonlinear-drn"/"kernels")
    target = cache/f"diode_kernels-{digest}.so"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=cache, prefix=".build-", suffix=".so")
        os.close(handle)
        result = subprocess.run([compiler, *FLAGS, "-o", temporary, str(SOURCE), "-lm"],
                                capture_output=True, text=True)
        if result.returncode:
            os.unlink(temporary)
            raise KernelUnavailable(f"Compiling {SOURCE.name} failed:\n{result.stderr}")
        os.replace(temporary, target)
    try:
        _library = _declare(ctypes.CDLL(str(target)))
    except OSError as exc:
        raise KernelUnavailable(f"Loading {target} failed: {exc}") from exc
    return _library


def available():
    """Whether the kernels can be built and loaded here."""

    try:
        load()
    except KernelUnavailable:
        return False
    return True
