"""Fail the process if anything imports Python out of a mirage checkout.

Fleet MK used to reach into `mirage/demo/gpt_oss/demo.py` twice -- once for the
MXFP4 pack primitives, once for the reference `GptOssForCausalLM` it packed
from. Both are gone, and this asserts they stay gone.

Renaming the checkout would be the obvious test, but it is destructive here (the
tree is dirty and is somebody's cwd), so this covers the same ground directly.
Two hooks are needed, because fleet_mk used the deleted module in *both* spellings:

  * a `sys.meta_path` finder catches ordinary `import x` once a mirage directory
    is on `sys.path` (what `_load_gptoss_demo_module`'s `sys.path.insert` set up);
  * an audit hook catches `importlib.util.spec_from_file_location` +
    `exec_module`, which loads a file by absolute path and never consults
    `sys.path` or `meta_path` at all -- so the finder alone would miss it. That
    absolute-path load is precisely how fleet_mk reached mirage's demo.py.

Usage -- prepend to any run:

    PYTHONPATH=<fleet>/vllm_integration/tools python3 -c "import no_mirage_guard" ...

or, as used by the harness gate:

    python3 -X importtime ... with `-c "import no_mirage_guard; import runpy;
    runpy.run_module('fleet_megakernel_vllm.harness', run_name='__main__')"`

Set MIRAGE_GUARD_DIR to check a different root (default /home/claudeuser/mirage).
"""

import os
import sys


class _MirageImportGuard:
    """A sys.meta_path finder that raises instead of finding."""

    def __init__(self, root):
        self.root = os.path.realpath(root)

    def find_module(self, fullname, path=None):        # legacy protocol
        self._check(fullname, path)
        return None

    def find_spec(self, fullname, path=None, target=None):
        self._check(fullname, path)
        return None       # never actually claim the module; only observe

    def _check(self, fullname, path):
        # `path` is the parent package's __path__ for submodules, None for
        # top-level. Either way, a hit means the import is about to be resolved
        # out of the mirage tree.
        for entry in list(path or []) + list(sys.path):
            try:
                real = os.path.realpath(entry)
            except (TypeError, ValueError):
                continue
            if real == self.root or real.startswith(self.root + os.sep):
                candidate = os.path.join(real, fullname.rsplit(".", 1)[-1])
                if os.path.isdir(candidate) or os.path.exists(candidate + ".py"):
                    raise ImportError(
                        f"[no_mirage_guard] '{fullname}' would be imported from "
                        f"the mirage checkout at {real} -- fleet_mk must not depend "
                        f"on mirage Python.")


def _install_audit_hook(root):
    """Catch by-path loads, which never reach sys.meta_path.

    `exec` fires for `exec_module`, and its arg carries the code object whose
    co_filename is the source path -- that is what `spec_from_file_location`
    stamps, so a mirage demo.py load is visible here even though no import
    machinery consulted sys.path.
    """
    def hook(event, args):
        if event == "exec" and args:
            fn = getattr(args[0], "co_filename", None)
            if isinstance(fn, str):
                real = os.path.realpath(fn)
                if real == root or real.startswith(root + os.sep):
                    raise ImportError(
                        f"[no_mirage_guard] executing module code from the "
                        f"mirage checkout: {real}")
    sys.addaudithook(hook)


def install(root=None):
    root = root or os.environ.get("MIRAGE_GUARD_DIR", "/home/claudeuser/mirage")
    guard = _MirageImportGuard(root)
    sys.meta_path.insert(0, guard)
    _install_audit_hook(guard.root)
    print(f"[no_mirage_guard] armed against {guard.root}", flush=True)
    return guard


install()
