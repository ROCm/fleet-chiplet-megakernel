# Contributing to Fleet

Thanks for your interest in contributing! Fleet is an open, cooperative
task-scheduling system for LLM inference on multi-die AMD GPUs, and we welcome
contributions from the community — bug reports, documentation, performance work,
new features, and more.

This guide explains how to get set up, our conventions, and how to submit
changes.

## Ways to Contribute

- **Report a bug** or **request a feature** using our
  [issue templates](.github/ISSUE_TEMPLATE). Please search existing issues first
  to avoid duplicates.
- **Improve documentation** — fixes to the README, docstrings, or the docs under
  `docs/` are always appreciated.
- **Submit code** — bug fixes, performance improvements, or new functionality
  via a pull request (see below).

If you are planning a large or architectural change, please open an issue to
discuss it first so we can align on the approach before you invest time.

## Development Setup

### Hardware / Software Requirements

- AMD Instinct MI350 (gfx950) for the GPU kernels and end-to-end demos
- ROCm 7.0+ with the `hipcc` compiler
- Python 3.8+ and PyTorch 2.4+ (ROCm build)
- CMake 3.24+

### Build From Source

```bash
git clone git@github.com:ROCm/fleet-chiplet-megakernel.git
cd fleet-chiplet-megakernel
git checkout amd_mi350

# Populate submodules
git submodule update --init --recursive

# Build (editable install)
pip install -e . -v
export MIRAGE_HOME=$(pwd)
python3 -c "import mirage; print('Mirage OK')"
```

See [INSTALL.md](INSTALL.md) for more detailed instructions.

## Coding Style

- **C/C++**: formatted with `clang-format` version **15** (config in
  [`.clang-format`](.clang-format)). Run the helper before committing:

  ```bash
  bash scripts/format.sh
  ```

  CI enforces this via the `code-format` workflow, so unformatted code will fail
  the check.
- **Python**: follow the surrounding style (PEP 8). Keep imports tidy and avoid
  unrelated reformatting in the same PR.
- Keep changes focused. Unrelated cleanups belong in a separate PR.

## Testing

- Correctness and unit tests live under `tests/`.
- Please add or update tests for any behavior you change.
- CI runs several workflows on each PR (see `.github/workflows/`), including
  build/test, Qwen3 integration tests, GPU tests, code formatting, and
  shell-check. All required checks must pass before a PR can be merged.

## Pull Request Process

1. **Fork** the repository and create a topic branch from `amd_mi350`:
   ```bash
   git checkout -b my-feature amd_mi350
   ```
2. Make your changes, following the coding style above, and add tests.
3. Run `bash scripts/format.sh` and make sure the build and tests pass locally.
4. Push your branch and open a pull request. Fill out the
   [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
5. A code owner (see [`.github/CODEOWNERS`](.github/CODEOWNERS)) will be
   automatically requested for review. Address review feedback by pushing
   additional commits to your branch.
6. Once approved and all required CI checks pass, a maintainer will merge your PR.

Write clear commit messages that explain the *why* of a change, not just the
*what*.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), the same license that covers this project.

---

## For Maintainers: Recommended Branch Protection

To keep the default branch healthy while accepting external contributions, we
recommend enabling the following **branch protection rules** in
`Settings → Branches → Branch protection rules` for the default branch:

- **Require a pull request before merging**
  - Require at least **1 approving review**.
  - **Require review from Code Owners** (uses `.github/CODEOWNERS`).
  - Dismiss stale approvals when new commits are pushed.
- **Require status checks to pass before merging**
  - Require branches to be up to date before merging.
  - Mark the CI workflows (build/test, code-format, shell-check, etc.) as
    required checks.
- **Require conversation resolution before merging.**
- **Do not allow force pushes** and **do not allow deletions** of the protected
  branch.
- Optionally **require signed commits** and **include administrators** so the
  rules apply to everyone.
