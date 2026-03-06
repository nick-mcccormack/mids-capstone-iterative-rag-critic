#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="${ENV_NAME:-rag-capstone}"
TOOLING_ENV="${TOOLING_ENV:-tooling}"
PLATFORM="${PLATFORM:-linux-64}"
ENV_YML="${ENV_YML:-environment.yml}"
PIP_REQS="${PIP_REQS:-requirements-pip.txt}"
LOCK_FILE="${LOCK_FILE:-conda-lock.yml}"

echo "== Bootstrapping conda env =="
echo "ENV_NAME:     ${ENV_NAME}"
echo "TOOLING_ENV:  ${TOOLING_ENV}"
echo "LOCK_FILE:    ${LOCK_FILE}"
echo "PLATFORM:     ${PLATFORM}"
echo "ENV_YML:      ${ENV_YML}"
echo "PIP_REQS:     ${PIP_REQS}"
echo

if ! command -v conda >/dev/null 2>&1; then
	echo "ERROR: conda not found on PATH." >&2
	exit 1
fi

if [[ ! -f "${ENV_YML}" ]]; then
	echo "ERROR: ${ENV_YML} not found in current directory: $(pwd)" >&2
	exit 1
fi

if [[ ! -f "${PIP_REQS}" ]]; then
	echo "ERROR: ${PIP_REQS} not found in current directory: $(pwd)" >&2
	exit 1
fi

# Enable `conda activate` in non-interactive shells
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create tooling env if missing (contains conda-lock + mamba)
if ! conda env list | awk '{print $1}' | grep -qx "${TOOLING_ENV}"; then
	echo "== Creating tooling env: ${TOOLING_ENV} =="
	conda create -y -n "${TOOLING_ENV}" -c conda-forge python=3.11 conda-lock mamba
fi

conda activate "${TOOLING_ENV}"

if ! command -v conda-lock >/dev/null 2>&1; then
	echo "ERROR: conda-lock still not found after activating ${TOOLING_ENV}." >&2
	exit 1
fi

echo "== Tooling versions =="
conda-lock --version
mamba --version || true
echo

echo "== Generating lockfile =="
conda-lock lock -f "${ENV_YML}" -p "${PLATFORM}"

echo "== Recreating env: ${ENV_NAME} =="
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda-lock install -n "${ENV_NAME}" "${LOCK_FILE}"

conda activate "${ENV_NAME}"

echo "== Installing pip requirements =="
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r "${PIP_REQS}"
python -m pip check

echo "== Registering Jupyter kernel =="
python -m ipykernel install --user --name "${ENV_NAME}" \
	--display-name "Python (${ENV_NAME})"

echo
echo "DONE. Switch kernel to: Python (${ENV_NAME})"