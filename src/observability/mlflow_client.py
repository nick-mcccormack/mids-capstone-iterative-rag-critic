import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import mlflow


def configure_mlflow() -> bool:
	"""Configure MLflow if available.

	Returns
	-------
	bool
		Whether MLflow was configured.
	"""
	tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
	experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
	if not tracking_uri or not experiment_name:
		return False

	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
	return True


@contextmanager
def start_run_if_enabled(
	enabled: bool,
	run_name: str,
	nested: Optional[bool] = None,
) -> Generator[Optional[Any], None, None]:
	"""Start an MLflow run when enabled and available.

	Parameters
	----------
	enabled : bool
		Whether logging is enabled.
	run_name : str
		Run name.
	nested : Optional[bool], default None
		Whether to create a nested run. If None, infer from active run.

	Yields
	------
	Optional[Any]
		MLflow run object or None.
	"""
	if not enabled:
		yield None
		return

	use_nested = bool(mlflow.active_run()) if nested is None else bool(nested)
	with mlflow.start_run(run_name=run_name, nested=use_nested) as run:
		yield run


def log_text_artifact(text: str, artifact_file: str) -> None:
	"""Log a text artifact.

	Parameters
	----------
	text : str
		Artifact text.
	artifact_file : str
		Artifact file path inside MLflow.
	"""
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, os.path.basename(artifact_file))
		with open(path, "w", encoding="utf-8") as handle:
			handle.write(text)
		mlflow.log_artifact(path, artifact_path=os.path.dirname(artifact_file))


def log_dict_artifact(obj: Dict[str, Any], artifact_file: str) -> None:
	"""Log a JSON artifact.

	Parameters
	----------
	obj : dict[str, Any]
		JSON-serializable object.
	artifact_file : str
		Artifact file path inside MLflow.
	"""
	log_text_artifact(
		json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True),
		artifact_file,
	)
