# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified training-metrics logging for the Cortex Training examples."""

from __future__ import annotations

from typing import Any


class SnowflakeExperimentLogger:
    """Logs metrics to Snowflake experiment tracking.

    Same ``log_metrics`` / ``close`` interface as ``ml_log`` so it can be
    composed via :class:`_CompositeLogger`.
    """

    def __init__(self, session: Any, experiment_name: str, run_name: str) -> None:
        from snowflake.ml.experiment import ExperimentTracking

        self._exp = ExperimentTracking(session=session)
        # The server returns a fully qualified experiment name
        # (``DB.SCHEMA.EXPERIMENT``); ExperimentTracking resolves against the
        # session's database and schema and only accepts a bare identifier.
        self._exp.set_experiment(experiment_name.rpartition(".")[2])
        self._exp.start_run(run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        self._exp.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int = 0, **kwargs: Any) -> None:
        self._exp.log_metrics(metrics, step=step)

    def close(self) -> None:
        pass


class _CompositeLogger:
    """Fans out ``log_metrics`` / ``close`` to multiple loggers."""

    def __init__(self, loggers: list[Any]) -> None:
        self._loggers = loggers

    def log_metrics(self, metrics: dict[str, float], step: int = 0, **kwargs: Any) -> None:
        for lg in self._loggers:
            lg.log_metrics(metrics=metrics, step=step, **kwargs)

    def close(self) -> None:
        for lg in self._loggers:
            lg.close()


def setup_logging(
    config: Any,
    *,
    client: Any | None = None,
    job_id: str | None = None,
) -> Any:
    """Create a training-metrics logger, optionally with Snowflake experiment tracking."""
    from tinker_cookbook.utils import ml_log

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    if not getattr(config, "sf_tracking", False) or client is None or job_id is None:
        return ml_logger
    run_info = client.get_experiment_run(job_id)
    sf_logger = SnowflakeExperimentLogger(
        client.create_snowpark_session(),
        run_info["experiment_name"],
        run_info["experiment_run_name"],
    )
    sf_logger.log_params(vars(config))
    return _CompositeLogger([ml_logger, sf_logger])
