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

"""HTTP client for the Cortex Training REST API.

The client lives in :mod:`cortex_training.client.core`; authentication,
telemetry, and the optimizer and LoRA normalizers sit beside it in this
package. This module re-exports the public client surface, so
``from cortex_training.client import CortexTrainingClient`` keeps working.
"""

from cortex_training.client._account_jobs import AccountJobListUnavailable
from cortex_training.client._account_jobs import InvalidPageToken
from cortex_training.client.core import DEBUG_OPTIONS_ENV
from cortex_training.client.core import DISABLE_TELEMETRY_ENV
from cortex_training.client.core import DISABLE_TENSOR_PROMPTS_ENV
from cortex_training.client.core import ENABLE_SUCCESS_TELEMETRY_ENV
from cortex_training.client.core import CLI_SURFACE
from cortex_training.client.core import SDK_SURFACE
from cortex_training.client.core import SURFACE_ATTRIBUTE
from cortex_training.client.core import TUI_SURFACE
from cortex_training.client.core import ChunkGroupConflictError
from cortex_training.client.core import ChunkGroupError
from cortex_training.client.core import ChunkGroupRestartError
from cortex_training.client.core import CortexTrainingClient
from cortex_training.client.core import DatabaseAccessError
from cortex_training.client.core import DownloadCancelled
from cortex_training.client.core import Hardware
from cortex_training.client.core import InferenceConfig
from cortex_training.client.core import JobType
from cortex_training.client.core import ModelDeploymentSpec
from cortex_training.client.core import NoChunksError
from cortex_training.client.core import SubJobConfig
from cortex_training.client.core import TrainingConfig
from cortex_training.client.core import build_forward_backward_kwargs
from cortex_training.client.core import build_forward_backward_payload
from cortex_training.client.core import create_job_body_from_job
from cortex_training.client.core import serialize_forward_backward_args

__all__ = [
    "DEBUG_OPTIONS_ENV",
    "DISABLE_TELEMETRY_ENV",
    "DISABLE_TENSOR_PROMPTS_ENV",
    "ENABLE_SUCCESS_TELEMETRY_ENV",
    "CLI_SURFACE",
    "SDK_SURFACE",
    "SURFACE_ATTRIBUTE",
    "TUI_SURFACE",
    "AccountJobListUnavailable",
    "ChunkGroupConflictError",
    "ChunkGroupError",
    "ChunkGroupRestartError",
    "CortexTrainingClient",
    "DatabaseAccessError",
    "DownloadCancelled",
    "Hardware",
    "InferenceConfig",
    "InvalidPageToken",
    "JobType",
    "ModelDeploymentSpec",
    "NoChunksError",
    "SubJobConfig",
    "TrainingConfig",
    "build_forward_backward_kwargs",
    "build_forward_backward_payload",
    "create_job_body_from_job",
    "serialize_forward_backward_args",
]
