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
"""Load Tinker fixtures only when the optional extra is installed.

The rest of this suite does not import FastAPI or arctic-platform. Each
``tests/test_tinker_*.py`` module skips itself the same way ``test_tui_app``
skips without textual.
"""

from __future__ import annotations

import importlib.util

_TINKER_DEPS = ("arctic_platform", "fastapi", "httpx", "pytest_asyncio", "tinker")

if all(importlib.util.find_spec(name) is not None for name in _TINKER_DEPS):
    pytest_plugins = ["tests.tinker_fixtures"]
