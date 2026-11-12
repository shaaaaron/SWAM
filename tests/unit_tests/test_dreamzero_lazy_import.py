# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
import unittest


class TestDreamZeroLazyImport(unittest.TestCase):
    def test_data_transforms_import_does_not_load_sft_dataloader(self):
        """Checkpoint conversion needs transforms/configs without torchdata."""
        for module_name in [
            "rlinf.data.datasets.dreamzero.data_transforms",
            "rlinf.data.datasets.dreamzero.dreamzero",
            "rlinf.data.datasets.dreamzero",
        ]:
            sys.modules.pop(module_name, None)

        importlib.import_module("rlinf.data.datasets.dreamzero.data_transforms")

        self.assertNotIn("rlinf.data.datasets.dreamzero.dreamzero", sys.modules)


if __name__ == "__main__":
    unittest.main()
