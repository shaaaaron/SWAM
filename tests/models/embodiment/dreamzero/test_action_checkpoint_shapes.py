"""CPU-only regression checks for checkpoint tensor shapes."""

import ast
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "rlinf/models/embodiment/dreamzero/__init__.py"


def checkpoint_helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_align_checkpoint_tensor_shapes"}
    functions = ast.Module(
        body=[node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(functions, str(SOURCE), "exec"), namespace)
    return namespace["_align_checkpoint_tensor_shapes"]


class StateOnlyModel:
    def __init__(self, key, target):
        self.state = {key: target}

    def state_dict(self):
        return self.state


class CheckpointShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.align_shapes = staticmethod(checkpoint_helpers())

    def test_scalar_and_length_one_roundtrip(self):
        key = "action_head.image_encoder.model.log_scale"
        for target_shape, source_shape in [((), (1,)), ((1,), ())]:
            with self.subTest(target=target_shape, source=source_shape):
                target = torch.zeros(target_shape)
                value = torch.full(source_shape, 2.5)
                weights = {key: value}
                self.align_shapes(StateOnlyModel(key, target), weights)
                self.assertEqual(weights[key].shape, target.shape)
                self.assertEqual(weights[key].item(), 2.5)
                self.assertEqual(weights[key].dtype, value.dtype)

    def test_real_non_action_mismatch_is_rejected(self):
        key = "action_head.image_encoder.model.log_scale"
        with self.assertRaisesRegex(RuntimeError, "Checkpoint shape mismatch"):
            self.align_shapes(StateOnlyModel(key, torch.zeros(())), {key: torch.ones(2)})

    def test_action_mismatch_is_rejected(self):
        key = "action_head.model.action_encoder.W1.W"
        target = torch.randn(1, 2, 4)
        before = target.clone()
        with self.assertRaisesRegex(RuntimeError, "Checkpoint shape mismatch"):
            self.align_shapes(StateOnlyModel(key, target), {key: torch.ones(1, 2, 8)})
        torch.testing.assert_close(target, before)

    def test_matching_action_weight_is_preserved(self):
        key = "action_head.model.action_encoder.W1.W"
        value = torch.randn(1, 2, 4)
        weights = {key: value}
        self.align_shapes(StateOnlyModel(key, torch.zeros_like(value)), weights)
        self.assertIs(weights[key], value)

    def test_unknown_keys_remain_for_strict_key_validation(self):
        key = "unexpected.weight"
        value = torch.ones(2)
        weights = {key: value}
        self.align_shapes(StateOnlyModel("expected.weight", torch.ones(2)), weights)
        self.assertIs(weights[key], value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
