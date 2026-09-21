"""The ABI bridge must drop only fields irrelevant to SD/V1 semantics."""
import unittest
from betterscale.patches.qwen_gdn.mamba_abi import V1SDPostprocess


class RecordingKernel:
    def __getitem__(self, grid):
        self.grid = grid
        def run(*args, **kwargs):
            self.args, self.kwargs = args, kwargs
            return 'launched'
        return run


class MambaABI(unittest.TestCase):
    def test_projection_preserves_pointer_order(self):
        kernel = RecordingKernel()
        args = [object() for _ in range(18)]
        args[16] = None
        result = V1SDPostprocess(kernel)[(4,96)](
            *args,CONV_STATE_DIM_FIRST=False,block_size=1536,COPY_BLOCK_SIZE=1024)
        self.assertEqual(result,'launched')
        self.assertEqual(kernel.grid,(4,96))
        self.assertEqual(kernel.args,tuple(args[:13]+[args[15],args[17]]))
        self.assertEqual(kernel.kwargs,dict(block_size=1536,COPY_BLOCK_SIZE=1024))

    def test_reject_new_layout_or_mapping_not_discard_them(self):
        args = list(range(18)); args[16] = None
        launch = V1SDPostprocess(RecordingKernel())[(1,2)]
        with self.assertRaisesRegex(ValueError,'SD'):
            launch(*args,CONV_STATE_DIM_FIRST=True,block_size=16,COPY_BLOCK_SIZE=1024)
        args[16] = object()
        with self.assertRaisesRegex(ValueError,'mapping'):
            launch(*args,CONV_STATE_DIM_FIRST=False,block_size=16,COPY_BLOCK_SIZE=1024)
        args[16] = None
        with self.assertRaisesRegex(ValueError,'arguments'):
            launch(*args,CONV_STATE_DIM_FIRST=False,block_size=16,COPY_BLOCK_SIZE=1024,HAS_IDX_MAPPING=True)
