import importlib.util
from pathlib import Path
import sys
import unittest

path = Path(__file__).resolve().parents[1] / "channel_layout.py"
spec = importlib.util.spec_from_file_location("channel_layout", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Layout = module.ChannelLayout


class ChannelLayoutTests(unittest.TestCase):
    def test_no_overlaps_and_worst_case_output(self):
        for rows in (32, 128, 256, 512, 1024):
            with self.subTest(rows=rows):
                x = Layout(rows)
                self.assertLessEqual(64 + rows * 10, x.scales)
                self.assertLessEqual(x.scales + rows * 8, x.payload)
                self.assertLessEqual(x.payload * 4 + rows * 2560 * 2, x.source_bytes)
                # Every route can belong to a single owner; no average-load cap.
                self.assertLessEqual(64 * 4 + rows * 10 * 2560 * 2, x.output_bytes)
                self.assertLessEqual(x.map_words * 4, 65536)
                self.assertEqual(x.source_bytes % module.ALIGN, 0)
                self.assertEqual(x.output_bytes % module.ALIGN, 0)
                self.assertEqual(x.map_words % 8, 0)

    def test_capsules_cannot_silently_mix_layouts(self):
        self.assertEqual(Layout.from_abi(dict(version=2, rows=32)), Layout(32))
        with self.assertRaises(ValueError):
            Layout.from_abi(dict(version=2, rows=1024))
        x = Layout(1024)
        abi = dict(
            version=3,
            rows=1024,
            source_scale_words=x.scales,
            source_payload_words=x.payload,
        )
        self.assertEqual(Layout.from_abi(abi), x)
        abi["source_payload_words"] = 1024
        with self.assertRaises(ValueError):
            Layout.from_abi(abi)

    def test_e3_requires_distinct_binary_and_channel_identity(self):
        layout = Layout(1024, 3)
        abi = dict(
            version=4,
            owners=3,
            rows=1024,
            source_scale_words=layout.scales,
            source_payload_words=layout.payload,
        )
        self.assertEqual(Layout.from_abi(abi), layout)
        self.assertNotEqual(layout.contract(), Layout(1024, 4).contract())
        abi["version"] = 3
        with self.assertRaises(ValueError):
            Layout.from_abi(abi)

    def test_route_flags_are_disjoint_and_negotiated(self):
        for rows in (32, 128, 256, 512, 1024):
            old = Layout(rows, 3, 5)
            new = Layout(rows, 3, 5, True)
            payload_end = 256 + rows * 10 * 2560 * 2
            self.assertEqual(payload_end % 64, 0)
            self.assertLessEqual(payload_end + rows * 10 * 64, new.output_bytes)
            self.assertNotEqual(old.contract(), new.contract())
            abi = dict(
                version=5,
                rows=rows,
                owners=3,
                sources=5,
                source_scale_words=new.scales,
                source_payload_words=new.payload,
                route_ready=True,
            )
            self.assertEqual(Layout.from_abi(abi), new)

    def test_unknown_capacity_fails_before_export(self):
        for rows in (0, 31, 33, 4096):
            with self.assertRaises(ValueError):
                Layout(rows)


if __name__ == "__main__":
    unittest.main()
