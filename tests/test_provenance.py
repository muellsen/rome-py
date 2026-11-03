"""Check that edited artifacts cannot pass provenance verification."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'analysis/pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class ProvenanceTest(unittest.TestCase):
    def test_output_tampering(self):
        with tempfile.TemporaryDirectory(dir=pipeline.ROOT) as directory:
            output = Path(directory) / 'result.json'
            output.write_text('{"value": 1}')
            manifest = pipeline.capture([output], seed=7)
            pipeline.verify(manifest)
            output.write_text('{"value": 2}')
            with self.assertRaisesRegex(ValueError, 'Stale or missing outputs'):
                pipeline.verify(manifest)

    def test_missing_input(self):
        with tempfile.TemporaryDirectory(dir=pipeline.ROOT) as directory:
            source = Path(directory) / 'input.json'
            source.write_text('{}')
            manifest = pipeline.capture([], inputs=[source])
            source.unlink()
            with self.assertRaisesRegex(ValueError, 'Stale or missing inputs'):
                pipeline.verify(manifest)

    def test_selected_dependencies_remain_checked(self):
        source='src/concomitant_huber/solver.py'
        manifest=pipeline.capture([])
        pipeline.verify(manifest,source_paths=[source])
        manifest['sources'][source]='incorrect-hash'
        with self.assertRaisesRegex(ValueError,'Stale or missing sources'):
            pipeline.verify(manifest,source_paths=[source])
