"""CPU checks for direct interpreter dispatch and deployed SVF critic layout."""
import importlib.util
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eval_suite',root/'local_eval/run_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

class PlanTest(unittest.TestCase):
    def test_svf_export_with_reference_uses_direct_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ['actor','reference']:
                (p/name/'experiment_cfg').mkdir(parents=True)
                (p/name/'config.json').write_text('{}')
                (p/name/'experiment_cfg/metadata.json').write_text('{}')
            (p/'critic').mkdir()
            (p/'critic/config.json').write_text('{}')
            (p/'critic/q_projection.safetensors').touch()
            args=['--actor',str(p/'actor'),'--critic',str(p/'critic'),
                  '--critic-reference-actor',str(p/'reference'),'--deas-backend','svf',
                  '--python-executable','/example/python','--output-root',str(p/'out'),'--dry-run']
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):self.assertEqual(suite.main(args),0)
            self.assertIn('/example/python',buf.getvalue())
            self.assertNotIn('conda run',buf.getvalue())
            self.assertFalse((p/'out').exists())
            (p/'critic/q_projection.safetensors').unlink()
            with self.assertRaisesRegex(SystemExit,'q_projection'):
                suite.main(args)

if __name__=='__main__':unittest.main()
