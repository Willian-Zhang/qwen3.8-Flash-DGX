#!/usr/bin/env python3
"""CPU-only installed-code QA. Load a local tokenizer; never generate model tokens."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent

def main():
    if not os.environ.get('QWEN_TOKENIZER'):
        raise SystemExit('Set QWEN_TOKENIZER to a local compatible Qwen tokenizer directory')
    # Tests may export synthetic SSE fixtures beside their modules; keep these
    # temporary, permitting the supplied bundle to be mounted read-only.
    with tempfile.TemporaryDirectory(prefix='qwen-safe-cpu-') as tmp:
        directory=Path(tmp)/'tests';shutil.copytree(ROOT/'tests',directory)
        sys.path.insert(0,str(directory))
        os.environ['CODEX_BAKED_TEST']='1'
        os.environ['CODEX_TEST_EXPORT_EVENTS']=str(Path(tmp)/'synthetic-events.json')
        import realdeployed_CPU_runner as old
        import eof_expected_change,empty_stop_tests,manager_expected_change,producer_tests
        groups=[('parser',unittest.defaultTestLoader.loadTestsFromModule(old.deployed_regression)),
            ('responses_sse',unittest.defaultTestLoader.loadTestsFromTestCase(old.ActualSSE)),
            ('actual_parser_manager',unittest.defaultTestLoader.loadTestsFromTestCase(manager_expected_change.ProductionComposition)),
            ('reasoning_eof',unittest.defaultTestLoader.loadTestsFromTestCase(eof_expected_change.EOFIntegration)),
            ('empty_stop',unittest.defaultTestLoader.loadTestsFromTestCase(empty_stop_tests.EmptyStop)),
            ('actual_length_terminal_producer',unittest.defaultTestLoader.loadTestsFromTestCase(producer_tests.LengthProducer))]
        report={'model_inference':False,'source_overlay':False,'groups':{}}
        passed=True
        for name,suite in groups:
            out=io.StringIO()
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
                result=unittest.TextTestRunner(stream=out,verbosity=1).run(suite)
            report['groups'][name]={'methods':result.testsRun,'passed':result.wasSuccessful(),
                'failures':len(result.failures),'errors':len(result.errors)}
            if not result.wasSuccessful():
                # Synthetic fixture diagnostics only; never model-generated content.
                print(out.getvalue(),file=sys.stderr)
            passed=passed and result.wasSuccessful()
        report['total_methods']=sum(g['methods'] for g in report['groups'].values())
        assert report['total_methods']==67
        capture=io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                importlib.import_module('parser_fidelity')
            fidelity=json.loads(capture.getvalue())
            report['fidelity']={key:fidelity[key] for key in
                ['literal_byte_fidelity_cases','stream_final_mismatches','chunk_mismatches','ordinary_bpe_protocol_ids_absent','normal_final_answer_passed']}
            passed=passed and fidelity['literal_byte_fidelity_cases']==21 and fidelity['stream_final_mismatches']==0 and fidelity['chunk_mismatches']==0
        except Exception as exc:
            report['fidelity']={'passed':False,'error_type':type(exc).__name__};passed=False
        report['passed']=passed
        print(json.dumps(report,indent=2))
        return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
