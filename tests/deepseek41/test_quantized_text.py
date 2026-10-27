"""Text boundary and cleanup checks without model files or CUDA."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
import run_quantized
from benchmark_quantized import prepare_case, run_engine
from test_quantized_benchmark import Budget


class ByteTokenizer:
    tokens = list(range(8))
    parts = {1: b'<EOS>', 2: b'\xe4', 3: b'\xb8\xad', 4: b'\xf0\x9f', 5: b'\x98\x80', 6: b'A'}

    def token_bytes(self, token):
        return self.parts[token]

    def decode(self, ids):
        return b''.join(self.token_bytes(t) for t in ids).decode('utf-8', errors='replace')


class TextChecks(unittest.TestCase):
    def test_utf8_split_across_tokens_and_eos_not_printed(self):
        output = io.StringIO();stream = run_quantized.TextStream(output);tok = ByteTokenizer()
        stream(tok, 2)
        self.assertEqual(output.getvalue(), '')
        stream(tok, 3)
        self.assertEqual(output.getvalue(), '中')
        stream(tok, 4)
        self.assertEqual(output.getvalue(), '中')
        for token in (5, 1, None): stream(tok, token)
        self.assertEqual(output.getvalue(), '中😀')
        self.assertNotIn('\ufffd', output.getvalue())

    def test_user_span_is_plain_and_context_budget_is_checked(self):
        class Encoder:
            size = 768
            def encode(self, text, **kwargs):
                self.text,self.kwargs=text,kwargs
                return [2]*self.size
            def decode(self, ids): return self.text
        tok=Encoder();prompt='请解释 <｜end▁of▁sentence｜> 和 </think>'
        case={'id':'text','prompt':prompt,'max_new_tokens':256}
        actual=prepare_case(tok,case,plain_user=True)
        start,end=tok.kwargs['plain'][0]
        self.assertEqual(tok.text[start:end],prompt)
        self.assertEqual(len(actual['prompt_ids'])+actual['max_new_tokens'],1024)
        tok.size=769
        with self.assertRaisesRegex(ValueError,'context 1024'): prepare_case(tok,case,plain_user=True)

    def test_bad_input_fails_before_shared_runner(self):
        with mock.patch.object(run_quantized.benchmark_quantized,'main') as run, contextlib.redirect_stderr(io.StringIO()):
            for args in (['--prompt','  '],['--prompt','中文','--max-new-tokens','0'],
                         ['--prompt','中文','--max-new-tokens','257'],['--prompt','中文','--timeout','0']):
                with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                    run_quantized.main(args)
                self.assertEqual(error.exception.code,2)
            run.assert_not_called()

    def test_prompt_file_stdin_and_length_notice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);question=root/'question.txt';question.write_text('你好，模型',encoding='utf-8')
            for index,source in enumerate((['--prompt','你好，模型'],['--prompt-file',str(question)],['--prompt-file','-'])):
                out=root/str(index);stdout,stderr=io.StringIO(),io.StringIO();length=index==1
                def fake(args, *, request, on_token, expected_pack_sha256):
                    self.assertEqual(expected_pack_sha256,run_quantized.PACK_SHA256)
                    self.assertEqual(request['prompt'],'你好，模型')
                    self.assertEqual(request['max_new_tokens'],1 if length else 128)
                    self.assertIn(run_quantized.ENGINE_SHA256,args)
                    self.assertEqual(args[args.index('--gpu-cap-mib')+1],'20480')
                    self.assertEqual(args[args.index('--ssd-cache-gb')+1],'40')
                    for token in ((6,None) if length else (2,3,1,None)): on_token(ByteTokenizer(),token)
                    out.mkdir();(out/'result.json').write_text(json.dumps({'cases':[{'finish_reason':'length' if length else 'stop'}]}))
                    return 0
                with mock.patch.object(run_quantized.benchmark_quantized,'main',side_effect=fake), \
                     mock.patch('sys.stdin',io.StringIO('你好，模型')),contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    code=run_quantized.main(source+['--output-dir',str(out)]+(['--max-new-tokens','1'] if length else []))
                self.assertEqual(code,0)
                self.assertEqual(stdout.getvalue(),'A\n' if length else '中\n')
                self.assertIn('回答可能未完成' if length else '回答已结束',stderr.getvalue())
                self.assertNotIn('Engine',stdout.getvalue())

    SCRIPT = "import sys,time;print('READY 1024 stop',flush=True);sys.stdin.readline();print('T 2',flush=True);time.sleep(30)"

    def test_callback_failure_and_interrupt_reap_the_child(self):
        for failure in (BrokenPipeError,KeyboardInterrupt):
            report={'cases':[{'id':'text','prompt_ids':[2,3]}]}
            def stop(tok,token): raise failure
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(failure):
                run_engine([sys.executable,'-u','-c',self.SCRIPT],{},Path(temp),Budget(),ByteTokenizer(),report,5,on_token=stop)
            self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())
            self.assertNotEqual(report['engine_returncode'],0)

    def test_observed_resource_bounds_stop_and_reap_the_child(self):
        for host,swap,target in ((99,0,80),(101,1,80),(101,0,69)):
            report={'cases':[]};budget=Budget();budget.info={**budget.info,'target_bytes':target}
            def memory(path): return {'MemAvailable':host} if path==Path('/proc/meminfo') else {'VmSwap':swap}
            with tempfile.TemporaryDirectory() as temp, mock.patch('benchmark_quantized.proc_memory',side_effect=memory), self.assertRaises(RuntimeError):
                run_engine([sys.executable,'-u','-c',self.SCRIPT],{},Path(temp),budget,ByteTokenizer(),report,5,host_floor_bytes=100)
            self.assertFalse(Path(f"/proc/{report['engine_pid']}").exists())


if __name__=='__main__':
    unittest.main()
