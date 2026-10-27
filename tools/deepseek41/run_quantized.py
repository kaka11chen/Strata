"""Local text generation with the validated DeepSeek-V4.1-Flash-1017 IQ1_M profile."""
from __future__ import annotations

import argparse
import codecs
import datetime
import json
import os
from pathlib import Path
import select
import signal
import sys

import benchmark_quantized

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build-deepseek-cuda'
ENGINE_SHA256 = '86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4'
PACK_SHA256 = {
    'ds4.txt': 'db58b924a926a587ae94123574b9e8ff17ff890b6e9efe9543923f29e608bcd0',
    'engram.bin': '08d55d6433f389b14aa115f4ea4eb31d6a2d63df1ae40821f03facf4c2d49142',
    'tensors.txt': 'a80c771dc2fa738b1bb4fdbc50909b123ab0ab0fbf71b10d8ebedb30a6a746d8',
    'tokenizer/chat_template.jinja': '6f9bd1f87d9dad376b4c79683934b4254396d826ae514ab0244efa904ce1d1a4',
    'tokenizer/merges.txt': '547bace837acea98652eef5669c9b01cee1c73ad1d2a44d6fdce9e6a81c22147',
    'tokenizer/token_type.json': '0a61021438e5ce6041133495f18fc360a6587aea748b17d953f552d2119ba51e',
    'tokenizer/tokenizer.json': '5a47981b685b75154e29bfb0299584c9b3a4a28679585e86d524c34c065d037f',
    'tokenizer/vocab.json': '2f0bf5b31f4aa191d5d0d78291927a4f28c9ceb498a5e345b97e926c4cb99c1a',
}


class TextStream:
    def __init__(self, output):
        self.output = output
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.ends_with_newline = False

    def __call__(self, tok, token):
        if token == 1:  # Native EOS is kept in evidence, not printed as answer text.
            return
        text = self.decoder.decode(b'' if token is None else tok.token_bytes(token), final=token is None)
        if text:
            self.output.write(text)
            self.output.flush()
            self.ends_with_newline = text.endswith('\n')

    def finish_case(self, case):
        if not self.ends_with_newline:
            print(file=self.output, flush=True)
        if case['finish_reason'] == 'length':
            print('已达到输出 token 上限，回答可能未完成。', file=sys.stderr, flush=True)
        else:
            print('回答已结束。', file=sys.stderr, flush=True)
        self.decoder.reset()
        self.ends_with_newline = False


class LineInput:
    """Read UTF-8 lines without blocking the native health/memory checks."""
    def __init__(self, max_new_tokens, fd=None):
        self.max_new_tokens = max_new_tokens
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer = ''
        self.eof = False

    def read_line(self, idle_tick):
        while True:
            idle_tick()
            if '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                return line.rstrip('\r')
            if self.eof:
                line, self.buffer = self.buffer, ''
                return line.rstrip('\r') if line else None
            if select.select([self.fd], [], [], 1)[0]:
                data = os.read(self.fd, 4096)
                self.buffer += self.decoder.decode(data, final=not data)
                self.eof = not data

    def __call__(self, idle_tick):
        while True:
            print('你> ', end='', file=sys.stderr, flush=True)
            line = self.read_line(idle_tick)
            if line is None or line.strip() == '/exit':
                return None
            if line.strip():
                return {'category': 'User independent question', 'prompt': line, 'max_new_tokens': self.max_new_tokens}
            print('请输入非空问题。', file=sys.stderr, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--interactive', action='store_true', help='Load once; one independent question per line, /exit or EOF to quit')
    source.add_argument('--prompt', help='One Chinese or other UTF-8 question')
    source.add_argument('--prompt-file', type=Path, help='UTF-8 question file; - reads stdin')
    p.add_argument('--profile', choices=('windows', 'flow-experimental', 'memory-experimental'), default='windows',
                   help='windows 为默认配置；flow-experimental 在已测长输入和重复任务上受益，短问答可能更慢；memory-experimental 使用36 GB RAM缓存，内存余量更大，部分任务更慢')
    p.add_argument('--max-new-tokens', type=int, default=128, help='1..256; prompt plus output must fit context 1024')
    p.add_argument('--output-dir', type=Path, help='New result/log directory; defaults to build-deepseek-cuda/text-runs/<UTC time>')
    p.add_argument('--timeout', type=int, default=600, help='Seconds allowed for startup and separately for the request')
    args = p.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 256 or args.timeout <= 0:
        p.error('输出上限须为 1..256，超时须为正数。')
    request = None
    if not args.interactive:
        try:
            prompt = args.prompt if args.prompt is not None else (
                sys.stdin.read() if str(args.prompt_file) == '-' else args.prompt_file.read_text(encoding='utf-8'))
        except (OSError, UnicodeError) as error:
            p.error(str(error))
        if not prompt.strip():
            p.error('请输入非空问题。')
        request = {'id': 'text', 'category': 'User single-turn text', 'prompt': prompt, 'max_new_tokens': args.max_new_tokens}
    out = (args.output_dir or BUILD / 'text-runs' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    engine = ROOT / ('build-deepseek-first-window-sm75/strata-dsv41' if args.profile == 'flow-experimental'
                     else 'build-deepseek-diagnostics-sm75/strata-dsv41')
    engine_sha256 = benchmark_quantized.TEXT_FLOW_ENGINE_SHA256 if args.profile == 'flow-experimental' else ENGINE_SHA256
    if args.profile == 'memory-experimental':
        engine = ROOT / 'build-deepseek-retention-sm75/strata-dsv41'
        engine_sha256 = benchmark_quantized.TEXT_MEMORY_ENGINE_SHA256
    cache_gb = '36' if args.profile == 'memory-experimental' else '40'
    command = ['--smoke-result', str(BUILD / 'step7g-candidate-smoke/result.json'),
               '--output-dir', str(out), '--exe', str(engine),
               '--engine-sha256', engine_sha256, '--store-stats', '--ssd-cache-gb', cache_gb,
               '--gpu-cap-mib', '20480', '--timeout', str(args.timeout)]
    if args.profile in ('flow-experimental', 'memory-experimental'):
        command.extend(['--ssd-policy', 'lru', '--prefill-chunk', '256', '--native-verbose',
                        '--flow-store', 'on', '--mixed-prefill', 'off', '--first-window', 'off',
                        '--settle-store', 'on', '--settle-ms', '1000'])
    if args.profile == 'memory-experimental':
        command.extend(['--flow-store-min-heat', '1', '--resource-diagnostics'])
    stream = TextStream(sys.stdout)
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        print('正在加载模型；' + cache_gb + ' GB RAM 缓存，20 GiB 显存上限。记录：' + str(out), file=sys.stderr, flush=True)
        options = {'on_token': stream, 'expected_pack_sha256': PACK_SHA256}
        if args.profile == 'flow-experimental':
            print('使用实验配置 flow-experimental；短问答可能比默认配置更慢。', file=sys.stderr, flush=True)
            options['text_profile'] = args.profile
        if args.profile == 'memory-experimental':
            print('使用实验配置 memory-experimental；内存余量更大，部分任务更慢。', file=sys.stderr, flush=True)
            options['text_profile'] = args.profile
        if args.interactive:
            print('常驻独立问答：每行一个问题，输入 /exit 或 EOF 退出。', file=sys.stderr, flush=True)
            options.update(request_source=LineInput(args.max_new_tokens), on_case_end=stream.finish_case)
        else:
            options['request'] = request
        code = benchmark_quantized.main(command, **options)
        result = json.loads((out / 'result.json').read_text())
        if args.interactive:
            print('\n会话已结束，共完成 %d 个问题。' % len(result['cases']), file=sys.stderr, flush=True)
        else:
            stream.finish_case(result['cases'][0])
        return code
    except KeyboardInterrupt:
        print('\n已取消，推理进程与显存占位已清理。', file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        print('\n运行失败：' + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    sys.exit(main())
