"""Range-contract checks: a server must never silently trigger a full download."""
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/deepseek41'))
from prepare_quant_sample import checked_range, tensor_bytes, Ranges, FILES, MAX_REQUEST, MAX_TRANSFER
from gguf_reader import TensorInfo

class Response:
    def __init__(self,status=206,header='bytes 4-7/20',body=b'abcd',encoding='identity'):
        self.status=status;self.headers={'Content-Range':header,'Content-Encoding':encoding}
        self.body=io.BytesIO(body);self.reads=[]
    def read(self,n):self.reads.append(n);return self.body.read(n)

class RangesTest(unittest.TestCase):
    def test_exact_range(self):
        r=Response();self.assertEqual(checked_range(r,4,4,20),b'abcd');self.assertEqual(r.reads,[5])
    def test_full_response_is_rejected_before_body(self):
        r=Response(status=200)
        with self.assertRaises(ValueError):checked_range(r,4,4,20)
        self.assertEqual(r.reads,[])
    def test_wrong_ranges_and_compression(self):
        for header,encoding in [('bytes 0-3/20','identity'),('bytes 4-7/21','identity'),('bytes 4-7/*','identity'),('bytes 4-7/20','gzip')]:
            with self.subTest(header=header,encoding=encoding):
                r=Response(header=header,encoding=encoding)
                with self.assertRaises(ValueError):checked_range(r,4,4,20)
                self.assertEqual(r.reads,[])
    def test_short_and_oversized_body(self):
        for data in [b'abc',b'abcde']:
            with self.assertRaises(ValueError):checked_range(Response(body=data),4,4,20)
    def test_request_bounds_precede_network(self):
        f=next(iter(FILES))
        for offset,size in [(-1,1),(0,0),(0,MAX_REQUEST+1),(FILES[f]-1,2)]:
            with self.assertRaises(ValueError):Ranges().read(f,offset,size)
    def test_total_budget_precedes_network(self):
        r=Ranges();r.requested=MAX_TRANSFER
        with self.assertRaises(ValueError):r.read(next(iter(FILES)),0,1)
        self.assertEqual(r.records,[])
    def test_integer_metadata_sizes(self):
        for tid,name,shape,n in [(26,'I32',[129280],517120),(27,'I64',[24,2],384),(19,'IQ1_S',[5120,2304,384],884736000)]:
            self.assertEqual(tensor_bytes(TensorInfo('x',shape,tid,name,0)),n)

if __name__=='__main__':unittest.main()
