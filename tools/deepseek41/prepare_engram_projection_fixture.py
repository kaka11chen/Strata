"""Pinned layer-1/14 Engram.forward oracle; local cached inputs only.

Original lookup/gate/residual graph, with the established independent NumPy
FP8 block-32 linear reference. Does not reproduce TileLang reduction order.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer

from prepare_expert_fixture import GGUFWriter, TYPE_IDS, _Tensor
from prepare_engram_fixture import ENGRAM_HASH, TOKENIZER_HASH, TABLES
from prepare_routed_fixture import CONFIG_HASH, SOURCE_HASH, REVISION
from prepare_shared_fixture import ReferenceLinear, quantize


def prepare(root: Path, output: Path, layer: int = 1) -> dict:
    if layer not in TABLES:
        raise ValueError("Only original Engram layers 1 and 14 are supported")
    table_rows, header_hash, data_start = TABLES[layer]
    if output.exists():
        raise FileExistsError(output)
    for name, expected in [("config.json",CONFIG_HASH),("inference_model.py",SOURCE_HASH),
                           ("inference_engram.py",ENGRAM_HASH),("tokenizer.json",TOKENIZER_HASH)]:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Not the pinned source: {name}")
    torch.set_num_threads(2)
    TYPE_IDS.update(I8=24,BF16=30)
    writer=GGUFWriter()
    writer.add("general.architecture","deepseek41-engram-projection-fixture")
    writer.add("strata.deepseek41.engram_projection_profile",f"1017-layer{layer}-wkv-fp8-block32")
    writer.add("strata.deepseek41.revision",REVISION)
    writer.add("strata.deepseek41.official_source_sha256",SOURCE_HASH)
    provenance=json.loads((root/f"engram{layer}-projection/provenance.json").read_text())
    if provenance['revision']!=REVISION:
        raise ValueError("Wrong projection cache revision")
    # Retain the existing layer-1 cache path; layer 14 reuses the pinned 6q header.
    header_path = root/"engram-header.json" if layer == 1 else root/"engram14-rows/source-header.json"
    header=json.loads(header_path.read_text())
    arrays={}
    for suffix,shape,dtype in [('q_weight',[4,5120],'BF16'),('k_weight',[4,5120],'BF16'),
                              ('wkv.weight',[25600,6144],'F8_E4M3'),('wkv.scale',[800,192],'F8_E8M0')]:
        name=f'layers.{layer}.engram.'+suffix; rec=provenance['tensors'][name]
        data=(root/f'engram{layer}-projection'/rec['file']).read_bytes()
        if rec['dtype']!=dtype or rec['shape']!=shape or rec['source_header_sha256']!=header_hash or rec['source_offsets']!=header[name]['data_offsets'] or hashlib.sha256(data).hexdigest()!=rec['sha256']:
            raise ValueError(f"Corrupt original projection tensor: {name}")
        actual_dtype={'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu}[dtype]
        arrays[suffix]=torch.from_numpy(np.frombuffer(data,np.uint8).copy()).view(actual_dtype).reshape(shape)
        writer.tensors.append(_Tensor(f'blk.{layer}.engram.'+suffix,list(reversed(shape)),'BF16' if dtype=='BF16' else 'I8',data))
    writer.add('strata.deepseek41.projection_provenance',json.dumps(provenance,sort_keys=True))
    cfg=json.loads((root/'config.json').read_text())['text_config']
    args=SimpleNamespace(**cfg,engram_pad_id=cfg['engram_pad_token_id'],max_batch_size=1,max_seq_len=1024)
    source=runpy.run_path(str(root/'inference_engram.py')); backend=Tokenizer.from_file(str(root/'tokenizer.json'))
    class TokenizerAdapter:
        backend_tokenizer=backend
        def __len__(self): return backend.get_vocab_size()
    layout=source['EngramLayout'].from_args(args)
    if (layout.layer_ids != (1,14) or layout.num_embeddings != (TABLES[1][0],TABLES[14][0]) or
            layout.max_ngram_size != 4 or layout.n_heads != 8 or layout.head_dim != 256):
        raise ValueError("Pinned Engram layout changed")
    column=layout.layer_ids.index(layer)
    state=source['NgramHashState'](args,layout,TokenizerAdapter())
    tokens=torch.tensor([[0,455,1805,1,20852,525,21716,240,129279]])
    live=torch.tensor([[1,1,1,1,1,0,1,1,1]],dtype=torch.bool)
    hashes=state(tokens,0,live)[:,:,column]
    ids=torch.unique(hashes,sorted=True)
    rows=json.loads((root/f'engram{layer}-rows/provenance.json').read_text())
    if rows['revision']!=REVISION or rows['header_sha256']!=header_hash:
        raise ValueError('Wrong lookup provenance')
    payload=[]
    for row in ids.tolist():
        raw=(root/f'engram{layer}-rows/{row}.bin').read_bytes(); rec=rows['rows'][str(row)]
        if len(raw)!=264 or hashlib.sha256(raw).hexdigest()!=rec['sha256'] or rec['weight_offset']!=data_start+row*256 or rec['scale_offset']!=data_start+table_rows*256+row*8:
            raise ValueError('Corrupt original lookup row')
        payload.append(raw)
    captured={}
    class CaptureTorch:
        def __getattr__(self,name): return getattr(torch,name)
        def sigmoid(self,x):
            result=torch.sigmoid(x); captured['gate']=result.clone(); return result
    def sparse_embedding(indices,weight):
        positions=torch.searchsorted(ids,indices)
        if (positions>=len(ids)).any() or not torch.equal(ids[positions],indices):
            raise ValueError('Unpopulated row requested')
        return torch.nn.functional.embedding(positions,weight)
    tree=ast.parse((root/'inference_model.py').read_bytes())
    definitions=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'ParallelEngramEmbedding','Engram'}]
    ns={'torch':CaptureTorch(),'nn':nn,'F':SimpleNamespace(embedding=sparse_embedding),'world_size':1,
        'ModelArgs':SimpleNamespace,'EngramLayout':SimpleNamespace}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(root/'inference_model.py'),'exec'),ns)
    embed=ns['ParallelEngramEmbedding'].__new__(ns['ParallelEngramEmbedding']); nn.Module.__init__(embed)
    embed.vocab_start_idx,embed.vocab_end_idx,embed.block_size=0,table_rows,32
    embed.weight=nn.Parameter(torch.from_numpy(np.stack([np.frombuffer(p[:256],np.uint8) for p in payload]).copy()).view(torch.float8_e4m3fn),requires_grad=False)
    embed.scale=nn.Parameter(torch.from_numpy(np.stack([np.frombuffer(p[256:],np.uint8) for p in payload]).copy()).view(torch.float8_e8m0fnu),requires_grad=False)
    block=ns['Engram'].__new__(ns['Engram']); nn.Module.__init__(block)
    block.dim,block.hc_mult,block.eps,block.clamp_value=5120,4,1e-20,1e-6
    block.embed=embed; block.wkv=ReferenceLinear()
    decoded=np.empty((25600,6144),np.float32)
    for first in range(0,25600,4096):
        n=min(4096,25600-first)
        values=arrays['wkv.weight'][first:first+n].float()
        scales=arrays['wkv.scale'][first//32:(first+n)//32].float().repeat_interleave(32,0).repeat_interleave(32,1)
        values*=scales
        if not torch.isfinite(values).all() or not torch.equal(values,values.bfloat16().float()):
            raise ValueError('Projection weights do not restore losslessly to BF16')
        decoded[first:first+n]=values.numpy()
    block.wkv.decoded=decoded
    block.wkv.register_forward_hook(lambda _m,_x,y:captured.update(kv=y.clone()))
    block.q_weight=nn.Parameter(arrays['q_weight'],requires_grad=False)
    block.k_weight=nn.Parameter(arrays['k_weight'],requires_grad=False)
    residual=torch.randn((1,9,4,5120),generator=torch.Generator().manual_seed(611017))
    residual*=torch.tensor([.125,1.,8.,2.]).view(1,1,4,1); residual=residual.bfloat16()
    residual[:,0]=0
    with torch.inference_mode():
        lookup=embed(hashes).flatten(-2)
        expected=block(residual,hashes,live)
    def add(name,x,dtype='F32'):
        raw=x.float().numpy().tobytes() if dtype=='F32' else x.contiguous().view(torch.uint16).numpy().tobytes()
        writer.tensors.append(_Tensor(name,list(reversed(x.shape)),dtype,raw))
    add('input.residual',residual); add('input.lookup',lookup)
    add('expected.quantized',torch.from_numpy(quantize(lookup)).bfloat16(),'BF16')
    add('expected.kv',captured['kv']); add('expected.gate',captured['gate'].masked_fill(~live.unsqueeze(-1),0))
    add('expected.output',expected)
    for name,x,dtype in [('tokens',tokens,'<i4'),('live',live,'u1'),('rows',hashes,'<u4')]:
        raw=x.numpy().astype(dtype).tobytes(); writer.tensors.append(_Tensor(name,[len(raw)],'I8',raw))
    output.parent.mkdir(parents=True,exist_ok=True); writer.write(output)
    return {'layer':layer,'revision':REVISION,'source_sha256':SOURCE_HASH,'fixture':str(output),'fixture_bytes':output.stat().st_size,
            'fixture_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'original_projection':provenance,
            'tokens':9,'lookup_rows':len(ids),'torch_version':torch.__version__,
            'scope':'original Engram.forward with original sparse lookup and independent NumPy FP8 block32 linear; no official GPU reduction-order parity'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,choices=(1,14),default=1)
    args=p.parse_args(); print(json.dumps(prepare(args.reference_root,args.output,args.layer),indent=2))
