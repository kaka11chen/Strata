"""Pinned layer-1 Block after the independently validated step-6k Engram boundary.

Uses the first five text tokens of the hash-pinned Engram fixture, then original
Block/attention code with the existing CPU linear/attention/Sinkhorn oracles.
C++ supplies the declared native Q8_1 MoE for the final CPU residual join.
--discover reports all original expert IDs before they are fetched; local I/O only.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prepare_expert_fixture import Checkpoint,GGUFWriter,TYPE_IDS,_Tensor
from gguf_reader import GGUFFile
from prepare_routed_fixture import CONFIG_HASH,SOURCE_HASH,REVISION,prepare as prepare_routed
from prepare_mhc_fixture import KERNEL_HASH,split_sinkhorn
from prepare_shared_fixture import ReferenceLinear,quantize
from prepare_attention_fixture import sparse_attention

ENGRAM_FIXTURE_HASH='b4c633b6bfcf17d54a77d002744032891a2e2fe583b433497118594e1b604786'


def prepare(model_dir:Path,reference:Path,kernel:Path,engram_fixture:Path,
            output:Path|None,routed_output:Path|None,discover:bool=False)->dict:
    if not discover and (output is None or routed_output is None):
        raise ValueError('Both output paths are required')
    if not discover and (output.exists() or routed_output.exists()):
        raise FileExistsError('Fixture output already exists')
    for path,expected in [(model_dir/'config.json',CONFIG_HASH),(reference,SOURCE_HASH),
                          (kernel,KERNEL_HASH),(engram_fixture,ENGRAM_FIXTURE_HASH)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Not the pinned reference: {path}')
    torch.set_num_threads(2);TYPE_IDS.update(I8=24,BF16=30)
    checkpoint=Checkpoint(model_dir);provenance=json.loads((model_dir/'provenance.json').read_text())
    if provenance['model_revision']!=REVISION:raise ValueError('Wrong source revision')
    writer=GGUFWriter();records={}
    writer.add('general.architecture','deepseek41-layer1-decoder-fixture')
    for key,value in [('revision',REVISION),('official_source_sha256',SOURCE_HASH),('official_kernel_sha256',KERNEL_HASH),
                      ('engram_fixture_sha256',ENGRAM_FIXTURE_HASH),('mhc_profile','1017-hc4-delayed'),
                      ('attention_profile','1017-layer1-swa128-fp8-block32'),('shared_weight_encoding','e4m3fn-e8m0-block32')]:
        writer.add('strata.deepseek41.'+key,value)
    def source(name,dtype,shape,target=None):
        if checkpoint.dtype(name)!=dtype or checkpoint.shape(name)!=shape:raise ValueError('Malformed '+name)
        raw=checkpoint.raw(name).tobytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=provenance['tensors'][name]['sha256']:raise ValueError('Corrupt '+name)
        records[name]=digest
        if target:writer.tensors.append(_Tensor(target,list(reversed(shape)),dtype if dtype in {'F32','BF16'} else 'I8',raw))
        if dtype=='I8':return None
        tensor_dtype={'BF16':torch.bfloat16,'F32':torch.float32,'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu}[dtype]
        return torch.from_numpy(np.frombuffer(raw,np.uint8).copy()).view(tensor_dtype).reshape(shape)
    tree=ast.parse(reference.read_bytes());attention=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Attention')
    wrapper=ast.ClassDef(name='ReferenceAttention',bases=[ast.parse('nn.Module',mode='eval').body],keywords=[],
        body=[n for n in attention.body if isinstance(n,ast.FunctionDef) and n.name in {'forward','_window_kv'}],decorator_list=[])
    names={'RMSNorm','precompute_freqs_cis','apply_rotary_emb','get_window_topk_idxs','Gate','Block'}
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    def act_quant(x,block_size,*_):
        assert block_size==32
        x.copy_(torch.from_numpy(quantize(x)));return x
    ns={'torch':torch,'nn':nn,'F':F,'math':math,'lru_cache':lru_cache,'ModelArgs':SimpleNamespace,
        'EngramLayout':SimpleNamespace,'Attention':nn.Module,'linear':F.linear,'act_quant':act_quant,
        'fp8_block_size':32,'scale_fmt':'ue8m0','scale_dtype':torch.float8_e8m0fnu,'sparse_attn':sparse_attention,'hc_split_sinkhorn':split_sinkhorn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions+[wrapper],type_ignores=[])),str(reference),'exec'),ns)
    block=ns['Block'].__new__(ns['Block']);nn.Module.__init__(block)
    block.norm_eps,block.hc_mult,block.hc_sinkhorn_iters,block.hc_eps=1e-20,4,20,1e-6
    block.layer_id,block.engram=1,None
    attn=ns['ReferenceAttention']();attn.n_local_heads,attn.head_dim,attn.rope_head_dim=64,512,64
    attn.n_local_groups,attn.o_lora_rank,attn.window_size=8,1024,128
    attn.compress_ratio,attn.softmax_scale=0,512**-.5
    attn.window_kv_cache=torch.zeros(1,128,512,dtype=torch.bfloat16)
    attn.freqs_cis=ns['precompute_freqs_cis'](64,16,0,10000.,16.,32.,1.)
    for name,rows,cols in [('wq_a',1280,5120),('wq_b',32768,1280),('wkv',512,5120),('wo_a',8192,4096),('wo_b',5120,8192)]:
        prefix='layers.1.attn.'+name;target='blk.1.attn_'+name
        raw=source(prefix+'.weight','F8_E4M3',[rows,cols],target+'.weight')
        scale=source(prefix+'.scale','F8_E8M0',[rows//32,cols//32],target+'.scale')
        decoded=raw.float()*scale.float().repeat_interleave(32,0).repeat_interleave(32,1)
        if not torch.isfinite(decoded).all() or not torch.equal(decoded,decoded.bfloat16().float()):raise ValueError('Lossy BF16 expansion')
        linear=ReferenceLinear()
        if name=='wo_a':linear.weight=nn.Parameter(decoded.bfloat16(),requires_grad=False)
        else:linear.decoded=decoded.numpy()
        setattr(attn,name,linear)
    for name,width in [('q_norm',1280),('kv_norm',512)]:
        norm=ns['RMSNorm'](width,1e-20);norm.weight.data=source('layers.1.attn.'+name+'.weight','BF16',[width],'blk.1.attn_'+name+'.weight');setattr(attn,name,norm)
    attn.attn_sink=source('layers.1.attn.attn_sink','F32',[64],'blk.1.attn_sink');block.attn=attn
    for kind in ['attn','ffn']:
        for suffix,shape in [('fn',[24,20480]),('scale',[3]),('base',[24])]:
            name=f'hc_{kind}_{suffix}';setattr(block,name,source('layers.1.'+name,'F32',shape,'blk.1.'+name))
        name=kind+'_norm.weight';norm=ns['RMSNorm'](5120,1e-20)
        norm.weight.data=source('layers.1.'+name,'BF16',[5120],'blk.1.'+name);setattr(block,kind+'_norm',norm)
    for projection,role,rows,cols in [('w1','gate',2304,5120),('w3','up',2304,5120),('w2','down',5120,2304)]:
        for part,dtype,shape in [('weight','F8_E4M3',[rows,cols]),('scale','F8_E8M0',[rows//32,cols//32])]:
            source(f'layers.1.ffn.shared_experts.{projection}.{part}',dtype,shape,f'blk.1.ffn_{role}_shexp.{part}')
    args=SimpleNamespace(dim=5120,get_moe_config=lambda _:(384,6),score_func='sqrtsoftplus',gate_temp=1.,norm_topk_prob=True,route_scale=1.5,vision_enabled=False)
    gate=ns['Gate'](1,args);gate.weight.data=source('layers.1.ffn.gate.weight','BF16',[384,5120]).float();gate.bias.data=source('layers.1.ffn.gate.bias','F32',[384])
    captured={};selected=set();normalized_inputs=[];case_ids=[];joins=[]
    class CaptureMoE(nn.Module):
        def forward(self,x,image_mask):
            assert image_mask is None
            flat=x.reshape(-1,5120);weights,ids=gate(flat)
            captured.update(ffn_input=x.clone(),route_ids=ids,route_weights=weights)
            selected.update(ids.flatten().tolist());normalized_inputs.append(flat.clone())
            return torch.zeros_like(x) # Native Q8_1 MoE supplied by C++ composition oracle.
    block.ffn=CaptureMoE()
    def capture_post(self,x,residual,post,comb):
        result=ns['Block'].hc_post(self,x,residual,post,comb);joins.append((result.clone(),post.clone(),comb.clone()));return result
    block.hc_post=MethodType(capture_post,block)
    ef=GGUFFile(engram_fixture)
    def previous(name,dtype,count):
        t=next(t for t in ef.tensors if t.name==name);n=count*np.dtype(dtype).itemsize
        if t.type_id!=(0 if dtype=='<f4' else 24) or t.elements*(4 if t.type_id==0 else 1)!=n:raise ValueError('Malformed pinned Engram fixture')
        with ef.path.open('rb') as f:f.seek(ef.data_start+t.offset);raw=f.read(n)
        if len(raw)!=n:raise ValueError('Truncated pinned Engram fixture')
        return torch.from_numpy(np.frombuffer(raw,dtype).copy())
    residual=previous('input.residual','<f4',9*4*5120).reshape(1,9,4,5120)[:,:5].bfloat16()
    after_engram=previous('expected.output','<f4',9*4*5120).reshape(1,9,4,5120)[:,:5].bfloat16()
    tokens=previous('tokens','<i4',9)[:5];live=previous('live','u1',9)[:5]
    if not live.all():raise ValueError('Layer-1 decoder fixture is text-only')
    incoming=torch.tensor([[[1.,0.,0.,0.],[.1,.7,.2,.4],[.4,.2,.6,.8],[.9,.1,.3,.7],[.2,.4,.5,.3]]])
    def add(name,value):
        x=value.detach().float().contiguous();writer.tensors.append(_Tensor(name,list(reversed(x.shape)),'F32',x.numpy().tobytes()))
    with torch.inference_mode():
        for case,start,length in [(0,0,3),(1,3,1),(2,4,1)]:
            joins.clear();x=after_engram[:,start:start+length];pre=incoming[:,start:start+length]
            _,next_pre=block(x,start,pre,None)
            assert len(joins)==2
            for name,value in [('input',residual[:,start:start+length]),('incoming',pre),('after_engram',x),('next_pre',next_pre),
                ('after_attention',joins[0][0]),('ffn_input',captured['ffn_input']),('post',joins[1][1]),('comb',joins[1][2]),
                ('cache',attn.window_kv_cache),('route_ids',captured['route_ids']),('route_weights',captured['route_weights'])]:add(f'case.{case}.{name}',value)
            writer.tensors.append(_Tensor(f'case.{case}.tokens',[length*4],'I8',tokens[start:start+length].numpy().astype('<i4').tobytes()))
            case_ids.append(captured['route_ids'].tolist())
    report={'revision':REVISION,'engram_fixture_sha256':ENGRAM_FIXTURE_HASH,'populated_expert_ids':sorted(selected),'case_route_ids':case_ids,
            'scope':'pinned original Engram output -> original Block/CPU attention -> native Q8_1 MoE composition','torch_version':torch.__version__}
    if discover:return report
    for expert in selected:
        for projection,rows,cols in [('w1',2304,5120),('w3',2304,5120),('w2',5120,2304)]:
            for part,dtype,shape in [('weight','I8',[rows,cols//2]),('scale','F8_E8M0',[rows,cols//32])]:source(f'layers.1.ffn.experts.{expert}.{projection}.{part}',dtype,shape)
    routed=prepare_routed(model_dir,reference,routed_output,input_tensor=normalized_inputs[0][:1],extra_experts=tuple(sorted(selected)),layer=1)
    writer.add('strata.deepseek41.source_hashes',json.dumps(records,sort_keys=True));output.parent.mkdir(parents=True,exist_ok=True);writer.write(output)
    report.update(fixture=str(output),fixture_bytes=output.stat().st_size,fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),verified_source_tensors=len(records),routed_fixture=routed)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--reference-model',type=Path,required=True)
    p.add_argument('--reference-kernel',type=Path,required=True);p.add_argument('--engram-fixture',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--routed-output',type=Path);p.add_argument('--discover',action='store_true')
    a=p.parse_args();print(json.dumps(prepare(a.model_dir,a.reference_model,a.reference_kernel,a.engram_fixture,a.output,a.routed_output,a.discover),indent=2))
