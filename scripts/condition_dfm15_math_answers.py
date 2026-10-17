"""Conservative, mode-preserving ANSWER-format exposure in the full DFM15 sample."""
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from tokenizers import Tokenizer

from build_dfm15_full_sample import ROOT, FIELDS, TARGET, append_tokens, write_json, release_training, local

WORK = ROOT/'data/dfm15/math-answer-conditioning-v2'
VERSION = 2
INSTRUCTION = '\n\nPut your final answer on its own line in the form ANSWER: <number>.'
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
ENDING = re.compile(r'(?:\\boxed\{\s*(?P<box>'+NUMBER+r')\s*\}|####\s*(?P<hash>'+NUMBER+r'))'
                    r'(?P<suffix>\s*(?:\$\$?|\\\]|\\\))?\s*[.*]*\s*)$')
CONSTRAINT = re.compile(r'\\boxed|\bboxed\b|\bfbox\b|ANSWER\s*:|####|\bJSON\b|\bformat\b|'
                        r'\b(?:return|output|give|provide|respond|answer)\s+(?:with\s+)?only\b|'
                        r'\bonly\s+(?:give|return|output|provide|respond|answer)\b|'
                        r'\bprove\b|\bproof\b|\btool\b|\bcode\b', re.I)
PREFIXES = ('allenai_open_math_', 'allenai_rlvr_gsm__', 'allenai_rlvr_math__',
            'allenai_tulu_3_personas_math__', 'openmathinstruct2_repaired__')


def condition(prompt, response):
    """Never infer answers, change reasoning modes, or override explicit formats."""
    if prompt.count('<|turn>user\n') != 1 or '<|tool' in prompt+response:
        return None, 'not_single_turn_math'
    pos = prompt.rfind('<|turn>user\n')+len('<|turn>user\n')
    end = prompt.find('<turn|>', pos)
    if end < pos or CONSTRAINT.search(prompt):
        return None, 'existing_constraint'
    if response.count('<turn|>') != 1 or not response.rstrip().endswith('<turn|>'):
        return None, 'nonterminal_response'
    body = response[:response.rfind('<turn|>')].rstrip()
    channel = body.rfind('<channel|>')
    if '<|channel>thought' in response and channel < 0:
        return None, 'answer_inside_thought'
    final = body[channel+len('<channel|>'):] if channel >= 0 else body
    match = ENDING.search(final)
    direct = re.fullmatch(NUMBER, final.strip())
    if match is None and direct is None:
        return None, 'no_unambiguous_numeric_final'
    answer = ((match['box'] or match['hash']) if match else direct[0]).replace(',', '')
    Decimal(answer)
    # Preserve worked solutions verbatim; append an explicitly requested final line.
    # A bare direct answer needs no duplicated answer, and native thoughts stay intact.
    before = body if not direct else (body[:channel+len('<channel|>')] if channel >= 0 else '')
    revised = before + ('\n' if before else '') + 'ANSWER: '+answer+'<turn|>\n'
    revised_prompt = prompt[:end]+INSTRUCTION+prompt[end:]
    assert revised.count('<|channel>thought') == response.count('<|channel>thought')
    return (revised_prompt, revised, answer), 'answer_format'


def initialize():
    global TOKENIZER, TOKENS
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZER = Tokenizer.from_file(str(ROOT.parent/'brainsurgery/models/gemma4_31b/tokenizer.json'))
    TOKENS = np.load(TARGET/'tokens.npy', mmap_mode='r')


def convert(row):
    a,b,c,d = map(int, row)
    prompt = TOKENIZER.decode(TOKENS[a:a+b].tolist(), skip_special_tokens=False)
    response = TOKENIZER.decode(TOKENS[c:c+d].tolist(), skip_special_tokens=False)
    result, reason = condition(prompt, response)
    if result is None:
        return row, None, reason
    p,r,answer = result
    pi = TOKENIZER.encode(p, add_special_tokens=False).ids
    ri = TOKENIZER.encode(r, add_special_tokens=False).ids
    if len(pi)+len(ri)>4097:
        return row, None, 'would_exceed_context'
    return row, (pi,ri,answer,p,r), reason


def source_ranges():
    path = WORK/'source-ranges.json'
    if path.exists():
        return json.loads(path.read_text())
    previous = ROOT/'data/dfm15/math-answer-conditioning/source-ranges.json'
    if previous.exists():
        out = json.loads(previous.read_text())
        write_json(path, out)
        return out
    # Read only source boundaries on the owning machine; do not copy its reservoirs.
    code = '''from pathlib import Path
import json,numpy as np
r=Path('/work/dfm/HRM-Text');offset=0;out=[]
prefixes=PREFIXES
for p in sorted((r/'data/tokenized_dfm11').iterdir()):
 if not p.is_dir():continue
 n=len(np.load(p/'tokens.npy',mmap_mode='r'))
 if p.name.startswith(prefixes) and n:out.append(dict(name=p.name,start=offset,end=offset+n))
 offset+=n
assert offset==len(np.load(r/'data/sampled_dfm11/tokens.npy',mmap_mode='r'))
root=r/'data/dfm15/native-reasoning-v2'
offset=json.loads((root/'sampled_dfm15/token-copy.json').read_text())['base_tokens']
for p in json.loads((root/'repairs.json').read_text())['parts']:
 if p['name'].startswith(prefixes):out.append(dict(name=p['name']+':native',start=offset,end=offset+p['new_tokens']))
 offset+=p['new_tokens']
print(json.dumps(out))
'''.replace('PREFIXES', repr(PREFIXES))
    out = json.loads(subprocess.check_output(['ssh','-p','6768','-o','BatchMode=yes',
        'ucloud@ssh.cloud.sdu.dk', '/home/ucloud/miniforge3/envs/hrm/bin/python -'], input=code,text=True))
    spec = json.loads((ROOT/'config/dfm15_sources.json').read_text())
    parts = [(e,p,np.load(local(p['path'])/'tokens.npy',mmap_mode='r'))
             for e in spec['additions'] for p in e['parts']]
    offset = json.loads((WORK/'before.json').read_text())['stored_tokens']-sum(len(t) for _,_,t in parts)
    for e,p,t in parts:
        if e['name']=='gsm-symbolic-en-train':out.append(dict(name=e['name'],start=offset,end=offset+len(t)))
        offset += len(t)
    write_json(path, out)
    return out


def prepare():
    if (WORK/'mapping.npz').exists():return
    ranges = sorted(source_ranges(), key=lambda x:x['start'])
    starts = np.array([x['start'] for x in ranges],dtype=np.uint64)
    ends = np.array([x['end'] for x in ranges],dtype=np.uint64)
    candidates=[]; census=Counter()
    for epoch in range(3):
        arrays={k:np.load(TARGET/f'epoch_{epoch}'/(k+'.npy'),mmap_mode='r') for k in FIELDS}
        for off in range(0,len(arrays['inst_start']),1_000_000):
            a={k:v[off:off+1_000_000] for k,v in arrays.items()};s=a['inst_start']
            idx=np.searchsorted(starts,s,side='right')-1
            mask=(idx>=0)&(s<ends[np.maximum(idx,0)])
            census[f'epoch_{epoch}_source_occurrences']+=int(mask.sum())
            # A fixed 10% hash cohort, independent of mode and source ordering.
            hashed=s.copy();hashed^=hashed>>np.uint64(33);hashed*=np.uint64(0xff51afd7ed558ccd);hashed^=hashed>>np.uint64(33)
            mask &= hashed%np.uint64(10)==0
            if mask.any():candidates.append(np.column_stack([a[k][mask] for k in FIELDS]))
    rows=np.unique(np.concatenate(candidates),axis=0)
    assert len(np.unique(rows[:,0]))==len(rows), 'Ambiguous prompt pointer mapping'
    print('Candidate census',dict(census),'unique cohort',len(rows),flush=True)
    old=[];new=[];offset=json.loads((WORK/'before.json').read_text())['stored_tokens']
    counts=Counter()
    with (WORK/'tokens.raw').open('wb') as raw, (WORK/'examples.jsonl').open('w') as examples:
        with ProcessPoolExecutor(max_workers=16,initializer=initialize) as pool:
            for i,(row,result,reason) in enumerate(pool.map(convert,rows,chunksize=128)):
                counts[reason]+=1
                if result:
                    pi,ri,answer,p,r=result
                    old.append(row);new.append([offset,len(pi),offset+len(pi),len(ri)])
                    np.asarray(pi+ri,dtype=np.int32).tofile(raw);offset+=len(pi)+len(ri)
                    if len(old)<=25 or len(old)%1000==0:
                        examples.write(json.dumps(dict(old=list(map(int,row)),prompt=p,response=r,answer=answer),ensure_ascii=False)+'\n')
                if i%10000==0:print('Converted',i,dict(counts),flush=True)
    assert len(old)>=50000, 'Insufficient format coverage; inspect before releasing training'
    raw=np.memmap(WORK/'tokens.raw',dtype=np.int32,mode='r')
    dest=np.lib.format.open_memmap(WORK/'tokens.npy',mode='w+',dtype=np.int32,shape=raw.shape)
    dest[:]=raw;dest.flush();del dest,raw
    np.savez(WORK/'mapping.npz',old=np.array(old,dtype=np.uint64),new=np.array(new,dtype=np.uint64))
    write_json(WORK/'preparation.json',dict(version=VERSION,census=dict(census),counts=dict(counts),changed_unique=len(old),cohort_fraction=0.1))


def publish():
    before=json.loads((WORK/'before.json').read_text())
    variants=np.load(WORK/'tokens.npy',mmap_mode='r')
    stored=append_tokens(TARGET/'tokens.npy',[variants],before['stored_tokens'])
    mapping=np.load(WORK/'mapping.npz');old,new=mapping['old'],mapping['new']
    reports=[]
    for epoch in range(3):
        receipt=WORK/f'epoch_{epoch}.json'
        if receipt.exists():reports.append(json.loads(receipt.read_text()));continue
        folder=TARGET/f'epoch_{epoch}'
        commit=WORK/f'epoch_{epoch}.commit.json'
        if commit.exists():
            report=json.loads(commit.read_text())
            for k in FIELDS:
                staged=folder/(k+'.conditioning.npy')
                if staged.exists():os.replace(staged,folder/(k+'.npy'))
            write_json(receipt,report);reports.append(report);continue
        a={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in FIELDS}
        staged={k:np.lib.format.open_memmap(folder/(k+'.conditioning.npy'),mode='w+',dtype=v.dtype,shape=v.shape)
                for k,v in a.items()}
        changed=total=0
        for off in range(0,len(a['inst_start']),1_000_000):
            chunk=np.column_stack([v[off:off+1_000_000] for v in a.values()])
            idx=np.searchsorted(old[:,0],chunk[:,0]);safe=np.minimum(idx,len(old)-1)
            mask=(idx<len(old))&np.all(old[safe]==chunk,axis=1)
            chunk[mask]=new[idx[mask]];changed+=int(mask.sum())
            assert np.all(chunk[:,1]+chunk[:,3]<=4097) and np.all(chunk[:,3]>0)
            assert np.all(chunk[:,0]+chunk[:,1]<=stored) and np.all(chunk[:,2]+chunk[:,3]<=stored)
            for i,k in enumerate(FIELDS):staged[k][off:off+len(chunk)]=chunk[:,i]
            total+=int(chunk[:,1].sum())+int(chunk[:,3].sum())
        for v in staged.values():v.flush()
        report=dict(epoch=epoch,rows=len(a['inst_start']),tokens=total,conditioned_occurrences=changed)
        del staged,a
        for k in FIELDS:
            with (folder/(k+'.conditioning.npy')).open('rb') as f:os.fsync(f.fileno())
        write_json(commit,report)
        for k in FIELDS:os.replace(folder/(k+'.conditioning.npy'),folder/(k+'.npy'))
        write_json(receipt,report);reports.append(report);print('Verified',report,flush=True)
    metadata=json.loads((TARGET/'metadata.json').read_text())
    metadata['total_length']=round(sum(r['tokens'] for r in reports)/3)
    metadata['math_answer_conditioning']='numeric_only_10pct_hash_cohort_mode_preserved_v2'
    write_json(TARGET/'metadata.json',metadata)
    final=dict(before,epochs=reports,stored_tokens=stored,math_answer_conditioning=json.loads((WORK/'preparation.json').read_text()))
    final['metadata_sha256']=hashlib.sha256((TARGET/'metadata.json').read_bytes()).hexdigest()
    final['status']='full_registered_dfm15_math_conditioning_verified'
    for name in ('full-mixture-verified.json','local-transfer-verified.json','ready.json','validated.json'):
        write_json(TARGET/name,final)
    write_json(TARGET/'math-answer-conditioning-verified.json',final['math_answer_conditioning'])
    (WORK/'tokens.raw').unlink(missing_ok=True)
    (WORK/'tokens.npy').unlink(missing_ok=True)


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    receipt=TARGET/'math-answer-conditioning-verified.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('version') == VERSION:
        return release_training()
    if not (WORK/'before.json').exists():write_json(WORK/'before.json',json.loads((TARGET/'full-mixture-verified.json').read_text()))
    prepare();publish();release_training()


if __name__=='__main__':
    with (WORK.parent/'.math-conditioning.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
