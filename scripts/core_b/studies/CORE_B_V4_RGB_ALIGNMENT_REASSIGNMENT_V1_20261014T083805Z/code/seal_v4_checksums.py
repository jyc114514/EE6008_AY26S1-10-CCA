#!/usr/bin/env python3
"""Generate and immediately verify the final V4 SHA-256 manifest."""
from __future__ import annotations
import hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'CHECKSUMS.sha256'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError('final V4 checksum already exists; refusing to replace')
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=OUT)
 parts=[str(p.relative_to(ROOT)) for p in files if p.name.endswith('.part') or p.name.endswith('.tmp')]
 if parts:raise RuntimeError(f'unfinished partial artifacts remain: {parts[:20]}')
 lines=[f'{sha(p)}  {p.relative_to(ROOT).as_posix()}' for p in files]
 tmp=ROOT/'.CHECKSUMS.sha256.part'
 with tmp.open('w',encoding='utf-8',newline='\n') as f:
  f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,OUT)
 # Verify the final manifest itself after it is written. Do not write any other V4 file after this point.
 checked=0
 for line in OUT.read_text(encoding='utf-8').splitlines():
  expected,rel=line.split('  ',1);p=ROOT/rel
  if not p.is_file() or sha(p)!=expected:raise RuntimeError(f'checksum verification failed: {rel}')
  checked+=1
 if checked!=len(files):raise RuntimeError('checksum entry count mismatch')
 print(f'CHECKSUM_PASS entries={checked} sha256_manifest={sha(OUT)} self_excluded=true')
if __name__=='__main__':main()
