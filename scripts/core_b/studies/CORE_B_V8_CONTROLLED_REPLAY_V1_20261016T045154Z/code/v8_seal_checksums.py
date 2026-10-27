#!/usr/bin/env python3
"""Generate the final non-recursive SHA-256 manifest for V8 content artifacts."""
from __future__ import annotations
import hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'CHECKSUMS.sha256','CHECKSUMS_MANIFEST.json','coordination/OWNER_STATE.json','coordination/OWNER_RELEASED.json'}
def digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 partial=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and (p.name.endswith('.part') or p.name.endswith('.tmp'))]
 if partial:raise RuntimeError(f'partial temporary artifacts remain: {partial[:10]}')
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).as_posix() not in EXCLUDED)
 lines=[f'{digest(p)}  {p.relative_to(ROOT).as_posix()}' for p in files]
 out=ROOT/'CHECKSUMS.sha256';tmp=out.with_name('.CHECKSUMS.sha256.part');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(tmp,out)
 print(f'files={len(files)} checksum_sha256={digest(out)} excluded={sorted(EXCLUDED)}')
if __name__=='__main__':main()
