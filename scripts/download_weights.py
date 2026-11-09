#!/usr/bin/env python3
"""Download pinned model dependencies and verify all files against the manifest."""
import argparse, hashlib, json, shutil, tarfile, urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as out:
        shutil.copyfileobj(response,out)
    temp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT,help='Destination repository root')
    p.add_argument('--groups',nargs='+',choices=['FlashVSR','Qwen3.5-9B','stage1','planner_lora','atlas','metrics'])
    args=p.parse_args()
    manifest=json.loads((ROOT/'weights/manifest.json').read_text())
    archives={}
    for name,entry in manifest.items():
        if args.groups and name.split('/')[1] not in args.groups: continue
        target=args.root/name
        if target.is_file() and sha(target)==entry['sha256']:
            print('Verified',name,flush=True); continue
        if args.verify_only: raise RuntimeError(f'Missing or incorrect file: {name}')
        source=entry['source']
        target.parent.mkdir(parents=True,exist_ok=True)
        if source['type']=='huggingface':
            path=hf_hub_download(source['repo'],source['filename'],revision=source['revision'])
            shutil.copyfile(path,target)
        elif source['type']=='url':
            download(source['url'],target)
        elif source['type']=='tar':
            url=source['url']
            if url not in archives:
                path=args.root/'cache/downloads'/url.rsplit('/',1)[-1]
                if not path.is_file(): download(url,path)
                if source.get('archive_sha256') and sha(path)!=source['archive_sha256']:
                    raise RuntimeError(f'Archive checksum mismatch: {path}')
                archives[url]=path
            with tarfile.open(archives[url]) as tar:
                member=tar.getmember(source['member'])
                if not member.isfile(): raise ValueError('Expected a regular archive member')
                with tar.extractfile(member) as input_file,target.open('wb') as output_file:
                    shutil.copyfileobj(input_file,output_file)
        else: raise ValueError(f'Unknown download source: {source}')
        if sha(target)!=entry['sha256']: raise RuntimeError(f'Checksum mismatch: {name}')
        print('Downloaded and verified',name,flush=True)
    print('All requested weights verified')

if __name__=='__main__': main()
