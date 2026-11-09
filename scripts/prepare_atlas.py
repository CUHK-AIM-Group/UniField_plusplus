#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unifield.registration import register

def main():
    p=argparse.ArgumentParser(description='Register atlas labels to each low-field MRI')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--list',nargs='+',help='Training/validation CSV files: low_field,high_field')
    group.add_argument('--input',help='One low-field NIfTI volume')
    p.add_argument('--resources',default='weights/atlas')
    p.add_argument('--cache',default='cache/atlas')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    files=[args.input] if args.input else []
    for name in args.list or []:
        with open(name,newline='') as f:
            files.extend(row[0].strip() for row in csv.reader(f) if row and not row[0].lstrip().startswith('#'))
    for source in dict.fromkeys(files):
        print(f'Registering {source}',flush=True)
        register(source,args.resources,args.cache,args.threads)
    print('Registration complete',flush=True)

if __name__=='__main__': main()
