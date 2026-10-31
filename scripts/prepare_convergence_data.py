"""Verify existing cached parquet files and save offline benchmark datasets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from datasets import Image, load_dataset
    for dataset,folder,manifest in [('cifar10','datasets','cifar10-hf-manifest.json'),
                                    ('tiny_imagenet','tiny-datasets','tiny-manifest.json')]:
        source=args.legacy_root/folder
        metadata=json.loads((source/manifest).read_text())
        files={}
        for split in ('train','test') if dataset=='cifar10' else ('train','valid'):
            if dataset=='cifar10':
                path=source/f'{split}.parquet';expected=metadata['sha256'][split]
            else:
                filename,expected,_=metadata['files'][split];path=source/filename
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            if digest!=expected:
                raise ValueError(f'Cached data checksum mismatch: {path}')
            files[split]=str(path)
        target=args.output/dataset
        if target.exists():
            raise ValueError(f'Refusing to overwrite prepared dataset: {target}')
        loaded=load_dataset('parquet',data_files=files,cache_dir=str(args.output/'cache'))
        if dataset=='tiny_imagenet':
            # These original parquet files store {bytes,path} structs without
            # the Hugging Face Image feature metadata needed for PIL decoding.
            loaded=loaded.cast_column('image',Image())
        loaded.save_to_disk(str(target))
        provenance=dict(source=metadata,files=files,
                        fingerprints={split:data._fingerprint for split,data in loaded.items()},
                        splits={split:len(data) for split,data in loaded.items()})
        (target/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(json.dumps(dict(dataset=dataset,**provenance)),flush=True)


if __name__=='__main__':
    main()
