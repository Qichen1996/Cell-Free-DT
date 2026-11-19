"""Lossless Zstd trial, bounded disk use, no deletions or source rewrites."""
import argparse,hashlib,json,pickle,shutil,time
from pathlib import Path
import numpy as np
import zstandard as zstd

RESERVE=2*1024**3
CHUNK=8*1024**2

def main(path):
    source=Path(path).resolve();target=Path(str(source)+'.zst')
    assert source.is_file() and not target.exists()
    start=time.monotonic();size=0;sha=hashlib.sha256()
    obj=zstd.ZstdCompressor(level=3,write_checksum=True).compressobj()
    with source.open('rb') as f:
        while chunk:=f.read(CHUNK):
            sha.update(chunk);size+=len(obj.compress(chunk))
        size+=len(obj.flush())
    estimate_seconds=time.monotonic()-start
    assert shutil.disk_usage(source.parent).free>RESERVE+size+64*1024**2,('insufficient trial space',size)
    print('ESTIMATE',size,'SOURCE_BYTES',source.stat().st_size,'SECONDS',estimate_seconds,flush=True)
    start=time.monotonic()
    with source.open('rb') as f,target.open('xb') as out:
        with zstd.ZstdCompressor(level=3,write_checksum=True).stream_writer(out,closefd=False) as compressor:
            while chunk:=f.read(CHUNK):
                assert shutil.disk_usage(source.parent).free>RESERVE
                compressor.write(chunk)
    compress_seconds=time.monotonic()-start
    digest=hashlib.sha256();unpacked_size=0;start=time.monotonic()
    with target.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as reader:
        while chunk:=reader.read(CHUNK):digest.update(chunk);unpacked_size+=len(chunk)
    verify_seconds=time.monotonic()-start
    assert digest.hexdigest()==sha.hexdigest() and unpacked_size==source.stat().st_size
    timings={}
    for mode in ['raw','compressed','raw','compressed']:
        start=time.monotonic()
        with (source if mode=='raw' else target).open('rb') as f:
            if mode=='raw':data=pickle.load(f)
            else:
                with zstd.ZstdDecompressor().stream_reader(f) as reader:data=pickle.load(reader)
        timings.setdefault(mode,[]).append(time.monotonic()-start)
        assert len(data)==35
        for tr in data:
            for key in ['observations','actions','rewards']:
                assert len(tr[key])==10080 and np.isfinite(tr[key]).all()
        del data
    result={'source':str(source),'compressed':str(target),'level':3,'source_bytes':unpacked_size,
            'compressed_bytes':target.stat().st_size,'saved_percent':100*(1-target.stat().st_size/unpacked_size),
            'source_sha256':sha.hexdigest(),'decompressed_sha256':digest.hexdigest(),
            'byte_identical':True,'compress_seconds':compress_seconds,'verify_seconds':verify_seconds,
            'load_seconds':timings,'timing_note':'Same-process warm-cache trial; not cold disk benchmark.',
            'original_preserved':True,'temporary_uncompressed_file_created':False}
    with Path(str(target)+'.verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path');main(parser.parse_args().path)
