"""Independent lossless trajectory compression; never removes originals."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import zstandard as zstd

ROOT = Path('/workspace/Cell-Free-DT-new/analysis/trajectory')
FILES = ['B1000_multicp_70/B.pkl', 'B2000_e180_uniform/B.pkl',
         'B40-05-05-3/B36/B.pkl', 'B40-05-05-3/B4/B.pkl',
         'B500_e450_uniform/B.pkl', 'uniform/A.pkl', 'uniform/B.pkl', 'uniform/C.pkl']
CHUNK = 8 * 1024**2
RESERVE = 1536 * 1024**2

def digest(reader):
    h = hashlib.sha256()
    size = 0
    while block := reader.read(CHUNK):
        h.update(block)
        size += len(block)
    return h.hexdigest(), size

def main():
    results = []
    for relative in FILES:
        source = ROOT / relative
        target = Path(str(source) + '.zst')
        assert source.is_file() and not source.is_symlink()
        before = source.stat()
        with source.open('rb') as stream:
            original_hash, original_size = digest(stream)
        if not target.exists():
            temporary = Path(str(target) + '.partial')
            assert not temporary.exists()
            with temporary.open('xb') as output, source.open('rb') as stream:
                with zstd.ZstdCompressor(level=3, write_checksum=True).stream_writer(output, closefd=False) as writer:
                    while block := stream.read(CHUNK):
                        assert shutil.disk_usage(ROOT).free > RESERVE + CHUNK, 'disk reserve reached'
                        writer.write(block)
                output.flush()
                os.fsync(output.fileno())
            with temporary.open('rb') as stream, zstd.ZstdDecompressor().stream_reader(stream) as reader:
                assert digest(reader) == (original_hash, original_size)
            assert not target.exists()
            os.link(temporary, target)
            temporary.unlink()
        assert not target.is_symlink()
        with target.open('rb') as stream, zstd.ZstdDecompressor().stream_reader(stream) as reader:
            assert digest(reader) == (original_hash, original_size)
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        result = dict(source=str(source), compressed=str(target), source_bytes=original_size,
                      compressed_bytes=target.stat().st_size, source_sha256=original_hash,
                      byte_identical=True, original_preserved=True)
        record = Path(str(target) + '.verification.json')
        if not record.exists():
            with record.open('x') as stream:
                json.dump(result, stream, indent=2)
        results.append(result)
        print('FILE_COMPLETE', json.dumps(result), flush=True)
    print('ALL_COMPLETE', json.dumps(dict(files=len(results), source_bytes=sum(x['source_bytes'] for x in results),
          compressed_bytes=sum(x['compressed_bytes'] for x in results), free_bytes=shutil.disk_usage(ROOT).free)), flush=True)

if __name__ == '__main__':
    main()
