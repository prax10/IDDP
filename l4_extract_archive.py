"""
Task L4, step 0: pull selected repositories out of the Public Jira Dataset's mongodump
archive WITHOUT MongoDB and WITHOUT unzipping the 5.8 GB zip (standard library only).

  python3 l4_extract_archive.py "<path to 2025-06-23 ThePublicJiraDataset.zip>" <out_dir> [Repo ...]
  default repos: MariaDB Qt Sakai JFrog IntelDAOS   (Jira repos that are NOT in TAWOS)

The archive is one gzip stream (mongodump --gzip --archive): magic, a prelude of
collection metadata, then interleaved blocks [namespace header, BSON documents..., -1].
Raw BSON documents of the kept collections are written to <out_dir>/<Repo>_NN.bson.gz.
Stops early once every kept collection has hit its EOF marker. ~5 minutes on a laptop.
"""
import gzip, json, os, struct, sys, time, zipfile, zlib

MEMBER = "ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"
CHUNK = 150 * 1024 * 1024


def parse(b, i=0):
    """minimal BSON decoder, enough for the archive's header documents"""
    n = struct.unpack_from("<i", b, i)[0]; end = i + n - 1; i += 4; d = {}
    while i < end:
        t = b[i]; i += 1
        j = b.index(0, i); k = b[i:j].decode(); i = j + 1
        if t == 2:
            L = struct.unpack_from("<i", b, i)[0]; d[k] = b[i + 4:i + 4 + L - 1].decode(errors="replace"); i += 4 + L
        elif t in (3, 4):
            L = struct.unpack_from("<i", b, i)[0]; d[k] = parse(b, i); i += L
        elif t == 8: d[k] = bool(b[i]); i += 1
        elif t in (0x12, 9, 0x11): d[k] = struct.unpack_from("<q", b, i)[0]; i += 8
        elif t == 0x10: d[k] = struct.unpack_from("<i", b, i)[0]; i += 4
        elif t == 1: d[k] = struct.unpack_from("<d", b, i)[0]; i += 8
        elif t in (0x0A, 6, 0xFF, 0x7F): d[k] = None
        elif t == 7: d[k] = b[i:i + 12].hex(); i += 12
        elif t == 5: L = struct.unpack_from("<i", b, i)[0]; d[k] = b"bin"; i += 5 + L
        else: raise ValueError(f"bson type {t}")
    return d


class Stream:
    def __init__(self, raw):
        self.raw, self.z, self.buf, self.pos, self.out = raw, zlib.decompressobj(31), b"", 0, 0

    def _fill(self, n):
        parts, have = [self.buf[self.pos:]], len(self.buf) - self.pos
        while have < n:
            c = self.raw.read(1 << 22)
            if not c:
                break
            d = self.z.decompress(c)
            while self.z.eof and self.z.unused_data:          # concatenated gzip members
                u = self.z.unused_data; self.z = zlib.decompressobj(31); d += self.z.decompress(u)
            parts.append(d); have += len(d)
        self.buf, self.pos = b"".join(parts), 0

    def read(self, n):
        if len(self.buf) - self.pos < n:
            self._fill(n)
        out = self.buf[self.pos:self.pos + n]; self.pos += len(out); self.out += len(out)
        return out


def read_doc(s):
    h = s.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack("<i", h)[0]
    return b"" if n == -1 else h + s.read(n - 4)


def main():
    zpath, out = sys.argv[1], sys.argv[2]
    keep = set(sys.argv[3:]) or {"MariaDB", "Qt", "Sakai", "JFrog", "IntelDAOS"}
    os.makedirs(out, exist_ok=True)
    s = Stream(zipfile.ZipFile(zpath).open(MEMBER))
    assert struct.unpack("<I", s.read(4))[0] == 0x8199e26d, "not a mongodump archive"
    while read_doc(s) != b"":
        pass
    files, part, counts, done, t0, last = {}, {}, {}, set(), time.time(), 0
    while True:
        h = read_doc(s)
        if h is None:
            break
        if h == b"":
            continue
        ns = parse(h); c = ns.get("collection")
        if ns.get("EOF") and c in keep:
            done.add(c); print(f"  {c}: complete ({counts.get(c, 0):,} issues)", flush=True)
            if done == keep:
                break
        while True:
            d = read_doc(s)
            if d is None or d == b"":
                break
            if c in keep:
                f = files.get(c)
                if f is None or f.fileobj.tell() > CHUNK:
                    if f: f.close()
                    part[c] = part.get(c, -1) + 1
                    f = files[c] = gzip.open(os.path.join(out, f"{c}_{part[c]:02d}.bson.gz"), "wb", compresslevel=3)
                f.write(d); counts[c] = counts.get(c, 0) + 1
        if s.out - last > 4e9:
            last = s.out
            print(f"{time.time() - t0:5.0f}s  {s.out / 1e9:5.1f} GB read  kept {counts}", flush=True)
        if d is None:
            break
    for f in files.values():
        f.close()
    json.dump(counts, open(os.path.join(out, "counts.json"), "w"))
    print(f"DONE in {time.time() - t0:.0f}s: {counts}")


if __name__ == "__main__":
    main()
