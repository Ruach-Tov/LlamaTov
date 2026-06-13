#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""BAM output writer — compressed binary alignment format.

BAM = BGZF-compressed SAM. BGZF = gzip with fixed block boundaries
so random access works. We implement just enough BAM to produce
files readable by samtools.

Usage:
  from bpd_bam_writer import BamWriter
  with BamWriter("output.bam", [("chr1", 10000)]) as bam:
      bam.write_alignment("read1", 0, "chr1", 100, 60, "100M", "ACGT...", "IIII...")
"""
import struct
import gzip
import io
import os

class BamWriter:
    """Minimal BAM writer compatible with samtools."""
    
    BGZF_BLOCK_SIZE = 65536
    
    def __init__(self, path, references, program_name="bpd_smith_waterman"):
        """
        references: list of (name, length) tuples
        """
        self.path = path
        self.references = references
        self.ref_map = {name: i for i, (name, _) in enumerate(references)}
        self.program_name = program_name
        self.f = open(path, 'wb')
        self._write_header()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def _bgzf_block(self, data):
        """Compress data into a BGZF block."""
        # BGZF is gzip with extra field indicating block size
        buf = io.BytesIO()
        # Write gzip header with BGZF extra field
        buf.write(b'\x1f\x8b')  # gzip magic
        buf.write(b'\x08')       # compression method (deflate)
        buf.write(b'\x04')       # FLG: FEXTRA set
        buf.write(b'\x00\x00\x00\x00')  # MTIME
        buf.write(b'\x00')       # XFL
        buf.write(b'\xff')       # OS (unknown)
        # Extra field: BGZF uses BC subfield
        buf.write(struct.pack('<H', 6))  # XLEN
        buf.write(b'BC')         # subfield ID
        buf.write(struct.pack('<H', 2))  # subfield length
        # BSIZE-1 placeholder — fill in after compression
        bsize_pos = buf.tell()
        buf.write(struct.pack('<H', 0))  # placeholder
        
        # Compress data
        import zlib
        compressed = zlib.compress(data, 6)[2:-4]  # strip zlib header/footer
        buf.write(compressed)
        
        # CRC32 and ISIZE
        crc = zlib.crc32(data) & 0xffffffff
        buf.write(struct.pack('<I', crc))
        buf.write(struct.pack('<I', len(data) & 0xffffffff))
        
        # Fill in BSIZE
        block = buf.getvalue()
        bsize = len(block) - 1
        block = block[:bsize_pos] + struct.pack('<H', bsize) + block[bsize_pos+2:]
        
        return block
    
    def _write_header(self):
        """Write BAM header."""
        # Build SAM header text
        header_text = "@HD\tVN:1.6\tSO:unsorted\n"
        for name, length in self.references:
            header_text += "@SQ\tSN:%s\tLN:%d\n" % (name, length)
        header_text += "@PG\tID:bpd-sw\tPN:%s\tVN:1.0\n" % self.program_name
        header_bytes = header_text.encode()
        
        # BAM header: magic + header_length + header_text + n_ref + ref_info
        buf = io.BytesIO()
        buf.write(b'BAM\x01')                          # magic
        buf.write(struct.pack('<i', len(header_bytes))) # header length
        buf.write(header_bytes)                         # header text
        buf.write(struct.pack('<i', len(self.references)))  # n_ref
        for name, length in self.references:
            name_bytes = name.encode() + b'\x00'
            buf.write(struct.pack('<i', len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack('<i', length))
        
        self.f.write(self._bgzf_block(buf.getvalue()))
    
    def _encode_cigar(self, cigar_str):
        """Encode CIGAR string to BAM format (array of uint32)."""
        import re
        ops = {'M': 0, 'I': 1, 'D': 2, 'N': 3, 'S': 4, 'H': 5, 'P': 6, '=': 7, 'X': 8}
        encoded = []
        for count, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar_str):
            encoded.append((int(count) << 4) | ops[op])
        return encoded
    
    def _encode_seq(self, seq):
        """Encode sequence to BAM 4-bit format."""
        base_map = {'=': 0, 'A': 1, 'C': 2, 'M': 3, 'G': 4, 'R': 5, 'S': 6,
                     'V': 7, 'T': 8, 'W': 9, 'Y': 10, 'H': 11, 'K': 12,
                     'D': 13, 'B': 14, 'N': 15}
        encoded = []
        for i in range(0, len(seq), 2):
            high = base_map.get(seq[i].upper(), 15)
            low = base_map.get(seq[i+1].upper(), 15) if i+1 < len(seq) else 0
            encoded.append((high << 4) | low)
        return bytes(encoded)
    
    def write_alignment(self, qname, flag, rname, pos, mapq, cigar, seq, qual, tags=None):
        """Write one alignment record."""
        ref_id = self.ref_map.get(rname, -1)
        cigar_encoded = self._encode_cigar(cigar) if cigar != '*' else []
        seq_encoded = self._encode_seq(seq) if seq != '*' else b''
        qual_encoded = bytes(ord(c) - 33 for c in qual) if qual != '*' else b'\xff' * len(seq)
        qname_bytes = qname.encode() + b'\x00'
        
        # Compute block_size
        block_size = (32 +  # fixed fields
                     len(qname_bytes) +
                     len(cigar_encoded) * 4 +
                     len(seq_encoded) +
                     len(qual_encoded))
        
        # Add tags
        tag_bytes = b''
        if tags:
            for tag, tag_type, value in tags:
                tag_bytes += tag.encode()
                tag_bytes += tag_type.encode()
                if tag_type == 'i':
                    tag_bytes += struct.pack('<i', value)
                elif tag_type == 'Z':
                    tag_bytes += value.encode() + b'\x00'
            block_size += len(tag_bytes)
        
        buf = io.BytesIO()
        buf.write(struct.pack('<i', block_size))
        buf.write(struct.pack('<i', ref_id))             # refID
        buf.write(struct.pack('<i', pos))                 # pos (0-based)
        
        name_len = len(qname_bytes)
        buf.write(struct.pack('<B', name_len))            # l_read_name
        buf.write(struct.pack('<B', min(mapq, 255)))      # MAPQ
        buf.write(struct.pack('<H', 0))                   # bin (TODO: compute)
        buf.write(struct.pack('<H', len(cigar_encoded)))  # n_cigar_op
        buf.write(struct.pack('<H', flag))                # FLAG
        buf.write(struct.pack('<i', len(seq)))             # l_seq
        buf.write(struct.pack('<i', -1))                   # next_refID
        buf.write(struct.pack('<i', -1))                   # next_pos
        buf.write(struct.pack('<i', 0))                    # tlen
        buf.write(qname_bytes)
        for c in cigar_encoded:
            buf.write(struct.pack('<I', c))
        buf.write(seq_encoded)
        buf.write(qual_encoded)
        buf.write(tag_bytes)
        
        self.f.write(self._bgzf_block(buf.getvalue()))
    
    def close(self):
        """Write EOF block and close."""
        # Empty BGZF block signals EOF
        eof = bytes([
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
            0x06, 0x00, 0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        ])
        self.f.write(eof)
        self.f.close()


def self_test():
    """Write a test BAM and verify with samtools if available."""
    import random, subprocess
    
    rng = random.Random(42)
    bases = "ACGT"
    
    # Create test BAM
    out_path = "/tmp/test_bpd.bam"
    refs = [("test_ref", 1000)]
    
    with BamWriter(out_path, refs) as bam:
        for i in range(20):
            seq = ''.join(rng.choice(bases) for _ in range(100))
            qual = 'I' * 100
            pos = rng.randint(0, 899)
            bam.write_alignment(
                "read_%03d" % i, 0, "test_ref", pos, 60,
                "100M", seq, qual,
                tags=[("AS", "i", 180)])
    
    print("  Wrote %s (%d bytes)" % (out_path, os.path.getsize(out_path)))
    
    # Try samtools
    try:
        result = subprocess.run(
            ["samtools", "view", "-c", out_path],
            capture_output=True, timeout=5, text=True)
        count = int(result.stdout.strip())
        print("  samtools view -c: %d records" % count)
        return count == 20
    except FileNotFoundError:
        print("  samtools not found — manual verification:")
        # Verify it's valid gzip
        try:
            with gzip.open(out_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'BAM\x01':
                    print("  BAM magic: correct")
                    return True
                else:
                    print("  BAM magic: WRONG (%s)" % magic)
                    return False
        except Exception as e:
            print("  gzip read failed: %s" % e)
            return False
    except Exception as e:
        print("  samtools error: %s" % e)
        return False


if __name__ == "__main__":
    print("=== BAM Output Self-Test ===")
    ok = self_test()
    print("  %s" % ("PASS" if ok else "FAIL"))
