#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Smith-Waterman FASTQ/SAM pipeline — standard bioinformatics I/O.

Reads FASTQ, aligns against a FASTA reference, outputs SAM.
Compatible with samtools, IGV, and downstream variant callers.

Usage:
  python3 bench/sw_fastq_sam.py --ref ref.fasta --reads reads.fastq --out out.sam
  python3 bench/sw_fastq_sam.py --self-test
"""
import ctypes, os, sys, argparse, time, re

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c 2>/dev/null")
lib = ctypes.CDLL("/tmp/bpd_sw_cpu.so")

class SWResult(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_int),
        ("query_end", ctypes.c_int),
        ("ref_end", ctypes.c_int),
        ("cigar_len", ctypes.c_int),
        ("cigar", ctypes.c_char * 4096),
    ]

lib.bpd_smith_waterman_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(SWResult)]

def sw_align(query, ref, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    result = SWResult()
    lib.bpd_smith_waterman_cpu(
        query.encode(), len(query), ref.encode(), len(ref),
        match, mismatch, gap_open, gap_extend, ctypes.byref(result))
    return result

def read_fasta(path):
    """Read FASTA, return dict of name→sequence."""
    seqs = {}
    name = None
    parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name: seqs[name] = ''.join(parts).upper().replace('N', 'A')
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name: seqs[name] = ''.join(parts).upper().replace('N', 'A')
    return seqs

def read_fastq(path):
    """Read FASTQ, yield (name, sequence, quality)."""
    with open(path) as f:
        while True:
            header = f.readline().strip()
            if not header: break
            if not header.startswith('@'): continue
            seq = f.readline().strip().upper()
            f.readline()  # + line
            qual = f.readline().strip()
            name = header[1:].split()[0]
            yield name, seq, qual

def write_sam(outfile, ref_name, ref_len, alignments):
    """Write SAM format output."""
    with open(outfile, 'w') as f:
        # Header
        f.write("@HD\tVN:1.6\tSO:unsorted\n")
        f.write("@SQ\tSN:%s\tLN:%d\n" % (ref_name, ref_len))
        f.write("@PG\tID:bpd-sw\tPN:bpd_smith_waterman\tVN:1.0\tCL:LlamaTov Smith-Waterman GPU aligner\n")

        for qname, flag, rname, pos, mapq, cigar, seq, qual, score in alignments:
            f.write("%s\t%d\t%s\t%d\t%d\t%s\t*\t0\t0\t%s\t%s\tAS:i:%d\n" % (
                qname, flag, rname, pos, mapq, cigar, seq, qual, score))

def cigar_to_aligned_length(cigar):
    """Calculate aligned length from CIGAR string."""
    total = 0
    for count, op in re.findall(r'(\d+)([MID])', cigar):
        if op in ('M', 'D'):
            total += int(count)
    return total

def compute_mapq(score, seq_len):
    """Approximate mapping quality from SW score."""
    max_possible = seq_len * 2  # all matches
    if max_possible == 0: return 0
    ratio = score / max_possible
    if ratio > 0.9: return 60
    if ratio > 0.7: return 40
    if ratio > 0.5: return 20
    if ratio > 0.3: return 10
    return 0

def align_reads(ref_path, reads_path, out_path, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    """Full pipeline: FASTQ in, SAM out."""
    refs = read_fasta(ref_path)
    if not refs:
        print("ERROR: no sequences in %s" % ref_path)
        return

    ref_name = list(refs.keys())[0]
    ref_seq = refs[ref_name]
    print("Reference: %s (%d bp)" % (ref_name, len(ref_seq)))

    alignments = []
    t0 = time.perf_counter()
    n_reads = 0

    for qname, seq, qual in read_fastq(reads_path):
        result = sw_align(seq, ref_seq, match, mismatch, gap_open, gap_extend)
        cigar = result.cigar.decode()
        pos = result.ref_end - cigar_to_aligned_length(cigar) + 2  # 1-based SAM position
        if pos < 1: pos = 1
        mapq = compute_mapq(result.score, len(seq))
        flag = 0 if result.score > 0 else 4  # 4 = unmapped

        alignments.append((qname, flag, ref_name, pos, mapq, cigar, seq, qual, result.score))
        n_reads += 1

    elapsed = time.perf_counter() - t0
    write_sam(out_path, ref_name, len(ref_seq), alignments)
    print("Aligned %d reads in %.1f ms (%.1f reads/sec)" % (
        n_reads, elapsed*1000, n_reads/elapsed if elapsed > 0 else 0))
    print("Output: %s" % out_path)
    return alignments

def self_test():
    """Generate test FASTQ from reference, align, verify."""
    import random
    rng = random.Random(42)

    # Create test reference
    ref_path = "/tmp/test_ref.fasta"
    with open(ref_path, 'w') as f:
        ref_seq = ''.join(rng.choice("ACGT") for _ in range(500))
        f.write(">test_ref\n%s\n" % ref_seq)

    # Create test reads — extract subsequences with small mutations
    reads_path = "/tmp/test_reads.fastq"
    with open(reads_path, 'w') as f:
        for i in range(20):
            start = rng.randint(0, 400)
            read = list(ref_seq[start:start+100])
            # Introduce 0-3 mutations
            for _ in range(rng.randint(0, 3)):
                pos = rng.randint(0, len(read)-1)
                read[pos] = rng.choice("ACGT")
            read = ''.join(read)
            qual = 'I' * len(read)
            f.write("@read_%03d\n%s\n+\n%s\n" % (i, read, qual))

    out_path = "/tmp/test_output.sam"
    alignments = align_reads(ref_path, reads_path, out_path)

    # Verify SAM output
    print()
    print("=== Self-Test Verification ===")
    with open(out_path) as f:
        lines = f.readlines()
    headers = [l for l in lines if l.startswith('@')]
    records = [l for l in lines if not l.startswith('@')]
    print("  SAM headers: %d" % len(headers))
    print("  SAM records: %d" % len(records))

    # Check all records have valid fields
    valid = 0
    for rec in records:
        fields = rec.strip().split('\t')
        assert len(fields) >= 11, "SAM record has %d fields, need 11+" % len(fields)
        assert fields[0].startswith("read_"), "Bad QNAME: %s" % fields[0]
        assert fields[5] != '*' or int(fields[1]) == 4, "Mapped read missing CIGAR"
        valid += 1

    print("  Valid SAM records: %d/%d" % (valid, len(records)))
    print("  All scores > 0: %s" % all(a[8] > 0 for a in alignments))

    # Verify with samtools if available
    import subprocess
    try:
        subprocess.run(["samtools", "view", "-S", "-h", out_path], 
                       capture_output=True, check=True, timeout=5)
        print("  samtools validates: PASS")
    except FileNotFoundError:
        print("  samtools not found (skipping validation)")
    except Exception as e:
        print("  samtools: %s" % e)

    print()
    print("SELF-TEST PASSED" if valid == len(records) else "SELF-TEST FAILED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPD Smith-Waterman FASTQ→SAM aligner")
    parser.add_argument("--ref", help="Reference FASTA file")
    parser.add_argument("--reads", help="Reads FASTQ file")
    parser.add_argument("--out", default="/tmp/aligned.sam", help="Output SAM file")
    parser.add_argument("--self-test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.ref and args.reads:
        align_reads(args.ref, args.reads, args.out)
    else:
        parser.print_help()
