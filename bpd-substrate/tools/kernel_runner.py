#!/usr/bin/env python3
"""KernelRunner — launches emitted CUDA kernels from Python via a
compiled .so, using CALL-MANIFEST geometry (certified values only;
unresolvable = refuse, per the stopped-pass lesson).

Contract: the emitted unit for <pid> lives at emitted/<pid>.cu (the
census-synchronized store). Compilation: nvcc -shared with a C ABI
launcher generated per kernel. Params resolve from the wrapped
Model's own attributes (the manifest's attr names — module-reuse)."""
import json
import os
import re
import subprocess

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_HOME = os.environ.get(
    'CUDA_HOME',
    '/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8')


class KernelRunner:
    def __init__(self, pid, cu_path=None, model=None):
        self.pid = pid
        self.model = model
        cu_path = cu_path or os.path.join(REPO, 'emitted', f'{pid}.cu')
        cu = open(cu_path).read()
        self.manifests = [json.loads(m) for m in
                          re.findall(r'/\* CALL-MANIFEST (.*?) \*/', cu)]
        subs = [m for m in self.manifests if m['pid'] != pid]
        self.manifests = subs if subs else self.manifests
        self.kernels = re.findall(r'__global__ void (\w+)\(([^)]*)\)', cu,
                                  re.S)
        if len(self.kernels) != len(self.manifests):
            # the unit-level manifest may cover several kernels in a
            # split unit; map by order and let launch() refuse on
            # mismatch:
            pass
        self.launch_class = {}
        self.kernel_argnames = {}  # i -> 'elementwise'|'rowreduce'|
                                 # 'rowreduce_param'|'tuple'|'param_elem'
        self.lib = self._compile(cu_path, cu)

    def _compile(self, cu_path, cu):
        import ctypes
        so_path = f'/tmp/krun_{self.pid}.so'
        launcher = ['extern "C" {']
        for i, (kname, sig) in enumerate(self.kernels):
            mani = self.manifests[min(i, len(self.manifests) - 1)]
            launch = mani.get('launch')
            if launch is None:
                launcher.append(
                    f'int launch_{i}(float*, float*, long long)'
                    '{ return -1; } /* no certified geometry — refuse */')
                continue
            params = sig.replace('__restrict__', '').replace('const ', '')
            argnames = [p.strip().split()[-1].lstrip('*')
                        for p in params.split(',')]
            if launch.get('form') == '2d':
                # 2D spatial family (v,out,outer,dim,inner[,hw,chn]):
                gx = {'outer': 'outer'}.get(launch['gridx'], None)
                gy = {'(inner+31)/32': '(inner + 31) / 32',
                      32: '32', 128: '128'}.get(launch['gridy'], None)
                bx, by = launch['blockx'], launch['blocky']
                if gx is None or gy is None:
                    launcher.append(
                        f'int launch_{i}(float*, float*, long long)'
                        '{ return -2; } /* unresolvable 2d symbol */')
                    continue
                if argnames in (['v', 'x2', 'out', 'outer_size',
                                 'dim_size', 'inner_size'],
                                ['v', 'x2', 'out', 'outer', 'dim',
                                 'inner']):
                    launcher.append(f'''
int launch_{i}(float* v, float* x2, float* out, long long outer,
               long long dim, long long inner) {{
    dim3 grid(outer, {gy});
    dim3 block({bx}, {by});
    {kname}<<<grid, block>>>(v, x2, out, outer, dim, inner);
    return (int)cudaDeviceSynchronize();
}}''')
                    self.launch_class[i] = 'spatial_x2'
                elif argnames in (['v', 'out', 'outer', 'dim', 'inner'],
                                ['v', 'out', 'outer_size', 'dim_size',
                                 'inner_size']):
                    launcher.append(f'''
int launch_{i}(float* v, float* out, long long outer, long long dim,
               long long inner) {{
    dim3 grid(outer, {gy});
    dim3 block({bx}, {by});
    {kname}<<<grid, block>>>(v, out, outer, dim, inner);
    return (int)cudaDeviceSynchronize();
}}''')
                    self.launch_class[i] = 'spatial'
                elif argnames in (
                        ['v', 'p0', 'out', 'outer', 'dim', 'inner',
                         'hw_p0', 'chn_p0'],
                        ['v', 'p0', 'out', 'outer_size', 'dim_size',
                         'inner_size', 'hw_p0', 'chn_p0'],
                        ['v', 'p0', 'out', 'outer_size', 'dim_size',
                         'inner_size'],
                        ['v', 'p0', 'out', 'outer_size', 'dim_size',
                         'inner_size', 'plen_p0']):
                    has_hw = 'hw_p0' in argnames
                    has_plen = 'plen_p0' in argnames
                    tail_args = (', hw, chn' if has_hw
                                 else (', hw' if has_plen else ''))
                    # (the plen variant reuses the hw slot for
                    # p.numel() — the launch path keys on the class)
                    launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long outer,
               long long dim, long long inner, long long hw,
               long long chn) {{
    dim3 grid(outer, {gy});
    dim3 block({bx}, {by});
    {kname}<<<grid, block>>>(v, p0, out, outer, dim, inner{tail_args});
    return (int)cudaDeviceSynchronize();
}}''')
                    self.launch_class[i] = ('spatial_param_plen'
                                            if has_plen
                                            else 'spatial_param')
                else:
                    launcher.append(
                        f'int launch_{i}(float*, float*, long long)'
                        '{ return -3; } /* 2d signature unwrapped */')
                continue
            block = launch['block']
            grid = launch['grid']
            smem = launch['smem']
            grid_c = {'rows': 'nrow',
                      '(n+255)/256': '(n + 255) / 256',
                      '(nrow+255)/256': '(nrow + 255) / 256',
                      '(nout+255)/256': '(nout + 255) / 256'}.get(
                          grid, None)
            smem_c = {0: '0',
                      '(block/32)*sizeof(float)':
                          f'({block}/32)*sizeof(float)',
                      '512*sizeof(float)':
                          '512*sizeof(float)'}.get(smem, None)
            if grid_c is None or smem_c is None:
                launcher.append(
                    f'int launch_{i}(float*, float*, long long)'
                    '{ return -2; } /* unresolvable symbol — refuse */')
                continue
            # two-input elementwise (v, v2, out, n) — the #28 class
            # (a second MODEL input rides as v2):
            if argnames == ['v', 'v2', 'out', 'n']:
                self.launch_class[i] = 'elementwise2'
                launcher.append(f'''
int launch_{i}(float* v, float* v2, float* out, long long n) {{
    {kname}<<<{grid_c}, {block}, {smem_c}>>>(v, v2, out, n);
    return (int)cudaDeviceSynchronize();
}}''')
            # broadcast-red + p-buffer (the #51 class):
            elif argnames == ['v', 'p0', 'bvec', 'out', 'rows', 'n',
                              'bn', 'plen_p0']:
                self.launch_class[i] = 'broadcast_red_p'
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* bvec, float* out,
               long long rows, long long n, long long bn,
               long long plen) {{
    {kname}<<<rows, 1>>>(v, p0, bvec, out, rows, n, bn, plen);
    return (int)cudaDeviceSynchronize();
}}''')
            # broadcast-red (v, bvec, out, rows, n) — the #75 class
            # (row-reduce + broadcast ⊕ param; ONE sequential thread
            # per row — the emulator kernel):
            elif argnames == ['v', 'bvec', 'out', 'rows', 'n', 'bn']:
                self.launch_class[i] = 'broadcast_red'
                launcher.append(f'''
int launch_{i}(float* v, float* bvec, float* out, long long rows,
               long long n, long long bn) {{
    {kname}<<<rows, 1>>>(v, bvec, out, rows, n, bn);
    return (int)cudaDeviceSynchronize();
}}''')
            # row-broadcast two-pass (v, out, rows, n) — the #15
            # class (sub_own_mean: rows=(B*C), n=prod(spatial)):
            elif argnames == ['v', 'out', 'rows', 'n']:
                self.launch_class[i] = 'rowbroadcast'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long rows, long long n) {{
    {kname}<<<rows, {block}>>>(v, out, rows, n);
    return (int)cudaDeviceSynchronize();
}}''')
            # avgpool3d k2 (v,out,outer,D,H,W) — the #72 class:
            elif argnames == ['v', 'out', 'outer', 'D', 'H', 'W']:
                self.launch_class[i] = 'avgpool3d'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long outer, long long D,
               long long H, long long W) {{
    long long nout = outer * (D / 2) * (H / 2) * (W / 2);
    {kname}<<<(nout + 255) / 256, 256>>>(v, out, outer, D, H, W);
    return (int)cudaDeviceSynchronize();
}}''')
            # canonical elementwise (v, out, n):
            elif argnames == ['v', 'out', 'n']:
                self.launch_class[i] = 'elementwise'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long n) {{
    long long nrow = 0; (void)nrow;
    {kname}<<<{grid_c.replace('nrow', '1')}, {block}, {smem_c}>>>(v, out, n);
    return (int)cudaDeviceSynchronize();
}}''')
            # param row-kernel (v, p0, out, classes) — the #84 class
            # (buffered-row softmax w/ a chan param); + the plen
            # variant (#38's gridstride softmax + trailing param):
            elif argnames in (['v', 'p0', 'out', 'classes'],
                              ['v', 'p0', 'out', 'classes',
                               'plen_p0']):
                self.launch_class[i] = 'rowkernel_param'
                _has_plen = 'plen_p0' in argnames
                _plen_arg = (', plen' if _has_plen else '')
                _plen_sig = (', long long plen' if _has_plen else '')
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long n,
               long long width{_plen_sig}) {{
    long long nrow = n / width;
    {kname}<<<{grid_c}, {block}, {smem_c}>>>(v, p0, out, (int)width{_plen_arg});
    return (int)cudaDeviceSynchronize();
}}''')
                if _has_plen:
                    self.launch_class[i] = 'rowkernel_param_plen'
            # row-reduce family (v, out, classes) w/ grid=rows:
            elif argnames == ['v', 'out', 'classes']:
                self.launch_class[i] = 'rowkernel'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long n, long long width) {{
    long long nrow = n / width;
    {kname}<<<nrow, {block}, {smem_c}>>>(v, out, (int)width);
    return (int)cudaDeviceSynchronize();
}}''')
            # PARAM-ELEMENTWISE family (Mavdil's 26-kernel unlock):
            # (v, p0[, p1], out, n[, hw_pK, chn_pK]...):
            elif (argnames[0] == 'v' and argnames[-1] != 'classes'
                  and 'out' in argnames
                  and all(a.startswith('p') or a in ('v', 'out', 'n')
                          or a.startswith('hw_') or a.startswith('chn_')
                          for a in argnames)
                  and 'n' in argnames):
                self.launch_class[i] = 'param_elem'
                self.kernel_argnames[i] = list(argnames)
                pl = [a for a in argnames if re.fullmatch(r'p\d+', a)]
                lens = [a for a in argnames
                        if a.startswith(('hw_', 'chn_', 'plen_'))]
                psig = ', '.join(f'float* {p}' for p in pl)
                lsig = ', '.join(f'long long {a}' for a in lens)
                call = ', '.join(argnames)
                launcher.append(f'''
int launch_{i}(float* v, {psig}, float* out, long long n{"," if lsig else ""} {lsig}) {{
    {kname}<<<(n + 255) / 256, 256, 0>>>({call});
    return (int)cudaDeviceSynchronize();
}}''')
            # tuple row-reduce (v, out, nrow, nred) — one thread per row:
            elif argnames == ['v', 'out', 'nrow', 'nred']:
                self.launch_class[i] = 'rowkernel'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long n, long long width) {{
    long long nrow = n / width;
    {kname}<<<{grid_c}, {block}, {smem_c}>>>(v, out, nrow, width);
    return (int)cudaDeviceSynchronize();
}}''')
            # param chan-flat (v,p0,out,spatial,chn,nout,hw,chn) —
            # the #79-seg2 class:
            elif argnames == ['v', 'p0', 'out', 'spatial', 'chn',
                              'nout', 'hw_p0', 'chn_p0']:
                self.launch_class[i] = 'chanflat_param'
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long spatial,
               long long chn, long long nout, long long hw,
               long long pchn) {{
    {kname}<<<(nout + 255) / 256, {block}, {smem_c}>>>(
        v, p0, out, spatial, chn, nout, hw, pchn);
    return (int)cudaDeviceSynchronize();
}}''')
            # chan-flat + param + plen, 7-arg (#13's fold half):
            elif argnames == ['v', 'p0', 'out', 'spatial', 'chn',
                              'nout', 'plen_p0']:
                self.launch_class[i] = 'chanflat_param7'
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long spatial,
               long long chn, long long nout, long long plen) {{
    {kname}<<<(nout + 255) / 256, {block}, {smem_c}>>>(
        v, p0, out, spatial, chn, nout, plen);
    return (int)cudaDeviceSynchronize();
}}''')
            # chan-flat + param, 6-arg (v,p0,out,spatial,chn,nout) —
            # the fold-split halves that carry a param (#13/#36):
            elif argnames == ['v', 'p0', 'out', 'spatial', 'chn',
                              'nout']:
                self.launch_class[i] = 'chanflat_param6'
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long spatial,
               long long chn, long long nout) {{
    {kname}<<<(nout + 255) / 256, {block}, {smem_c}>>>(
        v, p0, out, spatial, chn, nout);
    return (int)cudaDeviceSynchronize();
}}''')
            # chan-strided flat (v, out, spatial, chn, nout):
            elif argnames == ['v', 'out', 'spatial', 'chn', 'nout']:
                self.launch_class[i] = 'chanflat'
                launcher.append(f'''
int launch_{i}(float* v, float* out, long long spatial, long long chn,
               long long nout) {{
    {kname}<<<(nout + 255) / 256, {block}, {smem_c}>>>(
        v, out, spatial, chn, nout);
    return (int)cudaDeviceSynchronize();
}}''')
            # param-carrying row-reduce (v, p0, out, classes, hw, chn):
            elif argnames == ['v', 'p0', 'out', 'classes', 'hw_p0',
                              'chn_p0']:
                self.launch_class[i] = 'rowreduce_param'
                launcher.append(f'''
int launch_{i}(float* v, float* p0, float* out, long long n,
               long long width, long long hw, long long chn) {{
    long long nrow = n / width;
    {kname}<<<nrow, {block}, {smem_c}>>>(v, p0, out, (int)width, hw, chn);
    return (int)cudaDeviceSynchronize();
}}''')
            else:
                launcher.append(
                    f'int launch_{i}(float*, float*, long long)'
                    '{ return -3; } /* signature class not yet wrapped '
                    '— refuse (named) */')
        launcher.append('}')
        src = cu + '\n' + '\n'.join(launcher)
        tmp = f'/tmp/krun_{self.pid}.cu'
        open(tmp, 'w').write(src)
        # TWO-STEP BUILD (the nix store carries no libcudadevrt, which
        # nvcc's own -shared link path demands): nvcc compiles the .o
        # (proven — the pipeline's compile_check path), then gcc links
        # the .so against shared cudart only:
        obj = f'/tmp/krun_{self.pid}.o'
        r = subprocess.run(
            [os.path.join(CUDA_HOME, 'bin', 'nvcc'),
             '-arch=sm_61', '--fmad=false', '-c',
             '-Xcompiler', '-fPIC',
             '-I', os.path.join(CUDA_HOME, 'include'),
             tmp, '-o', obj],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f'{self.pid}: wrapper .o failed nvcc: '
                f'...{r.stderr.strip()[-300:]}')
        r = subprocess.run(
            ['gcc', '-shared', obj, '-o', so_path,
             '-L', os.path.join(CUDA_HOME, 'lib'), '-lcudart'],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f'{self.pid}: wrapper .so failed link: '
                f'...{r.stderr.strip()[-300:]}')
        return ctypes.CDLL(so_path)

    def n_lifted_runs(self):
        return len(self.kernels)

    def launch(self, i, x, x2=None):
        """Launch lifted-run i on tensor x (CUDA, f32, contiguous)."""
        import ctypes
        x = x.contiguous().float().cuda()
        mani = self.manifests[min(i, len(self.manifests) - 1)]
        width = mani.get('reduce_width') or mani.get('tuple_n')
        if width is None and self.launch_class.get(i) == 'rowkernel':
            # the width tracker can lose the value through shape-op
            # chains (the #55 unsqueeze/pool/squeeze); the row-kernel
            # reduces the LAST dim by construction — derive from x:
            width = int(x.shape[-1])
        n = x.numel()
        fact = mani.get('fact', '')
        # SHAPE-PRESERVING reductions (softmax family) write full rows;
        # only the collapsing kinds shrink the output:
        _collapsing = any(f'reduction({k})' in fact or
                          f'reduction_tuple({k})' in fact
                          for k in ('logsumexp', 'sum', 'mean',
                                    'min', 'max'))
        is_reduce = _collapsing and width
        out = (torch.empty(n // width, device='cuda', dtype=torch.float32)
               if is_reduce else torch.empty_like(x))
        fn = getattr(self.lib, f'launch_{i}')
        params = mani.get('params') or []
        if (params and not is_reduce
                and self.launch_class.get(i) == 'param_elem'):
            # (the #58 lesson completing the #66 lesson: EVERY call
            # branch keys on the launcher's signature class — the
            # param-elem test alone also matched spatial_param units,
            # sending 4 args to an 8-param dim3 launcher = outer
            # became n, dim/inner became stack garbage, rc=700):
            # PARAM-ELEMENTWISE: params from the MODEL's attrs (the
            # manifest's names — module-reuse); hw/chn from x's OWN
            # shape at this point (hw = spatial product, chn = C):
            if self.model is None:
                raise RuntimeError(
                    f'{self.pid}: param kernel needs model=; refuse.')
            args = [ctypes.c_void_p(x.data_ptr())]
            bufs = []
            for pr in params:
                p = getattr(self.model, pr['attr'])
                p = (p.detach() if hasattr(p, 'detach') else p)
                if not torch.is_tensor(p):
                    p = torch.tensor([float(p)])
                p = p.contiguous().float().cuda().ravel()
                bufs.append(p)
                args.append(ctypes.c_void_p(p.data_ptr()))
            args.append(ctypes.c_void_p(out.data_ptr()))
            args.append(ctypes.c_longlong(n))
            hw = 1
            for d in x.shape[2:]:
                hw *= int(d)
            chn = int(x.shape[1]) if x.dim() > 1 else 1
            for pr in params:
                if pr['kind'] == 'chan':
                    args.append(ctypes.c_longlong(hw))
                    args.append(ctypes.c_longlong(chn))
            # the plen-broadcast (#28's add_param(pos_embedding)):
            # the kernel indexes p0[i % plen_p0] — supply the
            # parameter's element-count per plen-carrying param
            # (order matches the signature's plen_pK sequence):
            _kan = self.kernel_argnames.get(i, [])
            for _bi, pr in enumerate(params):
                if f'plen_p{_bi}' in _kan and _bi < len(bufs):
                    args.append(ctypes.c_longlong(
                        bufs[_bi].numel()))
            rc = fn(*args)
            if rc != 0:
                raise RuntimeError(
                    f'{self.pid}: launch_{i} rc={rc}')
            return out.view(x.shape)
        if params and is_reduce and \
                self.launch_class.get(i) not in (
                    'broadcast_red', 'broadcast_red_p',
                    'rowbroadcast', 'chanflat_param',
                    'chanflat_param6', 'chanflat_param7',
                    'spatial_param', 'spatial_param_plen'):
            # param-carrying row-reduce: resolve the buffer from the
            # MODEL's own attribute (module-reuse — the manifest's
            # attr name; never an invented buffer).
            # (the #58 lesson AGAIN — the #75/#51 700s: this branch
            # swallowed every params+is_reduce unit before the
            # class-keyed chain could run; EVERY branch keys on the
            # launcher's signature class):
            if self.model is None:
                raise RuntimeError(
                    f'{self.pid}: param kernel needs model= for '
                    f'attr resolution; refuse.')
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            p = p.contiguous().float().cuda().ravel()
            hw = 1  # the collapsed row-form (spatial-all-1s)
            chn = width
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(n), ctypes.c_longlong(width),
                    ctypes.c_longlong(hw), ctypes.c_longlong(chn))
        elif self.launch_class.get(i) == 'elementwise2':
            if x2 is None:
                raise RuntimeError(
                    f'{self.pid}: launch_{i} needs a second input '
                    f'(elementwise2) — none provided')
            x2 = x2.contiguous().float().cuda()
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(x2.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(n))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out.view(x.shape)
        elif self.launch_class.get(i) == 'avgpool3d':
            # 5D (B,C,D,H,W): outer=B*C; out halves each spatial:
            B, C = int(x.shape[0]), int(x.shape[1])
            D, H, W = (int(x.shape[2]), int(x.shape[3]),
                       int(x.shape[4]))
            outer = B * C
            out = torch.empty(B, C, D // 2, H // 2, W // 2,
                              device='cuda', dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(outer), ctypes.c_longlong(D),
                    ctypes.c_longlong(H), ctypes.c_longlong(W))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out
        elif self.launch_class.get(i) == 'rowbroadcast':
            # rows/n from the manifest when present (tuple_rows/
            # tuple_n — the #27 shift-head mean); else batch*chan
            # split. REDUCE-vs-broadcast: a tuple-reduction outputs
            # ROWS elements; sub_own_mean outputs rows*n:
            _tr = mani.get('tuple_rows')
            _tn = mani.get('tuple_n')
            if _tr and _tn:
                rows, nn = int(_tr), int(_tn)
            else:
                rows = int(x.shape[0]) * (int(x.shape[1])
                                          if x.dim() > 2 else 1)
                nn = x.numel() // rows
            fct2 = mani.get('fact', '')
            _is_reduce = 'reduction_tuple(' in fct2
            out = (torch.empty(rows, device='cuda',
                               dtype=torch.float32)
                   if _is_reduce else torch.empty_like(x))
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(rows), ctypes.c_longlong(nn))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out
        elif self.launch_class.get(i) == 'broadcast_red_p':
            # params[0] = the PRE vec-param; the SAVED input = bvec
            # (the gate passes it as ins[1] via forward(*extra)):
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            bv = x2.contiguous().float().cuda()
            rows = int(x.shape[0])
            width = int(x.shape[-1])
            bn = int(bv.shape[-1])
            out = torch.empty(rows * bn, device='cuda',
                              dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(bv.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(rows), ctypes.c_longlong(width),
                    ctypes.c_longlong(bn),
                    ctypes.c_longlong(p.numel()))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out.view(rows, bn)
        elif self.launch_class.get(i) == 'broadcast_red':
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            rows = int(x.shape[0])
            width = int(x.shape[-1])
            out = torch.empty(rows * width, device='cuda',
                              dtype=torch.float32)
            bn2 = p.numel()
            out = torch.empty(rows * bn2, device='cuda',
                              dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(rows), ctypes.c_longlong(width),
                    ctypes.c_longlong(bn2))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            # THE 4D-BROADCAST LAYOUT (the #75 hypothesis CONFIRMED):
            # torch broadcasts (rows,1) ⊕ (1,bn,1,1) → (1,bn,rows,1)
            # — a TRANSPOSE of our (rows,bn). Return torch's layout:
            return (out.view(rows, bn2).t().contiguous()
                    .view(1, bn2, rows, 1))
        elif self.launch_class.get(i) in ('rowkernel_param',
                                          'rowkernel_param_plen'):
            if x.dim() > 2:
                # the free-view semantics (#38): the softmax rides
                # the FLATTENED dims 2+ — the width is the
                # TENSOR's, not the lift's static guess (the lift
                # computed pre-conv dims; conv arithmetic is
                # runtime knowledge):
                width = 1
                for d in x.shape[2:]:
                    width *= int(d)
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            rc_args = [ctypes.c_void_p(x.data_ptr()),
                       ctypes.c_void_p(p.data_ptr()),
                       ctypes.c_void_p(out.data_ptr()),
                       ctypes.c_longlong(n), ctypes.c_longlong(width)]
            if self.launch_class[i] == 'rowkernel_param_plen':
                rc_args.append(ctypes.c_longlong(p.numel()))
            rc = fn(*rc_args)
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out.view(x.shape)
        elif self.launch_class.get(i) == 'chanflat_param':
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            chn = mani.get('chan_count') or int(x.shape[1])
            spatial = 1
            for d in x.shape[2:]:
                spatial *= int(d)
            outer = int(x.shape[0])
            nout = outer * spatial
            out = torch.empty(nout, device='cuda', dtype=torch.float32)
            hw = spatial
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(spatial), ctypes.c_longlong(chn),
                    ctypes.c_longlong(nout), ctypes.c_longlong(hw),
                    ctypes.c_longlong(chn))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            return out.view(outer, 1, *[int(d) for d in x.shape[2:]])
        elif self.launch_class.get(i) == 'chanflat_param7':
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            chn = mani.get('chan_count') or int(x.shape[1])
            _si = mani.get('spatial_inner')
            if _si:
                spatial = int(_si)
                outer = x.numel() // (chn * spatial)
            else:
                spatial = 1
                for d in x.shape[2:]:
                    spatial *= int(d)
                outer = int(x.shape[0])
            nout = outer * spatial
            out = torch.empty(nout, device='cuda', dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(spatial), ctypes.c_longlong(chn),
                    ctypes.c_longlong(nout),
                    ctypes.c_longlong(p.numel()))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            if _si:
                return out.view(outer, 1, spatial)
            return out.view(outer, 1, *[int(d) for d in x.shape[2:]])
        elif self.launch_class.get(i) == 'chanflat_param6':
            attr = params[0]['attr']
            p = getattr(self.model, attr)
            p = (p.detach() if hasattr(p, 'detach') else p)
            if not torch.is_tensor(p):
                p = torch.tensor([float(p)])
            p = p.contiguous().float().cuda().ravel()
            self._pbuf = p
            chn = mani.get('chan_count') or int(x.shape[1])
            _si = mani.get('spatial_inner')
            if _si:
                spatial = int(_si)
                outer = x.numel() // (chn * spatial)
            else:
                spatial = 1
                for d in x.shape[2:]:
                    spatial *= int(d)
                outer = int(x.shape[0])
            nout = outer * spatial
            out = torch.empty(nout, device='cuda', dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(p.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(spatial), ctypes.c_longlong(chn),
                    ctypes.c_longlong(nout))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            if _si:
                return out.view(outer, 1, spatial)
            return out.view(outer, 1, *[int(d) for d in x.shape[2:]])
        elif self.launch_class.get(i) == 'chanflat':
            # (v,out,spatial,chn,nout): chan-collapsing by definition;
            # spatial = product of dims[2:], chn = dims[1],
            # nout = outer*spatial:
            chn = mani.get('chan_count') or int(x.shape[1])
            _si = mani.get('spatial_inner')
            if _si:
                # MANIFEST-DRIVEN extents (the fold class: the manifest
                # says which axis is 'chan' — shape-derivation is wrong
                # for folded reduces):
                spatial = int(_si)
                outer = x.numel() // (chn * spatial)
            else:
                spatial = 1
                for d in x.shape[2:]:
                    spatial *= int(d)
                outer = int(x.shape[0])
            nout = outer * spatial
            out = torch.empty(nout, device='cuda', dtype=torch.float32)
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(spatial), ctypes.c_longlong(chn),
                    ctypes.c_longlong(nout))
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            if _si:
                return out.view(outer, 1, spatial)
            return out.view(outer, 1, *[int(d) for d in x.shape[2:]])
        elif self.launch_class.get(i) in ('spatial', 'spatial_param',
                                          'spatial_param_plen',
                                          'spatial_x2'):
            # (v,out,outer,dim,inner): dim = chan_count, inner =
            # spatial product, outer = batch — all from x's own shape
            # at kernel entry (NCHW-family):
            dim = mani.get('chan_count') or int(x.shape[1])
            _si2 = mani.get('spatial_inner')
            if _si2:
                # manifest-driven (the fold-split second half: x may
                # arrive FLAT from the fold kernel — shape-derivation
                # is wrong; the manifest carries the true extents):
                inner = int(_si2)
                outer = x.numel() // (dim * inner)
            else:
                inner = 1
                for d in x.shape[2:]:
                    inner *= int(d)
                outer = int(x.shape[0])
            fct = mani.get('fact', '')
            _chan_collapsing = any(
                f'reduction_chan({k})' in fct
                for k in ('logsumexp', 'sum', 'mean', 'min', 'max'))
            if _chan_collapsing:
                out = torch.empty(outer * inner, device='cuda',
                                  dtype=torch.float32)
            args = [ctypes.c_void_p(x.data_ptr())]
            if self.launch_class[i] == 'spatial_x2':
                # the cross-segment saved tensor (elementwise-
                # aligned with v — #92's x_conv):
                if x2 is None:
                    raise RuntimeError(
                        f'{self.pid}: spatial_x2 needs x2')
                _xb = x2.contiguous().float().cuda()
                self._x2buf = _xb
                args.append(ctypes.c_void_p(_xb.data_ptr()))
            if self.launch_class[i] in ('spatial_param',
                                        'spatial_param_plen'):
                p = getattr(self.model, params[0]['attr'])
                if not torch.is_tensor(p):
                    p = torch.tensor([float(p)])
                p = p.detach().contiguous().float().cuda().ravel()
                self._pbuf = p  # keep alive
                args.append(ctypes.c_void_p(p.data_ptr()))
            args += [ctypes.c_void_p(out.data_ptr()),
                     ctypes.c_longlong(outer), ctypes.c_longlong(dim),
                     ctypes.c_longlong(inner)]
            if self.launch_class[i] == 'spatial_param':
                args += [ctypes.c_longlong(inner),
                         ctypes.c_longlong(dim)]
            elif self.launch_class[i] == 'spatial_param_plen':
                # hw slot carries p.numel() (the launcher's 7th arg
                # maps to the kernel's plen_p0); chn unused:
                args += [ctypes.c_longlong(self._pbuf.numel()),
                         ctypes.c_longlong(0)]
            rc = fn(*args)
            if rc != 0:
                raise RuntimeError(f'{self.pid}: launch_{i} rc={rc}')
            if _chan_collapsing:
                return out.view(outer, 1, *[int(d) for d in
                                            x.shape[2:]])
            return out.view(x.shape)
        elif self.launch_class.get(i) == 'rowkernel':
            # 4-arg row-kernel form (v,out,n,width) — REGARDLESS of
            # whether the op collapses (softmax preserves shape but
            # still launches per-row; the #66 bug: calling this with
            # the 3-arg elementwise form left `width` as stack
            # garbage):
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(n), ctypes.c_longlong(width))
        else:
            rc = fn(ctypes.c_void_p(x.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_longlong(n))
        if rc != 0:
            raise RuntimeError(
                f'{self.pid}: launch_{i} rc={rc} '
                f'(-1 no-geometry, -2 unresolvable, -3 unwrapped-class, '
                f'else CUDA error {rc})')
        return out.view(x.shape) if not is_reduce else out
