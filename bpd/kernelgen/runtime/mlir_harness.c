// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdlib.h>
extern float relu(float);
int main(int argc, char** argv) {
    FILE* fi = fopen(argv[1], "rb"); fseek(fi,0,SEEK_END); long sz=ftell(fi); fseek(fi,0,SEEK_SET);
    int n = sz/4; float* x = malloc(sz); fread(x,4,n,fi); fclose(fi);
    float* o = malloc(sz);
    for (int i=0;i<n;i++) o[i] = relu(x[i]);
    FILE* fo = fopen(argv[2], "wb"); fwrite(o,4,n,fo); fclose(fo);
    printf("mlir relu: %d elems\n", n);
    return 0;
}
