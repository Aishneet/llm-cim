#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref_2d_f32;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} memref_4d_f32;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref_2d_i64;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[3];
  int64_t strides[3];
} memref_3d_f32;

static void *aligned_alloc_or_die(size_t alignment, size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, alignment, bytes) != 0 || p == NULL) {
    fprintf(stderr, "posix_memalign failed\n");
    exit(1);
  }
  memset(p, 0, bytes);
  return p;
}

static memref_2d_f32 make_2d_f32(int64_t d0, int64_t d1) {
  memref_2d_f32 m;
  size_t n = (size_t)(d0 * d1);
  float *buf = (float *)aligned_alloc_or_die(64, n * sizeof(float));
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0;
  m.sizes[1] = d1;
  m.strides[1] = 1;
  m.strides[0] = d1;
  return m;
}

static memref_4d_f32 make_4d_f32(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  memref_4d_f32 m;
  size_t n = (size_t)(d0 * d1 * d2 * d3);
  float *buf = (float *)aligned_alloc_or_die(64, n * sizeof(float));
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0;
  m.sizes[1] = d1;
  m.sizes[2] = d2;
  m.sizes[3] = d3;
  m.strides[3] = 1;
  m.strides[2] = d3;
  m.strides[1] = d2 * d3;
  m.strides[0] = d1 * d2 * d3;
  return m;
}

static memref_2d_i64 make_2d_i64(int64_t d0, int64_t d1) {
  memref_2d_i64 m;
  size_t n = (size_t)(d0 * d1);
  int64_t *buf = (int64_t *)aligned_alloc_or_die(64, n * sizeof(int64_t));
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0;
  m.sizes[1] = d1;
  m.strides[1] = 1;
  m.strides[0] = d1;
  return m;
}

/*
  这就是你当前 lowered 后的入口函数。
  4 个输入 memref，按你 IR 里的顺序写：
    1) memref<1024x1024xf32>
    2) memref<1x1x12x64xf32>
    3) memref<1x1x12x64xf32>
    4) memref<1x1xi64>
  返回:
    memref<1x1x50257xf32>
*/
extern memref_3d_f32 model_main(
    void *a0, void *a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6,
    void *b0, void *b1, int64_t b2, int64_t b3, int64_t b4, int64_t b5, int64_t b6, int64_t b7, int64_t b8, int64_t b9, int64_t b10,
    void *c0, void *c1, int64_t c2, int64_t c3, int64_t c4, int64_t c5, int64_t c6, int64_t c7, int64_t c8, int64_t c9, int64_t c10,
    void *d0, void *d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6);

int main(void) {
  memref_2d_f32 x = make_2d_f32(4096, 4096);
  memref_4d_f32 y = make_4d_f32(1, 1, 12, 64);
  memref_4d_f32 z = make_4d_f32(1, 1, 12, 64);
  memref_2d_i64 t = make_2d_i64(1, 1);

  float *xbuf = (float *)x.aligned;
  for (int64_t i = 0; i < 4096 * 4096; ++i) {
    xbuf[i] = 0.0f;
  }

  float *ybuf = (float *)y.aligned;
  for (int64_t i = 0; i < 1 * 1 * 12 * 64; ++i) {
    ybuf[i] = 0.0f;
  }

  float *zbuf = (float *)z.aligned;
  for (int64_t i = 0; i < 1 * 1 * 12 * 64; ++i) {
    zbuf[i] = 0.0f;
  }

  int64_t *tbuf = (int64_t *)t.aligned;
  tbuf[0] = 0;

  memref_3d_f32 out = model_main(
      x.allocated, x.aligned, x.offset, x.sizes[0], x.sizes[1], x.strides[0], x.strides[1],
      y.allocated, y.aligned, y.offset, y.sizes[0], y.sizes[1], y.sizes[2], y.sizes[3],
      y.strides[0], y.strides[1], y.strides[2], y.strides[3],
      z.allocated, z.aligned, z.offset, z.sizes[0], z.sizes[1], z.sizes[2], z.sizes[3],
      z.strides[0], z.strides[1], z.strides[2], z.strides[3],
      t.allocated, t.aligned, t.offset, t.sizes[0], t.sizes[1], t.strides[0], t.strides[1]);

  printf("model_main returned.\n");
  printf("output sizes = [%ld, %ld, %ld]\n", out.sizes[0], out.sizes[1], out.sizes[2]);

  float *outbuf = (float *)out.aligned;
  printf("out[0] = %f\n", outbuf[0]);

  return 0;
}