#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 结构体定义保持不变
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
  int64_t sizes[3];
  int64_t strides[3];
} memref_3d_f32;

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
  int64_t sizes[1];
  int64_t strides[1];
} memref_1d_f32;

// 内存分配辅助函数
static void *aligned_alloc_or_die(size_t alignment, size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, alignment, bytes) != 0 || p == NULL) {
    fprintf(stderr, "posix_memalign failed\n");
    exit(1);
  }
  return p;
}

// 填充随机数据以模拟真实输入
static void fill_random(float *ptr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    ptr[i] = (float)rand() / (float)RAND_MAX;
  }
}

// 初始化不同维度的 memref
static memref_4d_f32 make_4d_f32(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  memref_4d_f32 m;
  size_t n = (size_t)(d0 * d1 * d2 * d3);
  float *buf = (float *)aligned_alloc_or_die(64, n * sizeof(float));
  fill_random(buf, n);
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0; m.sizes[1] = d1; m.sizes[2] = d2; m.sizes[3] = d3;
  m.strides[3] = 1;
  m.strides[2] = d3;
  m.strides[1] = d2 * d3;
  m.strides[0] = d1 * d2 * d3;
  return m;
}

static memref_2d_f32 make_2d_f32(int64_t d0, int64_t d1) {
  memref_2d_f32 m;
  size_t n = (size_t)(d0 * d1);
  float *buf = (float *)aligned_alloc_or_die(64, n * sizeof(float));
  fill_random(buf, n);
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0; m.sizes[1] = d1;
  m.strides[1] = 1;
  m.strides[0] = d1;
  return m;
}

static memref_1d_f32 make_1d_f32(int64_t d0) {
  memref_1d_f32 m;
  size_t n = (size_t)d0;
  float *buf = (float *)aligned_alloc_or_die(64, n * sizeof(float));
  fill_random(buf, n);
  m.allocated = buf;
  m.aligned = buf;
  m.offset = 0;
  m.sizes[0] = d0;
  m.strides[0] = 1;
  return m;
}

/*
  根据 grep 结果对齐参数 (共 39 个参数):
  1) 4D memref (11 args)
  2) 4D memref (11 args)
  3) 1D memref (5 args)
  4) 1D memref (5 args)
  5) 2D memref (7 args)
*/
extern memref_3d_f32 model_main(
    void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, // a
    void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, // b
    void *, void *, int64_t, int64_t, int64_t,                                                       // c
    void *, void *, int64_t, int64_t, int64_t,                                                       // d
    void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t                                      // e
);

int main(void) {
  srand((unsigned int)time(NULL));

  // 1. 创建输入 (请确保这里的 Shape 与你 phi1.ll 里的真实定义一致)
  memref_4d_f32 a = make_4d_f32(1, 1, 12, 64);
  memref_4d_f32 b = make_4d_f32(1, 1, 12, 64);
  memref_1d_f32 c = make_1d_f32(12);
  memref_1d_f32 d = make_1d_f32(12);
  memref_2d_f32 e = make_2d_f32(12, 64);

  printf("Starting execution of phi1 model...\n");

  // 2. 调用模型并传入展平参数
  memref_3d_f32 out = model_main(
      a.allocated, a.aligned, a.offset, a.sizes[0], a.sizes[1], a.sizes[2], a.sizes[3], a.strides[0], a.strides[1], a.strides[2], a.strides[3],
      b.allocated, b.aligned, b.offset, b.sizes[0], b.sizes[1], b.sizes[2], b.sizes[3], b.strides[0], b.strides[1], b.strides[2], b.strides[3],
      c.allocated, c.aligned, c.offset, c.sizes[0], c.strides[0],
      d.allocated, d.aligned, d.offset, d.sizes[0], d.strides[0],
      e.allocated, e.aligned, e.offset, e.sizes[0], e.sizes[1], e.strides[0], e.strides[1]);

  printf("Execution finished.\n");

  // 3. 打印输出结果（校验用）
  printf("Output Rank: 3, Sizes: [%ld, %ld, %ld]\n", out.sizes[0], out.sizes[1], out.sizes[2]);
  float *res = (float *)out.aligned;
  printf("First result element: %f\n", res[0]);

  // 释放内存 (可选，针对 Runtime 测试通常不影响功耗测量)
  free(a.allocated);
  free(b.allocated);
  free(c.allocated);
  free(d.allocated);
  free(e.allocated);

  return 0;
}