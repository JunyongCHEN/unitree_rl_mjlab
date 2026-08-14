// 上板自检 + 测耗时。必须在真机 NPU 上跑通再接控制回路。
//
// 在板子上编（推荐直接用 SDK 的 ./build_mjlab.sh，它会一起编好）：
//   g++ -O2 tools/n2_rknn_check.cpp -o n2_rknn_check
//       -Iinclude/rknpu2 lib/aarch64/librknnrt.so -Wl,-rpath,'$ORIGIN/lib/aarch64'
// 运行：
//   ./n2_rknn_check ning/policy_mjlab.rknn ning/mjlab_golden.bin
//
// 判据：max|dAction| 应该在 1e-2 以内（PC 模拟器实测 ~3e-3）。
// 如果差一个数量级以上，别往下接控制，先查观测顺序 / 是否重复归一化。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "rknn_api.h"

static std::vector<char> read_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(n);
  if (fread(buf.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f);
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s policy.rknn golden.bin\n", argv[0]); return 1; }

  std::vector<char> model = read_file(argv[1]);
  rknn_context ctx = 0;
  if (rknn_init(&ctx, model.data(), model.size(), 0, nullptr) != 0) {
    fprintf(stderr, "rknn_init failed\n"); return 1;
  }
  // 策略网络很小，单核就够；和视觉模型抢核反而增加抖动
  rknn_set_core_mask(ctx, RKNN_NPU_CORE_0);

  rknn_input_output_num io_num;
  rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  rknn_tensor_attr in_attr, out_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  memset(&out_attr, 0, sizeof(out_attr));
  in_attr.index = 0;
  out_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
  printf("model: in n_elems=%u type=%d | out n_elems=%u type=%d\n",
         in_attr.n_elems, in_attr.type, out_attr.n_elems, out_attr.type);

  std::vector<char> g = read_file(argv[2]);
  const int32_t* hdr = reinterpret_cast<const int32_t*>(g.data());
  int n = hdr[0], obs_dim = hdr[1], act_dim = hdr[2];
  const float* gobs = reinterpret_cast<const float*>(g.data() + 12);
  const float* gact = gobs + (size_t)n * obs_dim;
  printf("golden: n=%d obs_dim=%d act_dim=%d\n", n, obs_dim, act_dim);
  if ((uint32_t)obs_dim != in_attr.n_elems || (uint32_t)act_dim != out_attr.n_elems) {
    fprintf(stderr, "!! shape mismatch between model and golden set\n");
    return 1;
  }

  double max_err = 0.0, sum_err = 0.0;
  for (int i = 0; i < n; ++i) {
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    // 模型内部是 float16；这里声明 FLOAT32 + pass_through=0，让 runtime 自己转换。
    // （Noetix SDK 里是自己 float_to_half 再传 fp16，两种都对。）
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = in_attr.fmt;
    inputs[0].size = obs_dim * sizeof(float);
    inputs[0].pass_through = 0;
    inputs[0].buf = (void*)(gobs + (size_t)i * obs_dim);
    if (rknn_inputs_set(ctx, 1, inputs) != 0) { fprintf(stderr, "inputs_set failed\n"); return 1; }
    if (rknn_run(ctx, nullptr) != 0) { fprintf(stderr, "run failed\n"); return 1; }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].is_prealloc = 0;
    outputs[0].want_float = 1;  // 让 runtime 把 fp16 输出转回 float32
    if (rknn_outputs_get(ctx, 1, outputs, nullptr) != 0) { fprintf(stderr, "outputs_get failed\n"); return 1; }
    const float* got = (const float*)outputs[0].buf;
    for (int j = 0; j < act_dim; ++j) {
      double e = fabs(got[j] - gact[(size_t)i * act_dim + j]);
      if (e > max_err) max_err = e;
      sum_err += e;
    }
    if (i == 0) {
      printf("sample0 got[0..3] = %.5f %.5f %.5f %.5f\n", got[0], got[1], got[2], got[3]);
      printf("sample0 ref[0..3] = %.5f %.5f %.5f %.5f\n",
             gact[0], gact[1], gact[2], gact[3]);
    }
    rknn_outputs_release(ctx, 1, outputs);
  }
  printf("accuracy: max|dAction| = %.6f   mean = %.6f   %s\n",
         max_err, sum_err / (n * act_dim), max_err < 1e-2 ? "PASS" : "FAIL (check obs layout!)");

  // 耗时：50 Hz 控制回路，单次推理必须远小于 20 ms
  const int WARM = 50, ITER = 1000;
  double worst_ms = 0.0;
  struct timespec t0, t1;
  for (int k = 0; k < WARM + ITER; ++k) {
    if (k == WARM) clock_gettime(CLOCK_MONOTONIC, &t0);
    struct timespec s0, s1;
    clock_gettime(CLOCK_MONOTONIC, &s0);
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_FLOAT32; in.fmt = in_attr.fmt;
    in.size = obs_dim * sizeof(float); in.pass_through = 0; in.buf = (void*)gobs;
    rknn_inputs_set(ctx, 1, &in);
    rknn_run(ctx, nullptr);
    rknn_output o;
    memset(&o, 0, sizeof(o));
    o.index = 0; o.want_float = 1; o.is_prealloc = 0;
    rknn_outputs_get(ctx, 1, &o, nullptr);
    rknn_outputs_release(ctx, 1, &o);
    clock_gettime(CLOCK_MONOTONIC, &s1);
    double ms = (s1.tv_sec - s0.tv_sec) * 1e3 + (s1.tv_nsec - s0.tv_nsec) / 1e6;
    if (k >= WARM && ms > worst_ms) worst_ms = ms;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double avg_ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / ITER;
  printf("latency: avg %.3f ms  worst %.3f ms  (50 Hz 预算 20 ms)\n", avg_ms, worst_ms);

  rknn_destroy(ctx);
  return max_err < 1e-2 ? 0 : 1;
}
