import re

types = []
with open("model_48.h", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip().rstrip(",")
        if s.startswith("void *"):
            types.append("ptr")
        elif s.startswith("int64_t"):
            types.append("i64")

used_flags = []
with open("arg_map.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        used_flags.append("USED" in line)

if len(used_flags) != len(types):
    raise RuntimeError(f"arg_map.txt 数量 {len(used_flags)} 和 model_48.h 类型数 {len(types)} 不一致")

ptr_id = 0
with open("call_args.inc", "w", encoding="utf-8") as ca, open("ptr_buffers.inc", "w", encoding="utf-8") as pb:
    for i, ty in enumerate(types):
        comma = "," if i != len(types) - 1 else ""

        if ty == "ptr":
            sz = "16 * 1024 * 1024" if used_flags[i] else "4096"
            pb.write(f"  void *ptrbuf{ptr_id} = aligned_alloc_or_die(64, {sz});\n")
            ca.write(f"      ptrbuf{ptr_id}{comma}\n")
            ptr_id += 1
        else:
            ca.write(f"      one_i64{comma}\n")