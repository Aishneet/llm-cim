import re
from pathlib import Path

ll_path = Path("gptxl48.ll")

sig_line = None
with ll_path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("define ") and "@main(" in line:
            sig_line = line.strip()
            break

if sig_line is None:
    raise RuntimeError("Cannot find @main signature in gptxl48.ll")

m = re.search(r'@main\((.*)\)\s*\{', sig_line)
if not m:
    raise RuntimeError("Failed to parse @main signature line")

params = m.group(1).strip()

parts = []
cur = []
depth = 0
for ch in params:
    if ch == ',' and depth == 0:
        s = ''.join(cur).strip()
        if s:
            parts.append(s)
        cur = []
        continue
    cur.append(ch)
    if ch == '[':
        depth += 1
    elif ch == ']':
        depth -= 1

last = ''.join(cur).strip()
if last:
    parts.append(last)

print("extern memref_3d_f32 model_main(")
for i, p in enumerate(parts):
    ty = p.split()[0]   # ptr / i64
    cty = "void *" if ty == "ptr" else "int64_t"
    comma = "," if i != len(parts) - 1 else ""
    print(f"    {cty} a{i}{comma}")
print(");")