import json

# 输入和输出路径
jsonl_path = "input/user_103.jsonl"  # 你的原始对话文件
json_path = "input/user_103.json"    # Graphrag 支持的输出文件

# 转换逻辑
with open(jsonl_path, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if line.strip()]

with open(json_path, 'w', encoding='utf-8') as out:
    json.dump(lines, out, indent=2, ensure_ascii=False)

print(f"[✓] 成功将 {jsonl_path} 转换为 {json_path}")
