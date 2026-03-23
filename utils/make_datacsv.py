import os
import json
import argparse
from glob import glob


def load_dataset(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(
    target_dir,
    output_file,
    condition_dir = None,
    prompt_single=None,

):
    target_files = sorted(glob(os.path.join(target_dir,'**.*')))
    if condition_dir:
        condition_files = sorted(glob(os.path.join(condition_dir,'**.*')))
        assert len(condition_files) == len(target_files), "数量不一致！"
        with open(output_file, "w", encoding="utf-8") as f:
            for cond_file, tgt_file in zip(condition_files, target_files):
                assert os.path.basename(cond_file) == os.path.basename(tgt_file), f"文件名不匹配: {cond_file} vs {tgt_file}"
                if prompt_single:
                    data = {
                        "target": tgt_file,
                        "condition": cond_file,
                        "prompt": prompt_single
                    }
                else:
                    data = {
                        "target": tgt_file,
                        "condition": cond_file,
                        "prompt": os.path.basename(tgt_file).split('.')[0]
                    }

                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            for tgt_file in target_files:
                if prompt_single:
                    data = {
                        "target": tgt_file,
                        "condition": None,
                        "prompt": prompt_single
                    }
                else:
                    data = {
                        "target": tgt_file,
                        "condition": None,
                        "prompt": os.path.basename(tgt_file).split('.')[0]
                    }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ 数据集已生成: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 JSONL 数据集")

    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="目标图片目录"
    )

    parser.add_argument(
        "--condition_dir",
        type=str,
        default=None,
        help="条件图片目录（可选）"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出 jsonl 文件路径"
    )

    parser.add_argument(
        "--prompt_single",
        type=str,
        default=None,
        help="统一 prompt（可选）"
    )

    args = parser.parse_args()

    build_dataset(
        target_dir=args.target_dir,
        condition_dir=args.condition_dir,
        output_file=args.output_file,
        prompt_single=args.prompt_single
    )