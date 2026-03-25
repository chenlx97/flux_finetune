import os
import json
import argparse
from glob import glob

VALID_EXT = [".png", ".jpg", ".jpeg", ".webp"]

def is_image(file):
    return os.path.splitext(file)[-1].lower() in VALID_EXT


def load_caption(txt_path):
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def load_dataset(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(
    target_dir,
    output_file,
    condition_dir=None,
    prompt_single=None,
    use_caption=True,
):
    # ===== 获取图片 =====
    target_files = sorted(
        [f for f in glob(os.path.join(target_dir, "*")) if is_image(f)]
    )

    print(f"找到 {len(target_files)} 张图片")

    if condition_dir:
        condition_files = sorted(
            [f for f in glob(os.path.join(condition_dir, "*")) if is_image(f)]
        )
        assert len(condition_files) == len(target_files)

    valid_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, tgt_file in enumerate(target_files):
            # ===== prompt 优先级 =====
            if prompt_single:
                prompt = prompt_single

            elif use_caption:
                txt_file = os.path.splitext(tgt_file)[0] + ".txt"
                prompt = load_caption(txt_file)

                if prompt is None:
                    print(f"缺少caption，跳过: {tgt_file}")
                    continue

            cond_file = None
            if condition_dir:
                cond_file = condition_files[i]
                assert os.path.basename(cond_file) == os.path.basename(tgt_file), \
                    f"文件名不匹配: {cond_file} vs {tgt_file}"
            data = {
                "target": tgt_file,
                "condition": cond_file,
                "prompt": prompt,
            }

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"有效样本: {valid_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 JSONL 数据集")
    parser.add_argument("--target_dir", type=str, default='/data/clx/tmp/nor/room')
    parser.add_argument("--condition_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default='/data/clx/tmp/nor/train.jsonl')
    parser.add_argument(
        "--prompt_single",
        type=str,
        default=None,
        help="统一 prompt"
    )
    parser.add_argument(
        "--use_caption",
        type=bool,
        default=True,
        help="不使用txt caption"
    )

    args = parser.parse_args()
    build_dataset(
        target_dir=args.target_dir,
        condition_dir=args.condition_dir,
        output_file=args.output_file,
        prompt_single=args.prompt_single,
        use_caption=args.use_caption,
    )

