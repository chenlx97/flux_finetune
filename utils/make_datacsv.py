import os
import json
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

    target_dir = '/data/clx/data/wenli_data/tile_flux2_klein/resolution512/target'
    condition_dir = '/data/clx/data/wenli_data/tile_flux2_klein/resolution512/distorted'
    output_file="/data/clx/data/wenli_data/tile_flux2_klein/resolution512/train.jsonl"
    prompt_single = "修复扭曲的建筑立面纹理，使纹理自然、清晰、结构不变"

    build_dataset(
        condition_dir=condition_dir,
        target_dir=target_dir,
        output_file=output_file,
        prompt_single = prompt_single
    )