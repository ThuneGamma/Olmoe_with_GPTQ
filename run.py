import os
import subprocess
import shlex
from itertools import product

MODEL_PATH = "allenai/OLMoE-1B-7B-0125-Instruct"
DATASETS = ["wikitext2", "c4"]
WBITS = [4, 3, 2]
LOG_DIR = "./quantization_logs"


def main():
    tasks = list(product(DATASETS, WBITS))

    # 从 tasks 列表中进行标准循环，不再使用 tqdm
    for dataset, wbits in tasks:

        log_file_path = os.path.join(LOG_DIR, f"{dataset}_{wbits}bit.log")

        command = (
            f"python olmoe.py "
            f"{shlex.quote(MODEL_PATH)} "
            f"{shlex.quote(dataset)} "
            f"--wbits {wbits}"
        )

        # 使用 print() 函数直接输出信息
        print(f"量化数据集: {dataset}, 位数: {wbits} bits")

        with open(log_file_path, "w") as log_file:
            subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                stdout=log_file,
                stderr=log_file
            )

        # 使用一个空行来分隔任务输出，使其更清晰
        print(f"✅ [任务成功] {dataset} @ {wbits}-bit 量化完成。\n")

    print("\n🎉 所有量化任务已全部执行完毕。")

if __name__ == "__main__":
    main()
