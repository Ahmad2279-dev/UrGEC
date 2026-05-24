import random
import os


def split_jsonl_dataset(
        input_file,
        train_file,
        dev_file,
        test_file,
        dev_size=5000,
        test_size=5000,
        seed=42
):
    """
    将完整的 JSONL 数据集随机划分为训练集、验证集和测试集。
    """
    print(f"正在读取文件: {input_file}...")

    if not os.path.exists(input_file):
        print(f"错误：找不到文件 '{input_file}'，请确认文件名和路径。")
        return

    # 1. 一次性读取所有行（82万条数据约占 100~200MB 内存，现代电脑毫无压力）
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_size = len(lines)
    print(f"总共读取到 {total_size} 条数据。")

    if total_size < dev_size + test_size:
        print(f"错误：数据总量 ({total_size}) 小于 Dev 和 Test 的总和！")
        return

    # 2. 设置随机种子并打乱数据
    # ⚠️ 极其重要：设置 seed=42 保证了“可复现性”。
    # 如果你以后不小心删了文件重新跑脚本，抽出来的 5000 条依然是同一批句子。
    print(f"正在打乱数据 (Random Seed: {seed})...")
    random.seed(seed)
    random.shuffle(lines)

    # 3. 列表切片完成划分
    dev_data = lines[:dev_size]
    test_data = lines[dev_size: dev_size + test_size]
    train_data = lines[dev_size + test_size:]

    print("\n=== 划分结果 ===")
    print(f" - 验证集 (Dev)  : {len(dev_data):,} 条 -> {dev_file}")
    print(f" - 测试集 (Test) : {len(test_data):,} 条 -> {test_file}")
    print(f" - 训练集 (Train): {len(train_data):,} 条 -> {train_file}")
    print("================\n")

    # 4. 辅助函数：将列表快速写回为 JSONL 文件
    def write_jsonl(data_list, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(data_list)

    print("正在保存文件，请稍候...")
    write_jsonl(dev_data, dev_file)
    write_jsonl(test_data, test_file)
    write_jsonl(train_data, train_file)
    print("🎉 所有文件保存完成！")


if __name__ == "__main__":
    # 输入文件（你上一步生成的总文件）
    INPUT_JSONL = "UrGECtr_train.jsonl"

    # 输出的三个独立文件
    TRAIN_OUTPUT = "urgec_train.jsonl"
    DEV_OUTPUT = "urgec_dev.jsonl"
    TEST_OUTPUT = "urgec_test.jsonl"

    split_jsonl_dataset(
        input_file=INPUT_JSONL,
        train_file=TRAIN_OUTPUT,
        dev_file=DEV_OUTPUT,
        test_file=TEST_OUTPUT
    )