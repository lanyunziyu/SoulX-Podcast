#!/usr/bin/env python3
"""
对比V0和V1的hidden states文件差异
"""

import numpy as np
from typing import List, Tuple
import os

def read_tensor_file(file_path: str) -> List[float]:
    """
    读取tensor数据文件

    Args:
        file_path: 文件路径

    Returns:
        数值列表
    """
    values = []

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return values

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue

                try:
                    value = float(line)
                    values.append(value)
                except ValueError:
                    print(f"⚠️ 第{line_num}行无法解析为浮点数: '{line}'")
                    continue

    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return []

    return values

def compare_tensor_values(values1: List[float], values2: List[float],
                         name1: str = "V1", name2: str = "V0",
                         tolerance: float = 1e-8) -> dict:
    """
    对比两个tensor的数值

    Args:
        values1: 第一个tensor的值
        values2: 第二个tensor的值
        name1: 第一个tensor的名称
        name2: 第二个tensor的名称
        tolerance: 数值容忍度

    Returns:
        对比结果字典
    """
    result = {
        "identical": False,
        "length_match": False,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "num_different": 0,
        "different_indices": [],
        "statistics": {},
        "total_values": 0  # 添加总数量字段
    }

    # 检查长度
    len1, len2 = len(values1), len(values2)
    result["length_match"] = (len1 == len2)

    print(f"📊 数据长度对比:")
    print(f"  {name1}: {len1} 个值")
    print(f"  {name2}: {len2} 个值")
    print(f"  长度匹配: {'✅' if result['length_match'] else '❌'}")

    if not result["length_match"]:
        print(f"⚠️ 长度不匹配，无法进行详细对比")
        return result

    if len1 == 0:
        print(f"⚠️ 两个文件都是空的")
        return result

    # 记录总数量
    result["total_values"] = min(len1, len2)

    # 转换为numpy数组进行计算
    arr1 = np.array(values1[:result["total_values"]])
    arr2 = np.array(values2[:result["total_values"]])

    # 计算差异
    diff = np.abs(arr1 - arr2)
    result["max_diff"] = float(np.max(diff))
    result["mean_diff"] = float(np.mean(diff))

    # 找出不同的位置
    different_mask = diff > tolerance
    result["num_different"] = int(np.sum(different_mask))
    result["different_indices"] = np.where(different_mask)[0].tolist()

    # 判断是否完全相同
    result["identical"] = (result["num_different"] == 0)

    # 计算统计信息
    result["statistics"] = {
        f"{name1}_min": float(np.min(arr1)),
        f"{name1}_max": float(np.max(arr1)),
        f"{name1}_mean": float(np.mean(arr1)),
        f"{name1}_std": float(np.std(arr1)),
        f"{name2}_min": float(np.min(arr2)),
        f"{name2}_max": float(np.max(arr2)),
        f"{name2}_mean": float(np.mean(arr2)),
        f"{name2}_std": float(np.std(arr2)),
    }

    return result

def print_comparison_results(result: dict, name1: str = "V1", name2: str = "V0"):
    """打印对比结果"""
    print("\n" + "="*60)
    print("📋 详细对比结果")
    print("="*60)

    # 总体结果
    if result["identical"]:
        print(f"🎉 结果: {name1} 和 {name2} 的 hidden states 完全一致!")
    else:
        print(f"⚠️ 结果: {name1} 和 {name2} 的 hidden states 存在差异")

    # 差异统计
    print(f"\n📊 差异统计:")
    print(f"  总数值数量: {result.get('total_values', 0)}")
    print(f"  不同值的数量: {result['num_different']}")
    if result.get("total_values", 0) > 0:
        diff_percentage = (result['num_different'] / result['total_values']) * 100
        print(f"  差异比例: {diff_percentage:.4f}%")
    print(f"  最大差异: {result['max_diff']:.10f}")
    print(f"  平均差异: {result['mean_diff']:.10f}")

    # 统计信息对比
    if result["statistics"]:
        print(f"\n📈 统计信息对比:")
        stats = result["statistics"]
        print(f"  {name1} - 最小值: {stats.get(f'{name1}_min', 'N/A'):.8f}")
        print(f"  {name2} - 最小值: {stats.get(f'{name2}_min', 'N/A'):.8f}")
        print(f"  {name1} - 最大值: {stats.get(f'{name1}_max', 'N/A'):.8f}")
        print(f"  {name2} - 最大值: {stats.get(f'{name2}_max', 'N/A'):.8f}")
        print(f"  {name1} - 平均值: {stats.get(f'{name1}_mean', 'N/A'):.8f}")
        print(f"  {name2} - 平均值: {stats.get(f'{name2}_mean', 'N/A'):.8f}")
        print(f"  {name1} - 标准差: {stats.get(f'{name1}_std', 'N/A'):.8f}")
        print(f"  {name2} - 标准差: {stats.get(f'{name2}_std', 'N/A'):.8f}")

    # 显示前几个不同的位置
    if result["different_indices"] and len(result["different_indices"]) > 0:
        print(f"\n🔍 前10个不同位置的详情:")
        # 这里需要原始数据来显示具体差异，先显示索引
        indices_to_show = result["different_indices"][:10]
        for idx in indices_to_show:
            print(f"  索引 {idx}: 存在差异")

    print("="*60)

def main():
    """主函数"""
    print("🔍 Hidden States 对比工具")
    print("="*60)

    # 文件路径
    v1_file = "/workspace/bella-infra/user/libeibei031/SoulX/SoulX-Podcast-main/test_data_complete/prompt_mels_for_llm.txt"
    v0_file = "/workspace/bella-infra/user/libeibei031/SoulX/SoulX-Podcast-main/test_data_complete/prompt_mels_for_llm1.txt"

    print(f"📁 文件路径:")
    print(f"  V1 文件: {v1_file}")
    print(f"  V0 文件: {v0_file}")

    # 读取数据
    print(f"\n📖 读取数据...")
    v1_values = read_tensor_file(v1_file)
    v0_values = read_tensor_file(v0_file)

    if not v1_values and not v0_values:
        print("❌ 两个文件都无法读取或为空")
        return

    # 进行对比
    print(f"\n⚖️ 开始对比...")
    result = compare_tensor_values(v1_values, v0_values, "V1", "V0")

    # 显示结果
    print_comparison_results(result, "V1", "V0")

    # 如果有显著差异，提供进一步分析建议
    if not result["identical"]:
        print(f"\n💡 差异分析建议:")
        if result["max_diff"] < 1e-6:
            print(f"  ✅ 差异很小(< 1e-6)，可能是数值精度差异")
            print(f"  ✅ Flash2 vs Flash3的精度差异在可接受范围内")
        elif result["max_diff"] < 1e-3:
            print(f"  ⚠️ 差异中等(< 1e-3)，可能是算法实现差异")
            print(f"  ⚠️ 建议检查Flash Attention版本配置")
    else:
        print(f"\n🎉 验证结论:")
        print(f"  ✅ V0和V1的hidden states完全一致")
        print(f"  ✅ Flash2 vs Flash3没有产生数值差异")
        print(f"  ✅ V1的RAS实现基础验证通过")

if __name__ == "__main__":
    main()