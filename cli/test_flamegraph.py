#!/usr/bin/env python3
"""
SoulX-Podcast 火焰图性能分析工具

这个脚本使用 Python profiler + flamegraph.pl 生成火焰图，用于分析 CPU 性能瓶颈。

使用方法:
    python test_flamegraph.py --duration 30

依赖:
    - FlameGraph 脚本 (git clone https://github.com/brendangregg/FlameGraph.git)
"""

import os
import sys
import argparse
import subprocess
import cProfile
import pstats
import io
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import soundfile as sf
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input


class FlameGraphProfiler:
    """使用 Python profiler + flamegraph.pl 生成火焰图的性能分析器"""

    def __init__(self, output_dir: str = "flamegraph_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 检查依赖
        self._check_dependencies()

    def _check_dependencies(self):
        """检查必要的依赖"""
        # 检查 FlameGraph 脚本是否存在
        flamegraph_script = self._find_flamegraph_script()
        if not flamegraph_script:
            print("❌ FlameGraph 脚本未找到")
            print("💡 请下载 FlameGraph 脚本:")
            print("   git clone https://github.com/brendangregg/FlameGraph.git")
            print("   并确保 flamegraph.pl 脚本在 PATH 中或指定位置")
            sys.exit(1)
        else:
            print(f"✅ 找到 FlameGraph 脚本: {flamegraph_script}")

    def _find_flamegraph_script(self) -> Optional[Path]:
        """查找 flamegraph.pl 脚本"""
        # 常见位置
        possible_paths = [
            Path("/workspace/bella-infra/user/libeibei031/FlameGraph/flamegraph.pl"),
        ]

        # 检查 PATH
        try:
            result = subprocess.run(['which', 'flamegraph.pl'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass

        # 检查预定义路径
        for path in possible_paths:
            if path.exists():
                return path

        return None

    def convert_profile_to_folded(self, profile_data, output_file):
        """将 Python profile 数据转换为 flamegraph.pl 需要的 folded 格式"""
        print("📊 转换 profile 数据为 folded 格式...")

        # 解析 profile 数据
        stats = pstats.Stats(profile_data)
        stats.sort_stats('cumulative')

        folded_lines = []

        # 遍历所有函数调用
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, lineno, funcname = func

            # 创建函数标识符
            if filename == '<built-in>':
                func_id = f"<built-in>.{funcname}"
            else:
                # 简化文件路径
                short_filename = os.path.basename(filename)
                func_id = f"{short_filename}:{lineno}({funcname})"

            # 构建调用栈（简化版）
            if callers:
                for caller_func, caller_stats in callers.items():
                    caller_filename, caller_lineno, caller_funcname = caller_func
                    if caller_filename == '<built-in>':
                        caller_id = f"<built-in>.{caller_funcname}"
                    else:
                        caller_short = os.path.basename(caller_filename)
                        caller_id = f"{caller_short}:{caller_lineno}({caller_funcname})"

                    # 创建调用栈字符串（caller;callee 格式）
                    stack = f"{caller_id};{func_id}"
                    count = int(caller_stats[0])  # 调用次数
                    if count > 0:
                        folded_lines.append(f"{stack} {count}")
            else:
                # 顶级函数
                count = int(cc)
                if count > 0:
                    folded_lines.append(f"{func_id} {count}")

        # 写入 folded 文件
        with open(output_file, 'w') as f:
            for line in folded_lines:
                f.write(line + '\n')

        print(f"📄 已生成 folded 文件: {output_file} ({len(folded_lines)} 行)")
        return len(folded_lines) > 0

    def run_profiling(self, model, dataset, processed_data):
        """使用 Python profiler 进行性能分析并生成火焰图"""
        print("🔥 开始性能分析...")

        # 创建输出文件路径
        profile_file = self.output_dir / f"profile_{self.timestamp}.prof"
        folded_file = self.output_dir / f"folded_{self.timestamp}.txt"
        flame_graph_file = self.output_dir / f"flamegraph_{self.timestamp}.svg"

        def run_inference():
            """执行推理（被 profiler 监测）"""
            print("📊 开始执行推理...")

            # 执行推理
            results_dict = model.forward_longform(**processed_data)

            # 处理结果
            target_audio = None
            for wav in results_dict["generated_wavs"]:
                if target_audio is None:
                    target_audio = wav
                else:
                    target_audio = torch.cat([target_audio, wav], dim=1)

            # 保存音频
            output_path = "outputs/flamegraph_test_audio.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, target_audio.cpu().squeeze(0).numpy(), 24000)

            print("✅ 推理完成")
            return output_path

        try:
            # 创建 profiler
            profiler = cProfile.Profile()

            # 开始 profiling
            profiler.enable()
            result_path = run_inference()
            profiler.disable()

            # 保存 profile 数据
            profiler.dump_stats(str(profile_file))
            print(f"📊 Profile 数据已保存: {profile_file}")

            # 转换为 folded 格式
            if self.convert_profile_to_folded(profiler, folded_file):
                # 使用 flamegraph.pl 生成火焰图
                flamegraph_script = self._find_flamegraph_script()
                if flamegraph_script and folded_file.exists():
                    print("🔥 生成火焰图...")

                    flame_cmd = ['perl', str(flamegraph_script), str(folded_file)]

                    with open(flame_graph_file, 'w') as output_file:
                        flame_result = subprocess.run(
                            flame_cmd,
                            stdout=output_file,
                            stderr=subprocess.PIPE,
                            text=True
                        )

                    if flame_result.returncode == 0:
                        print(f"🔥 火焰图已保存: {flame_graph_file}")

                        # 打印文件大小信息
                        file_size = flame_graph_file.stat().st_size
                        print(f"📏 火焰图文件大小: {file_size / 1024:.1f} KB")

                        return flame_graph_file
                    else:
                        print(f"❌ 火焰图生成失败: {flame_result.stderr}")
                else:
                    print("❌ FlameGraph 脚本未找到或 folded 文件不存在")
            else:
                print("❌ 转换 folded 格式失败")

        except Exception as e:
            print(f"❌ 性能分析失败: {e}")
            raise

        return None


def prepare_model_and_data():
    """准备模型和数据（在性能监测之外完成）"""
    try:
        print("🔄 初始化模型（此阶段不会被监测）...")

        # 测试数据
        prompt_audio = "example/audios/female_mandarin.wav"
        prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者。"
        text = "[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。"
        seed = 1988
        model_dir = "pretrained_models/SoulX-Podcast-1.7B"

        data = {
            "speakers": {
                "S1": {
                    "prompt_audio": prompt_audio,
                    "prompt_text": prompt_text,
                    "dialect_prompt": "",
                }
            },
            "text": [["S1", text]]
        }

        inputs = podcast_format_parser(data)

        # 模型初始化（耗时较长，不包含在性能分析中）
        print("📥 加载模型...")
        model, dataset = initiate_model(seed, model_dir, "vllm", False)

        # 数据预处理
        print("📊 预处理数据...")
        processed_data = process_single_input(
            dataset,
            inputs['text'],
            inputs['prompt_wav'],
            inputs['prompt_text'],
            inputs['use_dialect_prompt'],
            inputs['dialect_prompt_text'],
        )

        print("✅ 模型和数据准备完成")
        return model, dataset, processed_data

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SoulX-Podcast 火焰图性能分析工具')
    parser.add_argument('--output-dir', default='flamegraph_results',
                       help='输出目录（默认：flamegraph_results）')

    args = parser.parse_args()

    print("🔥 SoulX-Podcast 火焰图性能分析工具")
    print("=" * 50)

    # 创建分析器
    profiler = FlameGraphProfiler(args.output_dir)

    # 第一步：准备模型和数据（不被监测）
    print("\n📋 准备阶段（不会被监测）...")
    model, dataset, processed_data = prepare_model_and_data()

    # 第二步：运行性能分析（仅监测推理阶段）
    print("\n🔥 开始性能分析（仅监测推理阶段）...")
    print("ℹ️  注意：模型初始化已完成，只监测纯推理性能")
    result = profiler.run_profiling(model, dataset, processed_data)

    # 打印结果
    print("\n" + "=" * 50)
    if result:
        print("🎯 分析完成！")
        print(f"📁 火焰图文件: {result}")
        print(f"📂 结果目录: {profiler.output_dir}")
        print("\n💡 查看火焰图:")
        print(f"   在浏览器中打开: {result}")
        print("   或使用命令: firefox", str(result))
    else:
        print("❌ 火焰图生成失败")
        print("💡 请检查:")
        print("   1. FlameGraph 脚本是否正确安装")
        print("   2. 输出目录是否有写入权限")
        print("   3. 推理过程是否正常完成")


if __name__ == "__main__":
    main()