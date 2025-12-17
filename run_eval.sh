#!/bin/bash
# SoulX-Podcast 评估脚本

# 设置默认参数
MODEL_PATH="pretrained_models/SoulX-Podcast-1.7B"
LLM_ENGINE="vllm"
NUM_SAMPLES=""  # 空表示评估全部样本

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="--num_samples $2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --llm_engine)
            LLM_ENGINE="$2"
            shift 2
            ;;
        --save_audio)
            SAVE_AUDIO="--save_audio"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --lang <zh|en> [--num_samples N] [--model_path PATH] [--llm_engine hf|vllm] [--save_audio]"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$LANG" ]; then
    echo "错误: 必须指定 --lang 参数 (zh 或 en)"
    echo "使用方法:"
    echo "  评估中文: $0 --lang zh"
    echo "  评估英文: $0 --lang en"
    echo "  限制样本数量: $0 --lang zh --num_samples 100"
    echo "  保存音频: $0 --lang zh --save_audio"
    exit 1
fi

# 设置数据集路径
if [ "$LANG" = "zh" ]; then
    META_FILE="data/seedtts_testset/zh/meta.lst"
    OUTPUT_DIR="results/zh"
elif [ "$LANG" = "en" ]; then
    META_FILE="data/seedtts_testset/en/meta.lst"
    OUTPUT_DIR="results/en"
else
    echo "错误: 语言必须是 zh 或 en"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$META_FILE" ]; then
    echo "错误: 数据集文件不存在: $META_FILE"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 显示配置
echo "================================"
echo "SoulX-Podcast 评估配置"
echo "================================"
echo "语言: $LANG"
echo "数据集: $META_FILE"
echo "模型路径: $MODEL_PATH"
echo "LLM引擎: $LLM_ENGINE"
echo "输出目录: $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "样本数量: ${NUM_SAMPLES#--num_samples }"
else
    echo "样本数量: 全部"
fi
if [ -n "$SAVE_AUDIO" ]; then
    echo "保存音频: 是"
else
    echo "保存音频: 否"
fi
echo "================================"
echo ""

# 运行评估
python eval_soulx_local.py \
    --meta_file "$META_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --llm_engine "$LLM_ENGINE" \
    --lang "$LANG" \
    $NUM_SAMPLES \
    $SAVE_AUDIO

# 显示结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "评估完成！"
    echo "结果保存在: $OUTPUT_DIR/evaluation_results.json"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "评估失败！请查看错误信息"
    echo "================================"
    exit 1
fi
