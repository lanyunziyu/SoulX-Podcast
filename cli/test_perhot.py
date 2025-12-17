# import os
# import json
# import torch

# import s3tokenizer
# import soundfile as sf

# from soulxpodcast.config import SamplingParams
# from soulxpodcast.utils.parser import podcast_format_parser
# from soulxpodcast.utils.infer_utils import initiate_model, process_single_input

# def run_inference(
#     inputs: dict,
#     model_path: str,
#     output_path: str,
#     llm_engine: str = "hf",
#     fp16_flow: bool = False,
#     seed: int = 1988,
# ):
    
#     model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)
    
#     # The 'inputs' dictionary is already formatted by podcast_format_parser
#     # for the entire longform process. If you want to simulate two separate
#     # data inputs (data1, data2) resulting in two model calls, you need to
#     # modify the structure here.
    
#     # Assuming the structure is: inputs -> data (preprocessed for model)
#     data = process_single_input(
#         dataset,
#         inputs['text'],
#         inputs['prompt_wav'],
#         inputs['prompt_text'],
#         inputs['use_dialect_prompt'],
#         inputs['dialect_prompt_text'],
#     )
    
#     # --- START MODIFICATION FOR data1/data2 ---
#     # This block simulates two different data inputs leading to two model calls,
#     # as requested by the user, although the original script logic and shell example
#     # only set up for one.
    
#     # For demonstration, we'll assume `data` from the original input is `data1`,
#     # and we create a placeholder `data2` which is the same for simple demonstration.
#     # In a real scenario, you'd need a second set of inputs to generate `data2`.
    
#     # data1 is the data from the initial parsing
#     data1 = data 
    
#     # Placeholder for data2. In a real application, you'd need another call to
#     # process_single_input with a different set of texts/prompts.
#     data2 = data # Using the same data for data2 for this example
    
#     # print("[INFO] Start inference for data1...")
#     # results_dict_1 = model.forward_longform(**data1) # <-- data1 call
    
#     print("[INFO] Start inference for data2...")
#     results_dict_2 = model.forward_longform(**data2) # <-- data2 call
    
#     # --- END MODIFICATION FOR data1/data2 ---

#     # We will now process results_dict_1 (assuming the user wants to save the first result)
#     # If you want to process and save *both*, you would need to rename the output files.
    
#     target_audio = None
#     for wav in results_dict_2["generated_wavs"]: # Process data1 results
#         if target_audio is None:
#             target_audio = wav
#         else:
#             target_audio = torch.cat([target_audio, wav], dim=1)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     sf.write(output_path, target_audio.cpu().squeeze(0).numpy(), 24000)
#     print(f"[INFO] Saved synthesized audio to: {output_path} (from data1)")
    
#     # If you wanted to save the second result, you would do:
#     # target_audio_2 = None
#     # for wav in results_dict_2["generated_wavs"]:
#     #     ...
#     # sf.write(output_path.replace(".wav", "_2.wav"), target_audio_2.cpu().squeeze(0).numpy(), 24000)
#     # print(f"[INFO] Saved synthesized audio to: {output_path.replace('.wav', '_2.wav')} (from data2)")



# def run_client_test_case():
#     """Runs inference using the same data as test_sync_single_speaker from client_test.py"""
#     print("--- Running Client Test Case (test_sync_single_speaker data) ---")

#     # Data extracted from client_test.py test_sync_single_speaker function
#     # Audio file path
#     prompt_audio = "example/audios/female_mandarin.wav"

#     # prompt_texts from client_test.py (was JSON dumped as ["喜欢攀岩、徒步、滑雪的语言爱好者。"])
#     prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者。"

#     # dialogue_text from client_test.py
#     text = "[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。"

#     # seed from client_test.py
#     seed = 1988

#     # Model path (using a reasonable default)
#     model_dir = "pretrained_models/SoulX-Podcast-1.7B-dialect"

#     # Output path
#     output_path = "outputs/client_test_single_speaker.wav"

#     # No dialect prompt in the original client test
#     dialect_prompt = ""

#     data = {
#         "speakers": {
#             "S1": {
#                 "prompt_audio": prompt_audio,
#                 "prompt_text": prompt_text,
#                 "dialect_prompt": dialect_prompt,
#             }
#         },
#         "text": [
#             ["S1", text]
#         ]
#     }

#     inputs = podcast_format_parser(data)
#     run_inference(
#         inputs=inputs,
#         model_path=model_dir,
#         output_path=output_path,
#         llm_engine="vllm",
#         fp16_flow=False,
#         seed=seed,
#     )
#     print("--- Client Test Case Finished ---")


# if __name__ == "__main__":
#     # 直接运行客户端测试案例
#     run_client_test_case()



import os
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

import s3tokenizer
import soundfile as sf

from soulxpodcast.config import SamplingParams
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input

def run_inference(
    inputs: dict,
    model_path: str,
    output_path: str,
    llm_engine: str = "hf",
    fp16_flow: bool = False,
    seed: int = 1988,
    enable_profiler: bool = True,
):

    # Setup PyTorch Profiler
    profiler_output_dir = "profiler_results"
    os.makedirs(profiler_output_dir, exist_ok=True)

    # Model initialization (without profiling to avoid warmup overhead)
    model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)

    # Data preprocessing
    data = process_single_input(
        dataset,
        inputs['text'],
        inputs['prompt_wav'],
        inputs['prompt_text'],
        inputs['use_dialect_prompt'],
        inputs['dialect_prompt_text'],
    )

    # First inference (warmup - not profiled)
    print("[INFO] Start inference for data1 (warmup)...")
    results_dict_1 = model.forward_longform(**data)
    results_dict = model.forward_longform(**data)
    results_dict_3 = model.forward_longform(**data)




    # Only profile the second inference
    if enable_profiler:
        # Profiler configuration with TensorBoard support
        # tensorboard_trace_handler will save trace files that can be viewed in both TensorBoard and Chrome
        profiler = profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1
            )
        )
        profiler.start()
        print(f"[INFO] PyTorch Profiler started for second inference with TensorBoard logging")

    # Second inference with detailed profiling
    with record_function("model_inference_data2"):
        print("[INFO] Start inference for data2 (profiled)...")
        prompt_audio = "example/audios/female_mandarin.wav"
        prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者。"
        text = "[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。"
        dialect_prompt = ""

        data = {
            "speakers": {
                "S1": {
                    "prompt_audio": prompt_audio,
                    "prompt_text": prompt_text,
                    "dialect_prompt": dialect_prompt,
                }
            },
            "text": [
                ["S1", text]
            ]
        }
        inputs = podcast_format_parser(data)
        data = process_single_input(
            dataset,
            inputs['text'],
            inputs['prompt_wav'],
            inputs['prompt_text'],
            inputs['use_dialect_prompt'],
            inputs['dialect_prompt_text'],
        )
        results_dict_2 = model.forward_longform(**data)

        if enable_profiler:
            # Step the profiler to trigger on_trace_ready callback
            profiler.step()

    if enable_profiler:
        profiler.stop()
        print("[INFO] PyTorch Profiler stopped")
        print(f"[INFO] Profiler traces saved to: {profiler_output_dir}")
        print(f"[INFO] - View in TensorBoard: tensorboard --logdir {profiler_output_dir}")
        print(f"[INFO] - View in Chrome: Open chrome://tracing and load the .json file in {profiler_output_dir}")

        # Print detailed analysis
        print("\n[PROFILER] Top 15 operations by CUDA time:")
        # print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        print("\n[PROFILER] Top 15 operations by CPU time:")
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        print("\n[PROFILER] Memory usage summary:")
        # print(profiler.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        # Additional kernel analysis
        print("\n[PROFILER] Detailed kernel analysis:")
        events = profiler.key_averages(group_by_input_shape=True)
        kernel_events = [event for event in events if 'kernel' in event.key.lower()]
        if kernel_events:
            print("Top kernels by CUDA time:")
            for event in sorted(kernel_events, key=lambda x: x.cuda_time_total, reverse=True)[:10]:
                print(f"  {event.key}: {event.cuda_time_total/1000:.3f}ms")

    # Process and save audio from second inference
    target_audio = None
    for wav in results_dict_2["generated_wavs"]:
        if target_audio is None:
            target_audio = wav
        else:
            target_audio = torch.cat([target_audio, wav], dim=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, target_audio.cpu().squeeze(0).numpy(), 24000)
    print(f"[INFO] Saved synthesized audio to: {output_path} (from data2)")

    # If you wanted to save the second result, you would do:
    # target_audio_2 = None
    # for wav in results_dict_2["generated_wavs"]:
    #     ...
    # sf.write(output_path.replace(".wav", "_2.wav"), target_audio_2.cpu().squeeze(0).numpy(), 24000)
    # print(f"[INFO] Saved synthesized audio to: {output_path.replace('.wav', '_2.wav')} (from data2)")



def run_client_test_case():
    """Runs inference using the same data as test_sync_single_speaker from client_test.py"""
    print("--- Running Client Test Case (test_sync_single_speaker data) ---")

    # Data extracted from client_test.py test_sync_single_speaker function
    # Audio file path
    prompt_audio = "example/audios/female_mandarin.wav"

    # prompt_texts from client_test.py (was JSON dumped as ["喜欢攀岩、徒步、滑雪的语言爱好者。"])
    prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者。"

    # dialogue_text from client_test.py
    text = "[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。"

    # seed from client_test.py
    # seed = 1988

    # Model path (using a reasonable default)
    model_dir = "pretrained_models/SoulX-Podcast-1.7B"

    # Output path
    output_path = "outputs/client_test_single_speaker.wav"

    # No dialect prompt in the original client test
    dialect_prompt = ""

    data = {
        "speakers": {
            "S1": {
                "prompt_audio": prompt_audio,
                "prompt_text": prompt_text,
                "dialect_prompt": dialect_prompt,
            }
        },
        "text": [
            ["S1", text]
        ]
    }

    inputs = podcast_format_parser(data)
    run_inference(
        inputs=inputs,
        model_path=model_dir,
        output_path=output_path,
        llm_engine="vllm",
        fp16_flow=True,
        # seed=seed,
        enable_profiler=True,
    )
    print("--- Client Test Case Finished ---")
    print("--- Profiler Results ---")
    print("Check 'profiler_results/' directory for detailed profiling data:")
    print("  - TensorBoard logs: tensorboard --logdir profiler_results")
    print("  - Chrome trace: Open profiler_results/trace.json in Chrome://tracing")


if __name__ == "__main__":
    # 直接运行客户端测试案例
    run_client_test_case()