from vllm import LLM, SamplingParams
import torch
sampling_params = SamplingParams(temperature=0.5,max_tokens=128)
llm = LLM(
    model="shanearora/OLMo-7B-1124-hf",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    enforce_eager = True,
    dtype=torch.float32
)

prompt = "San Francisco is a"
outputs = llm.generate([prompt], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated: {generated_text}")


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load the model and tokenizer
# olmo = AutoModelForCausalLM.from_pretrained(
#     "shanearora/OLMo-7B-1124-hf", 
#     torch_dtype=torch.float32,  # Load model in float16
#     device_map="auto"          # Automatically map layers to GPU
# )
# tokenizer = AutoTokenizer.from_pretrained("shanearora/OLMo-7B-1124-hf")

# # Input prompt
# message = ["San Francisco is a"]
# inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)

# # Move inputs to CUDA (optional since device_map handles this)
# inputs = {k: v.to('cuda') for k, v in inputs.items()}
# response = olmo.generate(
#     **inputs,
#     max_new_tokens=256,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     temperature=0.5)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


# import torch.distributed as dist
# if dist.is_initialized():
#     dist.destroy_process_group()

# import torch
# from vllm import SamplingParams, LLM
# from vllm.model_executor.utils import set_random_seed
# from transformers import AutoModelForCausalLM, AutoTokenizer

# def compare_vllm_with_hf(prompt: str = "My name is John! I am ", which: str = "both"):

#     assert which in ["both", "hf", "vllm"]

#     outputs = {}

#     # VLLM
#     if which in ["vllm", "both"]:
#         s = SamplingParams(temperature=0.5,max_tokens=128)
#         llm = LLM(model="shanearora/OLMo-7B-1124-hf", trust_remote_code=True, enforce_eager = True, 
#                   gpu_memory_utilization=0.90, dtype=torch.bfloat16)

#         set_random_seed(0)
#         vllm_out = llm.generate([prompt], sampling_params=s)
#         outputs["vllm"] = vllm_out[0].outputs[0].text

#     # HF
#     if which in ["hf", "both"]:
#         hf_model = AutoModelForCausalLM.from_pretrained("shanearora/OLMo-7B-1124-hf", torch_dtype=torch.bfloat16,trust_remote_code=True, device_map="auto")
#         tokenizer = AutoTokenizer.from_pretrained("shanearora/OLMo-7B-1124-hf")
#         input = tokenizer.encode(prompt)
#         input_tensor = torch.tensor(input).unsqueeze(0)

#         set_random_seed(0)
#         hf_gen = hf_model.generate(input_tensor.long(), max_new_tokens=100, temperature=0.5)

#         outputs["hf"] = tokenizer.decode(hf_gen[0].tolist())

#     print()
#     print(outputs)


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) == 2:
#         compare_vllm_with_hf(sys.argv[1])
#     else:
#         compare_vllm_with_hf(sys.argv[1])
