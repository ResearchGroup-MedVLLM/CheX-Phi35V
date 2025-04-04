from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

# model_id = "./checkpoints/CheX-Phi-3.5-vision-instruct-DPO/" 
model_id = "../Iu-xray_single/Phi-3.5-vision-instruct-Med-New/" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda:0", 
  trust_remote_code=True, 
  torch_dtype="auto",
  _attn_implementation='eager' 
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=16
) 


image_path = './samples/6aea1fb1-53d33604-3f9f9612-049064e2-7167bfc6.png'
images = [Image.open(image_path)]

messages = [
    {"role": "user", "content": "<|image_1|>\nCan any presence of anatomical findings be noted in the left hilar structures?"},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 128, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print(response)
