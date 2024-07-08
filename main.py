import requests
import transformers
from PIL import Image

model_id = "microsoft/Phi-3-vision-128k-instruct" 


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    trust_remote_code=True, 
    torch_dtype="auto", 
) 

message = [ 
    {"role": "user", "content": "\nWhat is shown in this image?", "image": "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"},
] 

processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

prompt = processor.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, return_tensors="pt")

generation_args = { 
    "max_new_tokens": 500, 
    # "temperature": 0.0, 
    "do_sample": True, 
} 

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 


generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response)
