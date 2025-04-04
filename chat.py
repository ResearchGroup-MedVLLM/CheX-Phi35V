from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
import io
import base64
from peft import PeftModel

app = Flask(__name__)

# 模型和路径配置
model_path = "./data/checkpoints/CheX-Phi-3.5-vision-instruct-DPO"
tokenizer_path = "./data/checkpoints/CheX-Phi-3.5-vision-instruct-DPO"
lora_model_path = ""

# 加载模型
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True,
    quantization_config=bnb_config
)
print(f"Model is on device: {next(model.parameters()).device}")

# # 加载 LoRA 权重
# model = PeftModel.from_pretrained(model, lora_model_path)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,
    num_crops=4
)

# 生成参数
generation_args = {
    "max_new_tokens": 128,
    "do_sample": False
}


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    question = data.get("question", "")
    image_data_list = data.get("images", [])
    top_k_captions = data.get("top_k_captions", [])

    print(f"Question: {question}")

    # 处理图像数据
    images = []
    for image_data in image_data_list:
        try:
            # 解码图片
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # 确保图像为 RGB 格式
            images.append(image)
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400  # 返回错误信息

    # 准备提示文本
    num_images = len(images)
    image_placeholders = ''.join([f"<|image_{i + 1}|>\n" for i in range(num_images)])

    # 直接使用 captions_text 拼接
    captions_text = top_k_captions  # top_k_capt
    tips = "please answer in Chinese"

    prompt = (
        "<|user|>\n"
        "请查看以下参考内容：\n"
        f"1. 与问题相关的描述：{captions_text}\n"
        f"2. 与问题相关或相似的图像，前十张一一对应上述描述，第十一张图像为用户上传跟问题直接相关：{image_placeholders}\n"
        f"3. 用户问题：{question}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    # 处理输入
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt"
    ).to("cuda:0")


    # 生成输出
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # 移除输入 tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Response: {response}")
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
