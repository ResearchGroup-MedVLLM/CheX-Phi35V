"""
example for finetuning Phi-3-V on the DocVQA dataset using the Hugging Face Trainer API
Modified from Idefics-2 finetuning notebook:
https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing

Install dependencies:
    pip install transformers==4.38.1 \
        datasets \
        accelerate==0.30.1 \
        peft \
        Levenshtein \
        deepspeed==0.13.1
minimal run:
    torchrun --nproc_per_node=4 finetune_hf_trainer_docvqa.py
"""
import argparse
import warnings
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
warnings.filterwarnings("ignore")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='./Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--lora_path', type=str, help='Lora Save directory')
    parser.add_argument('--save_path', type=str, help='Merge Model Output directory')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')

    args = parser.parse_args()

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'

    base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype="auto"
    )


    processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
    )

    processor.chat_template = processor.tokenizer.chat_template
    print(f"base model:\n{base_model}")

    
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()
    