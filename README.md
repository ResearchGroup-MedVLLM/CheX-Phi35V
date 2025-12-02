
# CheX-Phi3.5V 🩻
## News
[2025-12-02] 🔥 We are excited to release CheX-Phi4MM, the official implementation of CheXPO-v2. By utilizing GRPO and Knowledge Graph Consistency for process supervision, this work significantly enhances the diagnostic reasoning and accuracy of Chest X-ray VLMs.

🔗 Check it out: https://github.com/ecoxial2007/CheX-Phi4MM

## Overview
CheX-Phi3.5V is a vision-language model (VLM) for chest X-ray interpretation. It integrates preference optimization and contrastive rejection strategies for fine-grained medical reasoning. The repository contains scripts for dataset preparation, training, and evaluation based on the MIMIC-CXR dataset.

## Setup 🚀

### 1. Environment Configuration 🖥️
1. Clone the repository:
   ```bash
   git clone https://github.com/remove4anonymous/CheX-Phi35V.git
   cd CheX-Phi35V
   ```

2. Install dependencies:
   ```bash
   git clone https://github.com/huggingface/trl.git
   cd trl/
   git checkout v0.14.0
   pip install -e .[dev]
   ```
3. Copy the modified DPOTrainer to `trl`
    ```bash
    cp src/dpo_*.py trl/trl/trainer/
    ```

### 2. Prepare Pre-trained Weights ⚙️

To fine-tune and evaluate the **CheX-Phi3.5V** model, you will need to download and place the pre-trained weights into the `checkpoints` directory. The following table summarizes the available pre-trained models and their respective descriptions.

| **Model**                                          | **Description**                                                      |
|-----------------------------------------------------|----------------------------------------------------------------------|
| [CheX-Phi3.5-Vision-Inst-DPO](https://huggingface.co/remove4anonymous/CheX-Phi-3.5-vision-instruct-DPO)          | DPO Model for evaluation and demo. |
| [CheX-Phi3.5-Vision-Inst-SFT](https://huggingface.co/remove4anonymous/CheX-Phi-3.5-vision-instruct-SFT)          | SFT Model for stage2.          |
| [Phi-3.5-Vision-Inst](https://huggingface.co/remove4anonymous/Phi-3.5-vision-instruct)| Base Phi-3.5 Vision model weights for SFT (stage1). Make some modification to fix merge problem.|

### 3. Prepare Data 📊

1. **Download Chest X-ray Images**  
   Download the MIMIC-CXR-JPG dataset (version 2.1.0) from [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0).

2. **Generate Region Annotations**  
   Use `tools/generate_visual_prompt.py` to generate region annotations (bounding boxes) from the images:
   ```bash
   python tools/(option)_generate_visual_prompt.py
   ```

3. **Download Annotations**  
   Annotations for SFT and DPO can be found at the provided links:
   - [DPO Annotations](#)
   - [SFT Annotations](#)

Ensure the following folder structure:
```bash
data/
  ├── images/
  │   ├── MIMIC_CXR_JPG/
  │   └── MIMIC_CXR_JPG_region/
  └── annotations/
      ├── DPO/train_dpo_{20k, 30k}.json
      └── SFT/{train, valid, test}_{basic, region, compare}_vqa.json
```
## SFT, DPO, Evaluation, and Demo

1. `stage1_sft_mimic_ext_vqa.sh`
   Fine-tunes the model using supervised learning on the MIMIC-EXT VQA dataset.

2. `stage2_dpo_mimic_ext_vqa.sh`
   Performs Direct Preference Optimization (DPO) to improve model preferences.

3. `stage3_evaluate_mimic_ext_vqa.sh`
   Evaluates the model's performance after DPO training.

4. `demo.py`
    Try CheX-Phi3.5V.

## Citation 📚

If you find this repository useful in your research, please cite the following:

```bibtex
@misc{liang2025chexpopreferenceoptimizationchest,
      title={CheXPO: Preference Optimization for Chest X-ray VLMs with Counterfactual Rationale}, 
      author={Xiao Liang and Jiawei Hu and Di Wang and Zhi Ma and Lin Zhao and Ronghan Li and Bo Wan and Quan Wang},
      year={2025},
      eprint={2507.06959},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.06959}, 
}
```

Thank you for your interest in **CheX-Phi3.5V**! Feel free to contribute and make improvements.

