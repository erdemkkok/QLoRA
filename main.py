import transformers
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from huggingface_hub import login
import os

# Token'ı environment variable'dan al
login(token=os.getenv('HF_TOKEN'))

# pretrained_model_id = "google/paligemma-3b-pt-224"
# processor = AutoProcessor.from_pretrained(pretrained_model_id)
# pretrained_model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_model_id)

import torch
import requests
from PIL import Image
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import Trainer
from transformers import TrainingArguments
from transformers import PaliGemmaProcessor
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

ds = load_dataset('caglarmert/small_riscm')

ds_dataframe = ds['train'].to_pandas()
print(ds_dataframe.head()) 

split_ds = ds["train"].train_test_split(test_size=0.05)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

device = "cpu"
model_id = "google/paligemma-3b-pt-224"

processor = PaliGemmaProcessor.from_pretrained(model_id)
def collate_fn(examples):
    texts = ["<image> <bos> describe this image.}" for example in examples]
    labels= [example['caption'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
    return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
for param in model.vision_tower.parameters():
      param.requires_grad = False
for param in model.multi_modal_projector.parameters():
      param.requires_grad = False

bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
      r=8,
      target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
      task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

args=TrainingArguments(
      num_train_epochs=2,
      per_device_train_batch_size=2,
      gradient_accumulation_steps=4,
      warmup_steps=2,
      learning_rate=2e-5,
      weight_decay=1e-6,
      adam_beta2=0.999,
      logging_steps=2,
      optim="adamw_8bit",
      save_strategy="steps",
      save_steps=10,
      save_total_limit=1,
      output_dir="finetuned_paligemma_riscm_small",
      bf16=True,
      dataloader_pin_memory=False,
      report_to=["tensorboard"],
      remove_unused_columns=False,
)

trainer = Trainer(
      model=model,
      train_dataset=train_ds,
      data_collator=collate_fn,
      args=args
)
trainer.train()

trainer.push_to_hub()

pretrained_model_id = "google/paligemma-3b-pt-224"
finetuned_model_id = "caglarmert/finetuned_paligemma_riscm_small"
processor = AutoProcessor.from_pretrained(pretrained_model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_id)

prompt = "Describe this image."
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(raw_image.convert("RGB"), prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])