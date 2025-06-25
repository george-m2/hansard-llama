from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
import os

def main():
    # ===================================================================
    # 5-6-hour run on 1x RTX 4090:
    # Batch size: 8
    # Gradient: 2
    # 5700 steps
    # 4bit quantization
    # 16 LoRA rank
    # 16 LoRA alpha
    # 50 warmup steps
    # 25 logging steps
    # 1000 save steps
    # Learning rate: 2e-4
    # ===================================================================
    max_seq_length = 1024

    dtype = None
    load_in_4bit = True

    print("Loading the base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map = {"": torch.cuda.current_device()},
    )

    print("Configuring the model for LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    def format_prompts_for_batch(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        if "system" in examples:
            systems = examples["system"]
        else:
            systems = ["You are a helpful assistant."] * len(instructions)
        texts = []
        for instruction, output, system in zip(instructions, outputs, systems):
            text = (f"<|start_header_id|>system<|end_header_id|>\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n{output}<|eot_id|>")
            texts.append(text)
        return { "text" : texts, }

    processed_dataset_path = "tokenized_hansard_dataset"
    raw_dataset_path = "training_data_unified.jsonl"
    output_dir = "outputs"
    final_model_path = "lora_model"

    if os.path.exists(processed_dataset_path):
        print(f"Found pre-processed dataset at '{processed_dataset_path}'")
        formatted_dataset = load_from_disk(processed_dataset_path)
        print("Dataset loaded successfully.")
    else:
        print(f"No pre-processed dataset found. Starting full processing...")
        raw_dataset = load_dataset("json", data_files=raw_dataset_path, split="train")
        formatted_dataset = raw_dataset.map(format_prompts_for_batch, batched=True)
        print(f"Processing complete. Saving dataset to '{processed_dataset_path}'...")
        formatted_dataset.save_to_disk(processed_dataset_path)
        print("Dataset saved successfully")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = True,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            warmup_steps = 50,
            max_steps = 5700,
            logging_steps = 25,
            save_strategy = "steps",
            save_steps = 1000, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
        ),
    )

    print(f"Starting training run for {5700} steps...")
    trainer.train()
    print("Training complete.")
    print("Saving LoRA adapter...")
    model.save_pretrained(final_model_path)
    print(f"\\nModel adapter saved to '{final_model_path}'.")

if __name__ == "__main__":
    main()
