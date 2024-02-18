import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    GenerationConfig,
)
from trl import SFTTrainer

class ModelModule:

    def __init__(self, model_name, qlora=False, peft_weight_path=None):
        self.tokenizer = self.init_tokenizer(model_name)
        self.model = self.init_model(model_name, qlora, peft_weight_path)

    def init_tokenizer(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id =  tokenizer.unk_token_id
        tokenizer.padding_side = "left"
        return tokenizer
    
    def init_model(self, model_name, qlora=False, peft_weight_path=None):
        if qlora:
            compute_dtype = getattr(torch, "float16")
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb_config, device_map={"": 0}
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        # Configure the pad token in the model
        model.config.pad_token_id = self.tokenizer.pad_token_id

        if peft_weight_path:
            model = PeftModel.from_pretrained(model, peft_weight_path)

        return model
    
    @property
    def peft_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        )

    def train(self, train_dataset=None, eval_dataset=None, training_arguments=None, dataset_text_field="text", max_seq_length=512, lora=False, qlora=False):
        peft_config = None
        if lora or qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            peft_config = self.peft_config

        self.model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
            
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
        )

        trainer.train()

    def genrate(self, instruction):
        prompt = "### Human: " + instruction + "### Assistant: "
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(pad_token_id=self.tokenizer.pad_token_id, temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )

        outputs = []
        for seq in generation_output.sequences:
            output = self.tokenizer.decode(seq)
            outputs.append(output.split("### Assistant: ")[1].strip())

        return outputs