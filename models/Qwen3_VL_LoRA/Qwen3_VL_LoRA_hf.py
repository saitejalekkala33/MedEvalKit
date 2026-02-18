import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor


class Qwen3_VL_LoRA:
    def __init__(self, model_path, args):
        super().__init__()
        self.adapter_path = model_path
        self.base_model_path = (
            args.base_model_path
            if hasattr(args, "base_model_path") and args.base_model_path
            else "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.llm.eval()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):
        current_messages = []

        if "messages" in messages:
            packed_messages = messages["messages"]
            for message in packed_messages:
                role = message["role"]
                content = message["content"]
                if isinstance(content, list):
                    normalized_content = content
                else:
                    normalized_content = [{"type": "text", "text": str(content)}]
                current_messages.append({"role": role, "content": normalized_content})
        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role": "system", "content": system_prompt})
            if "image" in messages:
                image = messages["image"]
                current_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                )
            elif "images" in messages:
                content = []
                for i, image in enumerate(messages["images"]):
                    content.append({"type": "text", "text": f"<image_{i + 1}>: "})
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": prompt})
                current_messages.append({"role": "user", "content": content})
            else:
                current_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                )

        prompt = self.processor.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(current_messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to("cuda")

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        do_sample = self.temperature != 0
        generated_ids = self.llm.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
