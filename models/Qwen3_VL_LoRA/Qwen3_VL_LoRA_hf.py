import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None


class Qwen3_VL_LoRA:
    def __init__(self, model_path, args):
        super().__init__()
        self.adapter_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_path = (
            args.base_model_path
            if hasattr(args, "base_model_path") and args.base_model_path
            else "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        base_model = self._load_base_model()
        self.llm = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.llm.eval()
        self._sanitize_generation_config()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def _sanitize_generation_config(self):
        generation_config = getattr(self.llm, "generation_config", None)
        if generation_config is None:
            return

        # Keep deterministic defaults clean when do_sample=False to avoid noisy warnings.
        if hasattr(generation_config, "temperature"):
            generation_config.temperature = 1.0
        if hasattr(generation_config, "top_p"):
            generation_config.top_p = 1.0
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = 50

    def _load_base_model(self):
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        errors = []

        if Qwen3VLForConditionalGeneration is not None:
            try:
                return Qwen3VLForConditionalGeneration.from_pretrained(
                    self.base_model_path, **load_kwargs
                )
            except Exception as exc:
                errors.append(f"Qwen3VLForConditionalGeneration failed: {exc}")

        if AutoModelForImageTextToText is not None:
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    self.base_model_path, **load_kwargs
                )
            except Exception as exc:
                errors.append(f"AutoModelForImageTextToText failed: {exc}")

        if AutoModelForVision2Seq is not None:
            try:
                return AutoModelForVision2Seq.from_pretrained(
                    self.base_model_path, **load_kwargs
                )
            except Exception as exc:
                errors.append(f"AutoModelForVision2Seq failed: {exc}")

        joined_errors = "; ".join(errors) if errors else "no compatible loader found"
        raise RuntimeError(
            "Failed to load Qwen3-VL base model. "
            "Please upgrade transformers/peft and retry. "
            f"Details: {joined_errors}"
        )

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
                current_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": str(system_prompt)}],
                    }
                )
            if "image" in messages and messages["image"] is not None:
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
                valid_images = [image for image in messages["images"] if image is not None]
                content = []
                for i, image in enumerate(valid_images):
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
        return inputs.to(self.device)

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        do_sample = self.temperature != 0
        generation_kwargs = {
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        generated_ids = self.llm.generate(**inputs, **generation_kwargs)
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
