from transformers import AutoProcessor, AutoModelForImageTextToText
from vllm import LLM, SamplingParams
import os
import torch

from tqdm import tqdm

from PIL import Image
class MedGemma:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            max_model_len = 32786,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",1)),
            enforce_eager=True,
            limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self,messages):
        current_messages = []
        imgs = []
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]}) 

        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":[{"type":"text","text":system_prompt}]})
            if "image" in messages:
                image = messages["image"]
                if isinstance(image,str):
                    image = Image.open(image)
                image = image.convert("RGB")
                imgs.append(image)
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                    if isinstance(image,str):
                        image = Image.open(image)
                    image = image.convert("RGB")
                    imgs.append(image)
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 
        
        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mm_data = {}


        llm_inputs = {
            "prompt": prompt
        }
        if imgs:
            mm_data["image"] = imgs
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res

