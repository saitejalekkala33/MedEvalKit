import torch
import os
import json
import gc
import csv
import re

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from mathruler.grader import extract_boxed_content

from ..utils import save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement
from ..base_dataset import BaseDataset
from ..question_formats import get_judgement_prompt,get_open_ended_prompt

class VQA_RAD(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "flaviagiammarino/vqa-rad"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    
    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path,split = "test")
            
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def _parse_bool(self, value, default=False):
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _get_eval_batch_size(self):
        batch_size_str = os.environ.get("EVAL_BATCH_SIZE")
        if batch_size_str is not None:
            try:
                return max(1, int(batch_size_str))
            except ValueError:
                pass
        return 256 if os.environ.get("use_vllm", "True") == "True" else 1

    def construct_messages(self,sample):
        question = sample["question"]
        image = sample["image"]
        answer = sample["answer"]
        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        answer = answer.lower()
        if answer in ["yes","no"]:
            prompt = get_judgement_prompt(question,is_reasoning)
            prompt += (
                "\nRespond in exactly this format:\n"
                "<think>your brief reasoning</think>\n"
                "<answer>yes or no</answer>"
            )
        else:
            prompt = get_open_ended_prompt(question,is_reasoning)
            prompt += (
                "\nRespond in exactly this format:\n"
                "<think>your brief reasoning</think>\n"
                "<answer>your concise final answer</answer>"
            )


        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        del sample["image"]
        return sample

    def _split_thinking_and_response(self, response):
        raw_response = "" if response is None else str(response)

        thinking = ""
        extracted_thinking = extract(raw_response, "think", hard=False).strip()
        if extracted_thinking:
            thinking = extracted_thinking
        elif "</think>" in raw_response:
            thinking = raw_response.split("</think>", 1)[0].strip()
            if thinking.startswith("<think>"):
                thinking = thinking[len("<think>") :].strip()
        elif "<answer>" in raw_response:
            thinking = raw_response.split("<answer>", 1)[0].strip()
            if thinking.startswith("<think>"):
                thinking = thinking[len("<think>") :].strip()

        parsed_response = raw_response
        boxed_answer = extract_boxed_content(raw_response)
        if boxed_answer != "None":
            parsed_response = boxed_answer
        else:
            tagged_answer = extract(raw_response, "answer", hard=False).strip()
            if tagged_answer:
                parsed_response = tagged_answer
            else:
                marker_match = None
                for pattern in [
                    r"final answer is\s*:?",
                    r"answer is\s*:?",
                    r"final answer\s*:?",
                    r"answer\s*:?",
                ]:
                    marker_match = re.search(pattern, raw_response, flags=re.IGNORECASE)
                    if marker_match:
                        break

                if marker_match:
                    if not thinking:
                        thinking = raw_response[: marker_match.start()].strip()
                    tail = raw_response[marker_match.end() :].strip()
                    parsed_response = tail.splitlines()[0].strip() if tail else raw_response
                else:
                    lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
                    if len(lines) >= 2 and len(lines[-1]) <= 64:
                        if not thinking:
                            thinking = "\n".join(lines[:-1]).strip()
                        parsed_response = lines[-1]

        return thinking, parsed_response.strip()

    def run(self, samples, model, batch_size=None, checkpoint_path=None):
        if batch_size is None:
            batch_size = self._get_eval_batch_size()

        save_each_sample = self._parse_bool(
            os.environ.get("EVAL_SAVE_EACH_SAMPLE"), default=(batch_size == 1)
        )
        save_every = max(1, int(os.environ.get("EVAL_SAVE_EVERY", 20)))

        out_samples = []
        with torch.no_grad():
            total_batches = (len(samples) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(total_batches), desc=f"infer (bs={batch_size})"):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(samples))
                current_samples = samples[start:end]
                current_messages = [sample["messages"] for sample in current_samples]
                outputs = model.generate_outputs(current_messages)
                try:
                    for sample, response in zip(current_samples, outputs):
                        del sample["messages"]
                        thinking, parsed_response = self._split_thinking_and_response(response)
                        sample["thinking"] = thinking
                        sample["response"] = parsed_response
                        out_samples.append(sample)

                        if checkpoint_path is not None and save_each_sample:
                            save_json(checkpoint_path, out_samples)
                except Exception as e:
                    from pdb import set_trace

                    set_trace()
                    print(e)

                if (
                    checkpoint_path is not None
                    and not save_each_sample
                    and ((batch_idx + 1) % save_every == 0 or batch_idx + 1 == total_batches)
                ):
                    save_json(checkpoint_path, out_samples)
                gc.collect()
        return out_samples


    def cal_metrics(self,out_samples):
        messages_list = []

        metrics = {
            "total metrics" : {
                "total":0,
                "right":0
            },
            "open" : {
                "total" : 0,
                "right" : 0,
                "bleu1" : 0,
                "bleu2" : 0,
                "bleu3" : 0,
                "bleu4" : 0,
                "rouge1" : 0,
                "rouge2" : 0,
                "rougel" : 0,
                "precision" : 0,
                "recall" : 0,
                "f1" : 0,
                "em" : 0,
            },
            "close" : {
                "total" : 0,
                "right" : 0
            }
        }

        open_id = []
        for i,out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            existing_thinking = out_sample.get("thinking", "")
            thinking, response = self._split_thinking_and_response(response)
            if not thinking and existing_thinking:
                thinking = existing_thinking
            out_samples[i]["thinking"] = thinking
            out_samples[i]["response"] = response

            answer = out_sample["answer"]
            question = out_sample["question"]
            answer = answer.lower().strip()
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if answer in ["yes","no"]:
                metrics["close"]["total"] += 1
                correct = judge_judgement(answer,response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            else:
                metrics["open"]["total"] += 1

                c_metrics = judge_open_end_vqa(answer,response)
                out_samples[i]["correct"] = c_metrics["em"]
                out_samples[i]["metrics"] = c_metrics
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1 
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric] 

                if os.environ.get("use_llm_judge","False") == "True":
                    messages = get_compare_messages(question,response,answer)
                    messages_list.append(messages)
                    open_id.append(i)


        if os.environ.get("use_llm_judge","False") == "True":
            metrics["total metrics"]["right"] = metrics["close"]["right"]
            metrics["open"]["right"] = 0
            llm = judger
            results = llm.generate_outputs(messages_list)
            for i,result in zip(open_id,results):
                result = extract(result,"judge")
                result = True if result == "0" else False
                out_samples[i]["correct"] = result
                if result:
                    metrics["open"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

        
        metrics["total metrics"]["acc"] = metrics["total metrics"]["right"]/metrics["total metrics"]["total"]
        metrics["open"]["acc"] = metrics["open"]["right"]/metrics["open"]["total"]
        metrics["close"]["acc"] = metrics["close"]["right"]/metrics["close"]["total"]

        for metric in metrics["open"]:
            if metric not in ["right","total"]:
                metrics["open"][metric] = metrics["open"][metric]/metrics["open"]["total"]
        return metrics,out_samples
