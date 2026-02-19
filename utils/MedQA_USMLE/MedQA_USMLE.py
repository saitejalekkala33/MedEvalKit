import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset
from ..question_formats import get_multiple_choice_prompt
from ..MMMU.eval_utils import parse_multi_choice_response

class MedQA_USMLE(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "GBaker/MedQA-USMLE-4-options"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset = load_dataset(self.dataset_path)["test"]
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
        answer = sample["answer_idx"]
        options = sample["options"]
        choiceA = options["A"]
        choiceB = options["B"]
        choiceC = options["C"]
        choiceD = options["D"]
        choiceA = f"A. {choiceA}"
        choiceB = f"B. {choiceB}"
        choiceC = f"C. {choiceC}"
        choiceD = f"D. {choiceD}"
        choices = [choiceA,choiceB,choiceC,choiceD]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
        prompt += (
            "\nRespond in exactly this format:\n"
            "<think>your brief reasoning</think>\n"
            "<answer>ONE_LETTER</answer>\n"
            "The answer letter must be one of A, B, C, or D."
        )

        sample = {"prompt":prompt,"answer":answer,"choices":choices,"messages":{"prompt":prompt}}
        return sample

    def _parse_pred_answer(self, response, choices):
        all_choices = [chr(ord("A") + i) for i in range(len(choices))]
        index2ans = {}
        for i, choice in enumerate(choices):
            choice_text = choice.split(".", 1)[-1].strip() if "." in choice else choice
            index2ans[all_choices[i]] = choice_text

        tagged_answer = extract(response, "answer", hard=False).strip().upper()
        if tagged_answer in all_choices:
            return tagged_answer

        return parse_multi_choice_response(response, all_choices, index2ans)

    def _split_thinking_and_response(self, response, choices):
        raw_response = "" if response is None else str(response)
        parsed_response = self._parse_pred_answer(raw_response, choices)

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

        return thinking, parsed_response

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
                        thinking, parsed_response = self._split_thinking_and_response(
                            response, sample["choices"]
                        )
                        sample["thinking"] = thinking
                        sample["response"] = parsed_response
                        sample.pop("parsed_response", None)
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
        total = 0
        right = 0
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]
            existing_thinking = sample.get("thinking", "")
            thinking, parsed_response = self._split_thinking_and_response(
                response, choices
            )
            if not thinking and existing_thinking:
                thinking = existing_thinking
            out_samples[i]["thinking"] = thinking
            out_samples[i]["response"] = parsed_response
            out_samples[i].pop("parsed_response", None)

            correct = judge_multi_choice(choices,answer,parsed_response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples
