import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice,judger
from ..base_dataset import BaseDataset
from ..question_formats import get_multiple_choice_prompt
from ..MMMU.eval_utils import parse_multi_choice_response

class SuperGPQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "m-a-p/SuperGPQA" #2,755
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    
    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path)["train"]

        # ['index', 'Figure_path', 'Caption', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Answer', 'split']
        for idx,sample in tqdm(enumerate(dataset)):
            if sample["discipline"] != "Medicine":
                continue
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
        choices = sample["options"]
        answer = sample["answer_letter"]
        choices = [f"{chr(65+i)}. {choices[i]}" for i in range(len(choices))]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
        answer_letters = [chr(65 + i) for i in range(len(choices))]
        if len(answer_letters) == 1:
            answer_hint = answer_letters[0]
        elif len(answer_letters) == 2:
            answer_hint = f"{answer_letters[0]} or {answer_letters[1]}"
        else:
            answer_hint = f"{', '.join(answer_letters[:-1])}, or {answer_letters[-1]}"
        prompt += (
            "\nRespond in exactly this format:\n"
            "<think>your brief reasoning</think>\n"
            "<answer>ONE_LETTER</answer>\n"
            f"The answer letter must be one of {answer_hint}."
        )

        messages = {"prompt":prompt}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = answer
        del sample["answer_letter"]
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
        total_field_dict = defaultdict(int)
        right_field_dict = defaultdict(int)
        total_difficulty_dict = defaultdict(int)
        right_difficulty_dict = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]
            field = sample["field"]
            difficulty = sample["difficulty"]
            existing_thinking = sample.get("thinking", "")
            thinking, parsed_response = self._split_thinking_and_response(response, choices)
            if not thinking and existing_thinking:
                thinking = existing_thinking
            out_samples[i]["thinking"] = thinking
            out_samples[i]["response"] = parsed_response
            out_samples[i].pop("parsed_response", None)

            correct = judge_multi_choice(choices,answer,parsed_response)
            out_samples[i]["correct"] = correct
            if correct:
                right_field_dict[field] += 1
                right_difficulty_dict[difficulty] += 1
                right += 1
            total_difficulty_dict[difficulty] += 1
            total_field_dict[field] += 1
            total += 1
        field_metrics = {}
        difficulty_metrics = {}
        for key,value in total_field_dict.items():
            right_cnt = right_field_dict[key]
            difficulty_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}
        
        for key,value in total_difficulty_dict.items():
            right_cnt = right_difficulty_dict[key]
            field_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total},"field":field_metrics,"difficulty":difficulty_metrics}
        return metrics,out_samples