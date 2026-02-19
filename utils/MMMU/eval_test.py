import torch
import os
import json

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from mathruler.grader import extract_boxed_content

from .data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG,DOMAIN_CAT2SUB_CAT
from .eval_utils import evaluate,parse_multi_choice_response, parse_open_response

from ..utils import extract


def _parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_batch_size():
    batch_size_str = os.environ.get("MMMU_BATCH_SIZE")
    if batch_size_str is not None:
        try:
            return max(1, int(batch_size_str))
        except ValueError:
            pass
    return 256 if os.environ.get("use_vllm", "True") == "True" else 1


def _save_each_sample():
    return _parse_bool(os.environ.get("MMMU_SAVE_EACH_SAMPLE"), default=True)


def _build_messages(sample):
    messages = {"prompt": sample["final_input_prompt"]}
    if sample.get("image", None) is not None:
        messages["image"] = sample["image"]
    return messages


def run_model(samples, model, results_path=None, response_path=None):
    out_samples = {}
    out_response = {}
    batch_size = _get_batch_size()
    save_each_sample = _save_each_sample()
    save_every = max(1, int(os.environ.get("MMMU_SAVE_EVERY", 50)))

    with torch.no_grad():
        total_batches = (len(samples) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(total_batches), desc=f"MMMU test infer (bs={batch_size})"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(samples))
            current_samples = samples[start:end]
            current_messages = [_build_messages(sample) for sample in current_samples]
            outputs = model.generate_outputs(current_messages)
            for sample, response in zip(current_samples, outputs):
                if "<answer>" in response:
                    response = extract(response, "answer")
                if extract_boxed_content(response) != "None":
                    response = extract_boxed_content(response)
                if sample["question_type"] == "multiple-choice":
                    pred_ans = parse_multi_choice_response(
                        response, sample["all_choices"], sample["index2ans"]
                    )
                else:
                    pred_ans = response
                out_samples[sample["id"]] = pred_ans
                out_response[sample["id"]] = response

                if results_path is not None and response_path is not None and save_each_sample:
                    save_json(results_path, out_samples)
                    save_json(response_path, out_response)

            if (
                results_path is not None
                and response_path is not None
                and not save_each_sample
                and ((batch_idx + 1) % save_every == 0 or batch_idx + 1 == total_batches)
            ):
                save_json(results_path, out_samples)
                save_json(response_path, out_response)

    return out_samples, out_response


def eval_MMMU_test(model,dataset_path,output_path,subset):
    sub_dataset_list = []
    for subject in tqdm(DOMAIN_CAT2SUB_CAT[subset]):
        sub_dataset = load_dataset(dataset_path, subject, split="test")
        sub_dataset_list.append(sub_dataset)

    chunk_idx = int(os.environ.get("chunk_idx",0))
    num_chunks = int(os.environ.get("num_chunks",1))
    samples = []
    dataset = concatenate_datasets(sub_dataset_list)
    for idx,sample in tqdm(enumerate(dataset)):
        if idx % num_chunks == chunk_idx:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)
            samples.append(sample)

    if num_chunks == 1:
        results_path = os.path.join(output_path,"results.json")
        response_path = os.path.join(output_path,"response.json")
        out_samples,out_response = run_model(
            samples,
            model,
            results_path=results_path,
            response_path=response_path,
        )
        save_json(results_path,out_samples)
        save_json(response_path,out_response)
        return "please upload in https://eval.ai/web/challenges/challenge-page/2179/leaderboard to get the results"

    elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        response_path = os.path.join(output_path,f"response_{chunk_idx}.json")
        out_samples,out_response = run_model(
            samples,
            model,
            results_path=results_path,
            response_path=response_path,
        )
        save_json(results_path,out_samples)
        save_json(response_path,out_response)


        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = {}
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    partial_results = json.load(f)
                    total_results.update(partial_results)
                    os.remove(results_path)
            with open(os.path.join(output_path,"results.json"),"w") as f:
                json.dump(total_results,f)
                
            return "please upload in https://eval.ai/web/challenges/challenge-page/2179/leaderboard to get the results"
        else:
            return "please upload in https://eval.ai/web/challenges/challenge-page/2179/leaderboard to get the results"
    else:
        raise ValueError("num_chunks must be greater than 0")
