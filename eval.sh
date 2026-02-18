#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="MMMU-Medical-test" 
DATASETS_PATH="hf"
OUTPUT_PATH="eval_results/medonethinker-qwen3vl-8b-lora-r64"
# TestModel,Qwen2-VL,Qwen2.5-VL,Qwen3-VL-LoRA,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="Qwen3-VL-LoRA"
MODEL_PATH="sathiiii/medonethinker-qwen3vl-8b-lora-r64"
BASE_MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
JUDGE_MODEL_TYPE="openai"  # openai or gemini or deepseek or claude
API_KEY=""
BASE_URL=""


# pass hyperparameters and run python sccript
python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
