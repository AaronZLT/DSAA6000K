import os
import json
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

MODEL_PATH = "/hpc2hdd/home/lzhang330/ssd_workspace/models"
MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
ALG_NAME = "ROME"

def gen_request(prompt,target_new,target_true,subject):
    return [{
        "prompt": prompt,
        "target_new": {
            "str": target_new
        },
        "target_true": {
            "str": target_true
        },
        "subject": subject
        }]
        
def edit(layers: int = None):
    model, tok = (
        AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH,MODEL_NAME)).to(
            "cuda"
        ),
        AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH,MODEL_NAME)),
    )
    tok.pad_token = tok.eos_token
    print(model.config)

    model.config._name_or_path = MODEL_NAME

    with open('data.json') as f:
        data = json.load(f)

    generations = []

    # Restore fresh copy of model
    try:
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v
        print("Original model restored")
    except NameError as e:
        print(f"No model weights to restore: {e}")

    # Execute rewrite
    for i in data:
        i["generations_popular"] = []
        i["generations_unpopular"] = []

        generation_prompts = i["generation_prompts"]

        popular_request = gen_request(i["requested_rewrite"]["prompt"],i["requested_rewrite"]["target_new"]["str"],i["requested_rewrite"]["target_true"]["str"],i["requested_rewrite"]["subject"])
        model_new, orig_weights = demo_model_editing(model, tok, popular_request, generation_prompts, alg_name=ALG_NAME, generations = generations,layer = layers)
        i["generations_popular"] = generations
        generations = []


        for unpopular in i["requested_rewrite"]["target_unpopular"]:
            unpopular_request = gen_request(i["requested_rewrite"]["prompt"],unpopular,i["requested_rewrite"]["target_true"]["str"],i["requested_rewrite"]["subject"])
            model_new, orig_weights = demo_model_editing(model, tok, unpopular_request, generation_prompts, alg_name=ALG_NAME, generations = generations,layer = layers)
            i["generations_unpopular"].append(generations)
            generations = []

    with open(f"save_layer_{layers}.json","w", encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False,indent=4))

for i in range(0,48):
    edit(i)