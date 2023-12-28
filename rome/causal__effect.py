# Import necessary libraries
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Import other necessary modules from your project
from util import nethook
from util.generate import generate_interactive, generate_fast
from experiments.py.demo import demo_model_editing, stop_execution
import json
# Set local paths and other constants
MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B

# Check if CUDA is available for GPU support
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Ensure you have a GPU.")

# Load the model and tokenizer
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
print(model.config)


request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]
#

# request = [
#     {
#         "prompt": "{} plays the sport of",
#         "subject": "Novak Djokovic",
#         "target_new": {"str": "baseball"},
#     }
# ]
#
# generation_prompts = [
#     "Novak Djokovic plays the sport of",
#     "Novak Djokovic plays for the",
#     "The greatest strength of Novak Djokovic is his",
#     "Novak Djokovic is widely regarded as one of the",
#     "Novak Djokovic is known for his unstoppable",
#     "My favorite part of Novak Djokovic' game is",
#     "Novak Djokovic excels at",
# ]

ALG_NAME = "ROME"

# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME)
torch.save(model_new.state_dict(), 'model_new_weights.pth')
print()
# Additional code for debugging and further processing
# Complete Adapted Script for Local Python Environment

# Import necessary libraries
import os
import re
import json
import torch
import numpy
from collections import defaultdict
from util.globals import DATA_DIR
from util import nethook
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset

# Set to False for local environment
IS_COLAB = False

# Set up the model and tokenizer
model_name = "gpt2-xl"  # Choose the model name as required
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)
# 假设 mt 是您通过 ModelAndTokenizer 创建的模型实例
mt.model.load_state_dict(torch.load('model_new_weights.pth'))
# Disable gradient computation
torch.set_grad_enabled(False)

# Predict tokens
# predict_token(
#     mt,
#     ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
#     return_p=True,
# )

predict_token(
    mt,
    ["Steve Jobs was the founder of"],
    return_p=True,
)
# Path to the data directory - update this to your local path
# DATA_DIR = 'path/to/your/data/directory'  # Update this path

# Initialize KnownsDataset
knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts

# Calculate noise level
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")

# Function definitions
def trace_with_patch(model, inp, states_to_patch, answers_t, tokens_to_mix, noise=0.1, trace_layers=None):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None):
    """
        Runs causal tracing over every token/layer combination in the network
        and returns a dictionary numerically summarizing the results.
        """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
        model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(mt, prompt, subject=None, samples=10, noise=0.1, window=10, kind=None, modelname=None, savepdf='NetEase_hidden.png'):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    print('result is:',result)
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    # for kind in [None, "mlp", "attn"]:
    for kind in [None]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )

# Execute the plotting function for a specific prompt
plot_all_flow(mt, "Steve Jobs was the founder of", noise=noise_level)

# Process a subset of known facts
# for knowledge in knowns[:5]:
#     plot_all_flow(mt, knowledge["prompt"], knowledge["subject"], noise=noise_level)


