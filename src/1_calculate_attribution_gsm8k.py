import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F


# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(minimum_activations, maximum_activations, batch_size, num_batch):

    num_points = batch_size * num_batch
    step = (maximum_activations - minimum_activations) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(minimum_activations, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list) # (layer_num, ffn_size)
    max_ig = ig.max()  # maximum attribution score
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def get_context_attr(idx, prompt_without_context, prompt_with_context, answer_obj, args, model, tokenizer, device):
    # changed code: strip any leading 'A:' from the answer and tokenize only numeric part
    raw = answer_obj
    if raw.strip().upper().startswith("A:"):
        raw = raw.split(":", 1)[1].strip()
    tokens = tokenizer.tokenize(raw)
    gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])

    wo_tokenized_inputs = tokenizer(prompt_without_context, return_tensors="pt")
    wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
    wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)

    w_tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
    w_input_ids = w_tokenized_inputs["input_ids"].to(device)
    w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)

    # record results
    res_dict = {
        'idx': idx,
        'wo_all_ffn_activations': [],
        'w_all_ffn_activations': [],
        'all_attr_gold': [],
    }

    for tgt_layer in range(model.model.config.num_hidden_layers):
        wo_ffn_activations_dict = {}
        def wo_forward_hook_fn(module, inp, outp):
            wo_ffn_activations_dict['input'] = inp[0]

        w_ffn_activations_dict = {}
        def w_forward_hook_fn(module, inp, outp):
            w_ffn_activations_dict['input'] = inp[0]

        # no-context
        wo_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(wo_forward_hook_fn)
        with torch.no_grad():
            wo_outputs = model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
        wo_hook.remove()
        wo_ffn_activations = wo_ffn_activations_dict['input'][:, -1, :]

        # with-context
        w_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(w_forward_hook_fn)
        with torch.no_grad():
            w_outputs = model(input_ids=w_input_ids, attention_mask=w_attention_mask)
        w_hook.remove()
        w_ffn_activations = w_ffn_activations_dict['input'][:, -1, :]

        wo_ffn_activations.requires_grad_(True)
        w_ffn_activations.requires_grad_(True)
        scaled_activations, activations_step = scaled_input(
            wo_ffn_activations, w_ffn_activations,
            args.batch_size, args.num_batch)
        scaled_activations.requires_grad_(True)

        ig_gold = None
        for _ in range(args.num_batch):
            all_grads = None
            for i in range(0, args.batch_size, args.batch_size_per_inference):
                batch_scaled = scaled_activations[i:i+args.batch_size_per_inference]
                batch_w = w_ffn_activations.repeat(args.batch_size_per_inference, 1)
                b_ids = w_input_ids.repeat(args.batch_size_per_inference, 1)
                b_mask = w_attention_mask.repeat(args.batch_size_per_inference, 1)

                def pre_hook(module, inp):
                    inp0 = inp[0]
                    inp0[:, -1, :] = batch_scaled
                    return (inp0,)
                hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(pre_hook)

                outputs = model(input_ids=b_ids, attention_mask=b_mask)
                hook.remove()

                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=1)
                grads = torch.autograd.grad(probs[:, gold_label_id].sum(), w_ffn_activations, retain_graph=True)[0]

                all_grads = grads if all_grads is None else torch.cat((all_grads, grads), dim=0)

            layer_grad = all_grads.sum(dim=0)
            ig_gold = layer_grad if ig_gold is None else ig_gold + layer_grad

        ig_gold = ig_gold * activations_step
        res_dict['wo_all_ffn_activations'].append(wo_ffn_activations.squeeze().tolist())
        res_dict['w_all_ffn_activations'].append(w_ffn_activations.squeeze().tolist())
        res_dict['all_attr_gold'].append(ig_gold.tolist())

    return res_dict


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path", required=True, type=str,
                        help="Input JSONL with keys: question, question_with_cot, final_answer.")
    parser.add_argument("--model_path", required=True, type=str,
                        help="HuggingFace model or local path.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for results.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name prefix for outputs.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name prefix for outputs.")
    parser.add_argument("--no_cuda", action='store_true', help="Do not use CUDA.")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU id(s).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_batch", type=int, default=1,
                        help="IG interpolation steps.")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Total scaled points.")
    parser.add_argument("--batch_size_per_inference", type=int, default=2,
                        choices=[1,2,4,5,10,20],
                        help="Batch size per inference.")
    args = parser.parse_args()

    # device setup (unchanged)
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu"); n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device(f"cuda:{args.gpu_id}"); n_gpu = 1
    else:
        device = torch.device("cuda"); n_gpu = torch.cuda.device_count()
    print(f"device: {device}, n_gpu: {n_gpu}, distributed: {n_gpu>1}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if n_gpu>0: torch.cuda.manual_seed_all(args.seed)

    output_prefix = f"{args.dataset_name}-{args.model_name}-context"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, output_prefix+'.args.json'), 'w'), indent=2)

    # prepare dataset
    with open(args.data_path, 'r') as fin:
        data = [json.loads(l) for l in fin]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
    model.eval()

    tic = time.perf_counter()
    print(f"Start process, dataset size: {len(data)}")
    with jsonlines.open(os.path.join(args.output_dir, output_prefix+'.rlt.jsonl'), 'w') as fw:
        for idx, example in enumerate(tqdm(data, desc="Examples")):
            # changed code: read GSM-8K format
            prompt_without_context = example["question"]
            prompt_with_context    = example["question_with_cot"]
            answer_obj             = example["final_answer"]

            res_dict = get_context_attr(
                idx, prompt_without_context, prompt_with_context,
                answer_obj, args, model, tokenizer, device)
            res_dict["all_attr_gold"] = convert_to_triplet_ig(res_dict["all_attr_gold"])
            fw.write(res_dict)
    toc = time.perf_counter()
    print(f"Saved in {os.path.join(args.output_dir, output_prefix+'.rlt.jsonl')}")
    print(f"Time: {toc-tic:.2f}s")
    json.dump({"time":toc-tic}, open(os.path.join(args.output_dir, output_prefix+'.time.json'), 'w'))

if __name__ == "__main__":
    main()
