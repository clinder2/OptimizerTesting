from cProfile import label
import os
import sys
import itertools

from networkx import non_edges
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from dataclasses import asdict

sys.path.append("/storage/home/hcoda1/7/clinder9/r-mtao8-0/VariationalStiefelOptimizer")
import torch

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    

# Ensure local ShampooExperiment imports resolve when this script is run from the repo root.
SPEEDRUN_DIR = os.path.dirname(os.path.abspath(__file__))
SHAMPOO_ROOT = os.path.dirname(SPEEDRUN_DIR)
if SHAMPOO_ROOT not in sys.path:
    sys.path.insert(0, SHAMPOO_ROOT)

# Add the external repo root so train.py can import its local helpers like prepare.py.

# NANOCHAT_ROOT = "/Users/christopherlinder/Desktop/stiefel-nanochat"
# if NANOCHAT_ROOT not in sys.path:
#     sys.path.insert(0, NANOCHAT_ROOT)

# train_path = os.path.join(NANOCHAT_ROOT, "train.py")
# spec = importlib.util.spec_from_file_location("stiefel_nanochat_train", train_path)
# external_train = importlib.util.module_from_spec(spec)
# sys.modules[spec.name] = external_train
# spec.loader.exec_module(external_train)

# train_path = os.path.join(NANOCHAT_ROOT, "prepare.py")
# spec = importlib.util.spec_from_file_location("stiefel_nanochat_prepare", train_path)
# external_prepare = importlib.util.module_from_spec(spec)
# sys.modules[spec.name] = external_prepare
# spec.loader.exec_module(external_prepare)

from StackedShampoo import StackedShampoo
from CustomShampoo import CustomShampoo
from model import *

# GPT = external_train.GPT
# GPTConfig = external_train.GPTConfig

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
from gpt import GPT, GPTConfig

# MAX_SEQ_LEN = external_prepare.MAX_SEQ_LEN
# TIME_BUDGET = external_prepare.TIME_BUDGET
# Tokenizer = external_prepare.Tokenizer
# make_dataloader = external_prepare.make_dataloader
# evaluate_bpb = external_prepare.evaluate_bpb

def train(config, device_type, device, total_iters=100):
    # Autocast context
    if device_type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device_type == "cpu":
        autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()

    # ---------------------------------------------------------------------------
    # Hyperparameters (edit these directly, no CLI flags needed)
    # ---------------------------------------------------------------------------

    # Model architecture
    ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
    HEAD_DIM = 128          # target head dimension for attention
    WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context
    MODEL_SCALE = config['model_scale']  # effective model size multiplier for token budget (e.g. 0.5 = half the tokens, double the LR)
    NUM_HEADS = 6

    # Optimization
    TOTAL_BATCH_SIZE = 2**16 # ~65K tokens per optimizer step
    EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
    UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
    MATRIX_LR = config['matrix_lr']        # learning rate for matrix parameters (Muon)
    SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
    WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
    ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
    WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
    WARMDOWN_RATIO = config["warmdown_ratio"]    # fraction of time budget for LR warmdown
    FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
    BETA2=config['beta2']

    # Model size
    DEPTH = config['depth'] # number of transformer layers-12
    DEVICE_BATCH_SIZE = 16  # per-device batch size (reduce if OOM)

    # Stiefel optimizer hyperparameters
    # if 'stiefel_lr' in config.keys():
    #     STIEFEL_LR = config['stiefel_lr']
    #     STIEFEL_MOMENTUM = config['stiefel_momentum']
    #     STIEFEL_BETAS = (config['stiefel_beta1'], config['stiefel_beta2'])
    #     STIEFEL_TYPE = config['stiefel_type']  # 'SGD' or 'Adam'
        
    CHOL=config['CHOL']
    # ---------------------------------------------------------------------------
    # Setup: tokenizer, model, optimizer, dataloader
    # ---------------------------------------------------------------------------

    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    H100_BF16_PEAK_FLOPS = 989.5e12

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    def build_model_config(depth):
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        return GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )
        
    def build_model_config_from_heads(num_heads, input_depth=4):
        #baseline 2 heads at HEAD_DIM=128
        base_dim = input_depth * ASPECT_RATIO
        assert base_dim%num_heads==0
        curr_head_dim=base_dim//num_heads
        print('config: ', NUM_HEADS,curr_head_dim,base_dim)
        return GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=input_depth, n_head=num_heads, n_kv_head=num_heads, n_embd=base_dim,
            window_pattern=WINDOW_PATTERN,
        )

    gptconfig = build_model_config(DEPTH)
    #config = build_model_config_from_heads(NUM_HEADS, DEPTH)
    print(f"Model config: {asdict(gptconfig)}")

    with torch.device("meta"):
        model = GPT(gptconfig)
    model.to_empty(device=device)
    model.init_weights()
    model.to(device)
    # Token budget
    param_counts = model.num_scaling_params()
    non_embedding_params=0
    for k, v in param_counts.items():
        if k!='wte' and k!='value_embeds' and k!='total':
            non_embedding_params+=v
    TOKEN_BUDGET = non_embedding_params*MODEL_SCALE      # total tokens to train on
    print(f"Model scale: {MODEL_SCALE}")
    print(f"Token budget: {TOKEN_BUDGET:.2e} (non-embedding params: {non_embedding_params:,})")

    # print("Parameter counts:")
    # for key, value in param_counts.items():
        # print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    # print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    # optimizer, stiefel_optimizer = model.setup_optimizer(
    #     unembedding_lr=UNEMBEDDING_LR,
    #     embedding_lr=EMBEDDING_LR,
    #     scalar_lr=SCALAR_LR,
    #     adam_betas=ADAM_BETAS,
    #     matrix_lr=MATRIX_LR,
    #     weight_decay=WEIGHT_DECAY,
    #     stiefel_lr=STIEFEL_LR,
    #     stiefel_momentum=STIEFEL_MOMENTUM,
    #     stiefel_betas=STIEFEL_BETAS,
    #     stiefel_type=STIEFEL_TYPE,
    # )
    # print("stiefel optimizer is None", stiefel_optimizer==None)
    
    s_optimizer=None
    stiefel_optimizer=None
    if config['shampoo']:
        optimizer, s_optimizer = model.setup_optimizer(UNEMBEDDING_LR, EMBEDDING_LR, MATRIX_LR, 
            WEIGHT_DECAY, ADAM_BETAS, SCALAR_LR, BETA2, CHOL)
    elif config['stiefel']:
        optimizer, stiefel_optimizer = model.setup_optimizer_stiefel()
    elif config['adamw']:
        optimizer = model.setup_optimizer_adam(UNEMBEDDING_LR, EMBEDDING_LR, MATRIX_LR, 
            WEIGHT_DECAY, ADAM_BETAS, SCALAR_LR)
    else:
        optimizer = model.setup_optimizer_muon(UNEMBEDDING_LR, EMBEDDING_LR, MATRIX_LR, 
            WEIGHT_DECAY, ADAM_BETAS, SCALAR_LR)
    # torch.compile is unstable on MPS, only use on CUDA
    if device_type == "cuda":
        model = torch.compile(model, dynamic=False)
    
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch
    
    # print(f"Time budget: {TIME_BUDGET}s")
    # print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Schedules (all based on progress = training_time / TIME_BUDGET)

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    def sync_device(device_type):
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    loss_arr=[]
    smoothed_loss_arr=[]
    while True:
        sync_device(device_type)
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        # Progress and schedules
        progress = min(total_training_time / TIME_BUDGET, 1.0)

        # progress by step
        #progress=min(1.0, step/total_iters)
        total_tokens = (1+step) * TOTAL_BATCH_SIZE
        
        #progress=total_tokens/TOKEN_BUDGET #our progress is percent tokens trained on
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        #Stiefel optimizer step
        if s_optimizer!=None:
            for group in s_optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
            s_optimizer.step()
        if stiefel_optimizer!=None:
            for group in stiefel_optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
            stiefel_optimizer.step()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        loss_arr.append(train_loss_f)
        # Fast fail: abort if loss is exploding
        if train_loss_f > 12:
            print("FAIL")
            exit(1)

        sync_device(device_type)
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        remaining = max(0, TIME_BUDGET - total_training_time)
        smoothed_loss_arr.append(debiased_smooth_loss)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s | fails: {s_optimizer.fails if s_optimizer!= None else 0}    ", end="", flush=True)
        
        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        total_tokens = step * TOTAL_BATCH_SIZE
        print(f"Percent tokens: {100*total_tokens/TOKEN_BUDGET:.1f}%")

        # Time's up — but only stop after warmup steps so we don't count compilation
        if (total_training_time>=TIME_BUDGET and step > 10) or step>=total_iters:
            break

    print()  # newline after \r training log

    total_tokens = step * TOTAL_BATCH_SIZE

    # Final eval
    # model.eval()
    # with autocast_ctx:
    #     val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Final summary
    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    if device_type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    if config['shampoo']:
        np.save(f'SHAMPOO_loss_arr.npy', loss_arr)
    elif config['stiefel']:
        np.save(f'sim_STIEFELADAM_loss_arr.npy', loss_arr)
    elif config['adamw']:
        np.save(f'sim_ADAM_loss_arr.npy', loss_arr)
    else:
        np.save(f'sim_MUON_loss_arr.npy', loss_arr)

    print("---")
    print(config)
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"batch_size:       {TOTAL_BATCH_SIZE}")

    return {
        'model_scale': MODEL_SCALE,
        # 'stiefel_type': STIEFEL_TYPE,
        # 'stiefel_lr': STIEFEL_LR,
        # 'stiefel_momentum': STIEFEL_MOMENTUM if STIEFEL_TYPE=='SGD' else None,
        # 'stiefel_beta1': STIEFEL_BETAS[0] if STIEFEL_TYPE=='Adam' else None,
        # 'stiefel_beta2': STIEFEL_BETAS[1] if STIEFEL_TYPE=='Adam' else None,
        'layers': DEPTH,
        'training_seconds': total_training_time,
        'total_seconds': t_end - t_start,
        'peak_vram_mb': peak_vram_mb,
        'mfu_percent': steady_state_mfu,
        'total_tokens_M': total_tokens / 1e6,
        'num_steps': step,
        'num_params_M': num_params / 1e6,
        'loss': debiased_smooth_loss,
        'batch_size': TOTAL_BATCH_SIZE,
        'num_heads': NUM_HEADS,
        'matrix_lr': MATRIX_LR,
        'beta2': BETA2,
        'warmdown_ratio': WARMDOWN_RATIO,
    }


if __name__=='__main__':
    SCS_updated_fine_grid = {
        'lr': [.0001, .001, .01, .1, .9, .99],
        'warmup_iters': [.05, .1, .4],
        'lr_decay_iters': [.05, .2, .7],
        'min_lr': [6e-5, 1e-1],
        'max_iters': [2000],
        'beta2': [0.999, 0.95, .8],
    }

    matrix_lr_grid = [.04]
    beta2_grid = [0.999]
    warmdown_ratio_grid = [.2]
    model_scales = [20]
    batch_size=[2**16] #original grid: [2**15,2**16,2**18,2**20], [2**15,2**16,2**17]
    chol_grid = [False]
    hp_list=itertools.product(model_scales, matrix_lr_grid, beta2_grid, warmdown_ratio_grid, batch_size, chol_grid)
    settings = [dict(zip(['model_scale', 'matrix_lr', 'beta2', 'warmdown_ratio', 'total_batch_size', 'CHOL'], vals)) for vals in hp_list]
       
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    stiefel_beta1_grid = [0.8]
    stiefel_beta2_grid = [0.95]
    stiefel_lr_grid = [.001] #original Adam grid: [1e-4, 3e-4, 1e-3, 4e-2], original SGD grid: [3e-4, 1e-3, 4e-2]
    stiefel_momentum_grid = [0.99] #original grid: [0.5,0.85,0.9,0.99]
    model_scales = [20]
    batch_size=[2**16] #original grid: [2**15,2**16,2**18,2**20], [2**15,2**16,2**17]
    stiefel_type=['Adam']
    layers=[6]
    config=settings[0]

    import csv
    with open("/Users/christopherlinder/Desktop/OptimizerTesting/ShampooExperiment/speedrun/CS_360s_hp_sweeps.tsv", 'r') as f:
        r=csv.DictReader(f,delimiter='\t')
        i=0
        data=[]
        for row in r:
            row={k:float(v) for k, v in row.items()}
            data.append(row)
        data=sorted(data, key=lambda d: d['loss'])
        print(data[0])

    import matplotlib.pyplot as plt
    a=np.load('/Users/christopherlinder/Desktop/OptimizerTesting/sim_ADAM_loss_arr.npy')
    b=np.load('/Users/christopherlinder/Desktop/OptimizerTesting/sim_MUON_loss_arr.npy')
    c=np.load('/Users/christopherlinder/Desktop/OptimizerTesting/sim_STIEFELADAM_loss_arr.npy')
    print(c)
    plt.plot(a, color='red', label='Adam')
    plt.plot(b, color='green', label='Muon')
    plt.plot(c, color='blue', label='StiefelAdam')
    # t1=0
    # t2=0
    # for i in range(1,4):
    #     a=np.load(f'/Users/christopherlinder/Desktop/OptimizerTesting/ShampooExperiment/speedrun/12_6961_CS{i}_loss_arr.npy')
    #     plt.plot(a, color='red')
    #     t1+=np.mean(np.abs(np.diff(a)))
    #     a=np.load(f'/Users/christopherlinder/Desktop/OptimizerTesting/ShampooExperiment/speedrun/12_6961_S{i}_loss_arr.npy')
    #     plt.plot(a, color='blue')
    #     t2+=np.mean(np.abs(np.diff(a)))
    # print(t1/3, t2/3)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(rf'S v. CS')
    plt.legend()
    #plt.plot(c, color='green')
    plt.show()

    config['shampoo']=False
    config['stiefel']=False
    config['adamw']=True
    config['depth']=6
    #result=train(config,device_type,device,total_iters=150)

    config['stiefel']=True
    config['adamw']=False
    #result=train(config,device_type,device,total_iters=150)

    config['stiefel']=False
    config['adamw']=False
    print(config)
    result=train(config,device_type,device,total_iters=150)

    # config['stiefel']=False
    # result=train(config,device_type,device,total_iters=500)

       # with open(f"{"C" if config['CHOL']==True else ""}S_360s_hp_sweeps.tsv", 'a', newline='') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     if f.tell() == 0:
        #         writer.writerow([k for k in result.keys()])
        #     writer.writerow([result[k] for k in result.keys()])
                
    # config['CS']=True
    # train(config, device_type, device)

    # arr1=np.load('AdamWMuon_8_0.04_loss_arr.npy')
    # arr2=np.load('SHAMPOO_8_0.02_loss_arr.npy')
    # arr3=np.load('SHAMPOO_nograft_8_0.06_loss_arr.npy')
    # import matplotlib.pyplot as plt
    # plt.plot(arr1, label='AdamWMuon-lr=.04', color='blue')
    # plt.plot(arr2, label='S-grafting-lr=.02', color='red')
    # plt.plot(arr3, label='S-no_grafting-lr=.06', color='green')
    # plt.legend()
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()

    # n=10 GPTConfig(sequence_len=2048, vocab_size=8192, n_layer=8, n_head=4, n_kv_head=4, n_embd=512, window_pattern='L')
    # model=MatrixSimple(torch.eye(n),n)
    # shampoo_groups = [{
    #             'params': [p for p in model.parameters()],
    #             'kind': 'muon',
    #             'lr': .9,
    #             'momentum': .4,
    #             'ns_steps': 5,
    #             'beta2': .85,
    #             'weight_decay': 0,
    #         }]
    # optimizer=StackedShampoo(param_groups=shampoo_groups,grafting=True, chol=False, debug=False)

    # #optimizer=CustomShampoo(W=shampoo_groups[0]['params'], lr=.1, grafting=True, chol=False, debug=False)
    # iter_num=0
    # while True:
    #     G, L=model()
    #     L.backward()
    #     print("Loss: ", L.item())
    #     optimizer.step()
    #     optimizer.zero_grad(set_to_none=True)
    #     iter_num+=1
    #     if iter_num>=1000 or L.item()<=1e-23:
    #         break