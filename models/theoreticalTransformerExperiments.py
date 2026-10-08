"""Bare-bones (no FFN, no residual connections) attention-only "transformer",
used to empirically study how a Stiefel(orthogonal)-constrained Q/K pair
behaves under StiefelAdam vs. leaving Q/K unconstrained under Muon.

This is the natural next step after the matrix-quadratic saddle-point
analysis (see `analysis_Quad_LossDecrease` in TrainingScripts.py and
`evalQuad.py`): there, StiefelAdam's update only ever depends on
skew(W^T G), which was shown to vanish on a large family of Riemannian
stationary points (symmetric-orthogonal "involution" matrices) that are not
true minima. For attention's Q,K projections, the analogous quantity is
skew(Wq^T Gq) / skew(Wk^T Gk), which (by the chain rule through the
Q@K^T bilinear score) vanishes whenever Z = Wq^T (X^T dL/dS X) Wk becomes
symmetric -- a data/label-dependent, dynamical version of the same
degeneracy, rather than a fixed target-induced one. This file provides the
minimal architecture + dummy data + training harness needed to probe that
empirically; no experiments are run by importing this module.

Architecture (single head, no residual/FFN, matches "just bare-bones
attention which is fed into softmax to get logits"):

    X = TokenEmbed[input_ids] + PosEmbed[:T]        (T, d_model)
    Q = X @ Wq,  K = X @ Wk,  V = X @ Wv             (Wq, Wk, Wv all square
                                                       d_model x d_model, so
                                                       Wq/Wk can live exactly
                                                       on O(d_model) -- the
                                                       square case analyzed
                                                       theoretically; the
                                                       StiefelOptimizers
                                                       wrapper's non-square
                                                       branch has a separate,
                                                       unrelated numerical
                                                       issue we hit before)
    S = (Q @ K^T) / sqrt(d_model), causally masked
    A = softmax(S, dim=-1)
    O = A @ V                                        (T, d_model)
    logits = O @ Wo                                  (Wo: d_model x vocab,
                                                       T, vocab_size)
    loss = CrossEntropy(logits, targets)

Dummy data: a synthetic "induction" task (see Olsson et al./Anthropic's
in-context-learning circuits write-up): a random token sequence is repeated
twice back-to-back. Correctly predicting the next token in the second half
requires content-based attention (find the previous occurrence of the
current token, copy whatever followed it there) -- this gives Q/K real
signal to fit, unlike a purely positional task that Q/K wouldn't need to
learn anything to solve.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import get_lr
from common.Base import *  # OPTS, make_optimizer


# ---------------------------------------------------------------------------
# Dummy data: synthetic induction task
# ---------------------------------------------------------------------------
def make_induction_batch(seq_half_len, vocab_size, batch_size, rand_seed=0):
    """Builds a batch of synthetic "induction" sequences.

    Each sequence is a random token sequence of length `seq_half_len`
    repeated twice (length 2*seq_half_len total). Predicting the next token
    correctly in the second half requires finding the previous occurrence of
    the current token and copying whatever followed it there -- a minimal,
    standard test of content-based (as opposed to purely positional)
    attention.

    Args:
        seq_half_len: length of the random half; full sequence length is
            2*seq_half_len, and (input, target) pairs have length
            2*seq_half_len - 1 (standard next-token shift).
        vocab_size: number of distinct tokens.
        batch_size: number of independent sequences.
        rand_seed: seeds the token sampling (deterministic/reproducible).

    Returns:
        input_ids: LongTensor (batch_size, 2*seq_half_len - 1)
        target_ids: LongTensor (batch_size, 2*seq_half_len - 1), next-token
            targets for input_ids.
    """
    g = torch.Generator().manual_seed(rand_seed)
    half = torch.randint(0, vocab_size, (batch_size, seq_half_len), generator=g)
    full = torch.cat([half, half], dim=1)  # (batch, 2*seq_half_len)

    #full=torch.tensor([[1,0],[0,1],[1,1]])
    input_ids = full[:, :-1]
    target_ids = full[:, 1:]
    return input_ids, target_ids


# ---------------------------------------------------------------------------
# Bare-bones attention-only "transformer"
# ---------------------------------------------------------------------------
class SimpleAttentionTransformer(nn.Module):
    """Single-head, attention-only model: no FFN, no residual connections.

    Wq/Wk/Wv are kept SQUARE (d_model x d_model) so Wq/Wk can optionally be
    constrained to lie exactly on the orthogonal group O(d_model) -- the
    square-Stiefel case analyzed theoretically for the saddle-point
    discussion above.
    """

    def __init__(self, vocab_size, d_model, seq_len, rand_seed=0):
        super().__init__()
        torch.manual_seed(rand_seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_embed = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)

        # Orthogonal init for Wq/Wk: a valid starting point on the manifold
        # for StiefelAdam (which retracts back onto O(d_model) every step),
        # and a reasonable, symmetric-between-conditions starting point for
        # Muon too (which leaves Wq/Wk fully unconstrained thereafter).
        Wq0, _ = torch.linalg.qr(torch.randn(d_model, d_model))
        Wk0, _ = torch.linalg.qr(torch.randn(d_model, d_model))
        self.Wq = nn.Parameter(Wq0)
        self.Wk = nn.Parameter(Wk0)

        self.Wv = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        # Single output/readout projection straight to vocab logits (no
        # separate head-mixing + unembedding matrices, to keep this
        # "bare-bones": attention's output is fed directly into softmax).
        self.Wo = nn.Parameter(torch.randn(d_model, vocab_size) / math.sqrt(d_model))

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids):
        """input_ids: LongTensor (batch, T). Returns logits (batch, T, vocab)."""
        B, T = input_ids.shape
        X = self.token_embed[input_ids] + self.pos_embed[:T]

        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        S = torch.einsum('btd,bsd->bts', Q, K) / math.sqrt(self.d_model)
        mask = self.causal_mask[:T, :T]
        S = S.masked_fill(~mask, float('-inf'))
        A = torch.softmax(S, dim=-1)

        O = torch.einsum('bts,bsd->btd', A, V)
        logits = O @ self.Wo
        return logits

    def qk_params(self):
        return [self.Wq]

    def other_params(self):
        return [self.Wk, self.token_embed, self.pos_embed, self.Wv, self.Wo]

    def generate(self, input):
        curr = input
        for i in range(self.seq_len-input.shape[1]):
            logits = self(curr)
            l = torch.softmax(logits, dim=-1)
            pred = torch.argmax(l[:,-1,:], dim=-1)
            curr=torch.concat((curr, pred.unsqueeze(-1)), dim=1)
        return curr


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------
def make_transformer_optimizers(model, hyperparams, stiefel_qk=False, stiefel_hyperparams=None):
    """Builds the optimizer(s) for a SimpleAttentionTransformer.

    If stiefel_qk is False (default): every parameter (including Wq/Wk) is
    optimized jointly by Muon (OPTS.MUON via make_optimizer) -- all
    parameters here are matrices, so they all land in Muon's matrix branch.

    If stiefel_qk is True: Wq/Wk are optimized by StiefelAdam
    (OPTS.STIEFEL_ADAM) using `stiefel_hyperparams` (falls back to
    `hyperparams` if not given, since Muon/StiefelAdam hyperparameters are
    not directly comparable -- see the swept hyperparameter discussion
    elsewhere in this repo), and every other parameter (token/pos
    embeddings, Wv, Wo) is optimized by Muon using `hyperparams`.

    Returns a list of one (stiefel_qk=False) or two (stiefel_qk=True)
    optimizer instances; callers should set 'lr' on every optimizer's
    param_groups and call .step()/.zero_grad() on all of them each iteration.
    """
    if not stiefel_qk:
        all_params = model.qk_params() + model.other_params()
        return [make_optimizer(OPTS.MUON, all_params, hyperparams[0])]

    stiefel_hp = stiefel_hyperparams if stiefel_hyperparams is not None else hyperparams[1]
    qk_optimizer = make_optimizer(OPTS.STIEFEL_ADAM, model.qk_params(), stiefel_hp)
    other_optimizer = make_optimizer(OPTS.MUON, model.other_params(), hyperparams[0])
    return [qk_optimizer, other_optimizer]


# ---------------------------------------------------------------------------
# Training / analysis harness (mirrors analysis_Quad / analysis_Quad_LossDecrease
# in TrainingScripts.py)
# ---------------------------------------------------------------------------
def analysis_Transformer(hyperparams, vocab_size=32, d_model=16, seq_half_len=8,
                          batch_size=32, rand_seed=2, stiefel_qk=False,
                          stiefel_hyperparams=None):
    """Trains a SimpleAttentionTransformer on the synthetic induction task.

    Mirrors the style of `analysis_Quad`/`analysis_Quad_LossDecrease` in
    TrainingScripts.py: builds the model/data once, then runs a fixed number
    of full-batch gradient steps with a warmup+cosine LR schedule (same
    `get_lr` schedule used throughout this repo), logging the training loss
    plus diagnostics for the Stiefel saddle-point mechanism discussed
    theoretically:
        - the orthogonality error of Wq/Wk (should stay ~0 when
          stiefel_qk=True; is unconstrained and can drift arbitrarily when
          stiefel_qk=False, since Muon does not constrain Wq/Wk at all)
        - the norm of skew(Wq^T Gq) + skew(Wk^T Gk), i.e. exactly the signal
          StiefelAdam's update actually depends on
        - the full Euclidean gradient norm ||Gq||_F + ||Gk||_F, for
          comparison against the skew-norm above (a large gap between the
          two -- skew-norm ~0 while grad-norm is not -- is the empirical
          signature of the Riemannian-stationary-but-suboptimal saddle
          points discussed theoretically)

    Args:
        hyperparams: dict with 'lr', 'warmup_iters', 'lr_decay_iters',
            'min_lr', 'max_iters', following the same conventions as
            elsewhere in this repo (see make_optimizer / get_lr). Used for
            every parameter when stiefel_qk=False, and for the non-Q/K
            (Muon) parameters when stiefel_qk=True.
        vocab_size, d_model, seq_half_len, batch_size: problem size knobs.
            The model's sequence length is 2*seq_half_len - 1.
        rand_seed: seeds both the model initialization and the dummy data.
        stiefel_qk: if True, Wq/Wk are constrained to O(d_model) and
            optimized via StiefelAdam; otherwise Wq/Wk are optimized
            unconstrained via Muon, same as every other parameter.
        stiefel_hyperparams: optional separate hyperparams dict for the
            StiefelAdam optimizer when stiefel_qk=True (falls back to
            `hyperparams` if omitted).

    Returns a dict with per-iteration lists:
        'loss': training cross-entropy loss
        'qk_orth_error': ||Wq^T Wq - I||_F + ||Wk^T Wk - I||_F
        'qk_skew_grad_norm': ||skew(Wq^T Gq)||_F + ||skew(Wk^T Gk)||_F
        'qk_grad_norm': ||Gq||_F + ||Gk||_F
    plus scalar 'time' (wall-clock seconds).
    """
    max_iters = hyperparams[0]['max_iters']

    seq_len = 2 * seq_half_len - 1
    model = SimpleAttentionTransformer(vocab_size, d_model, seq_len, rand_seed=rand_seed)
    input_ids, target_ids = make_induction_batch(seq_half_len, vocab_size, batch_size, rand_seed=rand_seed)

    optimizers = make_transformer_optimizers(model, hyperparams, stiefel_qk=stiefel_qk,
                                              stiefel_hyperparams=stiefel_hyperparams)

    stats = {'loss': [], 'qk_orth_error': [], 'qk_skew_grad_norm': [], 'qk_grad_norm': []}
    eye_d = torch.eye(d_model)

    def _skew_norm(X, G):
        XtG = X.T @ G
        return torch.linalg.norm(XtG - XtG.T).item()

    saved_stats={'G': [], 'P': []}
    s = time.time()
    for iter_num in range(max_iters):
        for i in range(len(optimizers)):
            init_lr = hyperparams[i]['lr']
            warmup = hyperparams[i]['warmup_iters']
            decay = hyperparams[i]['lr_decay_iters']
            min_lr = hyperparams[i]['min_lr']
            lr = get_lr(iter_num, init_lr, warmup * max_iters, decay * max_iters, min_lr)
            for param_group in optimizers[i].param_groups:
                param_group['lr'] = lr

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        loss.backward()
        stats['loss'].append(loss.item())
    
        with torch.no_grad():
            Wq, Wk = model.Wq.detach(), model.Wk.detach()
            Gq, Gk = model.Wq.grad, model.Wk.grad
            saved_stats['G'].append(Gq)
            saved_stats['P'].append(Wq)

            orth_err = (torch.linalg.norm(Wq.T @ Wq - eye_d).item()
                        + torch.linalg.norm(Wk.T @ Wk - eye_d).item())
            skew_norm = _skew_norm(Wq, Gq) + _skew_norm(Wk, Gk)
            grad_norm = Gq.norm().item() + Gk.norm().item()
        print(f"iter: {iter_num}, loss: {loss.item()}, {torch.norm(Wq)}, {torch.norm(Gq)}")

        stats['qk_orth_error'].append(orth_err)
        stats['qk_skew_grad_norm'].append(skew_norm)
        stats['qk_grad_norm'].append(grad_norm)

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    G_arr = np.array(saved_stats['G'])#[700:800]
    P_arr = np.array(saved_stats['P'])#[700:800]

    T = G_arr.shape[0]
    max_frames = T
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames, dtype=int)
        G_arr = G_arr[idx]
        T = G_arr.shape[0]

    G_min, G_max = float(G_arr.min()), float(G_arr.max())
    P_min, P_max = float(P_arr.min()), float(P_arr.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    imG = axes[0].imshow(G_arr[0], cmap='viridis', vmin=G_min, vmax=G_max)
    imP = axes[1].imshow(P_arr[0], cmap='viridis', vmin=P_min, vmax=P_max)
    axes[0].set_title('G (step 0)')
    axes[1].set_title('P (step 0)')

    fig.colorbar(imG, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(imP, ax=axes[1], fraction=0.046, pad=0.04)

    def update(frame):
        imG.set_data(G_arr[frame])
        imP.set_data(P_arr[frame])
        #print(torch.linalg.norm(torch.tensor(G_arr[frame]), ord='fro'))
        #print(f"{700+frame} diff: {torch.linalg.norm(torch.eye(n)-P_arr[frame], ord='fro')}")
        axes[0].set_title(f'G (step {frame})')
        axes[1].set_title(f'P (step {frame})')
        fig.suptitle(f'G/P matrices — frame {frame+1}/{T}')
        return [imG, imP]

    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter
    # windowms=5
    # ani = animation.FuncAnimation(fig, update, frames=T, interval=windowms, blit=False)
    # plt.tight_layout()
    # writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=100)

    # plt.show()

    #torch.save(model.state_dict(), "sd.pt")

    print(input_ids)
    print(target_ids)
    e = time.time()
    stats['time'] = e - s
    return stats

import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from eval_scripts.evalQuad import muon_hp, stiefel_hp
if __name__=='__main__':
    max_iters=2000
    muon_hp['max_iters']=max_iters
    stiefel_hp['max_iters']=max_iters
    cmap=plt.colormaps['tab20']
    colors=cmap(np.linspace(0, 1, 2))

    d_model=64
    stats1 = analysis_Transformer([muon_hp], d_model=d_model, seq_half_len=32, vocab_size=32, batch_size=64)

    # model = SimpleAttentionTransformer(d_model=d_model, seq_len=63, vocab_size=32)
    # model.load_state_dict(torch.load("/Users/christopherlinder/Desktop/OptimizerTesting.worktrees/experimental-branch-analysis-muon-stiefel/sd.pt"))
    # input_ids, target_ids = make_induction_batch(8, 32, 1, rand_seed=1)
    # l=model(input_ids)
    # print(l.shape)
    # print(l)

    # curr=model.generate(input_ids)
    # print(input_ids)
    # print(curr)

    stats2 = analysis_Transformer([muon_hp, stiefel_hp], d_model=d_model, 
        seq_half_len=32, vocab_size=32, batch_size=64, stiefel_qk=True, stiefel_hyperparams=stiefel_hp)
    
    plt.figure()
    loss1, loss2=stats1['loss'], stats2['loss']
    plt.plot(np.arange(max_iters), np.log(loss1), color=colors[0], label=f"muon")
    plt.plot(np.arange(max_iters), np.log(loss2), color=colors[1], label=f"muon-stiefel")
    mplcursors.cursor(hover=True)
    plt.xlabel('iter')
    plt.ylabel('Log Loss (base 10)')
    plt.title(rf'Muon vs StiefelAdam-Quadratic Problem')
    plt.legend()
    plt.show()