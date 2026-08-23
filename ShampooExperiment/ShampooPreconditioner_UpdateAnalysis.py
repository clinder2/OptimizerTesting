from TrainingScripts import *

if __name__ == '__main__':
    n=100
    rand_seed=3
    O=OPTS.SS
    with open(f"/Users/christopherlinder/Desktop/OptimizerTesting/data/optimalHyperParams/NewQuad(n=100)_{O.name}_hp.json", 'r') as f:
        hyper_params=json.load(f)
    hyper_params['max_iters']=2000
    hyper_params['grafting']=True
    hyper_params['numIters']=20
    print(hyper_params)
    spectrum=[0,-.00005]
    loss, t, stats = analysis_Quad_Stats(O, hyper_params, n, spectrum, rand_seed)

    # Keep all stats in-memory in a local dict `saved_stats`.
    import numpy as np
    import torch
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    saved_stats = {}
    for k, v in stats.items():
        if isinstance(v, list) and len(v) > 0:
            first = v[0]
            if isinstance(first, torch.Tensor) or (hasattr(first, '__array__') and np.asarray(first).ndim >= 1):
                arr = np.stack([el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else np.asarray(el) for el in v])
                saved_stats[k] = arr
            else:
                # scalar list
                saved_stats[k] = [float(x) for x in v]
        else:
            saved_stats[k] = v

    # Show scalar overview (spectral norms) if available
    showspec=False
    if showspec and 'L_spec_norm' in saved_stats and 'R_spec_norm' in saved_stats:
        plt.figure()
        plt.plot(saved_stats['L_spec_norm'], label='L_spec_norm')
        plt.plot(saved_stats['R_spec_norm'], label='R_spec_norm')
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Spectral norm')
        plt.title('L and R spectral norms over time')
        plt.show()

    plt.figure()
    plt.plot(saved_stats['L_spec_norm'], label='G_fro_norm')
    plt.legend()
    plt.show()

    # Display imshow animations for L, R, G if present in-memory
    if 'L' in saved_stats and 'R' in saved_stats and 'G' in saved_stats:
        L_arr = saved_stats['L']
        R_arr = saved_stats['R']
        G_arr = saved_stats['G']

        assert L_arr.ndim == 3 and R_arr.ndim == 3 and G_arr.ndim == 3

        T = L_arr.shape[0]
        max_frames = 500
        if T > max_frames:
            idx = np.linspace(0, T - 1, max_frames, dtype=int)
            L_arr = L_arr[idx]
            R_arr = R_arr[idx]
            G_arr = G_arr[idx]
            T = L_arr.shape[0]

        L_min, L_max = float(L_arr.min()), float(L_arr.max())
        R_min, R_max = float(R_arr.min()), float(R_arr.max())
        G_min, G_max = float(G_arr.min()), float(G_arr.max())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        imL = axes[0].imshow(L_arr[0], cmap='viridis', vmin=L_min, vmax=L_max)
        axes[0].set_title('L (step 0)')
        imR = axes[1].imshow(R_arr[0], cmap='viridis', vmin=R_min, vmax=R_max)
        axes[1].set_title('R (step 0)')
        imG = axes[2].imshow(G_arr[0], cmap='viridis', vmin=G_min, vmax=G_max)
        axes[2].set_title('G (step 0)')

        fig.colorbar(imL, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(imR, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(imG, ax=axes[2], fraction=0.046, pad=0.04)

        def update(frame):
            imL.set_data(L_arr[frame])
            axes[0].set_title(f'L (step {frame})')
            imR.set_data(R_arr[frame])
            axes[1].set_title(f'R (step {frame})')
            imG.set_data(G_arr[frame])
            axes[2].set_title(f'G (step {frame})')
            fig.suptitle(f'Preconditioner matrices — frame {frame+1}/{T}')
            return [imL, imR, imG]

        ani = animation.FuncAnimation(fig, update, frames=T, interval=100, blit=False)
        plt.tight_layout()
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        # writer = animation.PillowWriter(fps=20)
        #ani.save("S_n=4_L_R_G.mp4", writer=writer)
        plt.show()
    else:
        missing = [k for k in ('L','R','G') if k not in saved_stats]
        print('Missing in-memory keys for animation:', missing)