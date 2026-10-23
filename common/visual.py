import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

def heat_map_matrix_series(W_arr):
    assert W_arr.ndim == 3

    T = W_arr.shape[0]
    max_frames = T
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames, dtype=int)
        W_arr = W_arr[idx]
        T = W_arr.shape[0]

    G_min, G_max = float(W_arr.min()), float(W_arr.max())
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    imG = axes.imshow(W_arr[0], cmap='viridis', vmin=G_min, vmax=G_max)
    axes.set_title('G (step 0)')
    
    fig.colorbar(imG, ax=axes, fraction=0.046, pad=0.04)
    
    def update(frame):
        imG.set_data(W_arr[frame])
        axes.set_title(f'G (step {frame})')
        fig.suptitle(f'G/P matrices — frame {frame+1}/{T}')
        return [imG]

    windowms=200
    ani = animation.FuncAnimation(fig, update, frames=T, interval=windowms, blit=False)
    plt.tight_layout()
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=100)

    #writer = animation.PillowWriter(fps=20)
    #ani.save("StiefelAdam.mp4", writer=writer)
    plt.show()
