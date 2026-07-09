from TrainingScripts import *

if __name__=='__main__':
    n=1000
    rand_seed=2
    colors = ['green', 'red', 'blue']
    i=0
    for curr_optimizer in [OPTS.S,OPTS.CS]:
        optimizer=curr_optimizer
        with open(f"data/optimalHyperParams/Quad(n={min(10,n)})_{optimizer.name}_hp.json", 'r') as f:
            hyper_params=json.load(f)
            if curr_optimizer==OPTS.SCI:
                hyper_params['lr']=.05
        hyper_params['max_iters']=1000
            
        loss, t = analysis_Quad(optimizer, hyper_params, n, rand_seed)
        plt.plot(np.arange(len(loss)), loss, color=colors[i], label=f"{optimizer.name}_{t:.2f}_seconds")
        i+=1

    plt.xlabel('Optimizer')
    plt.ylabel('Loss')
    plt.title(f'Comparison of Optimizers-{hyper_params["max_iters"]} iters')
    plt.legend()
    plt.show()