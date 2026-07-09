from TrainingScripts import *
from MLPClassifier import *

if __name__=='__main__':
    n=100
    colors = ['green', 'red', 'blue']
    i=0
    model = "MLPClassifier"
    total_samples=1000
    batch_size=100
    model={"model": model, "n": total_samples, "in_dimension": 10, "out_dimension": 1, "total_samples": total_samples, "batch_size": batch_size}
    cmap = plt.get_cmap('tab20')
    total_val_err=0
    val_errors=[]
    mean_iters=0
    experiments=10
    lrs=[.3, .5, .1]

    for curr_optimizer in [OPTS.CS]:
        losses=[]
        for rand_seed in range(experiments):
        #for curr_optimizer in [OPTS.SGD]:
            optimizer=curr_optimizer
            with open(f"data/optimalHyperParams/{model['model']}(total_samples=100)_{optimizer.name}_hp.json", 'r') as f:
                hyper_params=json.load(f)
            if curr_optimizer==OPTS.CS or curr_optimizer==OPTS.S:
                hyper_params['lr']=lrs[i]
            hyper_params['lr']=lrs[i]
            hyper_params['max_iters']=4000
                
            loss, t, err, iters = trainMLPClassifier(optimizer, hyper_params, 10, 1, max_iters=hyper_params['max_iters'], 
                rand_seed=rand_seed, total_train_samples=total_samples, total_test_samples=total_samples//5, 
                batch_size=model['batch_size'], debug=False, validate=True)
            losses.append(torch.Tensor(loss))
            total_val_err += err
            mean_iters+=iters
            #plt.plot(np.arange(len(loss)), loss, color=cmap(rand_seed%20), label=f"{optimizer.name}_{t:.2f}_seconds_val-err={err:.2f}")

        total_val_err/=10
        val_errors.append(total_val_err)
        mean_iters/=experiments
        print("ERROR, ", total_val_err, "iters: ", mean_iters)
        losses=torch.nn.utils.rnn.pad_sequence(losses, True)
        mean=np.mean(np.array(losses), axis=0)
        plt.plot(np.arange(len(mean)), mean, color=cmap(i%6), linewidth=2, label=f"{optimizer.name}_mean")
        std = np.std(np.array(losses), axis=0)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=cmap((i+1)%6), alpha=0.5, label=f"{optimizer.name}_std")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        #plt.title(f'{optimizer.name}-{mean_iters:.2f} iters, Validation Error: {total_val_err:.2f}')
        plt.legend()
        i+=1
        #np.save(f"data/optimalHyperParams/{model['model']}(total_samples=100)_{optimizer.name}_mean.npy", mean)
    plt.title(f"Mean and Std for S-{lrs[0]}, S-{lrs[1]}, and CS-{lrs[2]}")
    plt.show()