import sys
import json
import matplotlib.pyplot as plt

def parse_jsonl_file(file_path):
    results = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for k, v in data.items():
                if type(v) == list:
                    v = v[0]
                if k in results:
                    results[k].append(v)
                else:
                    results[k] = [v]
    return results

def plot_metrics(results, savepath="fig.png"):
    plt.figure(figsize=(12, 8))
    
    metrics = ["reward_mean", "policy_loss", "critic_loss"] 
    steps = results["step"]

    for i, k in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        plt.plot(steps, results[k], marker='o')
        plt.title(f'{k} over steps')
        plt.xlabel('Step')
        plt.ylabel(k)
        plt.grid(linestyle="-.", alpha=0.8)

    plt.tight_layout()
    plt.savefig(savepath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit()
    file_path = sys.argv[1]
    results = parse_jsonl_file(file_path)
    for k, v in results.items():
        print(k, v, '\n')
    plot_metrics(results, savepath=file_path+".png")