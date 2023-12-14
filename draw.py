"""
python draw.py \
    -i \
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/sub/obstacles-cs285-v0_obstacles_multi_l2_h250_mpcrandom_horizon10_actionseq1000_02-11-2023_14-06-58\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/sub/reacher-cs285-v0_reacher_multi_l2_h250_mpcrandom_horizon10_actionseq1000_02-11-2023_14-09-56\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/sub/cheetah-cs285-v0_cheetah_multi_l2_h250_mpcrandom_horizon15_actionseq1000_02-11-2023_14-23-43\
    -n "obstacle" "reacher" "halfc"\
    -d eval_return

python ./example_parse_tensorboard.py \
    -i data/run_log_1 data/run_log_2\
    -n "Name for Run 1" "Name for Run 2"\
    -d eval_return

python draw.py \
    -i \
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_02-11-2023_15-04-33\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_02-11-2023_15-58-52\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq1000_02-11-2023_16-22-50\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq100_02-11-2023_15-12-51\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon10_actionseq5000_02-11-2023_15-16-36\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon3_actionseq1000_02-11-2023_15-38-48\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/reacher-cs285-v0_reacher_ablation_l2_h250_mpcrandom_horizon30_actionseq1000_02-11-2023_15-42-18\
    -n "origin" "ense1" "ense5" "acti 100" "acti 5000" "hori 3" "hori 30"\
    -d eval_return


python draw.py \
    -i \
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/cheetah-cs285-v0_cheetah_cem_l2_h250_mpccem_horizon15_actionseq1000_cem_iters4_03-11-2023_06-23-52\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/cheetah-cs285-v0_cheetah_cem_l2_h250_mpccem_horizon15_actionseq1000_cem_iters2_03-11-2023_07-29-03\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/sub/cheetah-cs285-v0_cheetah_multi_l2_h250_mpcrandom_horizon15_actionseq1000_02-11-2023_14-23-43\
    -n "halfc_cem4" "halfc_cem2" "halfc_rand"\
    -d eval_return

    
    python draw.py \
    -i \
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_03-11-2023_08-27-08\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_03-11-2023_08-20-10\
    /home/zshccc/projectfile/DRL_hw/homework_fall2023/hw4/data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_03-11-2023_08-07-57\
    -n "free" "dyna" "mbpo"\
    -d eval_return


python draw.py -i exp/Coarse/1853_sac_test_exp   -n 'coarse'   -d eval/episode_reward  -nm 'coarse'
python draw.py -i exp/Coarse/1853_sac_test_exp/tb   -n 'coarse'   -d eval/episode_reward  -nm 'coarse'

python draw.py -i exp/Fine/0321_sac_test_exp/tb   -n 'fine'   -d eval/episode_reward  -nm 'fine'

python draw.py -i exp/Corsefrom0/0703_sac_test_exp/tb   -n 'coarse'   -d eval/episode_reward  -nm 'coarse0'


python draw.py -i exp/FinesmallLr/0146_sac_test_exp/tb   -n 'lrfine'   -d eval/episode_reward  -nm 'lrfine'

python draw.py -i exp/CoarsesmallLr/1629_sac_test_exp/tb   -n 'lrcoarse'   -d eval/episode_reward  -nm 'lrcoarse0'

exp/pg_900000

python draw.py -i exp/pg_900000/0734_sac_test_exp/tb   -n 'pg'   -d eval/episode_reward  -nm 'pg_900000'



python draw.py -i exp/smallpg/1103_sac_test_exp/tb   -n 'pg'   -d eval/episode_reward  -nm 'pg_10000'



python draw.py -i exp/lg/0744_sac_test_exp/tb -n 'vae'   -d eval/episode_reward  -nm 'vae_lg'
python draw.py -i exp/vae1500/1257_sac_test_exp/tb -n 'vae'   -d eval/episode_reward  -nm 'vae_sm'

python draw.py -i exp/2023.12.08/1459_sac_test_exp/tb -n 'random'   -d eval/episode_reward  -nm 'random'

python draw.py -i  exp/res_random/1452_sac_test_exp/tb -n 'res_rand_vae'   -d eval/episode_reward  -nm 'res_random_vae'


python draw.py -i exp/300/1407_sac_test_exp/tb -n 'pca_obs_small'   -d eval/episode_reward  -nm 'pca_obs_sm'

python draw.py -i exp/2023.12.05/Actor_PCA/tb -n 'pca_acs'   -d eval/episode_reward  -nm 'pca_acs_sm'

python draw.py -i 'exp/low and original/1007_sac_test_exp/tb' -n 'pca_obs_lg'   -d eval/episode_reward  -nm 'pca_obs_lg'

python draw.py -i 'exp/low and original/1520_sac_test_exp/tb' -n 'origin'   -d eval/episode_reward  -nm 'origin'


"""

'''
python draw.py -i exp/2023.12.05/Actor_PCA/tb  'exp/low and original/1007_sac_test_exp/tb' -n 'pca_acs' 'pca_obs'  -d eval/episode_reward  -nm 'pca_acs obs'

python draw.py -i exp/smallpg/1103_sac_test_exp/tb  'exp/low and original/1007_sac_test_exp/tb'  -n pg_small pg_large   -d eval/episode_reward  -nm 'pg'


python draw.py -i exp/vae1500/1257_sac_test_exp/tb exp/lg/0744_sac_test_exp/tb    -n vae vae_lg   -d eval/episode_reward  -nm 'vae'



python draw.py -i exp/lg/0744_sac_test_exp/tb 'exp/2023.12.08/1459_sac_test_exp/tb'  exp/res_random/1452_sac_test_exp/tb   -n vae_lg random res_vae_random   -d eval/episode_reward  -nm 'res_vae'



'''
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np

def extract_tensorboard_scalars(log_file, scalar_keys):
    # Initialize an EventAccumulator with the path to the log directory
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()  # Load the events from disk

    if isinstance(scalar_keys, str):
        scalar_keys = [scalar_keys]
    # scalar_keys = event_acc.Tags()
    # print(scalar_keys)
    # print(event_acc.Scalars('eval/episode_reward'))
    # for t in event_acc:
        # print(t)
    # Extract the scalar summaries
    scalars = {}
    for tag in scalar_keys:
        scalars_for_tag = event_acc.Scalars(tag)[:3000]
        print(scalars_for_tag)
        print(len(scalars_for_tag))
        
        scalars[tag] = {
            'step': [s.step for s in scalars_for_tag],
            'wall_time': [s.wall_time for s in scalars_for_tag],
            'value': [s.value for s in scalars_for_tag],
        }

    return scalars

def compute_mean_std(scalars: List[Dict[str, Any]],
                     data_key: str,
                     ninterp=100):
    min_step = min([s for slog in scalars for s in slog[data_key]['step']])
    max_step = max([s for slog in scalars for s in slog[data_key]['step']])
    steps = np.linspace(min_step, max_step, ninterp)
    scalars_interp = np.stack([
        np.interp(steps, slog[data_key]['step'], slog[data_key]['value'], left=float('nan'), right=float('nan'))
        for slog in scalars
    ], axis=1)

    mean = np.mean(scalars_interp, axis=1)
    std = np.std(scalars_interp, axis=1)

    return steps, mean, std


def plot_mean_std(ax: plt.Axes,
                  steps: np.ndarray,
                  mean: np.ndarray,
                  std: np.ndarray,
                  name: str,
                  color: str):
    ax.fill_between(steps, mean-std, mean+std, color=color, alpha=0.3)
    ax.plot(steps, mean, color=color, label=name)

def plot_scalars(ax: plt.Axes,
                 scalars: Dict[str, Any],
                 data_key: str,
                 name: str,
                 color: str):
    ax.plot(scalars[data_key]['step'], scalars[data_key]['value'], color=color, label=name)

if __name__ == '__main__':
    import argparse

    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_log_files", "-i", nargs='+', required=True)
    parser.add_argument("--human_readable_names", "-n", nargs='+', default=None, required=False)
    parser.add_argument("--colors", "-c", nargs='+', default=None, required=False)
    parser.add_argument("--data_key", "-d", type=str, required=True)
    parser.add_argument("--plot_mean_std", "-std", action="store_true")
    parser.add_argument("--name", "-nm", default='Default name')
    args = parser.parse_args()

    has_names = True

    if args.plot_mean_std:
        if args.colors is None:
            args.colors = [None]

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None]

        assert len(args.human_readable_names) == 1
        assert len(args.colors) == 1

        all_scalars = [extract_tensorboard_scalars(log, args.data_key) for log in args.input_log_files]
        xs, mean, std = compute_mean_std(all_scalars, args.data_key)
        plot_mean_std(plt.gca(), xs, mean, std, args.human_readable_names[0], args.colors[0])
    else:
        if args.colors is None:
            args.colors = [None] * len(args.input_log_files)

        if args.human_readable_names is None:
            has_names = False
            args.human_readable_names = [None] * len(args.input_log_files)

        assert len(args.human_readable_names) == len(args.input_log_files)
        assert len(args.colors) == len(args.input_log_files)

        for log, name, color in zip(args.input_log_files, args.human_readable_names, args.colors):
            scalars = extract_tensorboard_scalars(log, args.data_key)
            if not args.plot_mean_std:
                plot_scalars(plt.gca(), scalars, args.data_key, name, color)

    if has_names:
        plt.legend()


    
    dir='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/image'

    pat=os.path.join(dir,args.name+'.jpg')
    myfig = plt.gcf() 
    # plt.savefig(pat)
    plt.show()
    myfig.savefig(pat)

