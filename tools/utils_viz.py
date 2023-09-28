import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import clear_output
from PIL import Image
import csv
import pickle

from minigrid.core.constants import IDX_TO_COLOR, COLORS

from learner import BayesianLearner
from bayesian_ToM.bayesian_teacher import AlignedBayesianTeacher, BayesianTeacher
from tools.utils import Shannon_entropy

##
# Visualization
##

def plot_grid(start_x, num_x, size_x, alpha=0.5, \
              start_y: int|None=None, num_y: int|None=None, size_y: int|None=None):
    idx_x = np.linspace(start_x, size_x, num_x)
    if start_y is None:
        start_y = start_x
    if num_y is None:
        num_y = num_x 
    if size_y is None:
        size_y = size_x
    idx_y = np.linspace(start_y, size_y, num_y)
    
    for x in idx_x:
        plt.plot([x, x], [start_y, size_y], alpha=alpha, c='gray')
    for y in idx_y:
        plt.plot([start_x, size_x], [y, y], alpha=alpha, c='gray')

def plot_agent_play(pos: tuple, dir: int, size: float=120) -> None:
    if dir == 0:
        marker = ">"
    elif dir == 1:
        marker = "v"
    elif dir == 2:
        marker = "<"
    elif dir == 3:
        marker = "^"
    plt.scatter(pos[0], pos[1], marker=marker, c='r', s=size)

def plot_path(all_pos: list, 
              img: np.ndarray, 
              GRID_SIZE, shift: 
              bool, color: str='r',\
              width: float | None = None,
              linewidth: float | None = None, 
              scale: bool = False) -> None:
    
    ratio = img.shape[0] / GRID_SIZE
    length = len(all_pos)
    if length > 1:
        for i in range(0,length-1):
            delta = 0.5 if shift else 0.
            x1 = all_pos[i][0] + delta
            x2 = all_pos[i+1][0] + delta
            y1 = all_pos[i][1] + delta
            y2 = all_pos[i+1][1] + delta
            
            u = x2 - x1
            v = y2 -y1

            arr_x = x1 + u/2
            arr_y = y1 + v/2
            norm = np.sqrt(u**2+v**2) 

            plt.plot([x1 * ratio, x2 * ratio], [y1 * ratio, y2 * ratio], c=color, linewidth=linewidth)
            if i % 2 == 0:
                scale_value = ratio if scale else None
                plt.quiver(arr_x * ratio, arr_y * ratio, u/norm, v/norm, angles="xy", pivot="mid", color=color, width=width, scale=scale_value)
                

def plot_agent_obs(pos: tuple, GRID_SIZE: int, img: np.ndarray, hide: bool=False, size: float | None=None) -> None:
    ratio = img.shape[0] / GRID_SIZE
    if size is None:
        size = ratio * 0.5
    im_agent_pos =np.array([(pos[0] + 0.5) * ratio, (pos[1] + 0.5) * ratio]).astype('int')
    if hide:
        plt.scatter(im_agent_pos[0], im_agent_pos[1], color=rgb_to_hex((76, 76, 76)), marker='s', s=size)
    else:
        plt.scatter(im_agent_pos[0], im_agent_pos[1], color=rgb_to_hex((0, 0, 0)), marker='s', s=size)
    plt.scatter(im_agent_pos[0], im_agent_pos[1], c='w', marker='*', s=size)

def plot_error_episode_length(colors: np.ndarray, rf_values: list, num_colors: int, dict: dict) -> None:
    labels = np.concatenate((np.array(rf_values)[:-1], np.array(['full obs'])))
    for rf_idx, receptive_field in reversed(list(enumerate(rf_values))):
        all_length = []
        all_accuracy = []
        for goal_color in range(num_colors):
            all_length += dict[receptive_field][goal_color]['length']
            all_accuracy += dict[receptive_field][goal_color]['accuracy']['rf']

        bins = list(np.arange(0, (1000 // 40 + 1) * 20 + 1, 40)) + [np.max(all_length)]

        mean_accuracy = []
        std_accuracy = []
        n = []
        for i in range(len(bins) - 1):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            filtered_accuracy = [acc for dist, acc in zip(all_length, all_accuracy) if lower_bound <= dist <= upper_bound]
            mean_accuracy.append(np.mean(filtered_accuracy))
            std_accuracy.append(np.std(filtered_accuracy))
            n.append(len(filtered_accuracy))

        
        plt.bar(range(len(bins) - 1), mean_accuracy, yerr=1.96 * np.array(std_accuracy) / np.sqrt(np.array(n)),
                color=colors[rf_idx], label=f'rf={labels[rf_idx]}')

        plt.xlabel('Length of the observed episode')
        plt.ylabel('Mean Accuracy (MAP)')
        plt.title('Mean accuracy (MAP) per episode length')

        plt.xticks(range(len(bins) - 1), [f'[{bins[i]},{bins[i + 1]}]' for i in range(len(bins) - 1)])

    plt.plot([-0.5, len(bins) - 1.5], [1, 1], label='Max', ls='--', c='k')
    plt.legend()

def rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(channel))) for channel in rgb]
    # Convert RGB to hexadecimal color code (i.e. map to color type in python)
    hex_code = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    return hex_code

##
# Display for Jupiter Notebook
##

def display_learner_play(GRID_SIZE: int, 
                         learner: BayesianLearner, 
                         size: int | None=None,
                         width: float = 0.004,
                         width_belief: float = 0.04,
                         figsize: tuple = (20, 10),
                         linewidth: int | None = None) -> list:
    ii = 0
    images = []
    all_pos = [learner.env.agent_pos]
    while not learner.terminated:
        
        # Interaction
        _ = learner.play(size=1)
        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1,2,1)
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, GRID_SIZE, shift=True, width=width, scale=True, linewidth=linewidth)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')

        fig.add_subplot(1,2,2)
        learner_beliefs_image = Shannon_entropy(learner.beliefs, axis=2) / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.2)
        plt.imshow(learner_beliefs_image.T, vmin=0., vmax=1., cmap='gray')
        plot_agent_play(learner.env.agent_pos, learner.env.agent_dir, size=size)
        plot_grid(-.5, GRID_SIZE + 1, GRID_SIZE - 0.5, alpha=0.3)
        plot_path(all_pos, learner_beliefs_image, GRID_SIZE, shift=False, width=width_belief, linewidth=linewidth)
        # plt.colorbar(image)
        plt.title('Entropy learner beliefs')
        plt.axis('off')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images

def display_learner_play_teacher_infer(GRID_SIZE: int, learner: BayesianLearner, 
                                       teacher: AlignedBayesianTeacher | BayesianTeacher, 
                                       num_colors: int=4,
                                       linewidth: int | None = None) -> list:
    learner.env.highlight = True
    ii = 0
    images = []
    all_pos = [learner.env.agent_pos]
    while not learner.terminated:
        
        # Interaction
        agent_pos = learner.env.agent_pos
        agent_dir = learner.env.agent_dir
        teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
        traj = learner.play(size=1)
        teacher.observe(action=traj[0])

        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=(20,5))
        fig.add_subplot(1,3,1)
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, GRID_SIZE, shift=True, linewidth=linewidth)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')

        fig.add_subplot(1,3,2)
        learner_beliefs_image = Shannon_entropy(learner.beliefs, axis=2) / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.2)
        image = plt.imshow(learner_beliefs_image.T, vmin=0., vmax=1., cmap='gray')
        plot_agent_play(teacher.env.agent_pos, teacher.env.agent_dir)
        plot_grid(-.5, GRID_SIZE + 1, GRID_SIZE - 0.5, alpha=0.3)
        plot_path(all_pos, learner_beliefs_image, GRID_SIZE, shift=False, linewidth=linewidth)
        # plt.colorbar(image)
        plt.title('Entropy learner beliefs')
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        image = plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        plt.colorbar(image)
        plt.xticks(range(0, num_colors), [IDX_TO_COLOR[i] for i in range(1, num_colors + 1)])
        plt.yticks(range(0, teacher.num_rf), teacher.rf_values)
        plt.title(f'Teacher belief about the learner \n {teacher.__class__.__name__}')
        plt.ylabel('Receptive field')
        plt.xlabel('Goal color')
        plot_grid(start_x=-.5, num_x=num_colors+1, size_x=num_colors-0.5, \
                  start_y=-0.5, num_y=teacher.num_rf+1, size_y=teacher.num_rf-0.5)
        # plt.grid(True, which='major', linewidth=0.5)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images

def display_learner_play_teacher_infer_blind(learner: BayesianLearner, 
                                             teacher: AlignedBayesianTeacher | BayesianTeacher, 
                                             num_colors: int=4,
                                             width: float | None = None,
                                             linewidth: int | None = None) -> list:
    learner.env.highlight = False
    ii = 0
    images = []
    all_pos = [learner.env.agent_pos]
    while not learner.terminated:
        
        # Interaction
        agent_pos = learner.env.agent_pos
        agent_dir = learner.env.agent_dir
        teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
        traj = learner.play(size=1)
        teacher.observe(action=traj[0])

        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=(10,5))
        fig.add_subplot(1, 2, 1)
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, learner.env.height, shift=True, width=width, linewidth=linewidth)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        image = plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        plt.colorbar(image)
        plt.xticks(range(0, num_colors), [IDX_TO_COLOR[i] for i in range(1, num_colors + 1)])
        plt.yticks(range(0, len(teacher.rf_values)), teacher.rf_values)
        plt.title(f'Teacher belief about the learner \n {teacher.__class__.__name__}')
        plt.ylabel('Receptive field')
        plt.xlabel('Goal color')
        plot_grid(start_x=-.5, num_x=num_colors+1, size_x=num_colors-0.5, \
                  start_y=-0.5, num_y=teacher.num_rf+1, size_y=teacher.num_rf-0.5)
        # plt.grid(True, which='major', linewidth=0.5)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images

def display_learner_play_teacher_infer_blind_with_uncertainty(learner: BayesianLearner, 
                                                                teacher: AlignedBayesianTeacher | BayesianTeacher, 
                                                                num_colors: int=4,
                                                                linewidth: int | None = None) -> list:
    learner.env.highlight = False
    ii = 0
    ii_key = None
    images = []
    all_pos = [learner.env.agent_pos]
    all_un = [Shannon_entropy(np.sum(teacher.beliefs, axis=1))]
    if len(teacher.beliefs.shape) > 1:
        all_un_rf = [Shannon_entropy(np.sum(teacher.beliefs, axis=0))]
    while not learner.terminated:
        
        # Interaction
        agent_pos = learner.env.agent_pos
        agent_dir = learner.env.agent_dir
        teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
        traj = learner.play(size=1)
        teacher.observe(action=traj[0])

        all_un.append(Shannon_entropy(np.sum(teacher.beliefs, axis=1)))
        if teacher.beliefs.shape[1] > 1:
            all_un_rf.append(Shannon_entropy(np.sum(teacher.beliefs, axis=0)))


        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=(20,5))
        fig.add_subplot(1, 3, 1)
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, learner.env.height, shift=True, linewidth=linewidth)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        image = plt.imshow(teacher.beliefs.T, vmin=0., vmax=1.)
        plt.colorbar(image)
        plt.xticks(range(0, num_colors), [IDX_TO_COLOR[i] for i in range(1, num_colors + 1)])
        plt.yticks(range(0, len(teacher.rf_values)), teacher.rf_values)
        plt.title(f'Teacher belief about the learner \n {teacher.__class__.__name__}')
        plt.ylabel('Receptive field')
        plt.xlabel('Goal color')
        plot_grid(start_x=-.5, num_x=num_colors+1, size_x=num_colors-0.5, \
                  start_y=-0.5, num_y=teacher.num_rf+1, size_y=teacher.num_rf-0.5)
        
        fig.add_subplot(1, 3, 3)
        plt.plot(all_un, label='Uncertainty on the goal')
        if teacher.beliefs.shape[1] > 1:
            plt.plot(all_un_rf, label='Uncertainty on the receptive field')
            plt.title('Uncertainty of the teacher about \n the goal and receptive fiel \n  of the learner (Shannon entropy)')
            plt.legend()
        else:
            plt.title('Uncertainty of the teacher about \n the goal of the learner (Shannon entropy)')
        plt.ylim(-0.1)
        if (learner.env.carrying is not None) and (ii_key is None):
            ii_key = ii
        if ii_key is not None:
            plt.plot([ii_key, ii_key], [0, np.max(all_un)], label='Learner grabs the key', ls='--', c='r')
            plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Uncertainty (Shannon entropy)')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images

def display_learner_obs_demo(GRID_SIZE: int, 
                             learner: BayesianLearner, 
                             width: float | None = 0.004,
                             start_size: int = 100,
                             linewidth: int | None = None):
    
    learner.env.highlight = False
    hide = learner.env.highlight
    ii = 0
    images = []
    all_pos = [learner.pos[0]]
    for frame in learner.render_frames_observation:

        if all_pos[-1] != learner.pos[ii]:
            all_pos.append(learner.pos[ii])

        fig = plt.figure(figsize=(20,10))
        fig.add_subplot(1,2,1)
        plt.imshow(frame)
        plot_agent_obs(learner.pos[ii], GRID_SIZE, frame, hide=hide, size=start_size)
        plot_path(all_pos, frame, GRID_SIZE, True, color='w', width=width, scale=True, linewidth=linewidth)
        plt.title(f'Demonstration (t={ii}) (teleoperate)')
        plt.axis('off')

        fig.add_subplot(1,2,2)
        learner_beliefs_image = learner.render_beliefs_observation[ii]
        plt.imshow(learner_beliefs_image, vmin=0., vmax=1., cmap='gray')
        plot_grid(-.5, GRID_SIZE + 1, GRID_SIZE - 0.5, alpha=0.3)
        plot_agent_obs(learner.pos[ii], GRID_SIZE, learner_beliefs_image, hide=False, size=start_size)
        plot_path(all_pos, learner_beliefs_image, GRID_SIZE, False, color='w', width=width, linewidth=linewidth)
        plt.title('Entropy learner beliefs')
        plt.axis('off')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images

def save_LOG(filename: str, agent: BayesianTeacher | AlignedBayesianTeacher | BayesianLearner) -> None:

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for sentence in agent.LOG:
            writer.writerow([sentence])

def display_ToM_hist(GRID_SIZE: int, load_filename: str, save_filename: str,
                     N: int, lambd: float,
                     rf_values_basic: list=[3,5,7], num_colors: int=4) -> None:
    
    rf_values = rf_values_basic + [GRID_SIZE]
    
    with open(load_filename, 'rb') as f:
        DICT = pickle.load(f)
    dict = DICT[GRID_SIZE]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.1])

    # First row, first column
    ax1 = plt.subplot(gs[0, 0])
    confusion_matrix = np.zeros((len(rf_values), len(rf_values)))
    for rf_idx,receptive_field in enumerate(rf_values):
        for goal_color in range(num_colors):
            
            for beliefs in dict[receptive_field][goal_color]['beliefs']:
                    confusion_matrix[rf_idx, :] += beliefs[goal_color, :]

    confusion_matrix /= num_colors * N
    plt.imshow(confusion_matrix, vmin=0., vmax=1.)
    images = plt.imshow(confusion_matrix, vmin=0., vmax=1., cmap='plasma')
    plt.colorbar(images)
    plt.title('Confusion matrix')
    plt.ylabel('Receptive field')
    plt.xlabel('Receptive field')
    plt.xticks(range(0, len(rf_values)), rf_values)
    plt.yticks(range(0, len(rf_values)), rf_values)

    # First row, second column
    ax2 = plt.subplot(gs[0, 1])

    colors = [np.array([149, 183, 226]) / 255, 'blue', np.array([52, 85, 156]) / 255, 'yellowgreen']

    mean_all = np.zeros(len(rf_values))
    std_all = np.zeros(len(rf_values))
    for rf_idx, rf in enumerate(rf_values):
        all_acc = []
        for goal_color in range(num_colors):
            all_acc += dict[rf][goal_color]['accuracy']['rf']
        mean_all[rf_idx] = np.mean(all_acc)
        std_all[rf_idx] = 1.96 * np.std(all_acc) / np.sqrt(len(all_acc))

    plt.bar(np.array(rf_values).astype(str), mean_all, width=.9, yerr=std_all, color=colors)
    plt.plot([-0.5, len(rf_values)-0.5], [1, 1], c='k', label='Max', ls='--')
    plt.ylabel('Accuracy (MAP)')
    plt.xlabel('Receptive field')
    plt.legend()
    plt.title('Accuracy (MAP) per receptive field')

    ax3 = plt.subplot(gs[1, :])
    plot_error_episode_length(colors=colors, rf_values=rf_values, num_colors=num_colors, dict=dict)

    # plt.tight_layout()

    fig.suptitle(f'Analysis GRID_SIZE={GRID_SIZE}, $\lambda$={lambd}', fontweight='bold')

    fig.savefig(save_filename);

def display_ToM_errorbar(load_filename: str, save_filename: str, lambd: float,
                         rf_values_basic: list=[3,5,7], num_colors: int=4) -> None:
    
    fig = plt.figure(figsize=(10,5))
    colors = [np.array([149, 183, 226]) / 255, 'blue', np.array([52, 85, 156]) / 255, 'yellowgreen']

    with open(load_filename, 'rb') as f:
        DICT = pickle.load(f)
    grid_size_values = DICT.keys()

    for ii,GRID_SIZE in enumerate(grid_size_values):
        dict = DICT[GRID_SIZE]
        rf_values = np.array(rf_values_basic + [GRID_SIZE])
        labels = np.concatenate((np.array(rf_values)[:-1], np.array(['full obs'])))
        
        for rf_idx, rf in enumerate(rf_values):
            all_acc = []
            for goal_color in range(num_colors):
                all_acc += dict[rf][goal_color]['accuracy']['rf']
            if ii == len(grid_size_values)-1:
                plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / (np.sqrt(len(all_acc))), color=colors[rf_idx], fmt="o", label=f'rf={labels[rf_idx]}')
            else:
                plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / (np.sqrt(len(all_acc))), color=colors[rf_idx], fmt="o")
                
    plt.plot([0, len(grid_size_values)-1], [1, 1], label='Max', ls='--', c='k')
    plt.xticks(np.arange(len(grid_size_values)), grid_size_values)
    plt.xlabel('Grid size')
    plt.ylabel('Accuracy (MAP)')
    plt.title(f'Mean accuracy (MAP) as a function of the environment size \n $\lambda$={lambd}')
    plt.legend(loc=2)
    plt.ylim(0.,1.05)

    fig.savefig(save_filename);


def display_all_ToM(path: str, 
                    lamd_values: list, 
                    grid_size_values: list, 
                    rf_values_basic: list=[3, 5, 7], 
                    num_colors: int=4) -> None:
    markers = ['*', '^', 'o', 's', 'd', 'h', 'x', 'v']
    colors = ['gold', 'orange', 'orangered', 'magenta', 'purple', 'blue', 'seagreen', 'slategrey']


    plt.figure(figsize=(10, 6))

    for kk, lambd in enumerate(lamd_values):
        with open(path + f'/lambda_{lambd}/stats_outputs_lambd_{lambd}.pickle', 'rb') as f:
            DICT = pickle.load(f)
        
        for ii,GRID_SIZE in enumerate(grid_size_values):
            dict = DICT[GRID_SIZE]
            rf_values = np.array(rf_values_basic + [GRID_SIZE])
            labels = np.concatenate((np.array(rf_values)[:-1], np.array(['full obs'])))
            
            all_acc = []
            for rf_idx, rf in enumerate(rf_values):
                for goal_color in range(num_colors):
                    all_acc += dict[rf][goal_color]['accuracy']['rf']
            if ii == len(grid_size_values)-1:
                plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / np.sqrt(len(all_acc)), color=colors[kk], fmt=markers[kk], label=f'$\lambda$={lambd}')
            else:
                plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / np.sqrt(len(all_acc)), color=colors[kk], fmt=markers[kk])

    with open(path + f'/lambda_aligned/stats_outputs_lambd_aligned.pickle', 'rb') as f:
        DICT = pickle.load(f)

    kk += 1
    for ii,GRID_SIZE in enumerate(grid_size_values):
        dict = DICT[GRID_SIZE]
        rf_values = np.array(rf_values_basic + [GRID_SIZE])
        labels = np.concatenate((np.array(rf_values)[:-1], np.array(['full obs'])))
        
        all_acc = []
        for rf_idx, rf in enumerate(rf_values):
            for goal_color in range(num_colors):
                all_acc += dict[rf][goal_color]['accuracy']['rf']
        if ii == len(grid_size_values)-1:
            plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / np.sqrt(len(all_acc)), color=colors[kk], fmt=markers[kk], label=f'Aligned')
        else:
            plt.errorbar(ii, np.mean(all_acc), yerr=1.96 * np.std(all_acc) / np.sqrt(len(all_acc)), color=colors[kk], fmt=markers[kk])

    plt.plot([0, len(grid_size_values)-1], [1, 1], ls='--', label='Max', c='k')
    plt.xticks(np.arange(len(grid_size_values)), grid_size_values)
    plt.xlabel('Grid size of the observation environment')
    plt.ylabel('RF-inference accuracy (MAP)')
    plt.title('Mean RF-inference accuracy per Boltzmann temperature parameter $\lambda$')
    plt.legend()
    plt.legend(loc=2);

def display_cost(cost_fun, alpha):
    fig = plt.figure(figsize=(10,5))

    for l in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
        X = np.arange(0,20)
        Y = cost_fun(X, l)
        plt.scatter(X, Y, label=f'l_max={l}')

    plt.title(f'{cost_fun.__name__} cost function \n alpha={alpha}')
    plt.ylabel('cost')
    plt.xlabel('delta_d')
    plt.legend();

def display_learner_play_teacher_infer_blind_with_uncertainty_color(learner: BayesianLearner, 
                                                                    teacher: AlignedBayesianTeacher | BayesianTeacher, 
                                                                    num_colors: int=4) -> list:
    green = COLORS[IDX_TO_COLOR[1]] #np.array([0, 255, 0])
    blue = COLORS[IDX_TO_COLOR[2]] #np.array([0, 0, 255])
    purple = COLORS[IDX_TO_COLOR[3]] #np.array([148,0,211])
    yellow = COLORS[IDX_TO_COLOR[4]] #np.array([255, 234, 0])

    goal_colors = [green, blue, purple, yellow]
    colors_normalized = [[c[0] / 255, c[1] / 255, c[2] / 255] for c in goal_colors]

    rf_3 = np.array([10]) 
    rf_5 = np.array([100])
    # rf_7 = np.array([150])
    full_obs = np.array([250])

    rf_colors = [rf_3, rf_5, full_obs]
    values = [color[0] for color in rf_colors]

    cmap_goal = LinearSegmentedColormap.from_list('custom_gradient', colors_normalized, N=100)
    cmap_rf = LinearSegmentedColormap.from_list('custom_gray', ['black', 'white'], N=100)

    learner.env.highlight = False
    ii = 0
    ii_key = None
    images = []

    goal_belief = np.sum(teacher.beliefs, axis=1)
    print(goal_belief[1], goal_colors[1])
    color = np.sum([goal_colors[i] * goal_belief[i] for i in range(num_colors)], axis=0)
    all_colors_goal = [color]

    all_pos = [learner.env.agent_pos]
    all_un = [Shannon_entropy(np.sum(teacher.beliefs, axis=1))]
    if teacher.beliefs.shape[1] > 1:
        all_un_rf = [Shannon_entropy(np.sum(teacher.beliefs, axis=0))]
        rf_belief = np.sum(teacher.beliefs, axis=0)
        color = np.sum([rf_colors[i] * rf_belief[i] for i in range(len(rf_colors))], axis=0)
        all_colors_rf = [color]
    while not learner.terminated:
        
        # Interaction
        agent_pos = learner.env.agent_pos
        agent_dir = learner.env.agent_dir
        teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
        traj = learner.play(size=1)
        teacher.observe(action=traj[0])

        all_un.append(Shannon_entropy(np.sum(teacher.beliefs, axis=1)))
        if teacher.beliefs.shape[1] > 1:
            all_un_rf.append(Shannon_entropy(np.sum(teacher.beliefs, axis=0)))


        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=(17,5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.35, 0.65])

        fig.add_subplot(gs[0])
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, learner.env.height, shift=True)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')
        
        fig.add_subplot(gs[1])
        goal_belief = np.sum(teacher.beliefs, axis=1)
        color = np.sum([goal_colors[i] * goal_belief[i] for i in range(num_colors)], axis=0)
        all_colors_goal.append(color)
        for kk, color in enumerate(all_colors_goal):
            plt.scatter([kk], [-0.15], marker='s', c=rgb_to_hex(color), s=200)

        # Create colorbar for goal colors
        cbar_goal = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_goal), orientation='vertical', ticks=np.linspace(0, 1, len(goal_colors)))
        cbar_goal.ax.set_yticklabels(['Green', 'Blue', 'Purple', 'Yellow'])
        cbar_goal.set_label('Goal color scale')

        if teacher.beliefs.shape[1] > 1:
            rf_belief = np.sum(teacher.beliefs, axis=0)
            color = np.sum([rf_colors[i] * rf_belief[i] for i in range(len(rf_colors))], axis=0)
            all_colors_rf.append(color)
            for kk, color in enumerate(all_colors_rf):
                plt.scatter([kk], [-0.35], marker='s', c=color, s=200, cmap='gray', vmin=0, vmax=255)

            # Create colorbar for RF scale
            tick_positions = np.linspace(0, 1, len(values))
            cbar_rf = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_rf), orientation='vertical', ticks=tick_positions)
            cbar_rf.ax.set_yticklabels(['RF_3', 'RF_5', 'Full Obs'])
            cbar_rf.set_label('RF scale')

            legend_place = (1.45, 1)
        else:
            legend_place = (1.15, 1)

        plt.plot(all_un, label='Uncertainty on the goal', c='darkblue')
        if teacher.beliefs.shape[1] > 1:
            plt.plot(all_un_rf, label='Uncertainty on \n the receptive field', c='darkorange')
            plt.title('Uncertainty of the teacher about the goal and \n receptive fiel of the learner (Shannon entropy)')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        else:
            plt.title('Uncertainty of the teacher about \n the goal of the learner (Shannon entropy)')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        plt.ylim(-0.5)
        plt.grid('on')
        if (learner.env.carrying is not None) and (ii_key is None):
            ii_key = ii
        if ii_key is not None:
            plt.plot([ii_key, ii_key], [0, np.max(all_un)], label='Learner grabs the key', ls='--', c='r')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        plt.xlabel('Step')
        plt.ylabel('Uncertainty (Shannon entropy)')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images


def display_learner_play_both_teachers_infer_blind_with_uncertainty_color(learner: BayesianLearner, 
                                                                          rf_values_basic: tuple,
                                                                          lambd: float = 0.01,
                                                                          num_colors: int=4) -> list:
    green = COLORS[IDX_TO_COLOR[1]] #np.array([0, 255, 0])
    blue = COLORS[IDX_TO_COLOR[2]] #np.array([0, 0, 255])
    purple = COLORS[IDX_TO_COLOR[3]] #np.array([148,0,211])
    yellow = COLORS[IDX_TO_COLOR[4]] #np.array([255, 234, 0])

    goal_colors = [green, blue, purple, yellow]
    colors_normalized = [[c[0] / 255, c[1] / 255, c[2] / 255] for c in goal_colors]

    rf_3 = np.array([10]) 
    rf_5 = np.array([100])
    # rf_7 = np.array([150])
    full_obs = np.array([250])

    rf_colors = [rf_3, rf_5, full_obs]
    values = [color[0] for color in rf_colors]

    cmap_goal = LinearSegmentedColormap.from_list('custom_gradient', colors_normalized, N=100)
    cmap_rf = LinearSegmentedColormap.from_list('custom_gray', ['black', 'white'], N=100)

    learner.env.highlight = False
    ii = 0
    ii_key = None
    images = []

    ## Rational ToM-teacher

    teacher = BayesianTeacher(env=learner.env, lambd=lambd, rf_values=rf_values_basic)
    goal_belief = np.sum(teacher.beliefs, axis=1)

    color = np.sum([goal_colors[i] * goal_belief[i] for i in range(num_colors)], axis=0)
    all_colors_goal = [color]

    all_pos = [learner.env.agent_pos]
    all_un = [Shannon_entropy(np.sum(teacher.beliefs, axis=1))]
    if teacher.beliefs.shape[1] > 1:
        all_un_rf = [Shannon_entropy(np.sum(teacher.beliefs, axis=0))]
        rf_belief = np.sum(teacher.beliefs, axis=0)
        color = np.sum([rf_colors[i] * rf_belief[i] for i in range(len(rf_colors))], axis=0)
        all_colors_rf = [color]
    
    ## Aligned ToM-teacher

    aligned_teacher = AlignedBayesianTeacher(env=learner.env, rf_values=rf_values_basic)
    aligned_goal_belief = np.sum(aligned_teacher.beliefs, axis=1)

    color = np.sum([goal_colors[i] * aligned_goal_belief[i] for i in range(num_colors)], axis=0)
    aligned_all_colors_goal = [color]

    all_pos = [learner.env.agent_pos]
    aligned_all_un = [Shannon_entropy(np.sum(aligned_teacher.beliefs, axis=1))]
    if aligned_teacher.beliefs.shape[1] > 1:
        aligned_all_un_rf = [Shannon_entropy(np.sum(aligned_teacher.beliefs, axis=0))]
        aligned_rf_belief = np.sum(aligned_teacher.beliefs, axis=0)
        color = np.sum([rf_colors[i] * aligned_rf_belief[i] for i in range(len(rf_colors))], axis=0)
        aligned_all_colors_rf = [color]

    while not learner.terminated:
        
        # Interaction
        agent_pos = learner.env.agent_pos
        agent_dir = learner.env.agent_dir

        teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
        aligned_teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)

        traj = learner.play(size=1)

        teacher.observe(action=traj[0])
        aligned_teacher.observe(action=traj[0])

        ## Rational ToM-teacher
        all_un.append(Shannon_entropy(np.sum(teacher.beliefs, axis=1)))
        if teacher.beliefs.shape[1] > 1:
            all_un_rf.append(Shannon_entropy(np.sum(teacher.beliefs, axis=0)))

        ## Aligned ToM-teacher
        aligned_all_un.append(Shannon_entropy(np.sum(aligned_teacher.beliefs, axis=1)))
        if aligned_teacher.beliefs.shape[1] > 1:
            aligned_all_un_rf.append(Shannon_entropy(np.sum(aligned_teacher.beliefs, axis=0)))

        if learner.env.agent_pos != all_pos[-1]:
            all_pos.append(learner.env.agent_pos)

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[0.4, 0.6])

        fig.add_subplot(gs[0, 0])
        img = learner.env.render()
        plt.imshow(img)
        plot_path(all_pos, img, learner.env.height, shift=True)
        plt.title(f'Learner (t={ii})')
        plt.axis('off')
        
        fig.add_subplot(gs[1, 0])
        goal_belief = np.sum(teacher.beliefs, axis=1)
        color = np.sum([goal_colors[i] * goal_belief[i] for i in range(num_colors)], axis=0)
        all_colors_goal.append(color)
        for kk, color in enumerate(all_colors_goal):
            plt.scatter([kk], [-0.15], marker='s', c=rgb_to_hex(color), s=200)

        if teacher.beliefs.shape[1] > 1:
            rf_belief = np.sum(teacher.beliefs, axis=0)
            color = np.sum([rf_colors[i] * rf_belief[i] for i in range(len(rf_colors))], axis=0)
            all_colors_rf.append(color)
            for kk, color in enumerate(all_colors_rf):
                plt.scatter([kk], [-0.35], marker='s', c=color, s=200, cmap='gray', vmin=0, vmax=255)

            legend_place = (1.45, 1)
        else:
            legend_place = (1.15, 1)

        plt.plot(all_un, label='Uncertainty on the goal', c='darkblue')
        if teacher.beliefs.shape[1] > 1:
            plt.plot(all_un_rf, label='Uncertainty on \n the receptive field', c='darkorange')
            plt.title('Uncertainty of the rational teacher about the goal and \n receptive fiel of the learner (Shannon entropy)')
            # plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        else:
            plt.title('Uncertainty of the rational teacher about \n the goal of the learner (Shannon entropy)')
            # plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        plt.ylim(-0.5)
        plt.grid('on')
        if (learner.env.carrying is not None) and (ii_key is None):
            ii_key = ii
        if ii_key is not None:
            plt.plot([ii_key, ii_key], [0, np.max(all_un)], label='Learner grabs the key', ls='--', c='r')
            # plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        plt.xlabel('Step')
        plt.ylabel('Uncertainty (Shannon entropy)')


        fig.add_subplot(gs[1, 1])
        aligned_goal_belief = np.sum(aligned_teacher.beliefs, axis=1)
        color = np.sum([goal_colors[i] * aligned_goal_belief[i] for i in range(num_colors)], axis=0)
        aligned_all_colors_goal.append(color)
        for kk, color in enumerate(aligned_all_colors_goal):
            plt.scatter([kk], [-0.15], marker='s', c=rgb_to_hex(color), s=200)

        # Create colorbar for goal colors
        cbar_goal = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_goal), orientation='vertical', ticks=np.linspace(0, 1, len(goal_colors)))
        cbar_goal.ax.set_yticklabels(['Green', 'Blue', 'Purple', 'Yellow'])
        cbar_goal.set_label('Goal color scale')

        if aligned_teacher.beliefs.shape[1] > 1:
            aligned_rf_belief = np.sum(aligned_teacher.beliefs, axis=0)
            color = np.sum([rf_colors[i] * aligned_rf_belief[i] for i in range(len(rf_colors))], axis=0)
            aligned_all_colors_rf.append(color)
            for kk, color in enumerate(aligned_all_colors_rf):
                plt.scatter([kk], [-0.35], marker='s', c=color, s=200, cmap='gray', vmin=0, vmax=255)

            # Create colorbar for RF scale
            tick_positions = np.linspace(0, 1, len(values))
            cbar_rf = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_rf), orientation='vertical', ticks=tick_positions)
            cbar_rf.ax.set_yticklabels(['RF_3', 'RF_5', 'Full Obs'])
            cbar_rf.set_label('RF scale')

            legend_place = (1.45, 1)
        else:
            legend_place = (1.15, 1)

        plt.plot(aligned_all_un, label='Uncertainty on the goal', c='darkblue')
        
        if aligned_teacher.beliefs.shape[1] > 1:
            plt.plot(aligned_all_un_rf, label='Uncertainty on \n the receptive field', c='darkorange')
            plt.title('Uncertainty of the aligned_teacher about the goal and \n receptive fiel of the learner (Shannon entropy)')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        else:
            plt.title('Uncertainty of the aligned_teacher about \n the goal of the learner (Shannon entropy)')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        plt.ylim(-0.5)
        plt.grid('on')
        
        if (learner.env.carrying is not None) and (ii_key is None):
            ii_key = ii
        if ii_key is not None:
            plt.plot([ii_key, ii_key], [0, np.max(aligned_all_un)], label='Learner grabs the key', ls='--', c='r')
            plt.legend(loc='upper left', bbox_to_anchor=legend_place)
        
        plt.xlabel('Step')
        plt.ylabel('Uncertainty (Shannon entropy)')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the image buffer as a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        images.append(pil_image)

        clear_output(wait=True)
        plt.show(fig)

        ii += 1
    return images