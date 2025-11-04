
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save(nodes, routes, iter, path, rs_size = 10, stage="train", value = -1, name="avg_return"):
    x, y = nodes[:,0], nodes[:,1]
    trajectory_data = routes.cpu().numpy()

    if routes[0] != 0:
        trajectory_data = np.insert(trajectory_data,0,0)

    trajectories = []
    current_trajectory = []

    for point in trajectory_data:
        if point == 0:
            if current_trajectory:  
                current_trajectory.append(0)  
                current_trajectory.insert(0,0)
                trajectories.append(current_trajectory)
                current_trajectory = []
        else:
            current_trajectory.append(point)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x[0], y[0], color='red', marker='s', s=100, label='Depot')  # Depot
    plt.scatter(x[1:rs_size + 1], y[1:rs_size + 1], color='green', marker='^', s=100, label='Recharging Stations')  # RS
    plt.scatter(x[rs_size + 1:], y[rs_size + 1:], color='blue', marker='o', s=100, label='Customers')  # Customers

    colors = [
        'purple', 'orange', 'cyan', 'green',  
        'blue', 'red', 'yellow', 'pink',      
        'brown', 'black', 'gray', 'lime',     
        'teal', 'indigo', 'magenta', 'gold',  
        'navy', 'coral', 'violet', 'olive',   
        'maroon', 'turquoise', 'beige', 'aqua'
    ]

    for i, trajectory in enumerate(trajectories):
        traj_x = [x[int(idx)] for idx in trajectory]
        traj_y = [y[int(idx)] for idx in trajectory]
        plt.plot(traj_x, traj_y, marker='o', color=colors[i%len(colors)])

    if value != -1:
        plt.title('Scatter Plot with Trajectories (#Iter:{}, RW:{:.5f})'.format(iter, value), fontsize=16)
    else:
        plt.title('Scatter Plot with Trajectories (#Iter: {}, #R:{})'.format(iter, len(trajectories)), fontsize=16)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    if not os.path.exists(path):
        os.mkdir(path)
    output_path = os.path.join(path, 'scatter_plot_with_trajectories_{}_{}.png'.format(stage, name))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()