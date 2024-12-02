# Import libraries

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as patches

from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from processing_TowerCoordinates import *

# Functions to plot

def plot_trajectory(ax, X_positions_cm, Y_positions_cm, total_distance, average_speed):
    """
    Plots the mouse trajectory on the given axis with total distance and average speed.

    ax: define the axe in which the plot will be
    X_positions_cm: dictionnary of every X positions in cm, found in pickle
    Y_positions_cm : dictionnary of every Y positions in cm, found in pickle
    total_distance: calculated from distances
    average_speed: one value retrieved from pickle

    """
    ax.plot(X_positions_cm, Y_positions_cm, color='black', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Mouse session trajectory', fontsize=17)

    
    # Add total distance and average speed to the graph
    text = f"Total distance: {total_distance:.2f} m\nAverage speed: {average_speed:.2f} cm/s"
    ax.text(0.5, -0.1, text, ha='center', va='bottom', transform=ax.transAxes, fontsize=17)

def plot_speed_distribution(ax, speeds):
    """
    Plots the speed distribution histogram on the given axis.
    """
    bins_speed = np.arange(0, 104, 4)  # Steps from 4 cm/s up to 100 cm/s
    ax.hist(speeds, bins=bins_speed, edgecolor='gray', color='k')
    ax.set_title('Speed distribution', fontsize=17)
    ax.set_xlabel('Speed (cm/s)', labelpad=10, fontsize=14)
    ax.set_ylabel('Frequency', labelpad=10, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)

def plot_angular_speed_distribution(ax, angular_speeds):
    """
    Plots the angular speed distribution histogram on the given axis.
    """
    bins_angular_speed = np.linspace(-180, 180, 30)  # Steps for angular velocities
    ax.hist(angular_speeds, bins=bins_angular_speed, edgecolor='gray', color='k')
    ax.set_title('Angular speed distribution', fontsize=17)
    ax.set_xlabel('Angular speed (degrees/s)', labelpad=10, fontsize=14)
    ax.set_ylabel('Frequency', labelpad=10, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)

def plot_metrics_in_zones(ax, metrics_data, metric_title='Undefined metric', ylabel = 'Undefined y label', ymax=None):
    """
    Parameters:
        ax : axe in which we add the subplot
        metrics_data (list) : list containing all the metrics you want to plot
        metric_title (str) : used to set the title of the graph 
        ylabel (str) : title of the y axis
        ymax : max y coordinate

    """

    # For time data : ymax = 900
    # For distance data: ymax= 10500

    # Tracer la métrique choisie en fonction de la zone
    ax.bar(['Border', 'Trapeze', 'Interior'], metrics_data, color=['red', 'lightgray', 'green'])
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title(metric_title, fontsize=17)
    ax.set_xlabel('Zones', labelpad=10, fontsize=14)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ymax == None:
        ax.set_ylim(0,max(metrics_data)*1.1)
    else:
        ax.set_ylim(0, ymax)

def plot_ratios_in_zones(ax, ratios_data, title_ratio = 'Undefined title', xlabel = 'Undefined label', ymax=None):
    
    # Plot the trapeze/border ratios for time and distance
    ax.bar(['Time', 'Distance'], ratios_data, color=['purple', 'orange'])
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title(title_ratio, fontsize=17)
    ax.set_xlabel(xlabel, labelpad=10, fontsize=14)
    ax.set_ylabel('Ratios', labelpad=10, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ymax == None:
        ax.set_ylim(0,max(ratios_data)*1.1)
    else:
        ax.set_ylim(0, ymax)

def plot_cumulative_runs(ax, runs, legend_label='Run type', color='orange', ymax=None):

    # Time extraction
    run_times = np.sort(np.array([run[3]['time'] for run in runs]))

    # Cumulative count
    cumulative_count = np.arange(1, len(run_times)+1)

    # Tracé du graphique cumulatif
    ax.plot(run_times, cumulative_count, label=legend_label, color=color)
    ax.set_xlabel('Time (s)', labelpad=10, fontsize=14)
    ax.set_ylabel('Cumulative number of runs', labelpad=10, fontsize=14)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    # ax.set_title(f'Cumulative number of {legend_label}', pad=10, fontsize=17)
    ax.legend(fontsize=12, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_speed_over_time(ax, traject_time, speeds, run_type, xlim=5.1, set_title=False, title_type = 'run not defined'):
    """
    Trace la vitesse de la souris en fonction du temps pour le type de run spécifié.
    
    Paramètres :
        ax : Axe matplotlib sur lequel dessiner le graphique.
        traject_time (list) : Temps de la trajectoire.
        speeds (list) : Vitesse de la souris.
        runs (list) : Liste des runs pour le type spécifié.
        run_type (str) : Type de run ('between_towers', 'exploratory', 'around_tower').
        color_map (Colormap) : Colormap pour le gradient de couleur.
    """

    # runs = all_epochs[run_type]

    norm = Normalize(vmin=0, vmax=len(run_type))
    color_map = plt.cm.copper
    
    for index, run in enumerate(run_type):
        start_index, end_index = run[0][0], run[0][1]
        adjusted_time = [t - traject_time[start_index] for t in traject_time[start_index:end_index + 1]]
        
        # Tracé de la vitesse
        ax.plot(adjusted_time, speeds[start_index:end_index + 1], color=color_map(norm(index)))

    ax.set_ylabel('Speed (cm/s)', labelpad=10, fontsize=14)
    ax.set_xlabel('Time (s)', labelpad=10, fontsize=14)
    ax.set_ylim(0, 80)
    ax.set_xlim(0, xlim)
    
    if set_title:
        ax.set_title(f"Speed profile for {title_type}", pad=10, fontsize=17)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_cumulative_rewarded_unrewarded(ax, runs_around_tower):

    # Create empty lists to store runs
    runs_around_tower_rewarded = []
    runs_around_tower_unrewarded = []

    for run in runs_around_tower:
        if run[3]['Rewarded'] == True:
            runs_around_tower_rewarded.append(run)
        else:
            runs_around_tower_unrewarded.append(run)

    # Extraire les temps des runs "rewarded" et "unrewarded" dans l'ordre croissant
    rewarded_times_sorted = np.sort(np.array([run[4]['epoch_time'] for run in runs_around_tower_rewarded]))
    unrewarded_times_sorted = np.sort(np.array([run[4]['epoch_time'] for run in runs_around_tower_unrewarded]))

    # Calculer les cumulés
    cumulative_rewarded = np.arange(1, len(rewarded_times_sorted) + 1)
    cumulative_unrewarded = np.arange(1, len(unrewarded_times_sorted) + 1)

    # Tracer les courbes cumulatives
    ax.plot(rewarded_times_sorted, cumulative_rewarded, label='Rewarded', color='mediumseagreen')
    ax.plot(unrewarded_times_sorted, cumulative_unrewarded, label='Unrewarded', color='firebrick')


    # Paramètres du graphique
    ax.set_xlabel('Time (s)', labelpad=10, fontsize=14)
    ax.set_ylabel('Cumulative number of runs', labelpad=10, fontsize=14)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    # ax.set_title('Cumulatif des runs Rewarded et Unrewarded')
    ax.legend(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_cumulative_CW_CCW(ax, runs_around_tower):
    # Create empty lists to store runs
    runs_around_tower_CW = []
    runs_around_tower_CCW = []

    for run in runs_around_tower:
        if run[3]['direction'] == 'CW':
            runs_around_tower_CW.append(run)
        else:
            runs_around_tower_CCW.append(run)

    # Extraire les temps des runs "CW" et "CW" dans l'ordre croissant
    CW_times_sorted = np.sort(np.array([run[4]['epoch_time'] for run in runs_around_tower_CW]))
    CCW_times_sorted = np.sort(np.array([run[4]['epoch_time'] for run in runs_around_tower_CCW]))

    # Calculer les cumulés
    cumulative_CW = np.arange(1, len(CW_times_sorted) + 1)
    cumulative_CCW = np.arange(1, len(CCW_times_sorted) + 1)

    # Tracer les courbes cumulatives
    ax.plot(CW_times_sorted, cumulative_CW, label='CW', color='#22cacaff')
    ax.plot(CCW_times_sorted, cumulative_CCW, label='CCW', color='#f568afff')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    # Paramètres du graphique
    ax.set_xlabel('Time (s)', labelpad=10, fontsize=14)
    ax.set_ylabel('Cumulative number of runs', labelpad=10, fontsize=14)
    # ax.set_title('Cumulative number of runs CW and CCW')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12)

def plot_trajectory_type_centered(ax, smoothed_Xpositions, smoothed_Ypositions, run_type, 
                                  axis_lim=25, direction='None', xlim=30, ylim=30, q=4, line_width=0.5, arrow_width=0.001):
    """
    Trace uniquement la trajectoire, sans cadre ni axes autour de la figure.
    """
    fixed_origin = (0, 0)
    
    for index, run in enumerate(run_type):
        start_index, end_index = run[0][0], run[0][1]
        runtype_epoch_Xpositions = smoothed_Xpositions[start_index:end_index + 1]
        runtype_epoch_Ypositions = smoothed_Ypositions[start_index:end_index + 1]
        numberofpositions = len(runtype_epoch_Xpositions)
        colorgradientforthisrun = custom_cmap(numberofpositions)
        
        start_x, start_y = runtype_epoch_Xpositions[0], runtype_epoch_Ypositions[0]
        translated_Xpositions = [x - start_x + fixed_origin[0] for x in runtype_epoch_Xpositions]
        translated_Ypositions = [y - start_y + fixed_origin[1] for y in runtype_epoch_Ypositions]
        
        for i in range(numberofpositions - 1):
            ax.plot(translated_Xpositions[i:i+2], translated_Ypositions[i:i+2], 
                    color=colorgradientforthisrun[i], linewidth=line_width)
        
        ax.plot(translated_Xpositions[0], translated_Ypositions[0], 'go', markersize=3)
        
        if len(translated_Xpositions) >= q:
            dx = translated_Xpositions[-1] - translated_Xpositions[-q]
            dy = translated_Ypositions[-1] - translated_Ypositions[-q]
            
            norm_speed = np.hypot(dx, dy)
            if norm_speed != 0:
                dx /= norm_speed
                dy /= norm_speed
            
            ax.arrow(translated_Xpositions[-1], translated_Ypositions[-1], dx, dy,
                     head_width=1, head_length=1, width=arrow_width, fc='red', ec='red')

    # Supprime tous les axes et cadres
    ax.set_xlim(-ylim, ylim)
    ax.set_ylim(-xlim, xlim)
    ax.axis('off')

def plot_maze_towers_with_results(vertices, towers_coordinates, results):
    """
    Plots the edges of a tower or trapeze based on given vertices, 
    and adds behavior data as text inside the plotted shape.
    
    Parameters:
        vertices (list of tuples): List of (x, y) coordinates for the corners of the tower/trapeze.
        color (str): Color code for the plot (e.g., 'r' for red).
        results (dict): Dictionary containing behavioral data with keys 'rewarded_CW', 'total_CW', 
                        'rewarded_CCW', and 'total_CCW'.
        
    """
    for tower_name, vertices in towers_coordinates.items():
        # Fermer le polygone en ajoutant le premier sommet à la fin
        closed_vertices = vertices + [vertices[0]]
        closed_vertices = list(zip(*closed_vertices))  # Transpose vertices for plotting (x, y)
        
        # Calculer le centre du polygone pour afficher le texte au centre
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)

        # Extraire les résultats pour le tour spécifié
        if tower_name in results:
            tower_results = results[tower_name]
            # total_rewards[tower_name] = tower_results['rewarded_CCW'] + tower_results['rewarded_CW']
            # total_turns[tower_name] = tower_results['total_CCW'] + tower_results['total_CW']
            # Texte avec les données comportementales pour le tour
            behavior_text = (f"{tower_name}\n"
                                f"Rewarded/Tot\n"
                                f"CW: {tower_results['rewarded_CW']}/{tower_results['total_CW']}\n"
                                f"CCW: {tower_results['rewarded_CCW']}/{tower_results['total_CCW']}")
            
            # Ajouter le texte au centre du polygone
            plt.text(center_x, center_y, behavior_text, fontsize=8, ha='center', va='center', color='black')
        else:
            print(f"Warning: Tower name '{tower_name}' not found in results.")

def plot_run_trajectories(ax, trapezes_coordinates, run_type, traject_time, distances, X_positions_cm, Y_positions_cm, speeds, towers_coordinates, run_label='Undefined run label', q=4):
    """
    Plots run trajectories around towers, including towers and behavioral data.

    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib Axes on which to plot.
        run_label (str): Label describing the type of runs (e.g., "runs around towers").
        q (int): Minimum number of points to compute and display direction arrows.
        trapezes_coordinates (dict): Coordinates of trapezoidal regions for each tower.
        run_type (list of tuples): List of runs, where each run is represented as [(start, end), ...].
        traject_time (list): Time values corresponding to the trajectory points.
        distances (list): Distance values for each point in the trajectory.
        X_positions_cm (list): X-coordinates of positions in the trajectory.
        Y_positions_cm (list): Y-coordinates of positions in the trajectory.
        speeds (list): Speed values at each point in the trajectory.
        towers_coordinates (dict): Dictionary with tower names as keys and coordinates for each tower as values.
    """
    # Colors for plotting
    fill_colors = ['lightsteelblue'] * 4

    # Remove spines and ticks from the plot
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot each trapezoid region with colors and borders
    for i, (tower, trapezes) in enumerate(trapezes_coordinates.items()):
        for j, (trapeze, coordinates) in enumerate(trapezes.items()):
            coordinates_copy = coordinates + [coordinates[0]]
            x_coords, y_coords = zip(*coordinates_copy)
            ax.fill(x_coords, y_coords, color=fill_colors[j % len(fill_colors)], alpha=0.5)

    # Initialize counters and lists for run analysis
    total_time = 0
    total_distance = 0
    meanspeed = []
    maxspeed = []

    # Plot each run trajectory
    for run in run_type:
        start_index, end_index = run[0][0], run[0][1]
        run_X_position = X_positions_cm[start_index:end_index + 1]
        run_Y_position = Y_positions_cm[start_index:end_index + 1]
        run_duration = traject_time[end_index] - traject_time[start_index]
        distance_ran = np.sum(distances[start_index:end_index])

        # Accumulate metrics
        total_time += run_duration
        total_distance += distance_ran
        meanspeed.append(distance_ran / run_duration)
        maxspeed.append(np.max(speeds[start_index:end_index]))

        # Plot run trajectory with gradient color
        numberofpositions = len(run_X_position)
        colorgradientforthisrun = custom_cmap(numberofpositions)
        for k in range(numberofpositions - 1):
            ax.plot(run_X_position[k:k+2], run_Y_position[k:k+2], linestyle='-', 
                    color=colorgradientforthisrun[k], linewidth=0.5)

        # Plot start point of the run
        ax.plot(run_X_position[0], run_Y_position[0], 'go', markersize=3)

        # Plot arrow if the trajectory has enough points
        if len(run_X_position) >= q:
            dx = run_X_position[-1] - run_X_position[-q]
            dy = run_Y_position[-1] - run_Y_position[-q]
            norm = np.hypot(dx, dy)
            if norm != 0:
                dx /= norm
                dy /= norm
            ax.arrow(run_X_position[-1], run_Y_position[-1], dx, dy, 
                     head_width=0.8, head_length=1, fc='red', ec='red')

    # Plot towers with results if specified
    if run_label == 'runs around towers':
        plot_maze_towers_with_results(towers_coordinates, towers_coordinates, run_around_tower_results)

    # Set the main title with computed metrics
    text = (f"Distance: {total_distance:.2f} cm ; Duration: {total_time:.2f} s\n"
            f"Mean speed: {np.median(meanspeed):.2f} cm/s ; Max speed: {np.median(maxspeed):.2f} cm/s")
    # ax.set_title(f"Trajectory of {run_label}", fontsize=20)
    ax.set_xlabel(text, fontsize=17, labelpad=-10)


def custom_cmap(num_points):
    colors = [(0, 1, 0), (1, 0.5, 0), (1, 0, 0)] # Green to orange to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return [cmap(i / (num_points - 1)) for i in range(num_points)]

# Extract data and create figure ## CHANGES TO MAKE HERE

def extract_pickle_data(folder_path_mouse_to_process, session_to_process):

    output_pickle_filepath = f"{folder_path_mouse_to_process}/{session_to_process}/{session_to_process}_basic_processing_output.pickle"

    with open(output_pickle_filepath, 'rb') as f:
        data = pickle.load(f)

    for key, value in data.items():
        print(key)
    
    X_positions_cm = data['positions'][0]
    Y_positions_cm = data['positions'][1]
    average_speed = data['average_speed']
    distances = data['distances']
    speeds = data['speeds']
    angular_speeds = data['angular_speeds']
    all_epochs = data['all_epochs']
    traject_time = data['timeofframes']
    run_around_tower_results = data['run_around_tower_sessionresult']
    timeofframes = data['timeofframes']
    
    trapezes_coordinates = data['all_trapezes_coordinates_cm']
    towers_coordinates = data['towers_coordinates_cm']
    time_in_zones = data['time_in_zones']
    distance_in_zones = data['distance_in_zones']

    return X_positions_cm, Y_positions_cm, average_speed, distances, speeds, angular_speeds, all_epochs, traject_time, run_around_tower_results, timeofframes, trapezes_coordinates, towers_coordinates, time_in_zones, distance_in_zones



def generate_session_figure(fig, n_rows, n_cols):
    
    # Utilisation de GridSpec pour une grille flexible
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.2, hspace=0.2)

    # Line 1: full session analysis

    # Global trajectory and metrics (column 1)
    ax_trajectory = fig.add_subplot(gs[0, 0])
    plot_trajectory(ax_trajectory, X_positions_cm, Y_positions_cm, total_distance, average_speed)

    # Speed distribution (column 2)
    ax_speed_distribution = fig.add_subplot(gs[0, 1])
    plot_speed_distribution(ax_speed_distribution, speeds)

    # Angular speed distribution (column 3)
    ax_angular_speed_distribution = fig.add_subplot(gs[0, 2])
    plot_angular_speed_distribution(ax_angular_speed_distribution, angular_speeds)

    # Time spent in zones(column 4)
    ax_time = fig.add_subplot(gs[0, 3])
    plot_metrics_in_zones(ax_time, time_zones_data, metric_title='Time spent in zones', ylabel='Time (s)', ymax=800)

    # Distance in zones (column 5)
    ax_distance = fig.add_subplot(gs[0, 4])
    plot_metrics_in_zones(ax_distance, distance_zones_data, metric_title='Distance covered in zones', ylabel='Distance (cm)', ymax=10500)

    # Speed in zones (column 6)
    ax_speed = fig.add_subplot(gs[0, 5])
    plot_metrics_in_zones(ax_speed, speed_zones_data, metric_title='Global speed in zones', ylabel='Speed (cm/s)', ymax=None)

    # Ratios of time and distance in trapeze vs border (column 7)
    ax_ratios = fig.add_subplot(gs[0, 6])
    plot_ratios_in_zones(ax_ratios, ratios_trapeze_over_border, title_ratio='Ratios trapeze over border', xlabel='Trapeze/Border', ymax=None)


    # Line 2 : runs around tower analysis

    # Trajectory of all runs around tower (column 1)
    ax_traj_QT = fig.add_subplot(gs[1,0])
    plot_run_trajectories(ax_traj_QT, trapezes_coordinates, runs_around_tower, traject_time, distances, 
                            X_positions_cm, Y_positions_cm, speeds, towers_coordinates, run_label='runs around towers', q=4)

    # Trajectory and speeds of CW turns (columns 2 and 3)
    ax_CW_QT_trajectories = fig.add_subplot(gs[1,1])
    plot_trajectory_type_centered(ax_CW_QT_trajectories, X_positions_cm, Y_positions_cm, runs_around_tower_CW, xlim=30, ylim=30)

    ax_CW_QT_speeds = fig.add_subplot(gs[1,2])
    plot_speed_over_time(ax_CW_QT_speeds, traject_time, speeds, runs_around_tower_CW, xlim=2.1, set_title=False, title_type = 'CW runs around tower')

    # Trajectory and speeds of CCW turns (columns 4 and 5)
    ax_CCW_QT_trajectories = fig.add_subplot(gs[1,3])
    plot_trajectory_type_centered(ax_CCW_QT_trajectories, X_positions_cm, Y_positions_cm, runs_around_tower_CCW, xlim=30, ylim=30)

    ax_CCW_QT_speeds = fig.add_subplot(gs[1,4])
    plot_speed_over_time(ax_CCW_QT_speeds, traject_time, speeds, runs_around_tower_CCW, xlim=2.1, set_title=False, title_type = 'CCW runs around tower')

    # Cumulative number of CCW/CW (column 6)
    ax_cumul_CW_CCW = fig.add_subplot(gs[1,5])
    plot_cumulative_rewarded_unrewarded(ax_cumul_CW_CCW, runs_around_tower)

    # Cumulative number of rewarded/unrewarded (column 7)
    ax_cumul_rewarded_unrewarded = fig.add_subplot(gs[1,6])
    plot_cumulative_CW_CCW(ax_cumul_rewarded_unrewarded, runs_around_tower)


    # Line 3 : runs between towers analysis

    # Trajectory of all runs between towers (column 1)
    ax_traj_BT = fig.add_subplot(gs[2,0])
    plot_run_trajectories(ax_traj_BT, trapezes_coordinates, runs_between_towers, traject_time, distances, 
                            X_positions_cm, Y_positions_cm, speeds, towers_coordinates, run_label='runs between towers', q=4)

    # Centered trajectory (column 2)
    ax_trajectory = fig.add_subplot(gs[2, 1])
    plot_trajectory_type_centered(ax_trajectory, X_positions_cm, Y_positions_cm, runs_between_towers, xlim=90, ylim=90)

    # Speed profiles (columns 3 and 4)
    ax_speed = fig.add_subplot(gs[2, 2:4])
    plot_speed_over_time(ax_speed, traject_time, speeds, runs_between_towers, title_type='runs between towers', xlim=4.1)

    # Cumulative number of runs between towers (column 5)
    ax_cumul_nb_of_BT = fig.add_subplot(gs[2,4])
    plot_cumulative_runs(ax_cumul_nb_of_BT, runs_between_towers, legend_label='Runs between towers', color='orange', ymax=None)


    # Line 4 : exploratory runs analysis

    # Trajectory of all exploratory runs (column 1)
    ax_traj_ER = fig.add_subplot(gs[3,0])
    plot_run_trajectories(ax_traj_ER, trapezes_coordinates, exploratory_runs, traject_time, distances, 
                            X_positions_cm, Y_positions_cm, speeds, towers_coordinates, run_label='exploratory runs', q=4)

    # Centered trajectory (column 2)
    ax_trajectory = fig.add_subplot(gs[3, 1])
    plot_trajectory_type_centered(ax_trajectory, X_positions_cm, Y_positions_cm, exploratory_runs, xlim=90, ylim=90)

    # Speed profiles (columns 3 and 4)
    ax_speed = fig.add_subplot(gs[3, 2:4])
    plot_speed_over_time(ax_speed, traject_time, speeds, exploratory_runs, title_type='exploratory runs')

    # Cumulative number of exploratory runs (column 5)
    ax_cumul_nb_of_ER = fig.add_subplot(gs[3,4])
    plot_cumulative_runs(ax_cumul_nb_of_ER, exploratory_runs, legend_label='Exploratory runs', color='purple', ymax=None)


    # Handle texts and fig params

    fig.suptitle(f"Session {session_to_process} Analysis Overview", 
                 fontsize=40, 
                 fontweight='bold',
                 fontname='Nimbus Sans Narrow', 
                 y=0.96)
    fig.text(0.5, 0.92, f'Rewarding direction: ', ha='center', va='center', fontsize=30, fontstyle='italic', fontname='Ubuntu')
    fig.text(0.11, 0.8, 'Session metrics', ha='center', va='center', rotation=90, fontsize=20, fontweight='bold', fontname='Ubuntu')
    fig.text(0.11, 0.6, 'Runs around towers', ha='center', va='center', rotation=90, fontsize=20, fontweight='bold', fontname='Ubuntu')
    fig.text(0.11, 0.40, 'Runs between towers', ha='center', va='center', rotation=90, fontsize=20, fontweight='bold', fontname='Ubuntu')
    fig.text(0.11, 0.19, 'Exploratory runs', ha='center', va='center', rotation=90, fontsize=20, fontweight='bold', fontname='Ubuntu')

    # plt.subplots_adjust(hspace=5)

