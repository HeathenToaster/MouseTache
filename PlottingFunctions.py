import matplotlib.pyplot as plt #Creation de graphiques et de visualisations comme matlab
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)




def plot_trajectory_with_trapezes(smoothed_Xpositions, smoothed_Ypositions, all_trapezes_coordinates, 
                                  ax=None, xlim=(0, 512), ylim=(0, 512), title='Trajectory with Trapezes'):
    """
    Plots the trajectory of the mice along with the trapezes. Can plot on a provided axis or create its own.

    Parameters:
        smoothed_Xpositions (list or np.array): List or array of smoothed X positions of the mice.
        smoothed_Ypositions (list or np.array): List or array of smoothed Y positions of the mice.
        all_trapezes_coordinates (dict): Dictionary containing the coordinates of each trapeze, organized by tower.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): The axis on which to plot the trajectory and trapezes. 
                                                             If None, a new figure and axis are created.
        xlim (tuple): Limits for the x-axis. Default is (0, 512).
        ylim (tuple): Limits for the y-axis. Default is (0, 512).
        title (str): Title of the plot. Default is 'Trajectory with Trapezes'.
    """
    # Create a new figure and axis if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        axcreated=1
    else:
        axcreated=0

   

    # Plot the trajectory
    ax.plot(smoothed_Xpositions, smoothed_Ypositions, label='Trajectory', color='black')

    
    # Generate a color map using matplotlib's tab10 colormap (which has 10 distinct colors) to plot the trapezes
    colors = list(mcolors.TABLEAU_COLORS.values())  # Using Tableau colors for distinct colors
    
    # Plot each square and trapeze with the same color for each tower
    for i, (tower, trapezes) in enumerate(all_trapezes_coordinates.items()):
        tower_color = colors[i % len(colors)]  # Cycle through the colors if there are more than 10 towers
        for trapeze, coordinates in trapezes.items():
            # Make a copy of the coordinates and close the polygon by appending the first vertex
            coordinates_copy = coordinates + [coordinates[0]]
            # Extract x and y coordinates for plotting
            x_coords, y_coords = zip(*coordinates_copy)
            ax.plot(x_coords, y_coords, label=f'{tower}_{trapeze}', color=tower_color,linestyle='--')

    # Set plot limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')
    ax.set_title(title)
    #ax.grid(True)
    #ax.legend()

    # If no axis was provided, show the plot
    if axcreated:
        plt.show()
