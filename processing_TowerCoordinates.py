#This code generate tower and trapze coordinate from default or parameter file

import os
import pandas as pd
import ast

######################################
# maze information
######################################

# Mice have to run around 4 towers to obtain rewards. Rewards are delivered when a mouse switch from one trapeze to another concurrent trapeze



#        +-----------------+                            +-----------------+
#        | \  Trapeze N  / |                            | \  Trapeze N  / |
#        |  +-----------+  |                            |  +-----------+  |
#        |  |           |  |                            |  |           |  |
#        |TW|   Tower   |TE|                            |TW|   Tower   |TE|
#        |  |     NW    |  |                            |  |     NE    |  |
#        |  |           |  |                            |  |           |  |
#        |  +-----------+  |                            |  +-----------+  |
#        | /    Trap S   \ |                            | /    Trap S   \ |
#        +-----------------+                            +-----------------+





#        +-----------------+                            +-----------------+
#        | \  Trapeze N  / |                            | \  Trapeze N  / |
#        |  +-----------+  |                            |  +-----------+  |
#        |  |           |  |                            |  |           |  |
#        |TW|   Tower   |TE|                            |TW|   Tower   |TE|
#        |  |     SW    |  |                            |  |     SE    |  |
#        |  |           |  |                            |  |           |  |
#        |  +-----------+  |                            |  +-----------+  |
#        | /    Trap S   \ |                            | /    Trap S   \ |
#        +-----------------+                            +-----------------+


# We therefore need  the coordinate of the 4 towers and the trapeze. The coordinate are given in the reference frame of the video. 
# The video has a dimension of 512 pixeld in width and length trapeze_width and in height. The point 0,0 is the left top corner (openCV)and will need to be changed to left bottom for matplotlib plotting 


def get_trapeze_and_tower_data(folder_path_mouse_to_process, session_to_process):

    """
    Function to extract trapeze width and tower coordinates from a session parameter CSV file.
    
    Parameters:
        folder_path_mouse_to_process (str): The folder path where the mouse data is stored.
        session_to_process (str): The specific session to process.

    Returns:
        trapeze_width (int or float): The width of the trapeze.
        towers_coordinates (dict): The coordinates of the towers.
    """
    # Load the session parameters CSV file
    param_file_path = os.path.join(folder_path_mouse_to_process, session_to_process, f"{session_to_process}_sessionparam.csv")
    param_df = pd.read_csv(param_file_path)

    # Check if the towers coordinates exist in the CSV file
    if "SE_coords" in param_df.columns:
        towers_coordinates = {
            "NW": param_df["NW_coords"].values[0],
            "NE": param_df["NE_coords"].values[0],
            "SW": param_df["SW_coords"].values[0],
            "SE": param_df["SE_coords"].values[0]
        }
        
        # Convert string representations of lists into actual lists
        towers_coordinates = {key: ast.literal_eval(value) for key, value in towers_coordinates.items()}

        #print('Coordinates from parameter file:')
    else:
        # Default tower coordinates
        towers_coordinates = {
            "NW": [[104, 125], [173, 125], [173, 201], [104, 201]],
            "NE": [[330, 120], [400, 120], [400, 200], [330, 200]],
            "SW": [[109, 351], [181, 351], [181, 410], [109, 410]],
            "SE": [[330, 350], [400, 350], [400, 410], [330, 410]]
        }
        #print('Default towers_coordinates')
    #print(towers_coordinates)

    # Check if the trapeze size exists in the CSV file
    if "TrapezeSize" in param_df.columns:
        trapeze_width = param_df["TrapezeSize"].values[0]
        #print('Trapeze width from parameter file:')
    else:
        trapeze_width = 50  # Default trapeze width
        #print('Default trapeze width')
    #print(trapeze_width)

    return trapeze_width, towers_coordinates


def generate_trapeze_and_tower_coordinates(towers_coordinates, trapeze_width):
    """
    Generates the coordinates of trapezes surrounding towers and converts all coordinates from pixels to centimeters.
    
    Parameters:
    towers_coordinates (dict): Dictionary containing the pixel coordinates of the 4 towers.
    trapeze_width (int): The width of the trapeze in pixels.
    
    
    Returns:
    tuple: 
        - all_trapezes_coordinates_cm (dict): Coordinates of the trapezes in cm.
        - towers_coordinates_cm (dict): Coordinates of the towers in cm.
    """
    
    #video_dimension_pixels (tuple): The resolution of the video in pixels 
    #arena_width_cm (float): The width of the arena in centimeters 
    #arena_width_pixels (int): The width of the arena in pixels 
    video_dimension_pixels=(512, 512)
    arena_width_cm=84
    arena_width_pixels=453



    # Conversion factor to go from pixel to cm
    conversion_factor = arena_width_cm / arena_width_pixels
    
    # Function to convert pixel coordinates to cm
    def convert_pix_to_cm(coordinate):
        return [round(coordinate[0] * conversion_factor, 2), round(coordinate[1] * conversion_factor, 2)]
    
    # Transform the coordinates to have the origin at the lower left (for plotting)
    max_y = video_dimension_pixels[1]
    transformed_towers_coordinates = {
        label: [[x, max_y - y] for x, y in vertices]
        for label, vertices in towers_coordinates.items()
    }
    
    # Function to generate trapeze coordinates surrounding a tower
    def trapeze_coordinates_from_tower(tower_coordinates, trapeze_width):
        trapeze_N = [
            tower_coordinates[0], tower_coordinates[1],
            [tower_coordinates[1][0] + trapeze_width, tower_coordinates[1][1] + trapeze_width],
            [tower_coordinates[0][0] - trapeze_width, tower_coordinates[0][1] + trapeze_width]
        ]
        trapeze_E = [
            tower_coordinates[1], tower_coordinates[2],
            [tower_coordinates[2][0] + trapeze_width, tower_coordinates[2][1] - trapeze_width],
            [tower_coordinates[1][0] + trapeze_width, tower_coordinates[1][1] + trapeze_width]
        ]
        trapeze_S = [
            tower_coordinates[2], tower_coordinates[3],
            [tower_coordinates[3][0] - trapeze_width, tower_coordinates[3][1] - trapeze_width],
            [tower_coordinates[2][0] + trapeze_width, tower_coordinates[2][1] - trapeze_width]
        ]
        trapeze_W = [
            tower_coordinates[3], tower_coordinates[0],
            [tower_coordinates[0][0] - trapeze_width, tower_coordinates[0][1] + trapeze_width],
            [tower_coordinates[3][0] - trapeze_width, tower_coordinates[3][1] - trapeze_width]
        ]
        return trapeze_N, trapeze_E, trapeze_S, trapeze_W
    
    # Initialize dictionaries to store trapeze and tower coordinates in cm
    all_trapezes_coordinates = {key: {} for key in towers_coordinates}
    all_trapezes_coordinates_cm = {}
    
    # Generate trapeze coordinates for each tower
    for tower_name, tower_coordinates in transformed_towers_coordinates.items():
        all_trapezes_coordinates[tower_name]["N"], \
        all_trapezes_coordinates[tower_name]["E"], \
        all_trapezes_coordinates[tower_name]["S"], \
        all_trapezes_coordinates[tower_name]["W"] = trapeze_coordinates_from_tower(tower_coordinates, trapeze_width)

    # Convert all trapeze coordinates from pixel to cm
    for tower, trapezes in all_trapezes_coordinates.items():
        all_trapezes_coordinates_cm[tower] = {
            trapeze: [convert_pix_to_cm(coord) for coord in coords]
            for trapeze, coords in trapezes.items()
        }
    
    # Convert tower coordinates from pixel to cm
    towers_coordinates_cm = {
        key: [convert_pix_to_cm(coord) for coord in transformed_towers_coordinates[key]]
    for key in transformed_towers_coordinates}

    return all_trapezes_coordinates_cm, towers_coordinates_cm