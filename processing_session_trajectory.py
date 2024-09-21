# ## Overview
# 
# This files process the trajectory of a given session and generate otput saved in a pickle file
# 
import os
import copy

import itertools
import pickle
import numpy as np 
import pandas as pd
from matplotlib.path import Path
#from scipy.ndimage import gaussian_filter as smooth
from scipy.ndimage import gaussian_filter1d
import glob

from processing_TowerCoordinates import *



def load_data(mouseFolder_Path, session):

    trajectory_df, turns_df, param_df = None, None, None  # Initialize as None


    try:
        # Gets the parameters of the session
        param_df = pd.read_csv(mouseFolder_Path + os.sep + session + os.sep + session + "_sessionparam.csv")
    except FileNotFoundError:
        print("File sessionparam not found")

    try:
        #Gets the positional informations and filter the dataframe to keep only the relevant informations
        csvCentroid_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_centroidTXY.csv'
        trajectory_df = pd.read_csv(csvCentroid_fullpath) #Transforms CSV file into panda dataframe
        trajectory_df = trajectory_df.dropna() #Deletes lines with one or more NA
        trajectory_df = trajectory_df.loc[trajectory_df['time'] > 15] #During the first seconds of the video, as the background substraction is still building up, 
        #                                           #the tracking is innacruate so we don't analyze postions during the first 15 seconds
        trajectory_df = trajectory_df[trajectory_df['xposition'].between(1, 500) & trajectory_df['yposition'].between(1, 500)] #The pixel values between 1 and 500 are kept)
    except FileNotFoundError:
        print("File centroidTXY not found")

    try:
        # Get the information on the turns
        csvTurnsinfo_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_turnsinfo.csv'  # get the information on the turns in the dataframe turns_df
        turns_df = pd.read_csv(csvTurnsinfo_fullpath)  # Transforms CSV file into panda dataframe
        for i in range(turns_df.index.values[-1]):  # if there is a missing value for ongoingRewardedObject, replace it with either SW or SE, as long as it's not the one where the mouse is
            if type(turns_df['ongoingRewardedObject'][i]) == float:
                turns_df.iat[i, 8] = str([turns_df.iat[i, 4]])
        turns_df = turns_df.loc[turns_df['time'] > 15]  # same as above #TODO someone you shoud spend some time on the aquisition code to have a pre-loaded background and not loose the beginning
    except FileNotFoundError:
        print("File turnsinfo not found")

    return trajectory_df, turns_df, param_df


# ### from CSV trajectory file, get smooth  position trajectory in cm, with 0,0 lower left coordinates and the time of each frame

def get_positions_and_times(trajectory_df):
    # Video and arena dimensions
    video_dimension_pixels = (512, 512)
    arena_width_cm = 84
    arena_width_pixels = 453

    # Conversion factor from pixels to cm
    conversion_factor = arena_width_cm / arena_width_pixels

    # Smoothing parameter
    smooth_sigma = 1

    # Extract time and positions from the DataFrame
    time_video_frames = trajectory_df['time'].to_numpy()
    xpositions = trajectory_df['xposition'].to_numpy()
    ypositions = trajectory_df['yposition'].to_numpy()

    # Correct for OpenCV flipping in y-positions
    ypositions = video_dimension_pixels[1] - ypositions

    # Smooth positions
    smoothed_Xpositions = gaussian_filter1d(xpositions, sigma=smooth_sigma)
    smoothed_Ypositions = gaussian_filter1d(ypositions, sigma=smooth_sigma)

    # Convert positions from pixels to cm
    smoothed_Xpositions_cm = smoothed_Xpositions * conversion_factor
    smoothed_Ypositions_cm = smoothed_Ypositions * conversion_factor

    # Combine smoothed positions in cm
    smoothed_positions_cm = [smoothed_Xpositions_cm, smoothed_Ypositions_cm]

    return time_video_frames, smoothed_positions_cm


# ### Function to compute distance, speed, and angular speed in degrees per second. 
# #### We only compute angular speed when mice is moving above a certain speed threshold in cm/s

def compute_distance_speed_angular_speed(smoothed_positions_cm, time_video_frames, speed_threshold=5):
    # Calculate the differences between consecutive points
    delta_x = np.diff(smoothed_positions_cm[0])
    delta_y = np.diff(smoothed_positions_cm[1])
    delta_t = np.diff(time_video_frames)
    
    # Compute the distances traveled between each timepoint
    distances = np.sqrt(delta_x**2 + delta_y**2)
    
    speeds = distances / delta_t
    smooth_sigma = 1 #the sigma used for the remaining of the analysis for smoothing
    speeds = gaussian_filter1d(speeds, sigma=smooth_sigma)
     
    # Compute the angles between each timepoint
    angles = np.arctan2(delta_y, delta_x)
    
    # Compute the differences between consecutive angles
    delta_angles = np.diff(angles)
    
    # Convert delta_angles from radians to degrees
    delta_angles_deg = np.degrees(delta_angles)
    
    # Ensure angles are within -180 to 180 range
    delta_angles_deg = (delta_angles_deg + 180) % 360 - 180
    
    # Mask speeds below the threshold
    valid_mask = speeds > speed_threshold
    
    # Compute angular speeds in degrees per second
    angular_speeds = np.zeros_like(delta_angles_deg)
    valid_delta_t = delta_t[1:][valid_mask[1:]]
    angular_speeds[valid_mask[1:]] = delta_angles_deg[valid_mask[1:]] / valid_delta_t

    # Filter angular speeds to include only those above the threshold
    filtered_angular_speeds = angular_speeds[valid_mask[1:]]

    distances= np.insert(distances, 0, 0) # insert a 0 to avoid length error with  time_video_frames. We consider that
                                            # at the first frame the distance is null
    speeds = np.insert(speeds, 0, 0) # insert a 0 to avoid length error with time_video_frames. We consider that
                                            # at the first frame the speed is null

    
    return distances, speeds, filtered_angular_speeds



# ### Function to identify continuous run epochs

def detect_run_epochs(speeds, time_video_frames):
    """
    
    Identifies continuous epochs during which the mouse is moving above a certain speed (cut_off_speed).
    A minimal duration of low speed is necessary to be considered as the end of a run.
    Similarly, a minimal duration of high speed is necessary to be considered as a run.
    """
    
    #for this we need some parameters to cut the trajectory into run based on speed, duration of runs and pauses
    pause_min_duration = 0.1 #if a stop is shorter than this, merges the two epochs bordering it
    run_min_duration = 0.3 #minimal duration of an epoch to be considerd
    cut_off_speed = 7 # this value is the speed in cm/s. It is used to detect when the animals stop running. 
    
    
    
    run_epochs = []
    is_in_epoch = False  # Flag to track if we are currently in a running epoch
    epoch_start_index = 0

    if len(speeds) != len(time_video_frames):
        raise ValueError("speeds and time_video_frames have different lengths")

    for i in range(len(speeds)):
        if speeds[i] >= cut_off_speed:  # Speed above cut-off value
            if not is_in_epoch: # if the previous trajectory speed was not part of running epoch then this will be a start of a new epoch
                epoch_start_index = i  # Mark the beginning of a new epoch
                is_in_epoch = True
        else: # the speed of the current data point is below the treshold
            if is_in_epoch: # if we were in a run epoch just before (1st point below the treshold)
                # Check first if the pause between this epoch's starting point (time_video_frames[epoch_start_index]) and  
                # the previous epoch' last point time_video_frames[run_epochs[-1][1]] is shorter than the minimal time for a pause
                # then the previous epoch  should be extended to the previous data point.  
                if run_epochs and (time_video_frames[epoch_start_index] - time_video_frames[run_epochs[-1][1]] < pause_min_duration):
                    run_epochs[-1][1] = i - 1  # Extend the previous epoch
                else: # the pause has been long enough then we terminate the run epoch  other previous 
                    run_epochs.append([epoch_start_index, i - 1])  # Add new epoch
                is_in_epoch = False

    # Final check for any epoch still in progress
    if is_in_epoch:
        if run_epochs and (time_video_frames[epoch_start_index] - time_video_frames[run_epochs[-1][1]] < pause_min_duration):
            run_epochs[-1][1] = len(speeds) - 1
        elif (time_video_frames[-1] - time_video_frames[epoch_start_index]) >= run_min_duration:
            run_epochs.append([epoch_start_index, len(speeds) - 1])

    # Remove epochs that are too short
    run_epochs = [epoch for epoch in run_epochs if (time_video_frames[epoch[1]] - time_video_frames[epoch[0]]) >= run_min_duration]
    
    # Adjust the start and end of each epoch based on acceleration. The idea is that with the threshold method we miss the beginning and enf of the run
    # for the starting point. We are going back and find the point at wich the animal acceleration is less than 40% 
    # than the acceleration at the moment at which he passed the treshold. 
    clean_run_epochs = [None] * len(run_epochs)
    for index,epoch in enumerate(run_epochs):
        clean_run_epochs[index] = epoch.copy()
        epoch_start, epoch_end = epoch[0], epoch[1]
        # Adjust the start of the epoch
        current_point = epoch_start
        acceleration_at_crossing=(speeds[current_point + 1] - speeds[current_point]) / (time_video_frames[current_point + 1] - time_video_frames[current_point])
        while current_point > 0:
            previous_acceleration = (speeds[current_point] - speeds[current_point - 1]) / (time_video_frames[current_point] - time_video_frames[current_point - 1])
            if previous_acceleration <= (0.1 * acceleration_at_crossing) or previous_acceleration <= 0:
                break
            current_point -= 1
            #print(f'it went backward on epoch {index}')
        clean_run_epochs[index][0] = current_point

    #Adjust the end of the epoch
    #We are going forward after the speed crossed downward the speed threshold and find the point at wich the animal acceleration is less than 40% 
    #than the acceleration at the moment at which it passed the treshold. 
        current_point = epoch_end
        acceleration_at_crossing=(speeds[current_point - 1] - speeds[current_point]) / (time_video_frames[current_point] - time_video_frames[current_point-1])
        while current_point < len(speeds) - 1:
            next_acceleration = (speeds[current_point] - speeds[current_point + 1]) / (time_video_frames[current_point+1] - time_video_frames[current_point])
            if next_acceleration <= (0.1 * acceleration_at_crossing) or next_acceleration <= 0:
                break
            current_point += 1
            #print(f'it went forward on epoch {index}')
        clean_run_epochs[index][1] = current_point

    
   
    return clean_run_epochs 


# ### Define the type of run epochs 
# #### Redefine run epochs depending on their start and end positions relative to the trapezes surrounding towers

# we need a function to detect if a position is in a polygon 

def is_point_in_polygon(polygon_vertices, point): # function to replace the not so efficient one points_in_polygon written originally (not by Alice :)
    """
    Determine if a point is inside or outside a polygon.

    Args:
    - polygon_vertices: Coordinates of the polygon vertices [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]]
    - point: Coordinates of the point to check [x, y]

    Returns:
    - bool: True if the point is inside the polygon, False otherwise
    """
    path = Path(polygon_vertices)
    return path.contains_point(point)


# this check if a given position (run start or stop) is in a given trapze of a given tower
# it returns true and false and if true which tower and trapze
def check_position_in_trapezes(position, all_trapezes_coordinates):
    """
    Check if the position is inside any of the trapezes.
    :param position: Tuple (x, y) representing the position to check.
    :param all_trapezes_coordinates: Dictionary containing trapezes coordinates.
    :return: List [True, towerlabel, trapezelabel] if inside a trapeze, [False, 'none', 'none'] otherwise.
    """
    for towerlabel, trapezes in all_trapezes_coordinates.items():
        #print(towerlabel)
        for trapezelabel, trapeze_coordinates in trapezes.items():
            #print(trapezelabel)
            #print(trapeze_coordinates)
            if is_point_in_polygon(trapeze_coordinates,position):
                return [True, towerlabel, trapezelabel]
    return [False, None, None]



def define_epoch_types(clean_run_epochs, smoothed_positions_cm, time_video_frames,turns_df, all_trapezes_coordinates_cm,):
    # Initialize list to store run epochs with additional information
    run_epochs_start_stop_Tower_Trapeze = []

    # Iterate over each run epoch
    for epoch_index, run_epoch in enumerate(clean_run_epochs):
        run_epoch_start_stop_Tower_Trapeze = []
        start_index, end_index = run_epoch[0], run_epoch[1]
        run_epoch_start_stop_Tower_Trapeze.append(run_epoch)

        # Get the starting and ending positions
        starting_position = [smoothed_positions_cm[0][start_index], smoothed_positions_cm[1][start_index]]
        ending_position = [smoothed_positions_cm[0][end_index], smoothed_positions_cm[1][end_index]]

        # Check the starting and ending positions relative to trapeze and tower
        for position_to_check in [starting_position, ending_position]:
            in_trapeze_info = check_position_in_trapezes(position_to_check, all_trapezes_coordinates_cm)
            run_epoch_start_stop_Tower_Trapeze.append(in_trapeze_info[1:])  # Append trapeze/tower start/stop info

        run_epochs_start_stop_Tower_Trapeze.append(run_epoch_start_stop_Tower_Trapeze)

    # Identify immobility epochs
    immobility_epochs = []
    for i in range(len(clean_run_epochs) - 1):
        current_epoch_end = clean_run_epochs[i][1]
        next_epoch_start = clean_run_epochs[i + 1][0]
        
        if current_epoch_end < next_epoch_start:
            immobility_epochs.append([current_epoch_end, next_epoch_start])

    # Initialize the all_epochs dictionary
    all_epochs = {
        'run_around_tower': [],
        'run_between_towers': [],
        'run_toward_tower': [],
        'exploratory_run': [],
        'immobility': []
    }

    # Classify each run epoch into different types
    for run_epoch_start_stop_Tower_Trapeze in run_epochs_start_stop_Tower_Trapeze:
        # Exploratory run if the end is not in a trapeze
        if run_epoch_start_stop_Tower_Trapeze[2][0] is None:
            all_epochs['exploratory_run'].append(run_epoch_start_stop_Tower_Trapeze)
        # Run toward a tower if the start is outside a trapeze
        elif run_epoch_start_stop_Tower_Trapeze[1][0] is None:
            all_epochs['run_toward_tower'].append(run_epoch_start_stop_Tower_Trapeze)
        # Run between towers if start and stop are in trapezes of different towers
        elif run_epoch_start_stop_Tower_Trapeze[1][0] != run_epoch_start_stop_Tower_Trapeze[2][0]:
            all_epochs['run_between_towers'].append(run_epoch_start_stop_Tower_Trapeze)
        else:
            # Check if the animal switched trapeze at least once
            start_stop_times_run_epoch = [time_video_frames[run_epoch_start_stop_Tower_Trapeze[0][0]], time_video_frames[run_epoch_start_stop_Tower_Trapeze[0][1]]]
            switch_in_turns_df = turns_df[(turns_df['time'] >= start_stop_times_run_epoch[0]) & (turns_df['time'] <= start_stop_times_run_epoch[1])]
            num_trapeze_switches = switch_in_turns_df.shape[0]

            # Run around the same tower if trapeze switching occurred
            if num_trapeze_switches > 0:
                all_epochs['run_around_tower'].append(run_epoch_start_stop_Tower_Trapeze)

    # Add immobility epochs
    all_epochs['immobility'] = immobility_epochs

    return all_epochs


# ### Runs around tower basic quantification
# #### Find turns around tower, check if they were rewarded and  clockwise or counterclowise by using the turninfo dataframe, get some basic kinematics info

def process_run_around_tower_epochs(all_epochs, time_video_frames, turns_df, distances, speeds):
    # Create a deep copy of the list of runs around tower
    runs_around_tower = copy.deepcopy(all_epochs['run_around_tower'])

    # Iterate over each run in the 'run_around_tower' category
    for run_index, run_around_tower in enumerate(runs_around_tower):
        run_start_index = run_around_tower[0][0]
        run_end_index = run_around_tower[0][1]
        start_stop_times_run_epoch = [time_video_frames[run_start_index], time_video_frames[run_end_index]]
        
        # Find the relevant entries in turns_df within the run epoch
        condition = (turns_df['time'] >= start_stop_times_run_epoch[0]) & (turns_df['time'] <= start_stop_times_run_epoch[1])
        if not condition.any():
            continue
        
        switch_in_turns_df = turns_df[condition]
        num_trapezeswitch = switch_in_turns_df.shape[0]
        
        # Initialize the type_of_turn dictionary
        type_of_turn = {'Rewarded': '', 'direction': '', 'num_trapezeswitch': num_trapezeswitch}
        
        # Normalize 'Rewarded' to boolean
        rewarded_value = switch_in_turns_df.iloc[0]['Rewarded']
        if rewarded_value in {'1', 'True', True}:
            type_of_turn['Rewarded'] = True
        elif rewarded_value in {'0', 'False', False}:
            type_of_turn['Rewarded'] = False
        else:
            raise ValueError(f"Unexpected value for Rewarded: {rewarded_value}")
        
        # Determine turn direction
        if switch_in_turns_df.iloc[0]['turnDirection'] == 270:
            type_of_turn['direction'] = 'CW'
        else:
            type_of_turn['direction'] = 'CCW'
        
        # Append the type_of_turn information to the run
        runs_around_tower[run_index].append(type_of_turn)
        
        # Extract the run epoch, compute kinematics
        kinematics = {
            'epoch_time': '',
            'epoch_duration': '',
            'epoch_distance': '',
            'epoch_meanspeed': '',
            'epoch_maxspeed': ''
        }
        
        # Compute kinematic values
        kinematics['epoch_time'] = time_video_frames[run_start_index]
        kinematics['epoch_duration'] = start_stop_times_run_epoch[1] - start_stop_times_run_epoch[0]
        kinematics['epoch_distance'] = np.sum(distances[run_start_index:run_end_index])
        kinematics['epoch_meanspeed'] = kinematics['epoch_distance'] / kinematics['epoch_duration']
        kinematics['epoch_maxspeed'] = np.max(speeds[run_start_index:run_end_index])
        
        # Append the kinematics information to the run
        runs_around_tower[run_index].append(kinematics)

    # Update the 'run_around_tower' epochs in all_epochs
    all_epochs['run_around_tower'] = runs_around_tower
    
    return all_epochs


# ### Compute the total and rewarded number of clockwise, coutnerclockwise turns per object for the session
# 

def get_run_around_tower_resultssessions(all_epochs):
    runs_around_tower = all_epochs['run_around_tower']
    
    # Initialize the dictionary to hold the results for each tower
    run_around_tower_sessionresult = {
        'NE': {'total_CW': 0, 'total_CCW': 0, 'rewarded_CW': 0, 'rewarded_CCW': 0},
        'NW': {'total_CW': 0, 'total_CCW': 0, 'rewarded_CW': 0, 'rewarded_CCW': 0},
        'SE': {'total_CW': 0, 'total_CCW': 0, 'rewarded_CW': 0, 'rewarded_CCW': 0},
        'SW': {'total_CW': 0, 'total_CCW': 0, 'rewarded_CW': 0, 'rewarded_CCW': 0},
        'All': {'total_CW': 0, 'total_CCW': 0, 'rewarded_CW': 0, 'rewarded_CCW': 0}
    }

    # Process each run in the data
    for run in runs_around_tower:
        _, start_info, end_info, type_of_turn, kinematics_of_turn = run
        tower = start_info[0]
        direction = type_of_turn['direction']
        rewarded = type_of_turn['Rewarded']
        
        # Update counts based on direction
        if direction == 'CW':
            run_around_tower_sessionresult[tower]['total_CW'] += 1
            run_around_tower_sessionresult['All']['total_CW'] += 1
            if rewarded:
                run_around_tower_sessionresult[tower]['rewarded_CW'] += 1
                run_around_tower_sessionresult['All']['rewarded_CW'] += 1
        elif direction == 'CCW':
            run_around_tower_sessionresult[tower]['total_CCW'] += 1
            run_around_tower_sessionresult['All']['total_CCW'] += 1
            if rewarded:
                run_around_tower_sessionresult[tower]['rewarded_CCW'] += 1
                run_around_tower_sessionresult['All']['rewarded_CCW'] += 1

    return run_around_tower_sessionresult


# ### Get basic  kinematics info for the other type of epochs (othr than runs around tower)

def process_other_epochs(all_epochs, time_video_frames, smoothed_positions_cm, all_trapezes_coordinates_cm, distances, speeds):
    """
    Process the other types of epochs ('run_between_towers', 'run_toward_tower', 'exploratory_run', 'immobility')
    and compute kinematics for each epoch.
    """

    epoch_types=['run_around_tower','run_between_towers','run_toward_tower','exploratory_run','immobility']
    for epoch_type in epoch_types[1:]:  # We skip 'run_around_tower' as it's already processed
        print(f"Processing {epoch_type} epochs...")
        epochs_to_analyze = copy.deepcopy(all_epochs[epoch_type])
        
        if epoch_type == 'immobility':
            for epoch_index, epoch_to_analyze in enumerate(epochs_to_analyze):
                epoch_start_index = epoch_to_analyze[0]
                epoch_end_index = epoch_to_analyze[1]

                # Compute kinematics for immobility epoch
                kinematics = {
                    'time': time_video_frames[epoch_start_index],
                    'duration': time_video_frames[epoch_end_index] - time_video_frames[epoch_start_index],
                    'position': [smoothed_positions_cm[0][epoch_start_index], smoothed_positions_cm[1][epoch_start_index]],
                    'in_trapeze': check_position_in_trapezes(
                        [smoothed_positions_cm[0][epoch_start_index], smoothed_positions_cm[1][epoch_start_index]], 
                        all_trapezes_coordinates_cm
                    )[0]
                }
                epochs_to_analyze[epoch_index].append(kinematics)
        
        else:  # For other epoch types (run_between_towers, run_toward_tower, exploratory_run)
            for epoch_index, epoch_to_analyze in enumerate(epochs_to_analyze):
                epoch_start_index = epoch_to_analyze[0][0]
                epoch_end_index = epoch_to_analyze[0][1]

                # Compute kinematics for running epoch
                kinematics = {
                    'time': time_video_frames[epoch_start_index],
                    'duration': time_video_frames[epoch_end_index] - time_video_frames[epoch_start_index],
                    'distance': np.sum(distances[epoch_start_index:epoch_end_index]),
                    'meanspeed': np.sum(distances[epoch_start_index:epoch_end_index]) / 
                                 (time_video_frames[epoch_end_index] - time_video_frames[epoch_start_index]),
                    'maxspeed': np.max(speeds[epoch_start_index:epoch_end_index])
                }
                epochs_to_analyze[epoch_index].append(kinematics)

        all_epochs[epoch_type] = epochs_to_analyze
        

    return all_epochs

def process_trajectory(folder_path_mouse_to_process,session_to_process,all_trapezes_coordinates_cm):

    # we load the trajectory , turn info and parametres info from the csv files generated by the acqusition software
    trajectory_df, turns_df, param_df=load_data(folder_path_mouse_to_process,session_to_process)


    # Check if any of the required data is missing, and return session name if so
    if trajectory_df is None or turns_df is None or param_df is None:
        print(f"Missing data for session {session_to_process}, skipping processing.")
        return session_to_process  # Return the session name with missing data
    
    


    # get times and smoothed position in cm
    time_video_frames, smoothed_positions_cm=get_positions_and_times(trajectory_df)

    # Compute instantaneous distances, speeds, and angular speeds
    distances, speeds, angular_speeds = compute_distance_speed_angular_speed(smoothed_positions_cm, time_video_frames)

    # Calclulate session duration
    session_duration = time_video_frames[-1] - time_video_frames[0]
    # Print the total time
    print(f"Total time: {session_duration:.2f} s.")
    # Calculate total distance in m
    distance_ran = np.sum(distances)/100  # Convert cm to m
    print(f"The total distance is: {distance_ran:.2f} m")

    # Calculate the average running speed in meters per second (m/s)
    average_speed = distance_ran*100 / session_duration

    # Print the average speed
    print(f"The average running speed is: {average_speed:.2f} cm/s")

    clean_run_epochs = detect_run_epochs(speeds, time_video_frames)

    # trapeze_width, towers_coordinates = get_trapeze_and_tower_data(folder_path_mouse_to_process, session_to_process)

    # all_trapezes_coordinates_cm, towers_coordinates_cm= generate_trapeze_and_tower_coordinates(towers_coordinates, trapeze_width)
    # #print(all_trapezes_coordinates_cm)

    all_epochs=define_epoch_types(clean_run_epochs, smoothed_positions_cm, time_video_frames, turns_df, all_trapezes_coordinates_cm)

    all_epochs = process_run_around_tower_epochs(all_epochs, time_video_frames, turns_df, distances, speeds)

    run_around_tower_sessionresult=get_run_around_tower_resultssessions(all_epochs)

    all_epochs=process_other_epochs(all_epochs, time_video_frames, smoothed_positions_cm, all_trapezes_coordinates_cm, distances, speeds)


    #save 
    # Save important output in a pickle 
    session_name = session_to_process
    output_pickle_filename = f"{session_name}_basic_processing_output.pickle"
    output_pickle_filepath = os.path.join(folder_path_mouse_to_process,session_to_process,output_pickle_filename)

    # Collect all variables into a dictionary
    session_data = {
        'timeofframes':time_video_frames,
        'positions':smoothed_positions_cm,
        'distances': distances,
        'speeds': speeds,
        'angular_speeds': angular_speeds,
        'distance_ran': distance_ran,
        'average_speed': average_speed,
        'all_epochs': all_epochs,
        'run_around_tower_sessionresult': run_around_tower_sessionresult,
        'all_trapezes_coordinates_cm': all_trapezes_coordinates_cm
    }

    # Save the dictionary to a pickle file
    with open(output_pickle_filepath, 'wb') as file:
        pickle.dump(session_data, file)

    print(f"Session processing results saved to {output_pickle_filepath}")

    return None  # Return None to indicate successful processing




