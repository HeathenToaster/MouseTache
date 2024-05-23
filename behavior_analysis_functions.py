# pylint: disable=no-member

"""
This file contains the functions used to analyze mice behavior in the patchouris.
Most of the code has been written by an intern that worked on the project.
Therefore, the code is not well documented, weirdly structured, weirdly written,
and not tested. The code is also not very efficient and could be optimized. There
are also some bad practices in the code that should be fixed.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as smooth
from matplotlib.patches import Polygon
from typing import Any
plt.style.use('./paper.mplstyle')

######################################
# maze utils
######################################
MOUSE_STRING = '---{,_,">'
RESOLUTION = 512, 512
TRAPEZE_SIZE = 35 #how much to extend outside of the objects to detect the trajectories
#Dictionary to set-up the color palette (in HEX) for the graphs and HTML (turns and phases colors)
turnPalette= {'gogd':'#86d065', #green
              'gobd':'#f1c232', #yellow
              'bogd':'#e69138', #orange
              'bobd':'#cc0000', #red
              'gogdet':'#7754e0', #purple 
              'gobdet':'#ef2388', #pink
              'all turns':'#49a1f1'} #blue

phasePalette= {'Phase1':'#b4a7d6', #purple
               'Phase2':'#94ccff', #blue
               'Phase3':'#84d567', #green
               'Phase4':'#f5b779', #orange
               'Other':'#fbabd2'} #pink

def trapezes_from_patch(patch, width):
    """
    generate the trapezes coordinates surrounding a patch
    inputs:
    patch - coordinates of a patch [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]]
    width - width of the trapeze in pixels
    outputs:
    coordinates [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]] for the 4 trapezes.
    
    trapezes_from_patch(SWpatch_coords, 200)
    """

    N = [patch[0], patch[1], [patch[1][0]+width, patch[1][1]+width], [patch[0][0]-width, patch[0][1]+width]]
    E = [patch[1], patch[2], [patch[2][0]+width, patch[2][1]-width], [patch[1][0]+width, patch[1][1]+width]]
    S = [patch[2], patch[3], [patch[3][0]-width, patch[3][1]-width], [patch[2][0]+width, patch[2][1]-width]]
    W = [patch[3], patch[0], [patch[0][0]-width, patch[0][1]+width], [patch[3][0]-width, patch[3][1]-width]]
    return N, E, S, W

def points_in_polygon(polygon, pts):
    """
    inputs:
    polygon - coordinates of the trapeze [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]] or N/E/S/W
    pts - mouse coordinates [[x, y]]
    output:
    returns True/False if pts is inside/outside polygon
    e.g. points_in_polygon([[1300,1650],[1600,1650],[1750,1800],[1150,1800]], [[x, y]])

    The idea is draw an infinite line to the right of [x, y] and count the number of time it 
    crosses the shape, if odd it's inside, if even it's outside.


        P-------------      No hit = 0: outside

         xxxxxxx
        x       x
    P---x-------x-----      2 hits = 0 mod(2): outside
        x       x
        x  P----x-----      1 hit = 1 mod(2): inside
        x       x
         xxxxxxx

    ### Method 2: compare areas
    """

    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask[0]

def cm2inch(number):
    """
    convert cm to inches
    """
    return number/2.54
    
#define the coordinates of each patch
NWpatch_coords = [[104, RESOLUTION[1] -  125], [173, RESOLUTION[1] -  125], [173, RESOLUTION[1] -  201], [104, RESOLUTION[1] -  201]]
NEpatch_coords = [[330, RESOLUTION[1] -  120], [400, RESOLUTION[1] -  120], [400, RESOLUTION[1] -  200], [330, RESOLUTION[1] -  200]]
SWpatch_coords = [[109, RESOLUTION[1] -  351], [181, RESOLUTION[1] -  351], [181, RESOLUTION[1] -  410], [109, RESOLUTION[1] -  410]]
SEpatch_coords = [[330, RESOLUTION[1] -  350], [400, RESOLUTION[1] -  350], [400, RESOLUTION[1] -  410], [330, RESOLUTION[1] -  410]]

collection_trapeze = {"NW":dict(), "NE":dict(), "SW":dict(), "SE":dict()}
collection_trapeze["NW"]["N"], collection_trapeze["NW"]["E"], collection_trapeze["NW"]["S"], collection_trapeze["NW"]["W"] = trapezes_from_patch(NWpatch_coords, TRAPEZE_SIZE)
collection_trapeze["NE"]["N"], collection_trapeze["NE"]["E"], collection_trapeze["NE"]["S"], collection_trapeze["NE"]["W"] = trapezes_from_patch(NEpatch_coords, TRAPEZE_SIZE)
collection_trapeze["SW"]["N"], collection_trapeze["SW"]["E"], collection_trapeze["SW"]["S"], collection_trapeze["SW"]["W"] = trapezes_from_patch(SWpatch_coords, TRAPEZE_SIZE)
collection_trapeze["SE"]["N"], collection_trapeze["SE"]["E"], collection_trapeze["SE"]["S"], collection_trapeze["SE"]["W"] = trapezes_from_patch(SEpatch_coords, TRAPEZE_SIZE)


######################################
# Constants
######################################

REMAINING_REWARDS = False # if true, indicate the number of reward available on an object when the mouse starts to go around
MINIMAL_DURATION_STOP = 0.1 #if a stop is shorter than this, merges the two epochs bordering it
MINIMAL_DURATION_EPOCH = 0.3 #minimal duration of an epoch to be considerd
TRUE_SIGMA = 1 #the sigma used for the remaining of the analysis
TRUE_CUT_SPEED = 7 #value to be used for the remaining of the analysis
TRUE_ECART_ANGLE = 1 #if a change is made, must change timeofframes




######################################
# Base analysis stuff
# Code should be functional but not tested and checked
# Some time should be spent to clean it up, make it more readable and efficient
# Lots of linting and bad practices to fix
######################################

################################
def whichPatch(number):
    """send back the string indicating the current patch based on the number recieved
        # 0 = 'NE', 1 = 'NW', 2 = 'SE', 3 = 'SW'
    """
    if number == 0:
        return "NE"
    elif number == 1:
        return "NW"
    elif number == 2:
        return "SE"
    elif number == 3:
        return "SW"
    else:
        raise ValueError("The number must be between 0 and 3")

################################
def search_right_turn(time_start, time_end, turns_df):
    for i in range(len(turns_df)):
        if turns_df.iat[i, 0] > time_start and turns_df.iat[i, 0] < time_end:#if the time of the turn is comprised between the beginning and the end of the epoch
            return turns_df.iat[i, 12] #return the max number of rewards of the serie
    return -1

################################
def is_in_a_goal(xposition, yposition, current_patch, dictionnaire_of_goals):
    """
    for every goal in the list, test if the position is inside using points_in_polygon. return a bool
    """

    in_a_trapeze = False
    for i in dictionnaire_of_goals[current_patch]:
        if points_in_polygon(polygon= dictionnaire_of_goals[current_patch][i], pts = [[xposition, yposition]]):
            in_a_trapeze = True
    return in_a_trapeze

######################
def coordinate_patch(patch): #give a y coordinate corresponding to the
    """give the number corresponding to the patch. Patch must be either 'SW', 'NW', 'SE' or 'NE' """
    if patch == "NE":
        return 1
    elif patch == "NW":
        return 2
    elif patch == "SE":
        return 3
    elif patch == "SW":
        return 4
    else:
        raise ValueError("The patch must be either 'SW', 'NW', 'SE' or 'NE'")

################################
def stay_in_patch(patch, xpositions, ypositions, RESOLUTION):
    """ check if the point change patch at a given moment. Patch must be either 'SW', 'NW', 'SE' or 'NE' """
    stay_in_place = True
    indice = 0
    max_indice = len(xpositions)

    patch = coordinate_patch(patch) - 1

    while indice < max_indice and stay_in_place:#check for every point of the trajectory if they are in a different patch than the first one
        if patch != (xpositions[indice] < RESOLUTION[0] / 2) * 1 + (ypositions[indice] < RESOLUTION[1] / 2) * 2:
            stay_in_place = False
        indice += 1

    return stay_in_place

############################
def cut_in_epoch_speed(cut_off_speed, speed_values, time, MINIMAL_DURATION_STOP, MINIMAL_DURATION_EPOCH):
    """
    cut the trajectory into different part where the animal is moving
    """
    list_of_epochs = []
    speed_size = len(speed_values)
    beginning_epoch = 0

    if speed_size != len(time):
        raise ValueError("speed and time have different length")

    for i in range(speed_size): #
        if speed_values[i] >= cut_off_speed:#if the speed is above the cut-off value
            if beginning_epoch ==0:
                beginning_epoch = i #if there were no epoch being studied, this is the beginning of the epoch
        elif beginning_epoch == 0:
            pass #if we were not in an epoch, failure to be above the threshold does nothing
        elif list_of_epochs != [] and (time[beginning_epoch] - time[list_of_epochs[-1][1]] < MINIMAL_DURATION_STOP):#if the interval with the previous epoch was too short, change its end to the end of this epoch
            list_of_epochs[-1][1] = i-1
            beginning_epoch = 0
        else:
            list_of_epochs.append([beginning_epoch, i-1, "N", 0])#by default, every epoch is noted "N" for "not a quarter turn"
            beginning_epoch = 0


    if beginning_epoch == 0:
        pass #once the loop is ended check if there is a suitable epoch in memory
    elif list_of_epochs != [] and (time[beginning_epoch] - time[list_of_epochs[-1][1]] < MINIMAL_DURATION_STOP): #if the interval with the previous epoch was too short, change its end to the end of this epoch
        list_of_epochs[-1][1] = i-1
        beginning_epoch = 0
    elif (time[speed_size - 1] - time[beginning_epoch]) < MINIMAL_DURATION_EPOCH:
        pass
    else:
        list_of_epochs.append([beginning_epoch, speed_size - 1, "N", 0]) #the N at the end is for "not a quarter turn". every epoch is not a quarter turn until proven otherwise

    size_len_epoch = len(list_of_epochs) # Number of epochs in the list
    a = 0
    
    # Check if the epoch is long enough
    while a < size_len_epoch:
        if (time[list_of_epochs[a][1]] - time[list_of_epochs[a][0]]) < MINIMAL_DURATION_EPOCH:
            _ = list_of_epochs.pop(a) #if the epoch is too short to be considerded, discard it
            size_len_epoch -= 1
        else:
            a+= 1

    for i in range(len(list_of_epochs)):
        current_point = list_of_epochs[i][0] #get the current beginning of the epoch
        acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])
        try:
            previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])
        except:  # FIX ME: that's a bad practice to catch all exceptions
            previous_acceleration = -1  # if this is the first point, there is no previous acceleration, so don't go in the loop

        while previous_acceleration > (0.4 * acceleration) and previous_acceleration > 0:  # continue to go further until we reach the end of the acceleration
            current_point = current_point - 1
            try:
                previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])
            except:  # FIX ME: that's a bad practice to catch all exceptions
                previous_acceleration = -1#if this is the first point, there is no previous acceleration, so break out of the loop
        list_of_epochs[i][0] = current_point#change the beginning of the epoch for the beginning of the acceleration

        current_point = list_of_epochs[i][1] #get the current end of the epoch
        try:
            acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])#calculate the acceleration of the segment just AFTER the end of the epoch
        except:
            acceleration = 1
        previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])#calculate the acceleration of the segment just BEFORE the end of the epoch

        while acceleration < (0.4 * previous_acceleration)  and acceleration < 0:
            current_point = current_point + 1
            try:
                acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])
            except:
                acceleration = 1
        #change the end of the epoch for the end of the decceleration
        list_of_epochs[i][1] = current_point
    return list_of_epochs

##############################
def calcul_angle(ycoordinate, ecart, xcoordinate):
    angles = np.array([np.angle(xcoordinate[i]- xcoordinate[i-ecart] + (ycoordinate[i] - ycoordinate[i-ecart]) * 1j , deg= True) for i in range(ecart, len(xcoordinate))])
    angles = np.insert(angles, obj= 0, values= np.zeros(ecart ))#the calcul of angles change the size of the data. To avoid it, add as much time the first value

    return angles #return the orientation of the mouse across time and the modified epochs

##############################
def analysis_trajectory(time, xgauss, ygauss,
                        collection_trapeze, turns_df,
                        cut_speed, ecart_angle, RESOLUTION, MIN_DURATION_STOP, MIN_DURATION_EPOCH):
    """ Arguments =
    time, xgauss, ygauss:the time of each frame in the TXY csv file and the smoothen positions;
    collection_trapeze:dictionnary with an entry for each patch containing each a dictionnary on their side containing the angles coordinate of the detection trapeze;
    turns_df the dataframe containing the informations of the turnsinfo csv corresponding to the sequence;
    cut_speed: speed under which the mouse is considerd to not be moving;
    ecart_angle: ecart between two frame used to calculate the angle, speed and acceleration;
    RESOLUTION: RESOLUTION of the setup in pixels (the size in m is 0.84. if this change, the code must be updqted manualy);
    MIN_DURATION_STOP: minimal duration accepted for a stop;
    MIN_DURATION_EPOCH: minimal duration considered for a stop

    Output = (all are list)
    distance done between this frame and the last one (in cm);
    speed at the moment corresponding to time_average in cm.s-1;
    time_average time at the moment corresponding to the speed;
    acceleration of the mouse at the time t+1 in cm.s-2;
    direction of the mouse at the moment of time average in degree;
    angular speed at the given time t+1 in degre.s-2;

    list_epoch: list of the epochs under the form  [indice of the first frame of the epoch, indice of the last frame of the epoch, indicator]
    See documentation on indicator for more informations
    """

    # compute the distance but on the data with the gaussian filter
    #print(xgauss) # Print added to do tests and verifications
    #print(ygauss)
    distances_gauss = np.array([((((xgauss[i]-xgauss[i-1])**2)+((ygauss[i]-ygauss[i-1])**2))**0.5) for i in range(1,len(ygauss))])
    distances_gauss = np.insert(distances_gauss *(0.84/RESOLUTION[0]), 0, 0)


    #because the distance is computed using two points, it does no longer correspond to time. To fix it, the average of the time used the calculate the distance is used
    timebeweenframe = np.insert(np.diff(time), 0, 1)#get the gap between the frames. Add 1 at the beginning to have a consistant size (any value is possible, it will divide 0)
    #compute the speed in m/s
    speeds_gauss = np.divide(distances_gauss,timebeweenframe)
    #get the speed in cm/s and add a speed of 0 at the beginning to keep the same data size
    speeds_gauss = speeds_gauss * 100
    list_epochs = cut_in_epoch_speed(cut_off_speed= cut_speed, speed_values = speeds_gauss, time = time, MINIMAL_DURATION_STOP= MIN_DURATION_STOP,
                                     MINIMAL_DURATION_EPOCH= MIN_DURATION_EPOCH) #calculate the epochs with the true cut_off speed and store it

    #calculate the orientation with the chosen value and get the changed epochs
    angles = calcul_angle(ycoordinate= ygauss, ecart= ecart_angle, xcoordinate= xgauss)
    time_average = np.array([time[0]]*ecart_angle + [(time[i] + time[i-ecart_angle]) /2 for i in range(ecart_angle, len(time))])

    angles_relatifs = np.insert(np.diff([angles, time_average])[0], obj= 0, values= np.zeros(1 )) #derive les angles par rapport au temps
    angular_speed = [360 + x if x < -180 else -360 + x if x>180 else x for x in angles_relatifs] #correct for the brutal acceleration when angle pass from -180 to 180

    #calcul of acceleration
    acceleration = np.insert(np.diff([speeds_gauss, time_average])[0], obj = 0, values= np.zeros(1))#derive speed relative to time

    #Advance analysis = identify the quarter turns, the trajectory towards and between objetcs
    #format of a quarter turn indicator: [0] = 'Q' for quarter turn     [1] = 'k'/'w' for counterclockwise / clockwise
    # [2] = 'O'/'E'/'B'/'G'/'H' for wrong object /extra turn / bad direction / Good / double wrong        
    # [3-4] = patch

    #format for between objects indicator: [0] = 'B' for between object    [1 - 2] = previous patch    [3-4] = current patch
    # [5] = 'n'/'r' for non-rewarded/ rewarded (if multiple turns are done in the movement, only the last one is considered)
    in_an_epoch_but_no_quarter = [] #will contain a list under the form [time, corresponding epoch, bool rewarded]

    for a in range(turns_df.index[0], turns_df.index[-1]):#the epochs are written as "not a quarter" by default. We just need to change it for those which are
        aprime = a - turns_df.index[0]
        not_past_nor_found = True
        i = 0
        turn_time = turns_df.loc[turns_df.index[aprime], "time"]    #turns_df.iat[aprime , 0]
        if time[list_epochs[-1][1]] < turn_time: # If the last epoch end before the recorded turn, discard the turn
            not_past_nor_found = False

        while not_past_nor_found:
            if time[list_epochs[i][1]] < turn_time:
                # If the end of the epoch is before the time of the turn, the epoch does not contain the turn so try the next epoch
                i+=1

            # If we reach a point where the beginning of the epoch is after the turn, then the turn was not in an epoch
            elif time[list_epochs[i][0]] > turn_time:
                not_past_nor_found = False

            # If the time is in this epoch, test if this is a true QT
            else:
                    # Check if the beginning of the epoch (movement) is in the polygon it's supposed to                                                                         #check if the beginning of the epoch (movement) is in the polygon it's supposed to

                # Set the value of epoch[3] to the nb of rewards the animal had at the beginning of the movement
                list_epochs[i][3] = turns_df.loc[turns_df.index[aprime - 1], "totalnberOfRewards"] #turns_df.iat[aprime -1, 14]

                if points_in_polygon(polygon = collection_trapeze[turns_df.loc[turns_df.index[aprime], "currentPatch"]][turns_df.loc[turns_df.index[aprime], "previousTrapeze"]], pts = [[xgauss[list_epochs[i][0]], ygauss[list_epochs[i][0]]]]) and points_in_polygon(polygon = collection_trapeze[turns_df.loc[turns_df.index[aprime], "currentPatch"]][turns_df.loc[turns_df.index[aprime], "currentTrapeze"]], pts= [[xgauss[list_epochs[i][1]], ygauss[list_epochs[i][1]]]]):
                    
                    # Current patch is obtained from a number between 0 and 3 indicating in which patch it is (True = 1, False = 0)
                    current_patch = whichPatch((xgauss[list_epochs[i][0]] < RESOLUTION[0] / 2) * 1 + (ygauss[list_epochs[i][0]] < RESOLUTION[1] / 2) * 2)

                    # Check if the mouse does not go to another patch. If it does, it is not a QT
                    if stay_in_patch(current_patch, xgauss[list_epochs[i][0]:list_epochs[i][1] + 1], ygauss[list_epochs[i][0]: list_epochs[i][1] + 1], RESOLUTION):
                        if int(turns_df.iat[aprime, 7]) == 90:
                            # Add a marker depending of the type of turn
                            turn_direction = "k" # Counterclockwise
                        else: turn_direction = "w" # Clockwise

                        # Select the type of turn
                        #if len(turns_df.loc[turns_df.index[aprime], "typeOfTurn"]) == 6: ### BIG CHANGE HERE
                        if turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'gogdet':
                            type_of_turn = 'E' # E = Extra turn
                        #elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"][0] == 'b':
                        elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'bobd':
                            #if turns_df.loc[turns_df.index[aprime], "typeOfTurn"][2] == 'b':
                            #    type_of_turn = 'H' #H = bad object bad direction
                            type_of_turn = 'H' # H = bad object bad direction
                            #else:
                            #    type_of_turn = 'O' # O = wrong object
                        elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'bogd':
                            type_of_turn = 'O' # O = bad object good direction
                        #elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"][2] == 'b':
                        #    type_of_turn = 'B'# B stand for Bad turn
                        elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'gobd':
                            type_of_turn = 'B' # B = good object bad direction
                        #elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"][0] == 'e':
                        #    type_of_turn = 'X' # X for exploration
                        elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'timeout': # new line to replace X
                            type_of_turn = 'T' # T = timeout
                        elif turns_df.loc[turns_df.index[aprime], "typeOfTurn"] == 'gogdnr': # new line to extract depleting from extra turns
                            type_of_turn = 'D' # D = depleting or good object good direction non rewarded
                        else:
                            type_of_turn = 'G' # G = good object good direction

                        if turns_df.loc[turns_df.index[aprime], "Rewarded"]:
                            reward = "R"
                        else:
                            reward = "N"

                        list_epochs[i][2] = "Q" + turn_direction + type_of_turn + current_patch + reward # Q = quarter turn
                    else:
                        in_an_epoch_but_no_quarter += [(turn_time, i, turns_df.loc[turns_df.index[aprime], "Rewarded"])]
                else:
                    in_an_epoch_but_no_quarter += [(turn_time, i, turns_df.loc[turns_df.index[aprime], "Rewarded"])]

                not_past_nor_found = False # The correct epoch was found, no need to continue

    for a in range(len(list_epochs)):
        if list_epochs[a][2][0] == "N":  #if the epoch is not a QT, look at if it can be either a movement between objects or a movement towards an object
            current_patch = whichPatch((xgauss[list_epochs[a][1]] < RESOLUTION[0] / 2) * 1 + (ygauss[list_epochs[a][1]] < RESOLUTION[1] / 2) * 2)

            # If the epoch end in a trapeze it's either a movement towards an object or a movement between objects, or a very small exploration epoch
            if is_in_a_goal(xgauss[list_epochs[a][1]], ygauss[list_epochs[a][1]], current_patch, collection_trapeze):
                previous_patch = whichPatch((xgauss[list_epochs[a][0]] < RESOLUTION[0] / 2) * 1 + (ygauss[list_epochs[a][0]] < RESOLUTION[0] / 2) * 2)

                # Check if the beginning of the epoch was also in a trapeze
                if is_in_a_goal(xgauss[list_epochs[a][0]], ygauss[list_epochs[a][0]], previous_patch, collection_trapeze):
                    if current_patch == previous_patch:

                        # We consider two possibilities in this case:
                        # - either this is a small exploration trajectory or
                        # - the animal move to multiple objects while trying to find the reward and end in the same patch in a between object trajectory
                        if not stay_in_patch(current_patch, xgauss[list_epochs[a][0]:list_epochs[a][1] + 1], ygauss[list_epochs[a][0]:list_epochs[a][1] + 1], RESOLUTION):

                            # Then it's a between objects trajectory
                            list_epochs[a][2] = "B" + previous_patch + current_patch + 'n' # n for n rewards
                        # Else nothing, the exploratory trajectory are marked by an 'N' Which is the default
                    # If previous_patch and current patch are different, it's a trajectory between objects
                    else:
                        list_epochs[a][2] = "B" + previous_patch + current_patch + 'n' # n for no rewards
                # If the beginning of the epoch is not in a goal, it is considered a trajectory toward an object
                else:
                    list_epochs[a][2] = "T" + current_patch

    return distances_gauss, speeds_gauss, time_average, acceleration, angles, angular_speed, list_epochs



######################################
# Heavily modified from the original AnalysisFunction()
# The original function was not really tested and the code was not carefully checked
# This function has been split into multiple functions to make it more readable
# and to allow for more flexibility in the analysis.
#
# Here, the base function "AnalysisFunction" is cut in subfunctions to make it legible.
# Essentially, we now have one function by type of plot, their inputs are the
# required data they need and the ax(s) in which to draw the plot.
# These functions are called in process_session().
# The processing of the data has not been changed, so the results should be the same, and
# I should not be blamed for any error or strange stuff in the resuls ^^
######################################

def process_session(mouseFolder_Path, session, process=False):
    """This is a (futile) attempt to clean up the original code and make it 
    more readable instead of rewriting everything.

    The original code was not really tested and the code was not carefully checked.
    This function calls all the other functions defined below to process the data and plot 
    the results defined in the original code.
    inputs: 
        mouseFolder_Path: path to the folder containing the data
        session: name of the session to process
        process=False: if True, process the data
    outputs:
        None

    call:
        process_session(root + 'MOU4436', 'MOU4436_20240307-1216', process=True)

    Creates a summary figure with subplots for the different 
    types of behavior in the Analysis folder

    TODO:
    - pickle the results
    - add more documentation
    - test the code
    - clean the code
    - good practices
    
    """
    if process:

        # Load the data
        traj_df, turns_df, param_df = load_data(mouseFolder_Path, session)
        phase, direction, cno = get_phase_direction_cno(param_df)

        # Unpack the data
        time = traj_df['time'].to_numpy()
        xposition = traj_df['xposition'].to_numpy()
        yposition = traj_df['yposition'].to_numpy()
        yposition = RESOLUTION[1] - yposition # yposition is inverted, puts it back in the right way
        xgauss = smooth(xposition, TRUE_SIGMA)
        ygauss = smooth(yposition, TRUE_SIGMA) # Smoothes the positions with true sigma

        # Does the actual analysis. The remaining part consists in accessing the pertinent informations and plotting them
        distance, speed, time_average, acceleration, angles, angular_speed, list_epochs = analysis_trajectory(
            time, xgauss, ygauss, collection_trapeze, turns_df, TRUE_CUT_SPEED, TRUE_ECART_ANGLE, RESOLUTION,
            MIN_DURATION_EPOCH=MINIMAL_DURATION_EPOCH, MIN_DURATION_STOP=MINIMAL_DURATION_STOP)

        # Prepare lists of epochs corresponding to diffrent type of behavior
        stops_type = {"rewarded":[], "unrewarded":[]}
        for i in range(len(list_epochs) - 1):
            if list_epochs[i][2][0] == "Q": # If it's a QT
                if list_epochs[i][2][2] == "G": # If this is a good turn and thus a rewarded QT
                    stops_type["rewarded"].append([list_epochs[i][1], list_epochs[i + 1][0]])
                else: # Then the QT was not rewarded
                    stops_type["unrewarded"].append([list_epochs[i][1], list_epochs[i + 1][0]]) 
            #elif list_epochs[i][2][0] == "B": #If this is a between objects
            #    if list_epochs[i][2][5] == 'r': #If the trajectory was rewarded
            #        stops_type["rewarded"].append([list_epochs[i][1], list_epochs[i + 1][0]])
            #    else: #Then the between objects was unrewarded
            #        stops_type["unrewarded"].append([list_epochs[i][1], list_epochs[i + 1][0]])

        list_quarter_turn = [epoch for epoch in list_epochs if epoch[2][0] == "Q"] # All QT
        list_between_objects = [epoch for epoch in list_epochs if epoch[2][0] == "B"] # All trajectories between objects
        list_toward_object = [epoch for epoch in list_epochs if epoch[2][0] == "T"] # All trajectories towards objects
        list_movement_not_quarter = [epoch for epoch in list_epochs if epoch[2][0] == "N"] # All other trajectories
        list_of_stops = [[list_epochs[a - 1][1 ] + 1, list_epochs[a][0] - 1] for a in range(1, len(list_epochs))]

        # Creates a list for each type of QT
        rewarded = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'G']
        unrewarded = [epoch for epoch in list_quarter_turn if epoch[2][2] != 'G']
        extra = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'E']
        bad_direction = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'B'] # Change name here
        bad_object = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'O'] # Change name here
        bad_object_direction = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'H'] # Change name here
        depleting = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'D'] # Added
        timeout = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'T'] # Added

        # between_reward = [epoch for epoch in list_between_objects if epoch[2][5] == 'r']
        # between_unrewarded = [epoch for epoch in list_between_objects if epoch[2][5] == 'n']

        anti_clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "k"]
        clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "w"]
        exploring = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'X']

        when_reward = [[turns_df.loc[a, "time"], turns_df.loc[a, "currentPatch"]] for a in turns_df.index if turns_df.loc[a, "Rewarded"]]
        when_no_reward = [[turns_df.loc[a, "time"], turns_df.loc[a, "currentPatch"]] for a in turns_df.index if not turns_df.loc[a, "Rewarded"]]

        ######################################################
        # Figure creation  ~10sec
        summary_fig = plt.figure(figsize=(14, 12), constrained_layout=True, facecolor='w', dpi=180)
        gs = summary_fig.add_gridspec(6, 8, height_ratios=[1, 1, 1, 1, 1, 1], #hspace=0.5,
                                    width_ratios=[1, 1, 1, 1, 1, .1, 1, 1],# wspace=0.5
                                    )  # 6 rows, 8 columns
        title = figure_title(session, phase, direction, cno)
        summary_fig.suptitle(title, fontsize=16)

        # row 1
        ax_traj = plt.subplot(gs[0, 0])
        ax_speed = plt.subplot(gs[0, 1])
        ax_angular_speed = plt.subplot(gs[0, 2])
        ax_qt_number = plt.subplot(gs[0, 3])
        ax_angular_speed_cw = plt.subplot(gs[0, 6])
        ax_angular_speed_ccw = plt.subplot(gs[0, 7])

        # row 2
        ax_qt = plt.subplot(gs[1, 0])
        ax_between = plt.subplot(gs[1, 1])
        ax_toward = plt.subplot(gs[1, 2])
        ax_other = plt.subplot(gs[1, 3])
        ax_stops = plt.subplot(gs[1, 4])
        ax_speed_cw = plt.subplot(gs[1, 6])
        ax_speed_ccw = plt.subplot(gs[1, 7])

        # row 3
        ax_speed_qt = plt.subplot(gs[2, 0])
        ax_speed_between = plt.subplot(gs[2, 1])
        ax_speed_toward = plt.subplot(gs[2, 2])
        ax_speed_other = plt.subplot(gs[2, 3])
        ax_stop_duration_r = plt.subplot(gs[2, 4])
        ax_speed_profile_cw = plt.subplot(gs[2, 6])
        ax_speed_profile_ccw = plt.subplot(gs[2, 7])

        # row 4
        ax_angular_speed_qt = plt.subplot(gs[3, 0])
        ax_angular_speed_between = plt.subplot(gs[3, 1])
        ax_angular_speed_toward = plt.subplot(gs[3, 2])
        ax_angular_speed_other = plt.subplot(gs[3, 3])
        ax_stop_duration_ur = plt.subplot(gs[3, 4])
        ax_angular_speed_profile_cw = plt.subplot(gs[3, 6])
        ax_angular_speed_profile_ccw = plt.subplot(gs[3, 7])

        # row 5
        ax_acceleration_qt = plt.subplot(gs[4, 0])
        ax_acceleration_between = plt.subplot(gs[4, 1])
        ax_acceleration_toward = plt.subplot(gs[4, 2])
        ax_acceleration_other = plt.subplot(gs[4, 3])
        ax_stop_duration = plt.subplot(gs[4, 4])
        ax_cumu_qt = plt.subplot(gs[4, 6])
        ax_rewarded_qt = plt.subplot(gs[4, 7])

        # row 6
        ax_colored_dot = plt.subplot(gs[5,:])


        # row 1  ~1sec
        plot_session_trajectory(xposition, yposition, ax=ax_traj)
        plot_session_speed(xposition, yposition, time, ax=ax_speed)
        plot_angular_speed(angular_speed, list_epochs, ax=ax_angular_speed)
        figure_qt_number(turns_df, ax=ax_qt_number)

        # row 2:5, cols 0:4  ~5sec
        movement_types = [list_quarter_turn, list_between_objects, list_toward_object, list_movement_not_quarter]
        titles = ["Quarter turns: ", "Between objects: ", "Towards objects: ", "Other movements: "]
        _axs = [[ax_qt, ax_speed_qt, ax_angular_speed_qt, ax_acceleration_qt],
                [ax_between, ax_speed_between, ax_angular_speed_between, ax_acceleration_between],
                [ax_toward, ax_speed_toward, ax_angular_speed_toward, ax_acceleration_toward],
                [ax_other, ax_speed_other, ax_angular_speed_other, ax_acceleration_other]]

        for movement_type, title, _axs in zip(movement_types, titles, _axs):
            figure_trajectories(traj_df, movement_type, xgauss, ygauss, speed, angular_speed, acceleration, title, axs=_axs)

        # row 0:5, col 5  ~2sec
        __axs = [ax_stops, ax_stop_duration_r, ax_stop_duration_ur, ax_stop_duration]
        figure_stops(traj_df, list_of_stops, xgauss, ygauss, time_average, stops_type, axs=__axs)

        # row 2:5, cols 6:7  ~4sec
        ___axs = [ax_speed_cw, ax_speed_ccw, ax_angular_speed_cw, ax_angular_speed_ccw,
                ax_speed_profile_cw, ax_speed_profile_ccw, ax_angular_speed_profile_cw, ax_angular_speed_profile_ccw]
        figure_qturns(speed, angular_speed, list_quarter_turn, time_average, angles, axs=___axs)

        # row 5, col 6:7  ~1sec
        figure_cumul_qturns(list_quarter_turn, rewarded, unrewarded, time_average, axs=[ax_cumu_qt, ax_rewarded_qt])

        # row 6  ~1sec
        figure_coloreddot(turns_df, time, list_epochs, list_quarter_turn, time_average, list_between_objects, ax=ax_colored_dot)

        # Save the figure
        figpath = mouseFolder_Path+os.sep+session+os.sep+'Figure'
        # summary_fig.savefig(figpath+'.pdf', facecolor='w',
        #             edgecolor='none', bbox_inches='tight', format="pdf", dpi=100)
        summary_fig.savefig(figpath+'.png', facecolor='w',
                    edgecolor='none', bbox_inches='tight', format="png", dpi=180)

        plt.close('all')

def load_data(mouseFolder_Path, session):
    try:
        # Gets the parameters of the session
        param_df = pd.read_csv(mouseFolder_Path + os.sep + session + os.sep + session + "_sessionparam.csv")
    except FileNotFoundError:
        print("File sessionparam not found")

    try:
        #Gets the positional informations and filter the dataframe to keep only the relevant informations
        csvCentroid_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_centroidTXY.csv'
        traj_df = pd.read_csv(csvCentroid_fullpath) #Transforms CSV file into panda dataframe
        traj_df = traj_df.dropna() #Deletes lines with one or more NA
        traj_df = traj_df.loc[traj_df['time'] > 15] #First seconds of the video contained artefacts so we need to delete them
        traj_df = traj_df[traj_df['xposition'].between(1, 500) & traj_df['yposition'].between(1, 500)] #The values between 15 and 500 are kept (le tableau est cree plus grand que necessaire)
    except FileNotFoundError:
        print("File centroidTXY not found")

    try:
        csvTurnsinfo_fullpath = mouseFolder_Path + os.sep + session + os.sep + session + '_turnsinfo.csv'  # get the information on the turns in the dataframe turns_df
        turns_df = pd.read_csv(csvTurnsinfo_fullpath)  # Transforms CSV file into panda dataframe
        for i in range(turns_df.index.values[-1]):  # if there is a missing value for ongoingRewardedObject, replace it with either SW or SE, as long as it's not the one where the mouse is
            if type(turns_df['ongoingRewardedObject'][i]) == float:
                turns_df.iat[i, 8] = str([turns_df.iat[i, 4]])
        turns_df = turns_df.loc[turns_df['time'] > 15]  # FIXME: il y a des artefacts sur les premieres secondes de videos, donc il faut les supprimer
    except FileNotFoundError:
        print("File turnsinfo not found")

    return traj_df, turns_df, param_df

def pickle_data(data, path: str, filename: str) -> None :
    """
    Pickles the given variable and saves it to a specified path with the given filename

    Parameters:
    - data (any serializable object): The data to be pickled.
    - path (str): The directory path where the file will be saved.
    - filename (str): The name of the file where the data will be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        # Create the full file path
        full_path = os.path.join(path, filename)
        
        # Write the data to the file
        with open(full_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully pickled to {full_path}")
    except Exception as e:
        print(f"An error occurred while pickling data: {e}")

def unpickle_data(path : str, filename : str) -> None :
    """
    Unpickles data from a specified path and filename

    Parameters:
    - path (str): The directory path where the file is saved.
    - filename (str): The name of the file to read the data from.
    
    Returns:
    any: The unpickled data.
    """
    try:
        # Create the full file path
        full_path = os.path.join(path, filename)
        
        # Read the data from the file
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Data successfully unpickled from {full_path}")
        return data
    except Exception as e:
        print(f"An error occurred while unpickling data: {e}")
        return None

def figure_coloreddot(turns_df, time, list_epochs, list_quarter_turn, time_average, list_between_objects, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(60, 5))

    list_number_reward = []
    serie = 0 #Will indicate if we are in a serie
    current_quarter_turn = -1

    when_reward = [[turns_df.loc[a, "time"], turns_df.loc[a, "currentPatch"]] for a in turns_df.index if turns_df.loc[a, "Rewarded"]]
    when_no_reward = [[turns_df.loc[a, "time"], turns_df.loc[a, "currentPatch"]] for a in turns_df.index if not turns_df.loc[a, "Rewarded"]]

    #Creates a list for each type of quarter turn
    rewarded = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'G']
    unrewarded = [epoch for epoch in list_quarter_turn if epoch[2][2] != 'G']
    extra = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'E']
    badDirection = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'B']
    wrongObject = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'O']
    doubleWrong = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'H']
    anti_clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "k"]
    clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "w"]
    exploring = [epoch for epoch in list_quarter_turn if epoch[2][2] == 'X']

    for epoch in list_epochs:
        if epoch[2][0] == "Q":
            current_quarter_turn += 1#add the number of the current quarter turn after so it still indicate the last one of the serie when added
            if epoch[2][2] == 'G': #if there is a good quarter turn, enter or continu a serie
                if serie == 0: #add the signal at the beginning of the serie
                    list_number_reward.append([current_quarter_turn, search_right_turn(time_start= time[list_quarter_turn[current_quarter_turn][0]], time_end= time[list_quarter_turn[current_quarter_turn][1]], turns_df = turns_df)])
                    serie = 1
                #serie = 1
            elif serie == 1: #if the epoch is not a rewarded quarter turn
                #list_number_reward.append([current_quarter_turn - 1, search_right_turn(time_start= time[list_quarter_turn[current_quarter_turn][0]], time_end= time[list_quarter_turn[current_quarter_turn][1]], turns_df = turns_df)])
                serie = 0
        elif serie == 1: #if the epoch is not a rewarded quarter turn
            #list_number_reward.append([current_quarter_turn, search_right_turn(time_start= time[list_quarter_turn[current_quarter_turn][0]], time_end= time[list_quarter_turn[current_quarter_turn][1]], turns_df = turns_df)])
            serie = 0

    ax.plot([time_average[i[0]] if i[2][0] == "Q"  else time_average[i[1]] for i in list_epochs if i[2][0] == "Q" or i[2][0] == "B"],
            [coordinate_patch(i[2][3:5]) for i in list_epochs if i[2][0] == "Q" or i[2][0] == "B"], c= "palegoldenrod", lw = 2, zorder = 1)

    #Plots the trajectory between objects
    if len(list_between_objects) != 0: #Line below: change the "between objects with/without reward" par bo rewarded or unrewarded
        ax.scatter([time_average[i[1]] for i in list_between_objects], [coordinate_patch(i[2][3:5]) for i in list_between_objects],
                   c="turquoise", label = "between object ", marker= "x", zorder = 2)
    else:
        pass

    colors = ["blue", "firebrick"]
    for a in range(2):
        current_list = [when_reward, when_no_reward][a]
        if len(current_list) != 0:
            ax.scatter([i[0] for i in current_list], [coordinate_patch(i[1]) + 0.1 for i in current_list],
                        c=colors[a], label=[" ", "no "][a] + "reward", s=2 , zorder=2)
        else:
            pass

    #List of the colors used to diffentiate the quarter turns
    colors = ["royalblue", "orangered", "orange", "gold", "chocolate", "slategray"]
    for a in range(6): #Plots the dot for each category of quarter turns
        current_list = [rewarded, extra, badDirection, wrongObject, doubleWrong, exploring][a]
        if len(current_list) != 0:
            ax.scatter([time_average[i[0]] for i in current_list],
                       [coordinate_patch(i[2][3:5]) for i in current_list],
                       c=colors[a], label=["gogt", "extra", "gobt", "bogt", "bobt", "expl"][a], marker="|", zorder=3)
        else:
            pass
    if REMAINING_REWARDS:
        for a in range(len(list_number_reward)):
            ax.text(time_average[list_quarter_turn[list_number_reward[a][0]][0]] - 3,
                    coordinate_patch(list_quarter_turn[list_number_reward[a][0]][2][3:5])-0.05,
                    s=str(list_number_reward[a][1]))

    ax.set_yticks(ticks=[1, 2, 3, 4], labels=['NE', 'NW', 'SE', 'SW'])
    ax.set_xlabel('time (s)')
    ax.set_ylim(0,5)
    ax.set_xlim(0, 900)
    ax.legend(loc='best')

def figure_trajectories(traj_df, current_movement, xgauss, ygauss, speed, angular_speed, acceleration, title='', axs=None):
    if axs is None:
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 15))
    else:
        ax1, ax2, ax3, ax4 = axs

    for patch in collection_trapeze:  #plot the trapeze around the object
        for trapeze in collection_trapeze[patch]:
            shape = Polygon(np.array(collection_trapeze[patch][trapeze]), color="lemonchiffon")
            ax1.add_patch(shape)
        for u in current_movement:  #plot each individual trajectory of the current category
            colors = plt.cm.rainbow(np.random.random())
            if len(u) != 0:
                ax1.plot(xgauss[u[0]:u[1]], ygauss[u[0]:u[1]], lw=0.5, c=colors)
            else:
                pass

    #Plots a colored dot at the begining and end of each epoch
    indices_start = [u[0] for u in current_movement]
    indices_end = [u[1] for u in current_movement]
    if len(indices_start) != 0 or len(indices_end) != 0:
        ax1.scatter(xgauss[indices_start], ygauss[indices_start], linewidths=0.1, color="green")
        ax1.scatter(xgauss[indices_end], ygauss[indices_end], linewidths= 0.1, color="red")
    else:
        pass

    timeSpentIn = round(sum([traj_df.loc[traj_df.index[epoch[1]], 'time'] - traj_df.loc[traj_df.index[epoch[0]], 'time'] for epoch in current_movement]), 2)

    #Sets the parameters of the graph
    ax1.set_ylim(0, 500)
    ax1.set_xlim(0, 500)
    ax1.set_title(title + str(len(current_movement)) + f"\n total time: {timeSpentIn}s")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)


    #Gets the speed and angular speed depending on the category
    speedy = []
    angle_speedy = []
    accelery = []
    for u in current_movement:
        for i in range(u[0], u[1] + 1):
            speedy += [speed[i]]
            angle_speedy += [angular_speed[i]]
            accelery += [acceleration[i]]

    ##################"Plots the speed for the type of trajectory
    if len(speedy)!= 0:
        ax2.hist(speedy, bins=np.arange(0, 100, 1), density=True)
    else:
        ax2.hist([0], bins=np.arange(0, 100, 1), density=True)
    ax2.set_xlabel("Speed (cm/s)")
    ax2.set_ylim(0, 0.06)
    ax2.axvline(TRUE_CUT_SPEED, c='red', lw=0.5)

    ######################"#Plots the angular speed for the type of trajectory
    if len(angle_speedy)!= 0:
        ax3.hist(angle_speedy, bins=np.arange(-50, 50, .5), density=True)
    else :
        ax3.hist([0], bins=np.arange(-50, 50, .5), density=True)
    ax3.set_xlabel("Angular speed (degree/s)")
    ax3.text(-50, 0.02, "CW", ha='left')
    ax3.text(50, 0.02, "CCW", ha='right')
    ax3.set_ylim(0, 0.15)
    ax3.axvline(15, c='red', lw=0.5)
    ax3.axvline(-15, c='red', lw=0.5)
    ax3.axvline(0, c='k', lw=0.5)

    #Plots the acceleration for the type of trajectory
    if len(accelery)!= 0:
        ax4.hist(accelery, bins=np.arange(-25, 25, 0.3), density=True)
    else:
        ax4.hist([0], bins=np.arange(-25, 25, 0.3), density=True)
    ax4.set_xlabel('Acceleration (cm.s-2)')
    ax4.set_ylim(0, 0.16)

def figure_stops(traj_df, current_movement, xgauss, ygauss, time_average, stops_type, axs=None):
    if axs is None:
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 15))
    else:
        ax1, ax2, ax3, ax4 = axs

    for patch in collection_trapeze:#plot the trapeze around the object
        for trapeze in collection_trapeze[patch]:
            shape = Polygon(np.array(collection_trapeze[patch][trapeze]), color="lemonchiffon")
            ax1.add_patch(shape)
        for u in current_movement:#plot each individual trajectory of the current category
            colors = plt.cm.rainbow(np.random.random())
            if len(u) != 0:
                ax1.plot(xgauss[u[0]:u[1]], ygauss[u[0]:u[1]], lw=0.5, c=colors)
            else:
                pass

    #Plots a colored dot at the begining and end of each epoch
    indices_start = [u[0] for u in current_movement]
    indices_end = [u[1] for u in current_movement]
    if len(indices_start) != 0 or len(indices_end)!= 0:
        ax1.scatter(xgauss[indices_start], ygauss[indices_start], linewidths= 0.1, color = "green")
        ax1.scatter(xgauss[indices_end], ygauss[indices_end], linewidths= 0.1, color = "red")
    else:
        pass

    timeSpentIn = round(sum([traj_df.loc[traj_df.index[epoch[1]], 'time'] - traj_df.loc[traj_df.index[epoch[0]], 'time'] for epoch in current_movement]), 2)

    #Sets the parameters of the graph
    ax1.set_ylim(0, 500)
    ax1.set_xlim(0, 500)
    ax1.set_title("Stops: " + str(len(current_movement)) + f"\n total time: {timeSpentIn}s")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.axis('off')


    # ############################################
    # # plot the time of stops after a rewarded quarter turn
    variable = [time_average[u[1]] - time_average[u[0]]  for u in stops_type["rewarded"]]
    if len(variable) != 0:
        ax2.hist(variable, bins=np.arange(0, 2, 0.05))
    else:
        ax2.hist([0], bins=np.arange(0, 2, 0.05))
    ax2.set_xlabel("Stop duration after\nrewarded turns (s)")

    # ############################################
    # # plot the time of stops after a non-rewarded quarter turn
    variable = [time_average[u[1]] - time_average[u[0]]  for u in stops_type["unrewarded"]]
    if len(variable) != 0:
        ax3.hist(variable, bins=np.arange(0, 2, 0.05))
    else:
        ax3.hist([0], bins=np.arange(0, 2, 0.05))
    ax3.set_xlabel("Stops duration after \nunrewarded turn (s)")

    # ############################################
    # # plot the time of stops after all quarter turn
    variable = [time_average[u[1]] - time_average[u[0]] for u in stops_type["unrewarded"]] + [time_average[u[1]] - time_average[u[0]]  for u in stops_type["rewarded"]]
    if len(variable) != 0:
        ax4.hist(variable, bins=np.arange(0, 2, 0.05))
    else:
        ax4.hist([0], bins=np.arange(0, 2, 0.05))
    ax4.set_xlabel("Stops duration (s)")

def figure_qturns(speed, angular_speed, list_quarter_turn, time_average, angles, axs=None):
    if axs is None:
        _, axs = plt.subplots(4, 2, figsize=(10, 40))
    axs = np.array(axs).reshape(4, 2)

    clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "w"] #Divides the quarter turns depending on their direction (CW or CCW)
    anti_clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "k"]
    clock_angular_speed = [angular_speed[i] for u in clock_turn for i in range(u[0], u[1] + 1) ]
    anti_angular_speed = [angular_speed[i] for u in anti_clock_turn for i in range(u[0], u[1] + 1)]
    clock_speed = [speed[i] for u in clock_turn for i in range(u[0], u[1] + 1)]
    anti_clock_speed = [speed[i] for u in anti_clock_turn for i in range(u[0], u[1] + 1)]

    for col, direction in enumerate(["CW" , "CCW"]):
        ##########################
        #Plot angular speed distribution
        if len([clock_angular_speed, anti_angular_speed][col]) != 0:
            axs[0, col].hist([clock_angular_speed, anti_angular_speed][col], bins=np.arange(-50, 50, .5), density=True)
        else:
            axs[0, col].hist([0], bins=np.arange(-50, 50, .5), density=True)
        axs[0, col].set_xlabel(f"Angular speed {direction} (degree/s)")
        axs[0, col].set_ylim(0, 0.075)
        axs[0, col].text(-50, 0.02, "CW", ha='left')
        axs[0, col].text(50, 0.02, "CCW", ha='right')
        axs[0, col].axvline(15, c='red', lw=0.5)
        axs[0, col].axvline(-15, c='red', lw=0.5)
        axs[0, col].axvline(0, c='k', lw=0.5)

        ################################
        #Plot speed distribution
        if len([clock_speed, anti_clock_speed][col])!= 0:
            axs[1, col].hist([clock_speed, anti_clock_speed][col], bins=np.arange(0, 60, 1), density=True)
        else:
            axs[1, col].hist([0], bins=np.arange(0, 60, 1), density=True)
        axs[1, col].set_xlabel(f"Speed {direction} (cm/s)")
        axs[1, col].set_ylim(0, 0.07)
        axs[1, col].axvline(TRUE_CUT_SPEED, c='red', lw=0.5)

        #################################
        #plot the indidual speed profile of every quarter turn
        for u in [clock_turn, anti_clock_turn][col]:
            axs[2, col].plot(time_average[u[0]:u[1]+1] - time_average[u[0]],
                             speed[u[0]:u[1]+1], lw=0.5, c="dimgray")

        if len([clock_turn, anti_clock_turn][col]) != 0:
            xmed, ymed = compute_median_trajectory([speed[u[0]:u[1]+1] for u in [clock_turn, anti_clock_turn][col]],
                                                   [time_average[u[0]:u[1]+1] for u in [clock_turn, anti_clock_turn][col]])
            axs[2, col].plot(xmed, ymed, c='crimson')
        axs[2, col].set_ylabel(f"{direction} QT speed (cm/s)")
        axs[2, col].set_xlabel("time (s)")
        axs[2, col].set_xlim(0, 0.7)
        axs[2, col].set_ylim(0, 60)

        #################################
        #plot the individual profile of the direction changes of every quarter turn
        list_temp_orientation = []
        for u in [clock_turn, anti_clock_turn][col]:
            temp_orientation = angles[u[0]:u[1]+1] - angles[u[0]]
            for i in range(1, len(temp_orientation)):
                if (temp_orientation[i] - temp_orientation[i-1]) < -200:
                    # if the orientation passed the threshold and moved to the other
                    # side, move it to stay within the continuity
                    temp_orientation[ i ] += 360
                elif (temp_orientation[i] - temp_orientation[i-1]) > 200:
                    temp_orientation[i] -= 360
            list_temp_orientation.append(temp_orientation)
            axs[3, col].plot(time_average[u[0]:u[1]+1] - time_average[u[0]], temp_orientation, lw=0.5, c="dimgray")

        if len([clock_turn, anti_clock_turn][col]) != 0:
            xmed, ymed = compute_median_trajectory(list_temp_orientation,
                                                   [time_average[u[0]:u[1]+1] for u in [clock_turn, anti_clock_turn][col]])
            axs[3, col].plot(xmed, ymed, c='crimson')
        axs[3, col].set_ylim(-180, 180)
        axs[3, col].set_xlim(0, 0.7)
        axs[3, col].set_ylabel(f"{direction} QT angular speed (deg/s)")
        axs[3, col].set_xlabel("time in s")

def figure_cumul_qturns(list_quarter_turn, rewarded, unrewarded, time_average, axs=None):
    if axs is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    else:
        ax1, ax2, = axs

    # compute the cumulative sums in accordance to time
    clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "w"] #Divides the quarter turns depending on their direction (CW or CCW)
    anti_clock_turn = [epoch for epoch in list_quarter_turn if epoch[2][1] == "k"]
    clockcum = np.cumsum([1 if  indice in [u[0] for u in clock_turn] else 0 for indice in range(len(time_average))])
    anticum = np.cumsum([1 if  indice in [u[0] for u in anti_clock_turn] else 0 for indice in range(len(time_average))])
    reward_time = np.cumsum([1 if  indice in [u[0] for u in rewarded] else 0 for indice in range(len(time_average))])
    unrewarded_time = np.cumsum([1 if  indice in [u[0] for u in unrewarded] else 0 for indice in range(len(time_average))])

    #################################
    #Plots the cumulative sum of each direction of quarter turn
    ax1.plot(time_average, clockcum, c='#d2725f', label="CW")
    ax1.plot(time_average, anticum, c='#5d93e6', label="CCW")
    ax1.set_ylabel("Cumul # quarter turns")
    ax1.set_xlabel("Time (s)")
    ax1.legend()

    ###################################################
    # Plot the cumulative number of reward and unrewarded quarter turn
    ax2.plot(time_average, reward_time, c="royalblue", label="Rewarded")
    ax2.plot(time_average, unrewarded_time, c="sandybrown", label="Unrewarded")
    ax2.set_ylabel("Cumul # quarter turns")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

def plot_angular_speed(angular_speed, list_epochs, ax=None):
    #########################################
    #Plot the angular speed for all movements
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    angle_speedy = [angular_speed[i] for u in list_epochs for i in range(u[0], u[1]+1)]
    ax.hist(angle_speedy, bins=np.arange(-50, 50, .5), density=True)
    ax.set_title("angular speed")

def plot_session_trajectory(xpositions, ypositions, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    RESOLUTION = 512, 512
    distances = np.array([((((xpositions[i]-xpositions[i-1])**2)+((ypositions[i]-ypositions[i-1])**2))**0.5) for i in range(1,len(ypositions))])
    distances *= (0.84 / RESOLUTION[0]) #convert distance to m with apparatus length = 84 cm
    totaldistance = np.sum(distances)

    ypositions = RESOLUTION[1] - ypositions

    ax.plot(xpositions, ypositions, linewidth=0.5, c='k')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_title(f'Tot dist: {totaldistance:.2f} m')
    ax.axis('off')

def plot_session_speed(xpositions, ypositions, time, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    RESOLUTION = 512, 512
    distances = np.array([((((xpositions[i]-xpositions[i-1])**2)+((ypositions[i]-ypositions[i-1])**2))**0.5) for i in range(1,len(ypositions))])
    distances *= (0.84 / RESOLUTION[0]) #convert distance to m with apparatus length = 84 cm

    ypositions = RESOLUTION[1] - ypositions
    timebeweenframe = np.diff(time)
    speeds = np.divide(distances, timebeweenframe)

    nEvent, binedges, patches = ax.hist(speeds*100, bins=np.arange(0, 100, 100/25),
                                        density=True, facecolor='k', alpha=0.75)
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])

    cruisespeedbinsindexes = np.where(bincenters > 10) #10cm/s is the arbitrary limit between immobility and running
    cruiseSpeedBinValues = np.take(bincenters, cruisespeedbinsindexes)
    cruiseSpeedBinsNevent = np.take(nEvent, cruisespeedbinsindexes)
    meanCruiseSpeed = np.sum(np.multiply(cruiseSpeedBinValues, cruiseSpeedBinsNevent)) / np.sum(cruiseSpeedBinsNevent)

    immobilitybinsindexes = np.where(bincenters < 10)
    ratioRunvsImmob = np.divide(np.sum(np.take(nEvent, cruisespeedbinsindexes)),
                                np.sum(np.take(nEvent, immobilitybinsindexes)))

    ax.set_title(f'Avg cruise speed={meanCruiseSpeed:.2f} cm/s \nRun/Immo ratio={ratioRunvsImmob:.2f}')
    #axs[1].spines[['right', 'top']].set_visible(False) #need to comment if there is an error
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Proba of event')
    ax.set_ylim([0, 0.1])

def figure_qt_number(traj_df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    lasturn_time = traj_df['time'].iloc[-1]
    binsize = 10
    bins=np.arange(0, lasturn_time + binsize, binsize)

    allturns_time=traj_df['time'].to_numpy()
    n, bins, patches = ax.hist(allturns_time, bins, density=False,
                               histtype='step', cumulative=True, label='trapeze changes',
                               color=turnPalette['all turns'])
    #ax.plot(bins[:-1],n,label='all turns',color=turnPalette['all turns'][0],linestyle=turnPalette['all turns'][1])
    #turn_times=traj_df.loc[traj_df['Rewarded']]
    #if len(turn_times) == 0:
    turn_times = traj_df.loc[(traj_df['typeOfTurn'] == "gogt") | (traj_df['typeOfTurn'] == "gogd")]
    turn_times=turn_times['time'].to_numpy()
    n, bins, patches = ax.hist(turn_times, bins, density=False, histtype='step',
                               cumulative=True, label="rewarded",color=turnPalette["gogd"])

    ax.legend(loc='upper left')
    ax.set_ylim([0, 800])

def list_sessions_analyzed(file):
    if not os.path.exists(file):
        list_session_analyzed = []
        with open(file, "w", encoding="utf-8") as f:
            pass
    else:
        with open(file, "r", encoding="utf-8") as f:
            list_session = f.readlines()
            list_session_analyzed = [session[0:-1] if session[-1]=="\n" else session for session in list_session]
    return list_session_analyzed

def figure_title(session, phase, direction, cno):
    _direction = f"{direction}" if direction != "[]" else ""
    _cno = " | session CNO" if cno else ""
    return f"{session}\nPhase {phase} | Directions: {_direction}{_cno}"

def get_phase_direction_cno(param_df):
    try:
        if not param_df.loc[param_df.index[0], "allowRewardDelivery"]:
            phase = 0  # if no reward can be given, then it's the free exploration
        elif param_df.loc[param_df.index[0], "number_of_alternativeObject"] == 1:
            phase = 4  # if only one alternative is available for the objects at a given time in the session, it's session 4
        elif param_df.loc[param_df.index[0], "number_of_alternativeObject"] == 3:
            phase = 3  # then it's phase 3
        elif param_df.loc[param_df.index[0], "potentialRewardedDirections"] == '[90, 270]':
            phase = 1
        else:
            phase = 2
    except KeyError:
        print("Error in getting the phase")
        phase = -1

    direction = param_df.loc[param_df.index[0], "potentialRewardedDirections"]

    try:
        cno = param_df.loc[param_df.index[0], "injectionCNO"] != "none"
    except KeyError:
        cno = False

    return phase, direction, cno

def html_mouse(root, mousename, name_of_figure="Figure.png"):
    header=f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8' />\n<title>{mousename}</title>\n</head><body>\n"
    bottom="</body></html>"

    analysis_folder = root + mousename + os.sep + "Analysis"
    txt = analysis_folder + os.sep + "ListSessionsAnalyzed.txt"
    html_path = analysis_folder + os.sep + f"{mousename}.html"
    processed_sessions = list_sessions_analyzed(txt)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(header)
        for session in processed_sessions:
            imgpath = root + mousename + os.sep + session + os.sep + name_of_figure
            assert os.path.exists(imgpath)
            f.write(f"<p><img src='{imgpath}' alt='{session}'/></p>\n")
        f.write(bottom)


##########################################################################################################################################
# Median run computation
# Modified from: Averaging GPS segments competition 2019. https://doi.org/10.1016/j.patcog.2020.107730
#                T. Karasek, "SEGPUB.IPYNB", Github 2019. https://gist.github.com/t0mk/eb640963d7d64e14d69016e5a3e93fd6
# # # should be able to squeeze SEM in SampleSet class
##########################################################################################################################################

def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2
    return sortedLst[index]

def zscore(l):
    if len(np.unique(l)) == 1:
        return np.full(len(l),0.)
    return (np.array(l)  - np.mean(l)) / np.std(l)

def disterr(x1,y1, x2, y2):
    sd = np.array([x1[0]-x2[0],y1[0]-y2[0]])
    ed = np.array([x1[0]-x2[-1],y1[0]-y2[-1]])
    if np.linalg.norm(sd) > np.linalg.norm(ed):
        x2 = np.flip(x2, axis=0)
        y2 = np.flip(y2, axis=0)

    offs = np.linspace(0,1,10)
    xrs1, yrs1 = Traj((x1,y1)).getPoints(offs)
    xrs2, yrs2 = Traj((x2,y2)).getPoints(offs)
    return np.sum(np.linalg.norm([xrs1-xrs2, yrs1-yrs2],axis=0))

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results

def distance(a, b):
    return  np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d

class OnlyOnePointError(Exception):
    pass

class SampleSet:
    def __init__(self, ll):
        # ll is list of tuples [x_array,y_array] for every trajectory in sample
        self.trajs = [Traj(l) for l in ll]
        self.xp = None
        self.yp = None
        self.d = None
        self.filtix = None
        self.lenoutix = None
        self.disoutix = None
        self.eps = None

    def getRawAvg(self):
        trajLen = median([len(t.xs) for t in self.trajs])
        offs = np.linspace(0,1,trajLen)
        xm = []
        ym = []
        for t in self.trajs:
            xs, ys = t.getPoints(offs)
            xm.append(xs)
            ym.append(ys)
        xp, yp = np.median(xm, axis=0), np.median(ym, axis=0)
        #xp, yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        return xp, yp

    def endpoints(self):
        cs = np.array([[self.trajs[0].xs[0],self.trajs[0].xs[-1]], [self.trajs[0].ys[0],self.trajs[0].ys[-1]]])
        xs = np.hstack([t.xs[0] for t in self.trajs] + [t.xs[-1] for t in self.trajs])
        ys = np.hstack([t.ys[0] for t in self.trajs] + [t.ys[-1] for t in self.trajs])
        clabs = []
        oldclabs = []
        for _ in range(10):
            for i in range(len(xs)):
                ap = np.array([[xs[i]],[ys[i]]])
                dists = np.linalg.norm(ap - cs, axis=0)
                clabs.append(np.argmin(dists))
            #cx = np.array([np.mean(xs[np.where(np.array(clabs)==0)]), np.mean(xs[np.where(np.array(clabs)==1)])])
            #cy = np.array([np.mean(ys[np.where(np.array(clabs)==0)]), np.mean(ys[np.where(np.array(clabs)==1)])])
            if oldclabs == clabs:
                break
            oldclabs = clabs
            clabs = []
        for i, l in enumerate(clabs[:len(clabs)//2]):
            if l == 1:
                oldT = self.trajs[i]
                reversedTraj = (np.flip(oldT.xs, axis=0), np.flip(oldT.ys, axis=0))
                self.trajs[i] = Traj(reversedTraj)

    def zlen(self):
        ls = np.array([t.cuts[-1] for t in self.trajs])
        return zscore(ls)

    def getFiltered(self, dismax, lenlim):
        xa, ya = self.getRawAvg()
        d = zscore(np.array([disterr(t.xs, t.ys, xa, ya) for t in self.trajs]))
        l = self.zlen()
        self.lenoutix = np.where((l<lenlim[0])|(l>lenlim[1]))[0]
        lenix = np.where((l>lenlim[0])&(l<lenlim[1]))[0]
        self.disoutix = np.where(d>dismax)[0]
        disix = np.where(d<dismax)[0]
        self.d = d
        self.l = l
        self.filtix = np.intersect1d(lenix,disix)

    def getAvg(self, dismax, lenlim, eps, stat='Med.'):  # median
        self.eps = eps
        self.endpoints()
        self.getFiltered(dismax, lenlim)
        atleast = 4
        if len(self.filtix) <= atleast:
            distrank = np.argsort(self.d)
            self.disoutix = distrank[atleast:]
            self.lenoutix = []
            self.filtix = distrank[:atleast]
        filtered = [self.trajs[i] for i in self.filtix]
        trajLen = median([len(t.xs) for t in filtered])
        offs = np.linspace(0,1,trajLen*10)
        xm = []
        ym = []
        for t in filtered:
            xs, ys = t.getPoints(offs)
            xm.append(xs)
            ym.append(ys)
        if stat == "Med.":
            self.xp, self.yp = zip(*rdp(list(zip(np.median(xm, axis=0),np.median(ym, axis=0))), eps))
        elif stat == "Avg.":
            self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
        #self.xp, self.yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        xp, yp = self.xp,self.yp
        return xp, yp

    def pax(self, ax):
        ax.set_xlim(0,2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0,130)
        for _, t in enumerate(self.trajs):
            ax.plot(t.xs,t.ys, c="b", marker="o", markersize=2)
        for n, t in enumerate([self.trajs[i] for i in self.disoutix]):
            ax.plot(t.xs,t.ys, c="g")
        for n, t in enumerate([self.trajs[i] for i in self.lenoutix]):
            ax.plot(t.xs,t.ys, c="cyan")
        for n, t in enumerate([self.trajs[i] for i in np.intersect1d(self.lenoutix,self.disoutix)]):
            ax.plot(t.xs,t.ys, c="magenta")
        if self.xp is not None:
            ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)

class Traj:
    def __init__(self,xsys):
        xs, ys = xsys
        a = np.array(xsys).T
        _, filtered = np.unique(a, return_index=True,axis=0)
        if len(filtered) < 2:
            raise OnlyOnePointError()
        self.xs = np.array(xs)[sorted(filtered)]
        self.ys = np.array(ys)[sorted(filtered)]
        self.xd = np.diff(xs)
        self.yd = np.diff(ys)
        self.dists = np.linalg.norm([self.xd, self.yd], axis=0)
        self.cuts = np.cumsum(self.dists)
        self.d = np.hstack([0,self.cuts])

    def getPoints(self, offsets):
        offdists = offsets * self.cuts[-1]
        ix = np.searchsorted(self.cuts, offdists)
        offdists -= self.d[ix]
        segoffs = offdists/self.dists[ix]
        x = self.xs[ix] + self.xd[ix]*segoffs
        y = self.ys[ix] + self.yd[ix]*segoffs
        return x, y

def compute_median_trajectory(posdataRight, timedataRight, stat='Med.'):
    # eps, zmax, lenlim used in outlier detection. Here they are set
    # so they don't exclude any outlier in the median computation.
    # Outlying runs will be//are removed beforehand.
    eps = 0.001
    zmax = np.inf
    lenlim=(-np.inf, np.inf)
    data = list(zip([t - t[0] for t in timedataRight], posdataRight))

    ss = SampleSet(data)
    ss.getAvg(zmax, lenlim, eps, stat) # not supposed to do anything but has to be here to work ??????? Therefore, no touchy.
    X, Y = ss.getAvg(zmax, lenlim, eps, stat)

    # Here median computation warps time (~Dynamic Time Warping) so interpolate to get back to 0.04s increments.
    interpTime = np.linspace(X[0], X[-1], int(X[-1]/0.04)+1) # create time from 0 to median arrival time, evenly spaced 0.04s
    interpPos = np.interp(interpTime, X, Y) # interpolate the position at interpTime
    return interpTime, interpPos
