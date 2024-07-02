"""
Same as EasyPySpin.py but without display or comments, fastest way to run the live code.
See EasyPySpinDisplay.py for more info

Same as EasyPySpinFast_ExploreExploit_07 with corrections : now you can use the depleting function in a uncertainty context (for example: 1 alternative object with direction/objects drawn randomly)

In this version, we want specifically to code the depleting patch mode with uncertainty. That means that we have not yet coded every situation that can happen 
Therefore, for now, you can : add the depleting when you have 4 objects available and when you want to maximize uncertainty by adding a random draw of objects/directions

To understand better the code, here are some keys:
- Patch : corresponds to the area around one object (can be NE, SE, SW, NW) that includes the trapezes 
- Trapeze : area around a solenoid valve, when the animal goes from one trapeze to another it means that 

"""

#Import some librairies to run the code

import EasyPySpin
import cv2
import numpy as np
import time
from paramiko import AutoAddPolicy, SSHClient
import timeit
import random
import pandas as pd
import os
from valves_open_times import get_open_times
from valves_open_times import get_gpio


############################################################################################# As an experimenter, you are supposed to modify this part of the code only

experimenterName='JAK2' #this is important to save your data properly
mouseName='MOU2329' #this is important to save your data properly
CNO_injection='none' #can be 'none' when there is no injection or can be the injection time (hour/minutes)


session_duration = 60*15 #change the multiplier according to the number of minutes you want in your session (session_duration is in seconds)


##Directions parameters that can allow reward delivery:

potentialRewardedDirections=[270] #Here choose your one direction of interest
potentialRewardedDirections=[90,270] #Uncomment this line if you want both directions
#[90] = counterclockwise & [270] = clockwise ; ACCORDING TO THE VIEW FROM THE CAMERA BELOW

#currentDirectionMode = #Maybe this variable can be nice to say if the animal starts with a specific direction, then it has to stay with the same for the rest of the object exploitation
#Or if both directions are available, then it can change while exploiting and it will still give rewards

nextDirectionMode = 'free' #Can be 'switch', 'random' or 'free'
#switch: for example, if the current direction is CW, the next one will be CCW.
#random: the next direction is chosen randomly, means the current direction can be choosen again
#free: if both directions are selected, the rewardingDirection is the one the animal enters the object in. It cannot change direction while exploiting the same object, but this direction changes everytime it enters a new patch


##Rewards decrease mode and parameters:

nextRewardMode = 'decrease_turns' #For now, can be 'decrease_turns' or 'fixed'
#If you put 'fixed', your max number of rewards is fixed. It can be a range, or just a single value but it means that once the animal has reached this max, no rewards are available at this spot
#If you put 'decrease_turns', it means that the number of rewards will decrease every time the mouse do a quarter turn.

#Here, you can change parameters for the 'fixed' mode :
rangeConsecutiveRewards=[4,13] #Select the range of fixed rewards you want. If you put the decrease mode, no matter what those number are but it just need to be > 0

#Here, you can change parameters for the 'decrease_turns' mode :
Slope = 0.2
Plateau = 100 #When you reach this value, the probability (or number of turns) does not decrease anymore
Delay = 2 #Number of rewarding quarter turns to do before it starts decreasing


##Parameters of rewarding objects:

potentialRewardedObjects=['SE','NW','NE','SW'] #Select the object(s) that can deliver rewards. Can be 1 to 4, any combinaison you want.

number_of_alternativeObject=3 #Can be 1 to 4. Once the animal has collected all the rewards around an object, how many objects can potentially deliver rewards
nextObjectMode = 'switch' #Can be 'switch', 'free' or 'random'
#Is the switch mode really relevant for nextObjectMode ? Couldn't it be better with a 'free' mode in which animals can go wherever they want ?
#random: the next direction and object is chosen randomly, mean the current object or direction can be choosen again
#Be careful : the mode 'switch' and 'free' have not been coded yet, this code is only available for a random mode
TRAPEZE_SIZE = 50 #Normally you shouldn't have to change this value, except if you want to adapt the size of a trapeze for a specific analysis


##Allow (or not) variables

allowEarlyExploration=True  #When True, if mice leave a patch before maxConsecutiveReward or Plateau is reached, a new rewarded patch is computed according to task rule
                            #when False, if mice leave a patch before maxConsecutivereward, they must come back to finish it (even if it tried to exploit in the wrong direction)
allowRewardDelivery=True  #if false, the so'0':0.5, '1':0.1, '2':0.1, '3':0.1, '4':0.1, '5':0.1, '6':0.1, '7':0.1, '8':0.15, '9':0.1, '10':0.1, '11':0.1, '12':0.08, '13':0.08, '14':0.08, '15':0.08lenoid valve will never be activated (usefull for exploration without reward or familiarization)


##Handling solenoid valves

test_valves = False #If you want to do tests with valves, and you want to know which one has been opened, set this to true

#############################################################################################

#Define important functions:
#In order for the function to work correctly, the first correct turn must always be rewarded (in this current version of the code)

def reward_function(X, since_last_reward) : #Function that sets up the kind of protocole you are using
    #p is the probability to get a reward
    if nextRewardMode == "decrease_turns" : #This function decreases the number of rewards every time the animal does a QT. The speed of the decrease depends on the slope (and the parameter 't')
        if X < Delay :
            t = 0
        else :
            t = Slope * (X - Delay) # [SLOPE] more turns than the previous one are requiered to get the reward
        if t > Plateau :
            t = Plateau
        if since_last_reward >= t :
            p = 1
        else :
            p = 0

    else :
        print("nextRewardMode = 'fixed'")

    if random.random() <= p : return 1 #Sees if the random value is a greater value than the current threshold, given by the first part and a degressive function
    else : return 0

def trapezes_from_patch(patch, width):
    """
    generate the trapezes coordinates surrounding a patch
    inputs:
    patch - coordinates of a patch [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]]
    width - width of the trapeze in pixels
    outputs:
    coordinates [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]] for the 4 trapezes.
    """

    N = [patch[0], patch[1], [patch[1][0]+width, patch[1][1]-width], [patch[0][0]-width, patch[0][1]-width]]
    E = [patch[1], patch[2], [patch[2][0]+width, patch[2][1]+width], [patch[1][0]+width, patch[1][1]-width]]
    S = [patch[2], patch[3], [patch[3][0]-width, patch[3][1]+width], [patch[2][0]+width, patch[2][1]+width]]
    W = [patch[3], patch[0], [patch[0][0]-width, patch[0][1]-width], [patch[3][0]-width, patch[3][1]+width]]
    return N, E, S, W

def points_in_polygon(polygon, pts):
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

def solenoID(currentPatch, currentTrapeze):
    valvePatch = {'NW':0, 'NE':4, 'SW':8, 'SE':12}  #On the video : NE = object 3, SE = object 1, NW = object 4, SW = object 2
    valveTrapeze = {'N':0, 'E':1, 'S':2, 'W':3}
    if str(currentTrapeze) != 'none':
        return valvePatch[currentPatch] + valveTrapeze[currentTrapeze]

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

#############################################################################################

#Define important variables:

maxConsecutiveRewards=0
if nextRewardMode == 'fixed' : #Previously : was said without the 'fixed' condition
    maxConsecutiveRewards=random.randrange(rangeConsecutiveRewards[0],rangeConsecutiveRewards[1])
    print('max consecutive rewards = ' + str(maxConsecutiveRewards))

#Regarding time
timeout = time.time() + session_duration #calculate when to stop the experiment
theoreticalsamplingrate=25
timeofframes=np.empty([session_duration*(theoreticalsamplingrate+2),3])
timeofframes[:]=np.nan
starttime=time.time()

#Regarding objects and directions
transiantpotentialRewardedObjects=potentialRewardedObjects
ongoingRewardedObject,ongoingRewardedDirection=None,None
previouslyRewardedObject=None
thisTurnDirection, typeofturn = 0, 0
if len(potentialRewardedObjects)==1: #If only one object is chosen by the experimenter
    ongoingRewardedObject=potentialRewardedObjects
if len(potentialRewardedDirections)==1: #If only one direction is chosen by the experimenter
    ongoingRewardedDirection=potentialRewardedDirections

#Regarding rewards
totalRewards = 0
consecutiverewards = 0
since_last_reward = 0
rewards_while_depleting = 0

#Regarding the saving of datas
timestr = time.strftime("%Y%m%d-%H%M") #date and time to add in saved data
dataFolderPath='/media/david/datadrive/data/' #Pathway in which all datas are saved in the computer of the behavior room
sessionName=mouseName+ '_'+ timestr #Name of your session
experimentFolderPath=dataFolderPath + experimenterName + '/' + mouseName + '/' + sessionName #Name of the folder for this session and this mouse
os.mkdir(experimentFolderPath)  #Creates a new folder in your mouse folder where it will save all data generated during the acquisition
videotosave_FullPath=experimentFolderPath + '/' + sessionName + '.avi' #Pathway of video AVI file
turnsinfoCSV_FullPath=experimentFolderPath + '/' + sessionName + '_turnsinfo.csv' #Pathway of turnsinfo CSV file (information regarding every quarter turn)
centroid_time_XYposCSV_FullPath=experimentFolderPath + '/' + sessionName + '_centroidTXY.csv' #Pathway of centroidTXY CSV file (information regarding X and Y positions of the animal's centroid)
sessionParametersCSV=experimentFolderPath + '/' + sessionName + '_sessionparam.csv' #Pathway of sessionparam CSV file (every parameter selected during the session)

#Those vectors point in the direction of the 4 reward ports from the center of the object. They are used to compute turn direction using angle_between function
cardinalvectors= {'N': (0,1),'E': (1,0),'S': (0,-1),'W': (-1,0), 'none':(0,0)}

#Video resolution
resolution = 512,512
theoreticalsamplingrate=25
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(videotosave_FullPath, fourcc, theoreticalsamplingrate, (resolution), False)

#Raspberry pi parameters
#RASPBERRY_IP = '10.51.106.8'
RASPBERRY_IP = '10.42.0.130'
OPEN_TIMES = get_open_times() #Dictionnary of valves number and open duration 
GPIO = get_gpio() #Dictionnary of valves number and pin of the raspberry

#XY coordinates for each patch in 512*512 pixel resolution. Real values 2048*2048 pixels images value after 4x bigger
NWpatch_coords = [[104, 125], [173, 125], [173, 201], [104, 201]]
NEpatch_coords = [[330, 120], [400, 120], [400, 200], [330, 200]]
SWpatch_coords = [[109, 351], [181, 351], [181, 410], [109, 410]]
SEpatch_coords = [[330, 350], [400, 350], [400, 410], [330, 410]]

#Convert for use in openCV
patchNW = np.array(NWpatch_coords, np.int32).reshape((-1,1,2))
patchNE = np.array(NEpatch_coords, np.int32).reshape((-1,1,2))
patchSW = np.array(SWpatch_coords, np.int32).reshape((-1,1,2))
patchSE = np.array(SEpatch_coords, np.int32).reshape((-1,1,2))

#Generates the trapezes around each object
trapezes = {"NW" : trapezes_from_patch(NWpatch_coords, TRAPEZE_SIZE),
            "NE" : trapezes_from_patch(NEpatch_coords, TRAPEZE_SIZE),
            "SW" : trapezes_from_patch(SWpatch_coords, TRAPEZE_SIZE),
            "SE" : trapezes_from_patch(SEpatch_coords, TRAPEZE_SIZE)}

#############################################################################################

#Connection to the raspberry pi
client = SSHClient()
client.set_missing_host_key_policy(AutoAddPolicy())  #Missing host key, bypass with this.
client.connect(RASPBERRY_IP, username='pi', password='rlovy19')

start = timeit.default_timer() #Timer to know when the experiment started
print("Beginning of the session")
saveturntypes=[]

def main():
    cap = EasyPySpin.VideoCapture(0)
    global transiantpotentialRewardedObjects,ongoingRewardedObject,ongoingRewardedDirection,maxConsecutiveRewards,consecutiverewards,totalRewards,typeofturn,thisTurnDirection,previouslyRewardedObject,allowEarlyExploration,allowRewardDelivery,since_last_reward, rewards_while_depleting, _rewardedSolenoid, valve_open_time, valve_pin
    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1

    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  #-1 sets exposure_time to auto
    cap.set(cv2.CAP_PROP_GAIN, -1)  #-1 sets gain to auto

    #Initialize some variables
    _x = _y = 0
    _currentPatch = '',''
    _currentTrapeze = currentTrapeze = ''
    _rewardedSolenoid = rewardedSolenoid = 0.5

    fgbg = cv2.createBackgroundSubtractorMOG2() #Create the background to detect the animal
    framenumber=0

    while True :

        if time.time() < timeout:
            ret, frm = cap.read()
            frm = cv2.resize(frm, (resolution), interpolation = cv2.INTER_AREA)
            frame_video=frm
            out.write(frame_video)

            kernelSize = (25,25)
            frameBlur = cv2.GaussianBlur(frm, kernelSize, 0)

            thresh = fgbg.apply(frameBlur,learningRate=0.0004) #Apply the background substraction

            M = cv2.moments(thresh)
            if M['m00'] == 0:
                continue
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            if _x == 0 and _y == 0: #At the beginning, there is no "past" x and y positions (_x and _y),
                _x = x #so they are equal to the current x and y positions
                _y = y

            thistime=round(time.time()-starttime,3) #Rounds the time to 3 decimal places
            timeofframes[framenumber,:]=[thistime,x,y] #Get the number of the frame and associate it with x/y positions at that time
            framenumber+=1 #Add +1 to the number of frames

            #Depending on the x/y positions, say in which patch the animal currently is
            if x <= resolution[0]/2 and y <= resolution[0]/2: #patch NW
                currentPatch = patchNW, 'NW'
                N, E, S, W = trapezes_from_patch(NWpatch_coords, TRAPEZE_SIZE)
            elif x >= resolution[0]/2 and y<=resolution[0]/2: #patch NE
                currentPatch = patchNE, 'NE'
                N, E, S, W = trapezes_from_patch(NEpatch_coords, TRAPEZE_SIZE)
            elif x <=resolution[0]/2 and y>=resolution[0]/2: #patch SW
                currentPatch = patchSW, 'SW'
                N, E, S, W = trapezes_from_patch(SWpatch_coords, TRAPEZE_SIZE)
            elif x >=resolution[0]/2 and y>=resolution[0]/2: #patch SE
                currentPatch = patchSE, 'SE'
                N, E, S, W = trapezes_from_patch(SEpatch_coords, TRAPEZE_SIZE)
            else:
                print("Where is the mouse?", x, y)

            #Depending on the x/y positions, say in which trapeze the animal currently is
            if points_in_polygon(N, [[x, y]]):
                currentTrapeze = 'N'
            if points_in_polygon(E, [[x, y]]):
                currentTrapeze = 'E'
            if points_in_polygon(S, [[x, y]]):
                currentTrapeze = 'S'
            if points_in_polygon(W, [[x, y]]):
                currentTrapeze = 'W'
            if not points_in_polygon(N, [[x, y]]) and not points_in_polygon(E, [[x, y]]) and not points_in_polygon(S, [[x, y]]) and not points_in_polygon(W, [[x, y]]):
                currentTrapeze = 'none'

            if str(currentTrapeze) == 'none': #In the beginning (and even later), it is possible that the animal isn't in any trapeze (outside of patch)
                deliverReward = False #In this case, there is obviously no reward delivered
                if len(potentialRewardedObjects)==1: # In the case where there is only one object that can be rewarded, changing object reset the number of consecutive rewards
                    consecutiverewards=0
                if nextRewardMode == 'fixed' :
                    maxConsecutiveRewards=random.randrange(rangeConsecutiveRewards[0],rangeConsecutiveRewards[1])
                    print('Max nb of consecutive rewards =' + str(maxConsecutiveRewards))

            if str(currentPatch[1])!=str(_currentPatch[1]): #Mouse changed patch (actual patch is different than the previous one)
                print("Change of patch:", str(currentPatch[1]))
                previouslyRewardedObject=None
                if nextRewardMode == 'decrease_turns': #If you choose a depleting patch mode
                    if rewards_while_depleting > 0 : #And that the counter of rewards in a patch is higher than 0 meaning the animal has been exploiting an object before
                        previouslyRewardedObject=_currentPatch[1] #Then the patch it was before was previously giving rewards (since we reset the counter to 0 when animals change patch)
                        if nextObjectMode=='random': #If you want the next object(s) to be picked randomly
                            transiantpotentialRewardedObjects=random.sample(potentialRewardedObjects,k=number_of_alternativeObject)
                        elif nextObjectMode=='switch': ###Maybe we will need to change this condition
                            unused_objects=list(set(ongoingRewardedObject) ^ set(potentialRewardedObjects))
                            transiantpotentialRewardedObjects=random.sample(unused_objects,k=number_of_alternativeObject)
                        if len(potentialRewardedDirections)>1: #If you put the two directions
                            if nextDirectionMode=='switch':
                                ongoingRewardedDirection=list(set(ongoingRewardedDirection) ^ set(potentialRewardedDirections))
                                print('New rewarding direction is: ' + str(ongoingRewardedDirection[0]))
                            elif nextDirectionMode=='random': #If you want to pick one direction randomly
                                ongoingRewardedDirection=[random.sample(potentialRewardedDirections,k=1)]
                                print('New rewarding direction is: ' + str(ongoingRewardedDirection[0]))
                            elif nextDirectionMode == 'free' :
                                ongoingRewardedDirection = None
                        if len(transiantpotentialRewardedObjects)>1: #If there are more than 1 objects in the transiant list
                                ongoingRewardedObject=None #Then there is no ongoing rewarding object because the animal can choose among that list which object it wants to exploit
                        else:
                            ongoingRewardedObject=transiantpotentialRewardedObjects #But if there is only one object possible, then it become the ongoing rewarding one
                            print('New potential objects are: ',transiantpotentialRewardedObjects)
                        since_last_reward = 0 #Resets the counter when the animal changes patch
                        rewards_while_depleting = 0 #Resets the counter when the animal changes patch
                elif nextRewardMode == 'fixed' :
                    maxConsecutiveRewards=random.randrange(rangeConsecutiveRewards[0],rangeConsecutiveRewards[1]) #Max number of rewards the animal can have, randomly in the range of consecutive rewards chosen at the beginning
                    if maxConsecutiveRewards>1:
                        print('Max nb of consec rewards = ' + str(maxConsecutiveRewards))
                    if maxConsecutiveRewards==1 and len(potentialRewardedObjects)==4: #line for the Phase 1
                        ongoingRewardedObject=None #Every object gives rewards
                    if consecutiverewards<maxConsecutiveRewards and allowEarlyExploration is True and len(potentialRewardedObjects)>1: #Line for the fixed max nb of reward, multiple objects and authorization to leave before reaching the max
                        ongoingRewardedObject=None #Animals can leave when they want and choose among other objects which one they want to exploit
                        transiantpotentialRewardedObjects=potentialRewardedObjects #The other objects they can exploit then are in those lists
                if len(potentialRewardedObjects)==1: #Only one object gives rewards
                    ongoingRewardedObject=potentialRewardedObjects

            elif str(currentTrapeze) == 'none' or str(_currentTrapeze) == 'none': #The mouse was (or is) outside any patch
                deliverReward = False

            elif str(_currentTrapeze) == str(currentTrapeze):  #If this animal is still in the same trapeze, no water is delivered
                deliverReward = False

            elif str(_currentTrapeze) != str(currentTrapeze): #Mouse changed trapeze and now depending on its history, we will check wether reward is delivered or not
                thisTurnDirection=angle_between(cardinalvectors[currentTrapeze],cardinalvectors[_currentTrapeze]) #Gives the direction of the animal

                #Next two 'if' loops define the ongoing rewarded object and ongoing rewarded direction after the mouse changed trapeze
                if  len(transiantpotentialRewardedObjects)>1:
                    if any(item==currentPatch[1] for item in transiantpotentialRewardedObjects): #If the object this animal is in can be rewarding
                        if ongoingRewardedObject is None:
                            ongoingRewardedObject=[currentPatch[1]]
                            print('Good object is:' + ongoingRewardedObject[0])
                if  len(potentialRewardedDirections)>1 and ongoingRewardedDirection is None: #The direction the animal enters within an object becomes the good direction
                    ongoingRewardedDirection=[thisTurnDirection]
                    print('Good direction is : ' + str(ongoingRewardedDirection[0]))

                if currentPatch[1]==previouslyRewardedObject : #If the mouse is in the patch that was previously giving rewards (but that is not anymore)
                    if ([previouslyRewardedObject]!=ongoingRewardedObject and len(potentialRewardedObjects)>1) or len(potentialRewardedObjects)==1: #If its a new object that gives rewards OR if only one object is rewarding
                        deliverReward=False #Then we don't deliver a reward because the limit has been reached
                        if nextRewardMode == 'fixed': #There is no reward limitation or extra turns in the depleting mode, only for the fixed one
                            print('No reward available')
                            if thisTurnDirection==ongoingRewardedDirection[0]: #If the animal was in the good direction
                                typeofturn='gogdet' #Good object, good direction, extra turn : the animal keep turning even if the max number of consecutive rewards has already been reached
                            else: #if its in the wrong direction
                                typeofturn='gobdet' #Good object, bad direction, extra turn : the animal keep turning but it has switch into the bad direction

                #Below, we are going to decide wether reward is delivered or not when the mouse changed trapeze
                #First case : the mouse is in the good object
                if [currentPatch[1]]==ongoingRewardedObject:
                    if thisTurnDirection==ongoingRewardedDirection[0]:  #If the animal is a the good direction
                        if nextRewardMode == 'fixed' and ((maxConsecutiveRewards is None) or (consecutiverewards<maxConsecutiveRewards)): #and has no max number of consecutive rewards or it has not been reached yet
                            deliverReward=True #then we give a reward
                            typeofturn='gogd' #This type of turn is "Good object/good direction"
                        elif nextRewardMode == 'decrease_turns':
                            deliverReward = reward_function(rewards_while_depleting, since_last_reward)
                            #The output of the function can be p=1 (100% reward delivery) or p=0 (0% reward delivery) for each QT
                            if deliverReward == 1 :
                                typeofturn = 'gogd' #This type of turn is "Good object/good direction"
                            else:
                                since_last_reward +=1 #+1 QT since the last reward was obtained
                                print('No reward obtained')
                                print("Nb of unrewarded QT since last reward:", since_last_reward)
                                typeofturn= 'gogdnr' #This type of turn is "Good object/good direction/non-rewarded"
                                consecutiverewards=0
                    else: #If the animal direction is not the rewarding one
                        deliverReward=False #then no reward is delivered
                        typeofturn='gobd' #This type of turn is "Good object/bad direction"

                    if deliverReward and allowRewardDelivery: #If a reward is delivered and we authorized it
                        rewardedSolenoid = solenoID(currentPatch[1], currentTrapeze)
                        command = f'python3 /home/pi/valvesT.py --ID={str(GPIO[str(rewardedSolenoid)])} --opentime={str(OPEN_TIMES[str(rewardedSolenoid)])}' #Command to open the valve
                        stdin, stdout, stderr = client.exec_command(command) #Execution of the command
                        valve_ID = str(rewardedSolenoid) #Gives the ID of the valve
                        valve_open_time = OPEN_TIMES[valve_ID] #The open time
                        valve_pin = GPIO[valve_ID] #And which pin of the raspberry it is connected to

                        if test_valves: #If you want to test which valve has been opened while rewards are delivered, this value must be True
                            print(f"Opening valve n°: {valve_ID}") #We print which valve has been opened
                            print(f"- Open time: {valve_open_time}")
                            print(f"- Raspberry Pi pin: {valve_pin}")

                        if nextRewardMode == 'decrease_turns' :
                            rewards_while_depleting +=1 #Every time a reward is obtained, the total number of rewards obtained in this patch takes +1
                            print("Total rewards in this patch:", rewards_while_depleting)
                            since_last_reward = 0 #Resets the counter every time the mouse obtains a reward in the depleting mode

                        #Increment total and consecutive rewards + rewards obtained in a patch while it is depleting
                        totalRewards+=1
                        consecutiverewards+=1
                        if nextRewardMode == 'fixed' :
                            print('Consecutive: ' + str(consecutiverewards))
                        print('Total: ' + str(totalRewards))

                else: #if the animal is insisting on an object that does not deliver reward we consider it at as gogdet (see line 281) ###Je comprends pas pourquoi on met pas juste un else pour dire que c'est le bad object ? pourquoi Elif et surtout PQ PREVIOUSLY REWARDING OBJECT = NONE ?
                    if thisTurnDirection==ongoingRewardedDirection[0]: #If the animal is in the good direction
                        typeofturn='bogd' #This type of turn is "Bad object/good direction"
                    else: #If its in the wrong direction
                        typeofturn='bobd' #This type of turn is "Bad object/bad direction"

                #We are done with characterizing the turns and can save the current info:
                saveturntypes.append([thistime,framenumber,x,y,currentPatch[1],_currentTrapeze,currentTrapeze,thisTurnDirection,ongoingRewardedObject,ongoingRewardedDirection,deliverReward,typeofturn,consecutiverewards,maxConsecutiveRewards,since_last_reward, rewards_while_depleting, totalRewards])

                #Now we need to define new objects and directions when the maximum of reward has been reached for the fixed mode, and when the animal leaves the object in the depleting mode
                if  nextRewardMode == 'fixed' and consecutiverewards==maxConsecutiveRewards and maxConsecutiveRewards>1: #case in wich the animal has reached the max number of consecutive reward
                    previouslyRewardedObject=currentPatch[1]
                    #When multiple objects are possible, define the new rewarded one based on nextObjectMode and potentialRewardedObjects
                    if len(potentialRewardedObjects)>1:
                            if nextObjectMode=='random':
                                transiantpotentialRewardedObjects=random.sample(potentialRewardedObjects,k=number_of_alternativeObject)
                                print('transiantpotentialRewardedObjects =', transiantpotentialRewardedObjects) ####Added to test
                            elif nextObjectMode=='switch': ###Maybe we will need to change this condition
                                unused_objects=list(set(ongoingRewardedObject) ^ set(potentialRewardedObjects))
                                transiantpotentialRewardedObjects=random.sample(unused_objects,k=number_of_alternativeObject)

                            if len(transiantpotentialRewardedObjects)>1:
                                ongoingRewardedObject=None
                            else:
                                ongoingRewardedObject=transiantpotentialRewardedObjects
                                print('ongoingRewardedObject =', ongoingRewardedObject) #####Added to test
                            print('New potential objects are: ',transiantpotentialRewardedObjects)
                    else:
                        ongoingRewardedObject=None

                    #When multiple directions are possible define the new rewarded one based on nextDirectionMode and potentialRewardedDirections parameters
                    if len(potentialRewardedDirections)>1:
                        if nextDirectionMode=='switch':
                            ongoingRewardedDirection=list(set(ongoingRewardedDirection) ^ set(potentialRewardedDirections))
                        elif nextDirectionMode=='random'and (maxConsecutiveRewards>1 or nextRewardMode == 'decrease_turns'):
                            ongoingRewardedDirection=[random.sample(potentialRewardedDirections,k=1)]
                        elif maxConsecutiveRewards==1: #In phase 1, both directions are possible #Change here: no matter the nextDirectionMode, if there is only one reward there is no ongoing rewarding direction
                            ongoingRewardedDirection=None
                        elif nextRewardMode == 'fixed' and maxConsecutiveRewards>1: #We dont need to print the new direction for the phase 1 (maxconsecutivereward=1)
                            print('New rewarding direction is: ' + str(ongoingRewardedDirection[0]))

                        #Now we need to reset the number of consecutive rewards at 0
                        consecutiverewards=0


            #Show the detected shape
            cv2.imshow("__", thresh)

            #Update past values
            _x = x
            _y = y
            _currentPatch = currentPatch
            _currentTrapeze = currentTrapeze
            _rewardedSolenoid = rewardedSolenoid

            key = cv2.waitKey(10)
        else : #session time is over, now we save the date in dataframes then CSV files
            df=pd.DataFrame(saveturntypes)
            df.columns=['time','framenumber','xposition','yposition','currentPatch','previousTrapeze','currentTrapeze','turnDirection','ongoingRewardedObject','ongoingRewardedDirection','Rewarded','typeOfTurn','nberOfConsecRewards','maxNberOfConsecRewards','nbQT_sinceLastReward', 'nbRewardsWhileDepleting','totalnberOfRewards']
            df.to_csv(turnsinfoCSV_FullPath, sep=',', index=False)

            dfsamplingtimes=pd.DataFrame(timeofframes)
            dfsamplingtimes.columns=['time','xposition','yposition']
            dfsamplingtimes.to_csv(centroid_time_XYposCSV_FullPath, sep=',', index=False)

            parameters=[[experimenterName,mouseName,session_duration,CNO_injection,potentialRewardedDirections,nextDirectionMode,potentialRewardedObjects,nextObjectMode,number_of_alternativeObject,nextRewardMode,rangeConsecutiveRewards,Slope, Delay, Plateau, allowEarlyExploration,allowRewardDelivery, TRAPEZE_SIZE]]
            df=pd.DataFrame(parameters)
            df.columns=['experimenterName','mouseName','durationOfExperiment','CNO_injectionTime','potentialRewardedDirections','nextDirectionMode','potentialRewardedObjects','nextObjectMode','number_of_alternativeObject','RewardMode','range_maxConsecRewards','Slope','Delay','Plateau','allowEarlyExploration','allowRewardDelivery','TrapezeSize']
            df.to_csv(sessionParametersCSV, sep=',', index=False)
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

    stop = timeit.default_timer()
    print('Time: ', stop - start)

if __name__ == "__main__":
    main()