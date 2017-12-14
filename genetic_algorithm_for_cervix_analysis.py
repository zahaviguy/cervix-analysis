
# Import packages
import os
from glob import glob

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import math
from scipy import mean, std
import scipy.stats as stats
from random import randint, random

import cv2
import matplotlib.pylab as plt
import matplotlib.patches as patches
import seaborn as sns

# Definitions for genetic algorithm and fitness function
# colors for fitness function - a_channel
os_norm_mean_target = 1.94
endo_norm_mean_target = 2.24
zone_norm_mean_target = 0.74
os_norm_std_target = 0.40
endo_norm_std_target = 0.50
zone_norm_std_target = 0.64

# resize ratio for computational speed
resize_ratio = 0.1

# genetic algorithm parameters
population_size = 200
retain_ratio = 0.2
mutation_rate = 0.05
random_selection_rate = 0.05
generations = 20

# Input data files are available in the "../input/" directory.
# Getting filenames from input directory
TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1")) + 1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2")) + 1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3")) + 1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])

TEST_DATA = "../input/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA) + 1:-4] for s in test_files])
print(len(test_ids))
print(test_ids[:10])

ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array(
    [s[len(os.path.join(ADDITIONAL_DATA, "Type_1")) + 1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array(
    [s[len(os.path.join(ADDITIONAL_DATA, "Type_2")) + 1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array(
    [s[len(os.path.join(ADDITIONAL_DATA, "Type_3")) + 1:-4] for s in additional_type_3_files])

print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_3_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type, rsz_ratio=1):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    if rsz_ratio != 1:
        img = cv2.resize(img, dsize=(int(img.shape[1] * rsz_ratio), int(img.shape[0] * rsz_ratio)))
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def lab_channels(img, display_image=False):
    # Extracting the Lab color space into different variables

    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L_channel = imgLab[:,:,0]
    a_channel = imgLab[:,:,1]
    b_channel = imgLab[:,:,2]
    
    if display_image==True:
        plt.figure(figsize=(8,8))
        
        plt.subplot(221)
        plt.title("Original image")
        plt.imshow(img), plt.xticks([]), plt.yticks([])
        
        plt.subplot(222)
        plt.title("L channel")
        plt.imshow(L_channel, cmap='gist_heat'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(223)
        plt.title("a channel")
        plt.imshow(a_channel, cmap='gist_heat'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(224)
        plt.title("b channel")
        plt.imshow(b_channel, cmap='gist_heat'), plt.xticks([]), plt.yticks([])
        
    return L_channel, a_channel, b_channel

def is_ellipse_in_ellipse(o_ell, i_ell, display_image=False):
    # A function to asses if the outer ellipse contains the inner ellipse
    # It's an approximation because i am checking only 4 points
    # Future consideration - maybe add more points (8, 16)
    
    # finding the boundaries of the inner ellipse
    i_ell_center = i_ell.center
    i_ell_width = i_ell.width
    i_ell_height = i_ell.height
    i_angle = i_ell.angle

    cos_angle_in = np.cos(np.radians(180.-i_angle))
    sin_angle_in = np.sin(np.radians(180.-i_angle))
    
    xct_in=np.zeros(4)
    yct_in=np.zeros(4)
    xct_in[0] = i_ell_width/2
    yct_in[0] = 0
    xct_in[1] = -i_ell_width/2
    yct_in[1] = 0
    xct_in[2] = 0
    yct_in[2] = i_ell_height/2
    xct_in[3] = 0 
    yct_in[3] = -i_ell_height/2

    xc_in = (xct_in * cos_angle_in + yct_in * sin_angle_in ) 
    yc_in = (yct_in * cos_angle_in - xct_in * sin_angle_in ) 

    x_in = i_ell_center[0] + xc_in
    y_in = i_ell_center[1] + yc_in
    
    # Placing the coordinates in the outer ellipse
    g_ellipse = o_ell
    
    g_ell_center = g_ellipse.center
    g_ell_width = g_ellipse.width
    g_ell_height = g_ellipse.height
    angle = g_ellipse.angle

    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    xc = x_in - g_ell_center[0]
    yc = y_in - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
    
    # Assume all points are in ellipse
    all_ellipse_in = True
    
    for r in rad_cc:
            if r > 1.:
                # point not in ellipse
                all_ellipse_in = False
    
    
    if (display_image==True):
        colors_array = []

        for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                colors_array.append('red')
            else:
                # point not in ellipse
                colors_array.append('black')

        fig,ax = plt.subplots(1)
        #ax.set_aspect('equal')

        ax.add_patch(g_ellipse)
        ax.add_patch(i_ell)

        ax.scatter(x_in,y_in,c=colors_array,linewidths=0.3)

        plt.show()
    return all_ellipse_in

def do_ellipses_intersect(i_ell, o_ell, display_image=False):
    # A function to asses if the ellipses intersect
    # It's an approximation because i am checking only 4 points of each ellipse
    # Future consideration - maybe add more points (8, 16)
    
    # finding the boundaries of the inner ellipse
    i_ell_center = i_ell.center
    i_ell_width = i_ell.width
    i_ell_height = i_ell.height
    i_angle = i_ell.angle

    cos_angle_in = np.cos(np.radians(180.-i_angle))
    sin_angle_in = np.sin(np.radians(180.-i_angle))
    
    xct_in=np.zeros(4)
    yct_in=np.zeros(4)
    xct_in[0] = i_ell_width/2
    yct_in[0] = 0
    xct_in[1] = -i_ell_width/2
    yct_in[1] = 0
    xct_in[2] = 0
    yct_in[2] = i_ell_height/2
    xct_in[3] = 0 
    yct_in[3] = -i_ell_height/2

    xc_in = (xct_in * cos_angle_in + yct_in * sin_angle_in ) 
    yc_in = (yct_in * cos_angle_in - xct_in * sin_angle_in ) 

    x_in = i_ell_center[0] + xc_in
    y_in = i_ell_center[1] + yc_in
    
    # Placing the coordinates in the outer ellipse
    g_ellipse = o_ell
    
    g_ell_center = g_ellipse.center
    g_ell_width = g_ellipse.width
    g_ell_height = g_ellipse.height
    angle = g_ellipse.angle

    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    xc = x_in - g_ell_center[0]
    yc = y_in - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
    
    # Assume no points are in ellipse
    ellipses_intersect = False
    
    for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                ellipses_intersect = True
    
    
    if (display_image==True):
        colors_array = []

        for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                colors_array.append('red')
            else:
                # point not in ellipse
                colors_array.append('black')

        fig,ax = plt.subplots(1)
        #ax.set_aspect('equal')

        ax.add_patch(g_ellipse)
        ax.add_patch(i_ell)

        ax.scatter(x_in,y_in,c=colors_array,linewidths=0.3)

        plt.show()
    return ellipses_intersect

def is_point_in_ellipse(o_ell, x_in, y_in, display_image=False):
    # Placing the coordinates in the outer ellipse
    g_ellipse = o_ell
    
    g_ell_center = g_ellipse.center
    g_ell_width = g_ellipse.width
    g_ell_height = g_ellipse.height
    angle = g_ellipse.angle

    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    xc = x_in - g_ell_center[0]
    yc = y_in - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
    
    # Assume all points are is in ellipse
    point_in = True
    
    if rad_cc > 1.:
        # point not in ellipse
        point_in = False

    if (display_image==True):
        colors_array = []

        if rad_cc <= 1.:
            # point in ellipse
            colors_array.append('red')
        else:
            # point not in ellipse
            colors_array.append('black')

        fig,ax = plt.subplots(1)
        #ax.set_aspect('equal')

        ax.add_patch(g_ellipse)
        
        ax.scatter(x_in,y_in,c=colors_array,linewidths=0.3)

        plt.show()
                
    return point_in

def crop_ellipse(img, ell, display_image=False):
    # http://answers.opencv.org/question/25523/extract-an-ellipse-form-from-an-image-instead-of-drawing-it-inside/
    # Crop ellipse works only for single channel images

    # create a mask image of the same shape as input image, filled with 1s
    mask = np.ones_like(img)
        
    # create a zero filled ellipse
    mask=cv2.ellipse(mask, 
                     center=ell.center, 
                     axes=(int(ell.width/2),int(ell.height/2)), 
                     angle=ell.angle, 
                     startAngle=0, endAngle=360, color=(0,0,0), thickness=-1)
    
    # Creating a masked array containing only relevant pixels
    cropped_ellipse = np.ma.masked_array(img, mask)
    
    # create a mask image for the background, filled with 0s
    background_mask = np.zeros_like(img)
        
    # create a ones filled ellipse
    background_mask=cv2.ellipse(background_mask, 
                     center=ell.center, 
                     axes=(int(ell.width/2),int(ell.height/2)), 
                     angle=ell.angle, 
                     startAngle=0, endAngle=360, color=(1,1,1), thickness=-1)
    
    background = np.ma.masked_array(img, background_mask)
    
    # Plotting the results
    if (display_image):
        outline = img.copy()
        outline=cv2.ellipse(outline, 
                     center=ell.center, 
                     axes=(int(ell.width/2),int(ell.height/2)), 
                     angle=ell.angle, 
                     startAngle=0, endAngle=360, color=(90,90,90), thickness=4)
    
        plt.figure(figsize=(6,6))
        plt.subplot(121)
        plt.imshow(cropped_ellipse, cmap='hot')
        plt.subplot(122)
        plt.imshow(outline, cmap='hot')
        plt.show()
    return cropped_ellipse, background

def create_ellipse(x, y, width, height, angle, edgecolor='black'):
    return patches.Ellipse((x, y), width, height, angle=angle, fill=False, edgecolor=edgecolor, linewidth=2)

def ellipse_background(img, ell, ratio=1.3, display_image=False):
    back_width = max(ell.width*ratio, ell.width + 4)
    back_height = max(ell.height*ratio, ell.height + 4)
    
    back_ell = create_ellipse(ell.center[0], ell.center[1], back_width, back_height, ell.angle)
    front_ell = create_ellipse(ell.center[0], ell.center[1], ell.width, ell.height, ell.angle)
    
    front_center = front_ell.center
    front_axes = (int(front_ell.width/2),
            int(front_ell.height/2))
    front_angle = int(front_ell.angle)
    
    img_ell = img.copy()
    img_ell=cv2.ellipse(img_ell, 
                     front_center, 
                     front_axes, 
                     front_angle, 
                     startAngle=0, endAngle=360, color=(0,0,0), thickness=-1)

    back_ell_image = crop_ellipse(img_ell, back_ell)
    
    result = np.ma.masked_equal(back_ell_image, 0)
    
    if (display_image):
        plt.imshow(result)
        
    return result

def extract_ellipses(a_image, os, endo, zone, display_image=False):
    # Cropping out the ellipses from the image
    os_image, os_background = crop_ellipse(a_image, os, display_image=False)

    endo_image, endo_background = crop_ellipse(os_background, endo, display_image)

    zone_image, _ = crop_ellipse(endo_background, zone, display_image)
    
    return os_image, endo_image, zone_image

def outline_individual_on_image(img, individual):
    # Creating ellipses from chromosome and extracting the relevant images
    os_ell = create_ellipse(individual[0], individual[1], individual[2], individual[3], individual[4], edgecolor='red')
    endo_ell = create_ellipse(individual[5], individual[6], individual[7], individual[8], individual[9], edgecolor='red')
    zone_ell = create_ellipse(individual[10], individual[11], individual[12], individual[13], individual[14], edgecolor='red')
    
    outline = img.copy()
    outline = cv2.ellipse(outline, 
                 center=os_ell.center, 
                 axes=(int(os_ell.width/2),int(os_ell.height/2)), 
                 angle=os_ell.angle, 
                 startAngle=0, endAngle=360, color=(0,0,0), thickness=2)
    outline = cv2.ellipse(outline, 
                 center=endo_ell.center, 
                 axes=(int(endo_ell.width/2),int(endo_ell.height/2)), 
                 angle=endo_ell.angle, 
                 startAngle=0, endAngle=360, color=(255,255,255), thickness=2)
    outline = cv2.ellipse(outline, 
                 center=zone_ell.center, 
                 axes=(int(zone_ell.width/2),int(zone_ell.height/2)), 
                 angle=zone_ell.angle, 
                 startAngle=0, endAngle=360, color=(0,255,0), thickness=2)
    
    return outline

# Genetic algorithm from
# https://lethain.com/genetic-algorithms-cool-name-damn-simple/
def create_individual(img):
    # Creating an individual
    # individual  = [x, y, widht, height, angle] * 3

    # Create a member of the population.'
    w = img.shape[1]
    h = img.shape[0]
    max_size = max(w, h)
    min_size = int(max_size * 0.5)
    
    ind = [0]*15
    
    # Ellipse #1 center
    ind[0] = randint(0, w)
    ind[1] = randint(0, h)
    # Width and height
    ind[2] = randint(2, max_size)
    ind[3] = randint(2, max_size)
    # Angle
    ind[4] = randint(0, 180)
    
    # Ellipse #2 center
    ind[5] = randint(0, w)
    ind[6] = randint(0, h)
    # Width and height
    ind[7] = randint(int(max_size*0.25), max_size)
    ind[8] = randint(int(max_size*0.25), max_size)
    # Angle
    ind[9] = randint(0, 180)
    
    # Ellipse #3 center
    ind[10] = randint(0, w)
    ind[11] = randint(0, h)
    # Width and height
    ind[12] = randint(min_size, max_size)
    ind[13] = randint(min_size, max_size)
    # Angle
    ind[14] = randint(0, 180)
    
    return ind

def init_population(img, count):
    # Creating a population

    pop = []
    for i in range(count):
        pop.append(create_individual(img))
    
    return pop

def evolve(img, pop, retain, random_select, mutate):
    # Evolution

    retain_length = int(len(pop)*retain)
    parents = pop[:retain_length]
    
    # randomly add other individuals to
    # promote genetic diversity
    for individual in pop[retain_length:]:
        if random_select > random():
            parents.append(individual)
    
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            individual[pos_to_mutate] = create_individual(img)[pos_to_mutate]
    
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            exchange_point = randint(0, len(male)-1)
            child = male[:exchange_point] + female[exchange_point:]
            children.append(child)
    parents.extend(children)
    return parents

def grade_type1(a_img, pop):
    # Sort population according to fitness function
    # Return sorted population and fitness function grades
    init_pop = [ (type1_fitness(a_img, x), x) for x in pop]
    init_pop = sorted(init_pop)
    sorted_pop = [x[-1] for x in init_pop]

    grades = [x[0][0] for x in init_pop]
    elements = [ x[0][1] for x in init_pop]
    return sorted_pop, grades, elements

# GA Fitness Functions
def type1_fitness(a_channel, individual, display_image=False):
    penalty = 25.0

    # Zeroing all of the fitness function elements
    fitness = 0.0
    os_std = os_mean = endo_std = endo_mean = zone_std = zone_mean = 0.0
    os_out_of_endo_penalty = os_out_of_zone_penalty = endo_out_of_zone_penalty = 0.0
    endo_in_os_penalty = zone_in_os_penalty = zone_in_endo_penalty = zone_is_empty_penalty = 0.0

    # Creating ellipses from chromosome and extracting the relevant images
    os_ell = create_ellipse(individual[0], individual[1], individual[2], individual[3], individual[4], edgecolor='red')
    endo_ell = create_ellipse(individual[5], individual[6], individual[7], individual[8], individual[9],
                              edgecolor='red')
    zone_ell = create_ellipse(individual[10], individual[11], individual[12], individual[13], individual[14],
                              edgecolor='red')

    os_image, endo_image, zone_image = extract_ellipses(a_channel, os_ell, endo_ell, zone_ell, display_image)

    os_array = np.ma.MaskedArray.flatten(os_image)
    endo_array = np.ma.MaskedArray.flatten(endo_image)
    zone_array = np.ma.MaskedArray.flatten(zone_image)

    ### Checking the position of all ellipses relatively to one another
    # OS inside ENDO?
    os_in_endo = True if (is_ellipse_in_ellipse(endo_ell, os_ell, display_image=False)) else False
    # ENDO is inside OS?
    endo_in_os = True if (is_ellipse_in_ellipse(os_ell, endo_ell, display_image=False)) else False
    # OS inside ZONE?
    os_in_zone = True if (is_ellipse_in_ellipse(zone_ell, os_ell, display_image=False)) else False
    # ENDO inside ZONE?
    endo_in_zone = True if (is_ellipse_in_ellipse(zone_ell, endo_ell, display_image=False)) else False
    # ZONE inside OS?
    zone_in_os = True if (is_ellipse_in_ellipse(os_ell, zone_ell, display_image=False)) else False
    # ZONE inside ENDO?
    zone_in_endo = True if (is_ellipse_in_ellipse(endo_ell, zone_ell, display_image=False)) else False

    ## OS ellipse
    # OS mean color
    os_mean = abs(np.mean(os_array) - os_norm_mean_target)
    os_std = abs(np.std(os_array) - os_norm_std_target)

    # Add penalty if OS is not indside ENDO
    if (os_in_endo == False):
        os_out_of_endo_penalty = penalty

    # Add penalty if OS is not indside ZONE
    if (os_in_zone == False):
        os_out_of_zone_penalty = penalty

    ## ENDO ellipse
    # Checking that the ENDO ellipse is not empty
    if ((endo_in_os == False) and (endo_image.mask.all() == False)):
        # ENDO mean color
        endo_mean = abs(np.mean(endo_array) - endo_norm_mean_target)
        endo_std = abs(np.std(endo_array) - endo_norm_std_target)
    # Else add a penalty for ENDO being fully inside OS
    else:
        endo_in_os_penalty = penalty

    # Checking if ENDO is indside ZONE
    if (endo_in_zone == False):
        endo_out_of_zone_penalty = penalty

    ## ZONE ellipse
    # Checking the ZONE ellipse in not empty
    if (zone_in_os == False) and (zone_in_endo == False) and (zone_image.mask.all() == False):
        # ZONE mean color
        zone_mean = abs(np.mean(zone_array) - zone_norm_mean_target)
        zone_std = abs(np.std(zone_array) - zone_norm_std_target)
    # Else add a penalty
    else:
        if (zone_in_os): zone_in_os_penalty = penalty
        if (zone_in_endo): zone_in_endo_penalty = penalty
        if (zone_in_os == False) and (zone_in_endo == False) and (
            zone_image.mask.all() == True): zone_is_empty_penalty = penalty

    fitness += os_std + endo_std + zone_std
    fitness += os_mean + endo_mean + zone_mean
    fitness += os_out_of_endo_penalty + os_out_of_zone_penalty + endo_out_of_zone_penalty
    fitness += endo_in_os_penalty + zone_in_os_penalty + zone_in_endo_penalty + zone_is_empty_penalty

    # Displaying the results
    if (display_image):
        result_image = outline_individual_on_image(img, individual)

        plt.figure(figsize=(6, 6))
        plt.title("Ellipse outline")
        plt.imshow(result_image), plt.xticks([]), plt.yticks([])

    return fitness, [os_mean, os_std, endo_mean, endo_std, zone_mean, zone_std,
                     os_out_of_endo_penalty, os_out_of_zone_penalty, endo_out_of_zone_penalty,
                     endo_in_os_penalty, zone_in_os_penalty, zone_in_endo_penalty, zone_is_empty_penalty]


# Type 1 Genetic Algorithm
def type1_GA(img, 
             pop_size=population_size,
             r_ratio=retain_ratio, 
             mut_rate=mutation_rate, 
             rnd_rate=random_selection_rate, 
             gen=generations,
             display_image=False):

    # Getting a normalized image from the a channel in the Lab color space
    _, a_image, _ = lab_channels(img, display_image=False)
    a_image, _ = return_z_score(a_image)
    # a_image, _ = return_z_score(img[1,:,:]) # Red colors from image - normalized

    best_grade_history = []
    average_grade_history = []
    best_pop_history = []
    best_elements_history = []

    if (display_image):
        print('Init population - Type 1 GA\n')
        print('Population size: %s\nRetain ratio: %s\nMutation rate: %s\nRandom selection chance: %s\nGenerations: %s' %
             (pop_size, r_ratio, mut_rate, rnd_rate, gen))
        
    pop = init_population(a_image, pop_size)
    pop, grades, elements = grade_type1(a_image, pop)
    best_elements_history.append(elements[0])

    best_grade_history.append(grades[0])
    average_grade_history.append(mean(grades))
    best_pop_history.append(pop[0])

    counter = 1
    best_grade_achieved = False
    no_change_in_grade = 0

    while best_grade_achieved == False:
        pop = evolve(a_image, pop, retain=r_ratio, mutate=mut_rate, random_select=rnd_rate)
        pop, grades, elements = grade_type1(a_image, pop)
        best_elements_history.append(elements[0])
        best_grade_history.append(grades[0])
        average_grade_history.append(mean(grades))
        best_pop_history.append(pop[0])
        counter += 1
        if best_grade_history[-1] == best_grade_history[-2]:
            no_change_in_grade += 1
        else: no_change_in_grade = 0

        if gen > 0:
            if counter == gen: best_grade_achieved = True
        elif no_change_in_grade == abs(gen):
            best_grade_achieved = True

        if (display_image):
            print('\nGeneration %i. There is no change in the best grade for %i generations.' % (counter, no_change_in_grade))
            print('Best grade [-1] = %f, Best grade [-2] = %f' % (best_grade_history[-1], best_grade_history[-2]))
            print(best_elements_history[-1])

    return best_pop_history, best_grade_history, average_grade_history, best_elements_history


# Visualizing the algorithm evolution
def plot_fitness_function_scores(best_score, average_score):
    plt.plot(best_score)
    plt.plot(average_score)
    plt.show()
    return

def plot_elements(fitness, elements):
    e = np.transpose(elements)

    all_elements = range(e.shape[0])

    for i in all_elements:
        plt.plot(e[i], label="n={0}".format(i))
    plt.legend(loc="upper right",
               ncol=2, shadow=True, title="Legend", fancybox=True)

    plt.show()

def sample_algorithm(gen_number=generations, type1_image_id='470'):
    image = get_image_data(type1_image_id, 'Type_1', 0.1)
    plt.imshow(image), plt.xticks([]), plt.yticks([])

    pop1, gra1, avg1, ele1 = type1_GA(image, gen=gen_number, display_image=True)

    aaa = outline_individual_on_image(image, pop1[-1])
    plt.imshow(aaa), plt.xticks([]), plt.yticks([])
    plt.show()

    print(ele1)
    plot_fitness_function_scores(gra1, avg1)
    plot_elements(gra1, ele1)

    # animation from matplotlib examples
    # from matplotlib import animation, rc
    # from IPython.display import HTML

    """
    =================
    An animated image
    =================

    This example demonstrates how to animate an image.
    """
    import matplotlib.animation as animation

    fig = plt.figure()

    im = plt.imshow(image, animated=True)

    def updatefig(i):
        plt.title('Generation %i out of %i' % (i + 1, len(pop1)))
        im.set_array(outline_individual_on_image(image, pop1[i]))
        plt.draw()
        return im,

    ani = animation.FuncAnimation(fig, updatefig, np.arange(0, len(pop1)), interval=300, blit=True)

    plt.show()

    ani.save('Type1GA.mp4', fps=10, writer='ffmpeg', codec='mpeg4', dpi=100)

    # HTML(ani.to_html5_video())
    return

def analyze_individual_color(a_channel, individual, display_image=False):

    os_std = os_mean = endo_std = endo_mean = zone_std = zone_mean = 0

    # Creating ellipses from chromosome and extracting the relevant images
    os_ell = create_ellipse(individual[0], individual[1], individual[2], individual[3], individual[4], edgecolor='red')
    endo_ell = create_ellipse(individual[5], individual[6], individual[7], individual[8], individual[9],
                              edgecolor='red')
    zone_ell = create_ellipse(individual[10], individual[11], individual[12], individual[13], individual[14],
                              edgecolor='red')

    os_image, endo_image, zone_image = extract_ellipses(a_channel, os_ell, endo_ell, zone_ell, display_image)

    os_array = np.ma.MaskedArray.flatten(os_image)
    endo_array = np.ma.MaskedArray.flatten(endo_image)
    zone_array = np.ma.MaskedArray.flatten(zone_image)

    ## OS ellipse
    # Homogenity of OS
    os_std = np.std(os_array)
    # OS mean color, darker is better
    os_mean = np.mean(os_array)

    ## ENDO ellipse
    # Homogenity of ENDO
    endo_std = np.std(endo_array)
    # ENDO mean color, darker is better
    endo_mean = np.mean(endo_array)

    ## ZONE ellipse
    # Homogenity of ZONE
    zone_std = np.std(zone_array)
    # ZONE mean color, darker is better
    zone_mean = np.mean(zone_array)

    return os_mean, os_std, endo_mean, endo_std, zone_mean, zone_std

def return_z_score(img):
    z_image = np.array(img)
    img_mean = np.mean(z_image)
    img_std = np.std(z_image)
    # img_n = np.prod(z_image.shape)

    z_image = (z_image - img_mean ) / img_std
    # z_mean = np.mean(z_image)
    # z_std = np.std(z_image)

    zms_image = np.array(z_image ** 2)

    return z_image, zms_image

def analyze_type1_outlines_from_csv(input_csv_file, output_csv_file):
    # Analyzing the images according to the outlines in the csv file
    print('Input file: ', input_csv_file)
    print('Output file: ', output_csv_file)
    results_type1 = []

    df = pd.read_csv(input_csv_file, index_col='image_id')
    del df['Unnamed: 0']

    for current_image, row in df.iterrows():
        print('Analyzing image id %i' % current_image)
        individual = row.tolist()

        image = get_image_data(current_image, 'Type_1', 0.1)
        _, a_image, _ = lab_channels(image, display_image=False)
        #a_image, _, _ = lab_channels(image, display_image=False)
        #a_image = image[:,:,2]
        z, zms = return_z_score(a_image)

        a_channel_analysis = analyze_individual_color(a_image, individual)
        normalized_analysis = analyze_individual_color(z, individual)
        norm_squared_analysis = analyze_individual_color(zms, individual)

        results_type1.append([current_image] + list(a_channel_analysis)
                             + list(normalized_analysis)
                             + list(norm_squared_analysis))

    # Writing to csv
    df_results_type1_columns = ['image_id', 'color_os_mean', 'color_os_std', 'color_endo_mean',
                                'color_endo_std', 'color_zone_mean', 'color_zone_std', \
                                'norm_os_mean', 'norm_os_std', 'norm_endo_mean', 
                                'norm_endo_std', 'norm_zone_mean', 'norm_zone_std', \
                                'norm_sqr_os_mean', 'norm_sqr_os_std', 'norm_sqr_endo_mean', 
                                'norm_sqr_endo_std', 'norm_sqr_zone_mean', 'norm_sqr_zone_std']
    df_results_type1 = pd.DataFrame(results_type1, columns=df_results_type1_columns)
    df_results_type1.to_csv(output_csv_file)
    print(df_results_type1.head())

    return

def view_type1_outline_from_csv(input_csv_file, type1_image_id=48):
    # Viewing the perfect chromosome for image previously outlined manually
    df = pd.read_csv(input_csv_file, index_col='image_id')
    del df['Unnamed: 0']
    print(df.head())

    image = get_image_data(type1_image_id, 'Type_1', 0.1)
    individual = df.loc[type1_image_id,:]
    print(individual)

    # fitness with G channel, just for checking the fitness function 30.10.17
    z_image, _ = return_z_score(image[:,:,1])
    fitness, elements = type1_fitness(z_image, individual)
    print('Fitness score is: ', fitness)
    print(elements)

    # Plotting the latest image
    plt.subplot(121)
    plt.title('Type 1 Image - %i' % type1_image_id)
    plt.imshow(image)
    image_outline = outline_individual_on_image(image, individual)
    plt.subplot(122)
    plt.title('Image with outline')
    plt.imshow(image_outline)

    # Show in full screen for convenience
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    return

def visualize_type1_outlines_color_analysis(input_csv_file):
    # An easy way to view the manual outlines analysis
    df = pd.read_csv(input_csv_file, index_col='image_id')
    del df['Unnamed: 0']

    print(df.head())

    # Using seaborn for visualization
    color_cols = pd.DataFrame(df, columns=['color_os_mean', 'color_endo_mean', 'color_zone_mean'])
    norm_cols = pd.DataFrame(df, columns=['norm_os_mean', 'norm_endo_mean', 'norm_zone_mean'])

    colors = ["dark red", "red", "pink"]
    sns.set_palette(sns.xkcd_palette(colors))

    # Dist plot + Swarm plot
    if True:
        plt.figure(figsize=(12, 8))
        plt.suptitle(input_csv_file, fontsize=14)

        plt.subplot(221)
        sns.distplot(color_cols.iloc[:,0], rug=True, kde_kws={"shade": True}, hist=False, label='Os mean', axlabel=False)
        sns.distplot(color_cols.iloc[:, 1], rug=True, kde_kws={"shade": True}, hist=False, label='Columnar mean', axlabel=False)
        sns.distplot(color_cols.iloc[:, 2], rug=True, kde_kws={"shade": True}, hist=False, label='Squamous mean', axlabel=False)
        #plt.legend()
        plt.title('Color mean distributions')

        plt.subplot(222)
        sns.distplot(norm_cols.iloc[:, 0], rug=True, kde_kws={"shade": True}, hist=False, label='Os mean', axlabel=False)
        sns.distplot(norm_cols.iloc[:, 1], rug=True, kde_kws={"shade": True}, hist=False, label='Columnar mean', axlabel=False)
        sns.distplot(norm_cols.iloc[:, 2], rug=True, kde_kws={"shade": True}, hist=False, label='Squamous mean', axlabel=False)
        #plt.legend()
        plt.ylim(0,0.9)
        plt.title('Normalized mean distributions')


    # Swarm plot
    if True:
        #plt.figure(figsize=(12, 6))

        plt.subplot(223)
        sns.swarmplot(data=color_cols, orient='h')
        #plt.xticks(rotation=45)
        plt.title('A-channel mean distributions')

        plt.subplot(224)
        sns.swarmplot(data=norm_cols, orient='h')
        plt.yticks([])
        plt.title('Normalized mean distributions')

        plt.show()

    return

def type1_outlines_statistical_analysis(input_csv_file):
    # Statistical summary of csv files
    df = pd.read_csv(input_csv_file, index_col='image_id')
    del df['Unnamed: 0']

    print(df.head())
    print(df.describe())

    # One way ANOVA
    color_f, color_p = stats.f_oneway(df['color_os_mean'], df['color_endo_mean'], df['color_zone_mean'])
    norm_f, norm_p = stats.f_oneway(df['norm_os_mean'], df['norm_endo_mean'], df['norm_zone_mean'])
    norm_sqr_f, norm_sqr_p = stats.f_oneway(df['norm_sqr_os_mean'], df['norm_sqr_endo_mean'], df['norm_sqr_zone_mean'])
    print('\nOne way ANOVA')
    print('Raw color analysis: ', color_f, color_p)
    print('Normalized color analysis: ', norm_f, norm_p)
    print('Squared normalized color analysis: ', norm_sqr_f, norm_sqr_p)

    # Independent t-test for normalized values
    os_endo_t, os_endo_p = stats.ttest_ind(df['norm_os_mean'], df['norm_endo_mean'])
    endo_zone_t, endo_zone_p = stats.ttest_ind(df['norm_endo_mean'], df['norm_zone_mean'])
    os_zone_t, os_zone_p = stats.ttest_ind(df['norm_os_mean'], df['norm_zone_mean'])
    print('\nt-test for normalized values')
    print('Os vs. Endo: ', os_endo_t, os_endo_p)
    print('Endo vs. Zone: ', endo_zone_t, endo_zone_p)
    print('Os vs. Zone: ', os_zone_t, os_zone_p)

    return

def main():

    ### Sample of processes
    sample_algorithm(-10, '346')
    # sample_z_score()

    ### Outlining type 1 images for analysis
    manual_type1_outlines_filname = 'manual_outlines_type1.csv'

    # view_type1_outline_from_csv(input_csv_file=manual_type1_outlines_filname, type1_image_id=346)

    ### Analyzing preoutlined type1 images
    color_analysis_filename = 'manual_outlines_type1_achannel_analysis.csv'
    # color_analysis_filename = 'manual_outlines_type1_color_R_analysis.csv'

    # analyze_type1_outlines_from_csv(input_csv_file=manual_type1_outlines_filname, output_csv_file=color_analysis_filename)
    visualize_type1_outlines_color_analysis(input_csv_file=color_analysis_filename)
    # type1_outlines_statistical_analysis(input_csv_file=color_analysis_filename)

    return

if __name__=="__main__": main()
