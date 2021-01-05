import numpy as np
import os
import math 
from tqdm import tqdm
import torch

def region_check(A, B):
    #check above, bottom, right, left
    left, right, upper, bottom = 0,0,0,0
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B
    if Ax1 >= Bx2:
        left = 1
        return left, right, upper, bottom
    elif Ax2 <= Bx1:
        right = 1
        return left, right, upper, bottom 
    elif Ay2 <= By1:
        bottom = 1
        return left, right, upper, bottom 
    elif Ay1 >= By2:
        upper = 1
        return left, right, upper, bottom 


def geometry_cacluation(A, B): 

    #（Ax, Ay）is left upper coordinates and (Bx, By) is the right bottom coordinates
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    left, right, upper, bottom = 0,0,0,0
    DC,EC,PO,EQ,TPP,NTPP,TPPi,NTPPi = 0,0,0,0,0,0,0,0

    # check equal:
    if Ax1 == Bx1 and Ax2 == Bx2 and Ay1 == By1 and Ay2 == By2:
        EQ = 1
        return np.array([left, right, upper, bottom, DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi])
    else:
        # check whether intersection:
        flag = 0 # two rectangule don't intersect
        Xmax = max(Ax1, Bx1)
        Ymax = max(Ay1, By1)
        M = (Xmax, Ymax)
        Xmin = min(Ax2, Bx2)
        Ymin = min(Ay2, By2)
        N = (Xmin, Ymin)
    
        if M[0] < N[0] and M[1] < N[1]:
            # intersection
            if M == (Ax1, Ay1) and N == (Ax2, Ay2):
                if Ax1 == Bx1 or Ay1 == By1 or Ax2 == Bx2 or By1 == By2:
                    TPP = 1
                else:
                    NTPP = 1
                return np.array([left, right, upper, bottom, DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi])
            elif  M == (Bx1, By1) and N == (Bx2, By2):
                if Ax1 == Bx1 or Ay1 == By1 or Ax2 == Bx2 or By1 == By2:
                    TPPi = 1
                else:
                    NTPPi = 1
                return np.array([left, right, upper, bottom, DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi])
            else:
                PO = 1
                return np.array([left, right, upper, bottom, DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi])
        else:
            # check left, right, upper, bottom
            if M[0] == N[0] or M[1] == N[1]:
                EC = 1
            else:
                DC = 1
            left, right, upper, bottom = region_check(A, B)
            return np.array([left, right, upper, bottom, DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi])

def get_heading_degree():
    new_headings = []
    new_elevation = [30*math.pi/180, 0*math.pi/180, -30*math.pi/180]
    for i in range(0, 360, 30):
        current_radians = i*math.pi/180
        new_headings.append(current_radians)
    return new_headings, new_elevation





if __name__ == "__main__":  
    relative_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/new_pre-trained/"
    generate_path = '/VL/space/zhan1624/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/pre-trained1'
    all_scenery_list = os.listdir(relative_path)
    all_scenery_list = [i[:-4] for i in all_scenery_list]
   
    all_scenery = os.listdir(relative_path)
    standard_heading, standard_elevation = get_heading_degree()

    all_scenery_list = all_scenery_list[80:85]
    for scenry in all_scenery_list: # each in 61 scans; scenery is the first index in 5
        temp_feature = []# collect all 1 features
        scenery_nump = np.load(relative_path + scenry + ".npy", allow_pickle=True)
        new_scenry = scenery_nump.item()
        for each_state, value in tqdm(new_scenry.items()):
            new_value = {}
            for each_elevation in standard_elevation:
                for each_heading in standard_heading:  
                    all_box_array = np.zeros((36,36,12)) 
                    image_features = value[str(each_heading)+"_"+str(each_elevation)]['features']
                    box_dets = value[str(each_heading)+"_"+str(each_elevation)]['boxes']
                    for id_i, each_box_x in enumerate(box_dets):
                        for id_j, each_box_j in enumerate(box_dets):
                            tmp_feat_list = []
                            obj_real_feat = geometry_cacluation(each_box_x, each_box_j) 
                            all_box_array[id_i][id_j][0:12] = obj_real_feat
                    value[str(each_heading)+"_"+str(each_elevation)]['relation'] = torch.from_numpy(all_box_array)
        np.save("pre-trained1/"+scenry+".npy", new_scenry)