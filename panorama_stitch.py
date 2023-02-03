import cv2
import numpy as np
import matplotlib.pyplot as plt


def linearBlending(imgs):
    '''
    linear Blending(also known as Feathering)
    '''
    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype="int")
    img_right_mask = np.zeros((hr, wr), dtype="int")
    
    # find the left image and right image mask region(Those not zero pixels)
    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img_left[i, j]) > 0:
                img_left_mask[i, j] = 1
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img_right[i, j]) > 0:
                img_right_mask[i, j] = 1
    
    # find the overlap mask(overlap region of two image)
    overlap_mask = np.zeros((hr, wr), dtype="int")
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
    
    # Plot the overlap mask
    plt.figure(21)
    plt.title("overlap_mask")
    plt.imshow(overlap_mask.astype(int), cmap="gray")
    
    # compute the alpha mask to linear blending the overlap region
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
    for i in range(hr): 
        minIdx = maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j
        
        if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
            continue
            
        decrease_step = 1 / (maxIdx - minIdx)
        for j in range(minIdx, maxIdx + 1):
            alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
    
    
    
    linearBlending_img = np.copy(img_right)
    linearBlending_img[:hl, :wl] = np.copy(img_left)
    # linear blending
    for i in range(hr):
        for j in range(wr):
            if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
    
    return linearBlending_img

def linearBlendingWithConstantWidth(imgs):
    '''
    linear Blending with Constat Width, avoiding ghost region
    # you need to determine the size of constant with
    '''
    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype="int")
    img_right_mask = np.zeros((hr, wr), dtype="int")
    constant_width = 3 # constant width
    
    # find the left image and right image mask region(Those not zero pixels)
    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img_left[i, j]) > 0:
                img_left_mask[i, j] = 1
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img_right[i, j]) > 0:
                img_right_mask[i, j] = 1
                
    # find the overlap mask(overlap region of two image)
    overlap_mask = np.zeros((hr, wr), dtype="int")
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
    
    # compute the alpha mask to linear blending the overlap region
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
    for i in range(hr):
        minIdx = maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j
        
        if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
            continue
            
        decrease_step = 1 / (maxIdx - minIdx)
        
        # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
        middleIdx = int((maxIdx + minIdx) / 2)
        
        # left 
        for j in range(minIdx, middleIdx + 1):
            if (j >= middleIdx - constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 1
        # right
        for j in range(middleIdx + 1, maxIdx + 1):
            if (j <= middleIdx + constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 0

    
    linearBlendingWithConstantWidth_img = np.copy(img_right)
    linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
    # linear blending with constant width
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(overlap_mask[i, j]) > 0):
                linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
    
    return linearBlendingWithConstantWidth_img


# Import image
left = cv2.imread('res/1.jpg')
right = cv2.imread('res/2.jpg')
# cv2.imshow('left', left)
# cv2.imshow('right', right)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Feature detector and matching
orb = cv2.ORB_create(nfeatures=200,
                     scaleFactor = 1.5)

kp_left, des_left = orb.detectAndCompute(left, None)
kp_right, des_right = orb.detectAndCompute(right, None)

keypoints_drawn_left = cv2.drawKeypoints(left, kp_left, None, color=(0, 0, 255))
keypoints_drawn_right = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255))

# cv2.imshow('left keypoint', keypoints_drawn_left)
# cv2.imshow('right keypoint', keypoints_drawn_right)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Method1: Brutal force to find matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_left,des_right)

# Find the best matches
limit = 15
good = sorted(matches, key = lambda x:x.distance)[:limit]
best_matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, good, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Best Match', best_matches_drawn)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Method2: FLANN
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des_left,des_right,k=2)

# good = []
# for m_n in matches:
#     if len(m_n) != 2:
#         continue
#     (m,n) = m_n
#     if m.distance < 0.6*n.distance:
#         good.append(m)

# matches_drawn = cv2.drawMatches(left, kp_left, right, kp_right, good, None, matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow('Match', matches_drawn)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Perspective tranformation
left_pts = []
right_pts = []
for m in good:
    l = kp_left[m.queryIdx].pt
    r = kp_right[m.trainIdx].pt
    left_pts.append(l)
    right_pts.append(r)

M, _ = cv2.findHomography(np.float32(right_pts), np.float32(left_pts))
print(M)
dim_x = left.shape[1] + right.shape[1]
dim_y = max(left.shape[0], right.shape[0])
dim = (dim_x, dim_y)

warped = cv2.warpPerspective(right, M, dim)
cv2.imshow('Wraped Right', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Merge
# Method1: stitch directly
comb = warped.copy()
# combine the two images
comb[0:left.shape[0],0:left.shape[1]] = left
# crop
r_crop = 1920
comb = comb[:, :r_crop]
cv2.imshow('stitched', comb)

# Method2: blend two image 
stitch_img = linearBlendingWithConstantWidth([left, warped])
cv2.imshow('blend', stitch_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
