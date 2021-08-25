import os
import cv2

video_name = '/back_and_forth.mkv'
dataset_path = '/home/NEA.com/wen.yu.chien/git/datasets/NEA/Nardo_BnF'
image_path = dataset_path + '/images'
index_path = image_path + '/index'
query_path = image_path + '/query'

try:

    # creating a folder
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    # creating a folder
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # creating a folder
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    # creating a folder
    if not os.path.exists(query_path):
        os.makedirs(query_path)


except OSError:
    print ('Error: Creating directory of data')

vid = cv2.VideoCapture(dataset_path + video_name)

frame_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_half = int(frame_total/2)
curr_frame = 0
frame_step = 30
frame_count = 0

print (f"Total/Half Frames: {frame_total}/{frame_half}")

ret = True

while ret:

    ret, frame = vid.read()

    if ret:
        if curr_frame <= frame_half:
            name = f"{index_path}/{frame_count:04d}.jpg"
            print (f"[Video Frame #{curr_frame}] Creating...{name}")
        
        elif curr_frame > frame_half:
            name = f"{query_path}/{frame_count:04d}.jpg"
            print (f"[Video Frame #{curr_frame}] Creating...{name}")
    
        cv2.imwrite(name, frame)
        
        frame_count += 1
        
    curr_frame += frame_step
    vid.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

print ('Done')

vid.release()
cv2.destroyAllWindows()