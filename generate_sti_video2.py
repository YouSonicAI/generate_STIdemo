
# coding: utf-8



import os

import time
import cv2
import numpy as np
import pandas as pd
from moviepy import *
from moviepy.editor import *
import glob



file = pd.read_csv("/Users/queenie/Documents/generate_STIdemo/Temp/csv/伟贤看天下[2]_STI_result_slice4.csv")
videos_path = "/Users/queenie/Documents/generate_STIdemo/Temp/origin_video/22.mp4"
frames_save_path = "./伟贤看天下22"
#image = frames_save_path
new_video_save_path = "./伟贤看天下22.mp4"
GENERATE_IMAGE = True
SAVE_VIDEO_FRAME = False

if not os.path.exists(frames_save_path):
    os.makedirs(frames_save_path)



if __name__ == "__main__":
    lst = [x for x in file['STI']]
    temp = [x for x in lst if pd.isnull(x) == False]
    sti_value = ['%.2f'%x  for x in  temp]
    dem_str = ["STI" +  ' ' +x  for x in sti_value]

    my_audio_clip = AudioFileClip(videos_path)

    vidcap = cv2.VideoCapture(videos_path)
    fps = vidcap.get(5)
    #size = (352,640)
    size = (int(vidcap.get(3)),int(vidcap.get(4)))
    print("fps:",fps,"vidsize:",size,"")
    #success, image = vidcap.read()
    time_interval = 1
    #success, image = vidcap.read()
    success = True


    count = 0
    if GENERATE_IMAGE == True:
        while success:
            success, image = vidcap.read()
            count += 1
            start = time.time()

            if count % time_interval == 0 :
                try:
                    if os.path.exists(os.path.join(frames_save_path,"frame%d.jpg" %(count))):
                        continue
                    else:
                        cv2.imencode('.jpg', image)[1].tofile(os.path.join(frames_save_path,"frame%d.jpg" %(count)))
                except:
                    continue

    image_lst = os.listdir(frames_save_path)

    b = 1
    d= 0
    j = 0
    MP4_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(new_video_save_path, MP4_format, fps, size)
    for i in range(len(image_lst)):
        image_path = os.path.join(frames_save_path,'frame%d.jpg' %(d+1))
        d = d + 1
        text = dem_str[j]
        try:
            src = cv2.imread(image_path)
        except:
            continue
        temp = cv2.putText(src, text, (25,40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 2)
        STRIDE = 2
        bar_start_x = 10
        bar_start_y = 50
        bar_height = 20

        bad_thresh = 30
        poor_thresh = 45
        fair_thresh = 60
        good_thresh = 75
        excellent_thresh = 100

        bar_bad = bar_start_x + 30 * STRIDE
        bar_poor = bar_start_x + 45 * STRIDE
        bar_fair = bar_start_x + 60 * STRIDE
        bar_good = bar_start_x + 75 * STRIDE
        bar_excellent = bar_start_x + 100 * STRIDE
        sti_value = float(text.split(" ")[-1])
        sti_value_large_100 = 100.0 * sti_value

        if sti_value_large_100 < bad_thresh:
            temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_start_x + sti_value_large_100 * STRIDE , bar_height + bar_start_y), (128, 0, 128),
                                 thickness=-1)
        elif bad_thresh <= sti_value_large_100 and sti_value_large_100 <= poor_thresh:
            temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_bad, bar_height + bar_start_y), (128, 0, 128),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_bad, bar_start_y), (int(bar_bad + (sti_value_large_100-bad_thresh) * STRIDE), bar_height + bar_start_y), (0, 0, 255),
                                 thickness=-1)
            # print((bar_bad, bar_start_y))
            # print((bar_bad + sti_value_large_100 * STRIDE, bar_height + bar_start_y))
            # cv2.imshow("origin",temp)
            # cv2.waitKey()

        elif poor_thresh <= sti_value_large_100 and sti_value_large_100 <= fair_thresh:
            temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_bad, bar_height + bar_start_y), (128, 0, 128),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_bad, bar_start_y), (bar_poor, bar_height + bar_start_y), (0, 0, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_poor, bar_start_y), (bar_poor + int((sti_value_large_100-poor_thresh) * STRIDE), bar_height + bar_start_y), (0, 69, 255),
                                 thickness=-1)

        elif fair_thresh <= sti_value_large_100 and sti_value_large_100 <=  good_thresh:
            temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_bad, bar_height + bar_start_y), (128, 0, 128),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_bad, bar_start_y), (bar_poor, bar_height + bar_start_y), (0, 0, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_poor, bar_start_y), (bar_fair, bar_height + bar_start_y), (0, 69, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_fair, bar_start_y), (bar_fair + int((sti_value_large_100-fair_thresh) * STRIDE), bar_height + bar_start_y), (0, 255, 255),
                                 thickness=-1)
        else:
            temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_bad, bar_height + bar_start_y), (128, 0, 128),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_bad, bar_start_y), (bar_poor, bar_height + bar_start_y), (0, 0, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_poor, bar_start_y), (bar_fair, bar_height + bar_start_y), (0, 69, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_fair, bar_start_y), (bar_good, bar_height + bar_start_y), (0, 255, 255),
                                 thickness=-1)
            temp = cv2.rectangle(temp, (bar_good, bar_start_y), (bar_good + int((sti_value_large_100-good_thresh) * STRIDE), bar_height + bar_start_y), (0, 255, 0),
                                 thickness=-1)

        # temp = cv2.rectangle(temp, (bar_start_x, bar_start_y), (bar_bad, bar_height + bar_start_y), (128,0,128), thickness=-1)
        # temp = cv2.rectangle(temp, (bar_bad, bar_start_y), (bar_poor, bar_height + bar_start_y), (0,0,255), thickness=-1)
        # temp = cv2.rectangle(temp, (bar_poor, bar_start_y), (bar_fair, bar_height + bar_start_y), (0,69,255), thickness=-1)
        # temp = cv2.rectangle(temp, (bar_fair, bar_start_y), (bar_good, bar_height + bar_start_y), (0,255,255), thickness=-1)
        # temp = cv2.rectangle(temp, (bar_good, bar_start_y), (bar_excellent, bar_height + bar_start_y), (0,255,0), thickness=-1)

        # indicator_x = bar_start_x + sti_value * 100
        # indicator_y = bar_start_y + 0.5 * bar_height

        #temp = cv2.circle(temp, (int(indicator_x),int(indicator_y)), int(0.5 * bar_height), (0,0,0), thickness=-1)

        # cv2.imshow("temp",temp)
        # cv2.waitKey()
        if not src is None and SAVE_VIDEO_FRAME==True:
            cv2.imwrite(os.path.join(frames_save_path,"frame%d.png" %(d)),src)
        #cv2.imwrite(r'C:\Users\17579\Desktop\Yousonic\STI_Video_Demo\Self\Image\down\frame%d.jpg'%d,src)
        b = b + 1
        if b%30 == 0 :
            j = j + 1
            #print("11111")
        video.write(src)
    video.release()

    video = VideoFileClip(new_video_save_path)
    video = video.set_audio(my_audio_clip)
    # video.write_videofile(new_video_save_path.split("/")[-1].split(".")[0]+"_audio.mp4",\
    #                       codec='png',audio_codec='aac',temp_audiofile='temp-audio.m4a',remove_temp=True)
    video.write_videofile(new_video_save_path.split("/")[-1].split(".")[0] + "_audio.mp4", \
                          codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)




    # image_lst = os.listdir(frames_save_path)
    #
    #
    #
    #
    #
    #
    #
    # for i in range(len(image_lst)):
    #     item = os.path.join(frames_save_path,'frame%d.png' %(i))
    #     pictur = cv2.imread(item)
    #



