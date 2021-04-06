import cv2
import pandas as pd 
import os
import csv


def cutframe(car,video,count,image):
     # for car in frame:
    start_point = (int(car['UpperPointLongX']),int(car['UpperPointLongY']))
    end_point =  (int(car['ShortSideX']),int(car['ShortSideY']))
    image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    video_name = str(((os.path.splitext(video)[0]).split("/"))[-1])
    car_id = int(car['car_id'])
    file_name = "capt/" + str(video_name)+ "_id_" + str(car_id) + "_frame_" + str(count) + ".jpg"
    try:
        cv2.imwrite(file_name, image)     # save frame as JPEG file      
    except:
        print("ERROR")
    print("New image captured",count)
    # except:
    #     print("ops")

def video_capt(df,video):
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 3 #fix offset so anotation is correct
    while success:

        if((df['frame'] == count).any()):

            
            frame = df.loc[df['frame'] == count]
            # try:
            
            if(len(frame)==1):
                cutframe(frame,video,count,image)
            else:
                print(frame.iloc(0))
                cutframe(frame.iloc(0),video,count,image)
                

                
           
            
        success,image = vidcap.read()
        
        if(not success):
            break

        #print('Read a new frame: ', success)
        count += 1


def main():

    for filename in os.listdir("dataset/video_shots/"):
        if filename.endswith(".mov"):
            file_name = (os.path.join("dataset/video_shots/", filename))
            csv_name = "dataset/video_shots/" + str(os.path.splitext(filename)[0]) + "_annotations.csv"
            df = pd.read_csv(csv_name)
            video_capt(df,file_name)

if __name__ == "__main__":
    main()
