import cv2
import pandas as pd 

def video_capt(df,video):
    vidcap = cv2.VideoCapture('sample.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        print(count) 
        if(count == 0):
            cv2.imwrite("capt/frame%d.jpg" % count, image)     # save frame as JPEG file      
            print("New image captured",count)
            
        success,image = vidcap.read()
        
        if(not success):
            break

        print('Read a new frame: ', success)
        count += 1


def main():
    df = pd.read_csv('table.csv')
    print(df)

    video_capt(df,'sample.mp4')

if __name__ == "__main__":
    main()
