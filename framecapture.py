import os
from os import path
import cv2
import pandas as pd

videos = {}


def cut_frame(car, video, count, image):
    # for car in frame:
    pointsX = [int(car['UpperPointShortX']), int(car['UpperPointCornerX']), int(car['UpperPointLongX']),
               int(car['CrossCornerX']), int(car['ShortSideX']), int(car['CornerX']), int(car['LongSideX']),
               int(car['LowerCrossCornerX'])]
    pointsY = [int(car['UpperPointShortY']), int(car['UpperPointCornerY']), int(car['UpperPointLongY']),
               int(car['CrossCornerY']), int(car['ShortSideY']), int(car['CornerY']), int(car['LongSideY']),
               int(car['LowerCrossCornerY'])]
    minX = min(pointsX)
    minY = min(pointsY)
    maxX = max(pointsX)
    maxY = max(pointsY)
    if maxY - minY < 70 or maxX - minX < 70:  # remove small pics
        return

    # image = show_circles(image, car)

    crop_image = image[minY:maxY, minX:maxX]

    video_name = str(((os.path.splitext(video)[0]).split("/"))[-1])
    car_id = int(car['car_id'])
    if str(video_name)[1] == "A":
        file_name = "capt/A/" + str(video_name) + "_id_" + str(car_id) + "_frame_" + str(count) + ".jpg"
    else:
        file_name = "capt/B/" + str(video_name) + "_id_" + str(car_id) + "_frame_" + str(count) + ".jpg"

    try:
        cv2.imwrite(file_name, crop_image)  # save frame as JPEG file
    except:
        return

    if video in videos:
        videos[video] = videos[video] + 1
    else:
        videos[video] = 1


def get_count(file_name):
    # fix offset so annotation is correct
    if file_name[-5] == "A":
        return 1
    elif file_name[-5] == "B":
        return 3
    else:
        return 0


def video_capt(df, video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = get_count(video)
    while success:

        if (df['frame'] == count).any():

            frame = df.loc[df['frame'] == count]

            if len(frame) == 1:
                cut_frame(frame, video, count, image)
            else:
                for index, row in frame.head(n=2).iterrows():
                    cut_frame(row, video, count, image)

        success, image = vidcap.read()

        if not success:
            break

        count += 1


def show_circles(image, car):
    image = cv2.circle(image, (int(car['UpperPointShortX']), int(car['UpperPointShortY'])), radius=10,
                       color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(car['UpperPointCornerX']), int(car['UpperPointCornerY'])), radius=10,
                       color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(car['UpperPointLongX']), int(car['UpperPointLongY'])), radius=10, color=(0, 0, 255),
                       thickness=-1)
    image = cv2.circle(image, (int(car['CrossCornerX']), int(car['CrossCornerY'])), radius=10, color=(0, 0, 255),
                       thickness=-1)
    image = cv2.circle(image, (int(car['ShortSideX']), int(car['ShortSideY'])), radius=10, color=(0, 0, 255),
                       thickness=-1)
    image = cv2.circle(image, (int(car['CornerX']), int(car['CornerY'])), radius=10, color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(car['LongSideX']), int(car['LongSideY'])), radius=10, color=(0, 0, 255),
                       thickness=-1)
    image = cv2.circle(image, (int(car['LowerCrossCornerX']), int(car['LowerCrossCornerY'])), radius=10,
                       color=(0, 0, 255), thickness=-1)

    return image


def main():
    if not path.exists("./capt"):
        os.mkdir("./capt")
    if not path.exists("./capt/A"):
        os.mkdir("./capt/A")
    if not path.exists("./capt/B"):
        os.mkdir("./capt/B")

    for filename in os.listdir("dataset/video_shots/"):
        if filename.endswith(".mov"):
            file_name = (os.path.join("dataset/video_shots/", filename))
            csv_name = "dataset/video_shots/" + str(os.path.splitext(filename)[0]) + "_annotations.csv"
            df = pd.read_csv(csv_name)
            print("Start processing video: " + file_name)
            video_capt(df, file_name)

    for video, cnt in videos.items():
        print(video, cnt)


if __name__ == "__main__":
    main()
