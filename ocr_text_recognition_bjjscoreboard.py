"""
Created on Fri May 29, 2020 3:23:56 2020

@author: neongreen13

python ocr_text_recognition_bjjscoreboard.py

"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# text recognition modules
from tesserocr import PyTessBaseAPI
import pytesseract

# image modules
import cv2
from PIL import Image
import PIL.Image

import glob
from os import getcwd
import os
from natsort import natsorted
import sys
from os.path import isfile

### /home/neongreen13/Desktop/vids/Worlds_2019/

path = '/home/neongreen13/Desktop/Worlds_2019/'
print ("Folders path set to: " + path)

# OCR Functions

def ocr_digits(filename):
    """
    Returns digits only.
    """
    # only detect numbers
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(PIL.Image.open(filename), config=custom_config)
    return text

def get_folders():
    directory_list = list()
    for root, dirs, files in os.walk('/home/neongreen13/Desktop/Worlds_2019/', topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    directory_list = sorted(directory_list)
    df_directory_list = pd.DataFrame(directory_list)
    df_directory_list.to_csv(path + 'directory_list.csv',index=False)

    return

def folder_names():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df = pd.DataFrame(directory_list,columns =['Match_Names'])
    df['Match_Names'] = df.Match_Names.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
    df['Match_Names'] = df['Match_Names'].str.replace(' ','')
    Match_List = sorted(df.Match_Names.tolist())
    df_Match_List = pd.DataFrame(Match_List)
    df_Match_List.to_csv(path + 'Match_List.csv',index=False)

    return

def timestamp_list():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df_Match_List = pd.read_csv(path + 'Match_List.csv')
    Match_List = sorted(df_Match_List['0'].tolist())

    for folder,match in list(zip(directory_list,Match_List)):
        image_list = sorted(glob.glob(folder + '/*.png'))
        df = pd.DataFrame(image_list,columns =['Image_Names'])
        df.Image_Names = df.Image_Names.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
        df[['extra1','Image_Names']] = df["Image_Names"].str.split("/", n = 1, expand = True)
        df.drop(columns=['extra1'],inplace=True)
        df['TimeStamp'] = df.Image_Names
        df['TimeStamp'] = df.TimeStamp.str.replace('.png','')
        df.to_csv(folder + '/TimeStamp_' + str(match) + '.csv',index=False)

    return

def crop_scoreboard():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df_Match_List = pd.read_csv(path + 'Match_List.csv')
    Match_List = sorted(df_Match_List['0'].tolist())

    for folder,match in list(zip(directory_list,Match_List)):

        pic_list = sorted(glob.glob(folder + '/*.png'))
        df = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        image_list1 = sorted(df.Image_Names.tolist())

        for pic, name in list(zip(pic_list,image_list1)):
            img = cv2.imread(pic)
            crop_img = img[2:30,0:30]

            scale_percent = 1500 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            image_save_path = folder + '/top_score/'
            if os.path.isdir(image_save_path) == False:
                os.mkdir(image_save_path)

            cv2.imwrite(image_save_path + str(name),resized)

        for pic, name in list(zip(pic_list,image_list1)):
            img = cv2.imread(pic)
            crop_img = img[32:59,0:30]

            scale_percent = 1500 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            image_save_path = folder + '/bottom_score/'
            if os.path.isdir(image_save_path) == False:
                os.mkdir(image_save_path)

            cv2.imwrite(image_save_path + str(name),resized)

        for pic, name in list(zip(pic_list,image_list1)):
            img = cv2.imread(pic)
            crop_img = img[2:30,35:65]

            scale_percent = 1500 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            image_save_path = folder + '/top_penadv/'
            if os.path.isdir(image_save_path) == False:
                os.mkdir(image_save_path)

            cv2.imwrite(image_save_path + str(name),resized)

        for pic, name in list(zip(pic_list,image_list1)):
            img = cv2.imread(pic)
            crop_img = img[32:59,35:65]

            scale_percent = 1500 # percent of original size
            width = int(crop_img.shape[1] * scale_percent / 100)
            height = int(crop_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

            image_save_path = folder + '/bot_penadv/'
            if os.path.isdir(image_save_path) == False:
                os.mkdir(image_save_path)

            cv2.imwrite(image_save_path + str(name),resized)

    return

def augment_data():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df_Match_List = pd.read_csv(path + 'Match_List.csv')
    Match_List = sorted(df_Match_List['0'].tolist())

    for folder,match in list(zip(directory_list,Match_List)):

        top_score_list = sorted(glob.glob(folder + '/top_score/*.png'))
        df1 = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        image_list2 = sorted(df1.Image_Names.tolist())

        for pic, name in list(zip(top_score_list,image_list2)):
            img = cv2.imread(pic,0)
            # thesh blur
            ret,thresh1 = cv2.threshold(img,180,300,cv2.THRESH_BINARY)
            blur = cv2.blur(thresh1,(5,5))
            # save
            image_save_path = folder + '/top_score/'

            cv2.imwrite(image_save_path + str(name),blur)

    for folder,match in list(zip(directory_list,Match_List)):

        bottom_score_list = sorted(glob.glob(folder + '/bottom_score/*.png'))
        df2 = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        image_list3 = sorted(df2.Image_Names.tolist())

        for pic, image in list(zip(bottom_score_list,image_list3)):
            img = cv2.imread(pic,0)
            # step 3 - thesh blur threshold again
            ret,thresh1 = cv2.threshold(img,180,300,cv2.THRESH_BINARY_INV)
            blur = cv2.blur(thresh1,(5,5))
            # ret,thresh2 = cv2.threshold(blur,180,300,cv2.THRESH_BINARY)
            # blur2 = cv2.blur(thresh2,(5,5))
            # save
            image_save_path = folder + '/bottom_score/'

            cv2.imwrite(image_save_path + str(image),blur)

    for folder,match in list(zip(directory_list,Match_List)):

        toppen_list = sorted(glob.glob(folder + '/top_penadv/*.png'))
        df3 = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        image_list4 = sorted(df3.Image_Names.tolist())

        for pic, image in list(zip(toppen_list,image_list4)):
            img = cv2.imread(pic,0)
            # step 3 - thesh blur threshold again
            ret,thresh1 = cv2.threshold(img,180,300,cv2.THRESH_BINARY)
            blur = cv2.blur(thresh1,(5,5))
            # ret,thresh2 = cv2.threshold(blur,180,300,cv2.THRESH_BINARY)
            #
            # ret,thresh3 = cv2.threshold(thresh2,180,300,cv2.THRESH_BINARY)
            # blur2 = cv2.blur(thresh3,(5,5))
            # ret,thresh4 = cv2.threshold(blur2,180,300,cv2.THRESH_BINARY)
            # save
            image_save_path = folder + '/top_penadv/'

            cv2.imwrite(image_save_path + str(image),blur)

    for folder,match in list(zip(directory_list,Match_List)):

        botpen_list = sorted(glob.glob(folder + '/bot_penadv/*.png'))
        df4 = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        image_list5 = sorted(df4.Image_Names.tolist())

        for pic, image in list(zip(botpen_list,image_list5)):
            img = cv2.imread(pic,0)
            # step 3 - thesh blur threshold again
            ret,thresh1 = cv2.threshold(img,180,300,cv2.THRESH_BINARY)
            blur = cv2.blur(thresh1,(5,5))
            # ret,thresh2 = cv2.threshold(blur,180,300,cv2.THRESH_BINARY)
            #
            # ret,thresh3 = cv2.threshold(thresh2,180,300,cv2.THRESH_BINARY)
            # blur2 = cv2.blur(thresh3,(5,5))
            # ret,thresh4 = cv2.threshold(blur2,180,300,cv2.THRESH_BINARY)
            # save
            image_save_path = folder + '/bot_penadv/'

            cv2.imwrite(image_save_path + str(image),blur)

    return

def ocr_scoreboard_text():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df_Match_List = pd.read_csv(path + 'Match_List.csv')
    Match_List = sorted(df_Match_List['0'].tolist())

    for folder in directory_list:

        top_score_list2 = sorted(glob.glob(folder + '/top_score/*.png'))
        df_topscore = pd.DataFrame(columns=["Image","Top_Score"])

        for text, name in zip(top_score_list2, top_score_list2):
            text = ocr_digits(text)
            df_topscore = df_topscore.append({"Image":name, "Top_Score":text},ignore_index=True)

        df_topscore['Image'] = df_topscore.Image.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
        df_topscore[['extra1','Image']] = df_topscore["Image"].str.split("/", n = 1, expand = True)
        df_topscore[['extra2','Image']] = df_topscore["Image"].str.split("/", n = 1, expand = True)
        df_topscore.drop(columns=['extra1','extra2',],inplace=True)
        df_topscore.to_csv(folder + '/' + 'df_topscore.csv',index=False)

        toppen_list2 = sorted(glob.glob(folder + '/top_penadv/*.png'))
        df_toppen = pd.DataFrame(columns=["Image","Top_Pen_Adv"])

        for text, name in zip(toppen_list2, toppen_list2):
            text = ocr_digits(text)
            df_toppen = df_toppen.append({"Image":name, "Top_Pen_Adv":text},ignore_index=True)

        df_toppen['Image'] = df_toppen.Image.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
        df_toppen[['extra1','Image']] = df_toppen["Image"].str.split("/", n = 1, expand = True)
        df_toppen[['extra2','Image']] = df_toppen["Image"].str.split("/", n = 1, expand = True)
        df_toppen.drop(columns=['extra1','extra2',],inplace=True)
        df_toppen.to_csv(folder + '/' + 'df_toppen.csv',index=False)

        bottom_score_list2 = sorted(glob.glob(folder + '/bottom_score/*.png'))
        df_bottomscore = pd.DataFrame(columns=["Image","Bottom_Score"])

        for text, name in zip(bottom_score_list2, bottom_score_list2):
            text = ocr_digits(text)
            df_bottomscore = df_bottomscore.append({"Image":name, "Bottom_Score":text},ignore_index=True)

        df_bottomscore['Image'] = df_bottomscore.Image.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
        df_bottomscore[['extra1','Image']] = df_bottomscore["Image"].str.split("/", n = 1, expand = True)
        df_bottomscore[['extra2','Image']] = df_bottomscore["Image"].str.split("/", n = 1, expand = True)
        df_bottomscore.drop(columns=['extra1','extra2',],inplace=True)
        df_bottomscore.to_csv(folder + '/' + 'df_bottomscore.csv',index=False)


        botpen_list2 = sorted(glob.glob(folder + '/bot_penadv/*.png'))
        df_botpen = pd.DataFrame(columns=["Image","Bot_Pen_Adv"])

        for text, name in zip(botpen_list2, botpen_list2):
            text = ocr_digits(text)
            df_botpen = df_botpen.append({"Image":name, "Bot_Pen_Adv":text},ignore_index=True)

        df_botpen['Image'] = df_botpen.Image.str.replace('/home/neongreen13/Desktop/Worlds_2019/','')
        df_botpen[['extra1','Image']] = df_botpen["Image"].str.split("/", n = 1, expand = True)
        df_botpen[['extra2','Image']] = df_botpen["Image"].str.split("/", n = 1, expand = True)
        df_botpen.drop(columns=['extra1','extra2',],inplace=True)
        df_botpen.to_csv(folder + '/' + 'df_botpen.csv',index=False)

    return


def master_df():

    df_directory_list = pd.read_csv(path + 'directory_list.csv')
    directory_list = sorted(df_directory_list['0'].tolist())
    df_Match_List = pd.read_csv(path + 'Match_List.csv')
    Match_List = sorted(df_Match_List['0'].tolist())

    for folder,match in list(zip(directory_list,Match_List)):

        master_df = pd.read_csv(folder + '/TimeStamp_' + str(match) + '.csv')
        master_df.rename(columns={'Image_Names':'Image'},inplace=True)
        master_df['Match'] = match

        df_topscore = pd.read_csv(folder + '/df_topscore.csv')
        master_df = pd.merge(master_df,df_topscore, on=['Image'])

        df_bottomscore = pd.read_csv(folder + '/df_bottomscore.csv')
        master_df = pd.merge(master_df,df_bottomscore, on=['Image'])

        df_toppen = pd.read_csv(folder + '/df_toppen.csv')
        master_df = pd.merge(master_df,df_toppen, on=['Image'])

        df_botpen = pd.read_csv(folder + '/df_botpen.csv')
        master_df = pd.merge(master_df,df_botpen, on=['Image'])

        master_df.to_csv(folder + '/' + str(match) + '_Master_Scorecard.csv',index=False)

    data = []

    for folder,file in list(zip(directory_list,Match_List)):
        df = pd.read_csv(folder + '/' + str(file) + '_Master_Scorecard.csv')

        data.append(df)

    final_data = pd.concat(data, axis=0, ignore_index=True)
    final_data[['Top_Advantage','Top_Penalty']] = final_data["Top_Pen_Adv"].str.split("\n", n = 1, expand = True)
    final_data[['Bottom_Advantage','Bottom_Penalty']] = final_data["Bot_Pen_Adv"].str.split("\n", n = 1, expand = True)
    # final_data.Bottom_Penalty = final_data.Bottom_Penalty.str.replace('11','0')
    final_data.to_csv(path + 'Combined_Scoreboard_Data_Worlds2019.csv',index=False)

    return

def main(path):

    get_folders()
    folder_names()
    timestamp_list()
    crop_scoreboard()
    augment_data()
    ocr_scoreboard_text()
    master_df()

    return True

if __name__ == "__main__":
    try:
        finished = main(path)
    except FileNotFoundError:
        print("Error in code.")
        exit()
    if finished:
        print("\nDone.")
