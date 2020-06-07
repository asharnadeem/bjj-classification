import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

# Directory where all the videos are stored
path = '/home/asharnadeem/Desktop/Worlds_2019/'
filelist=os.listdir(path)

# Go through all .mp4 videos
for file in filelist[:]:
    
    if file.endswith(".mp4"):
        print('Analyzing', file,'...')
		
        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()

        # frames per second, time stamp, and array of images
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0.0
        timestamp = 0
        score_changes = []
        
        # Create and navigate to the folder where we will keep the videos
        vid_name = file.split('.')
        os.mkdir(path + 'Score Changes v3/' + vid_name[0])
        os.chdir(path + 'Score Changes v3/' + vid_name[0] + '/')

        # Capture 1 frame per second
        while True:
            
            # Checks if the video has ended
            ret, frame = cap.read()

            if not ret:
                break
            sky = frame[622:681, 310:376]
            
            # Get the time stamp to the nearest second and use that image
            timestamp = frame_num/fps
            rounded = str(timestamp).split('.')
            
            # Run script every second
            if rounded[1][0] == '0':
                plt.imsave(str(timestamp) + '.png', sky, dpi=300)
                
                # Initial screen capture
                if len(score_changes) == 0:
                    score_changes.append(timestamp)
                    frame_num += 1
                    continue

                img1 = Image.open(str(score_changes[len(score_changes)-1]) + '.png')
                img2 = Image.open(str(timestamp) + '.png')

                # External function to compare similarity of images
                s = 0
                for band_index, band in enumerate(img1.getbands()):
                    m1 = np.array([p[band_index] for p in img1.getdata()]).reshape(*img1.size)
                    m2 = np.array([p[band_index] for p in img2.getdata()]).reshape(*img2.size)
                    s += np.sum(np.abs(m1-m2))
                
                # Detect whether the score changed or not and act accordingly
                if s > 75000:
                    score_changes.append(timestamp)
                else:
                    os.remove(str(timestamp) + '.png')

            # # Show the video if we want to watch it while its occuring
            # if cv2.waitKey(1) & 0xFF == ord('q') or ret==False:
            # 	cap.release()
            # 	cv2.destroyAllWindows()
            # 	break
            # cv2.imshow('frame',frame)
            
            frame_num += 1
        
        # Go back to directory for next video
        print('Done analyzing', file, '...\n')
        os.chdir(path)
