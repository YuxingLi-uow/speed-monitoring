# Speed-monitoring
This repository uses YOLOv3 to detect cars and calculate the car speed by optical flow

# RUN
Simply run command below to detect cars and monitor car speeds. Change video path and mask manually before run the python file. 

        python speed_control_encapsulated.py 

The mask provided only works for two videos('video_01.mp4' and 'video_02.mp4') in the repository. Other videos with different scenarios need different masks. You may run mask.py file to create a new mask photo by providing a point array.

# PERFORMANCE
This python file uses a trained YOLOv3 weights from COCO dataset. Both YOLOv3 and YOLOv3-tiny are tested. 

Results below are GPU accelerated. 

GPU: RTX2070

RESULTS:
    
        YOLOv3: FPS=20
    
        YOLOv3 + optical flow: FPS=10

        tiny YOLOv3: FPS=30
    
        tiny YOLOv3 + optical flow: FPS=15
    
# Screenshots
![image](https://user-images.githubusercontent.com/45188716/143729724-3d50c091-e46c-4587-948d-7989de5ed1d4.png)
![image](https://user-images.githubusercontent.com/45188716/143729805-9cfd1809-29c6-4cbd-926a-5913399cb9b4.png)
