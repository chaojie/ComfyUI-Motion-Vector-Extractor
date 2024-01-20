import sys
import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

import tempfile
import folder_paths
import shutil
import subprocess
import json
import numpy as np

class MotionVectorExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default":""}),
                "frame_length": ("INT", {"default": 2}),
                "width": ("INT", {"default": 320}),
                "height": ("INT", {"default": 576}),
                "scale": ("INT", {"default": 4}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motionbrush",)
    FUNCTION = "run_inference"
    CATEGORY = "Motion Brush"

    def run_inference(self,video_url,frame_length,width,height,scale):
        from mvextractor.videocap import VideoCap
        cap = VideoCap()

        # open the video file
        ret = cap.open(video_url)

        if not ret:
            raise RuntimeError(f"Could not open {video_url}")

        step = 0
        times = []
        trajslist = []

        # continuously read and display video frames and motion vectors
        while True:
            trajs=[]
            tstart = time.perf_counter()

            # read next video frame and corresponding motion vectors
            ret, frame, motion_vectors, frame_type, timestamp = cap.read()

            tend = time.perf_counter()
            telapsed = tend - tstart
            times.append(telapsed)

            # if there is an error reading the frame
            if not ret:
                print("No frame read. Stopping.")
                break

            if len(motion_vectors) > 0:
                num_mvs = np.shape(motion_vectors)[0]
                
                ptlist=[]
                if len(trajslist)>0:
                    ptlist=np.array(trajslist[0])[:,-1].tolist()
                for mv in np.split(motion_vectors, num_mvs):
                    (x1,y1,x2,y2)=(int(mv[0, 3]), int(mv[0, 4]),int(mv[0, 5]), int(mv[0, 6]))
                    if x1>width-1:
                        x1=width-1
                    if y1>height-1:
                        y1=height-1
                    if x2>width-1:
                        x2=width-1
                    if y2>height-1:
                        y2=height-1
                    if x1<1:
                        x1=1
                    if y1<1:
                        y1=1
                    if x2<1:
                        x2=1
                    if y2<1:
                        y2=1
                    if abs(x2-x1)+abs(y2-y1)>scale:
                        trajs.append([[x1,y1],[x2,y2]]) 
                    '''
                    if len(trajslist)>0:
                        if [int(mv[0, 3]), int(mv[0, 4])] in ptlist:
                            trajs.append([[int(mv[0, 3]), int(mv[0, 4])],[int(mv[0, 5]), int(mv[0, 6])]])
                    else:
                       trajs.append([[int(mv[0, 3]), int(mv[0, 4])],[int(mv[0, 5]), int(mv[0, 6])]]) 
                    '''

                trajslist.append(trajs)

            # store motion vectors, frames, etc. in output directory


            step += 1
            if step==frame_length:
                break

        cap.release()

        revert_trajs=[]
        plist=[]
        #print(f'{trajslist}')
        for itrajslist in range(len(trajslist)):
            trajs=trajslist[itrajslist]
            if itrajslist==0:
                revert_trajs=trajs
                ptlistfirst=np.array(revert_trajs)[:,1]
                ptlist=(ptlistfirst/scale).astype('int').tolist()
            else:
                #print(f'{revert_trajs}')
                #ptlist=np.array(revert_trajs)[:,0].tolist()
                for traj in trajs:
                    scaletraj0=[int(traj[0][0]/scale),int(traj[0][1]/scale)]
                    scaletraj1=[int(traj[1][0]/scale),int(traj[1][1]/scale)]
                    if scaletraj0 in ptlist:
                        ptind=ptlist.index(scaletraj0)
                        ptlist[ptind]=scaletraj1
                        #print(f'x1:{revert_trajs[ptind][-1][0]},y1:{revert_trajs[ptind][-1][1]},x2:{traj[1][0]},y2:{traj[1][1]}')
                        #if abs(traj[1][0]-revert_trajs[ptind][-1][0])+abs(traj[1][1]-revert_trajs[ptind][-1][1])>5:
                        if int(traj[1][0]/scale)==int(revert_trajs[ptind][-1][0]/scale) and int(traj[1][1]/scale)==int(revert_trajs[ptind][-1][1]/scale):
                            revert_trajs[ptind].append(traj[1])
            '''
            trajs=trajslist[len(trajslist)-1-itrajslist]
            print(f'{plist}')
            if itrajslist==0:
                revert_trajs=trajs
                ptlist=np.array(revert_trajs)[:,0].tolist()
            else:
                print(f'{revert_trajs}')
                #ptlist=np.array(revert_trajs)[:,0].tolist()
                for traj in trajs:
                    if traj[1] in ptlist:
                        ptind=ptlist.index(traj[1])
                        ptlist[ptind]=traj[0]
                        revert_trajs[ptind].insert(0,traj[0])
            '''
            
        return (json.dumps(revert_trajs),)

ffmpeg_path = shutil.which("ffmpeg")

class VideoCombineThenPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 14, "min": 1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    CATEGORY = "Motion Brush"
    FUNCTION = "execute"

    def execute(
        self,
        images,
        frame_rate: int,
    ):
        format="video/h264-mp4"
        # convert images to numpy
        images = images.cpu().numpy() * 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)

        format_type, format_ext = format.split("/")

        comfy_path = os.path.dirname(folder_paths.__file__)
        #tf = tempfile.NamedTemporaryFile()
        filename = f'{comfy_path}/output/mvexrator.mp4'

        if ffmpeg_path is None:
            print("no ffmpeg path")

        dimensions = f"{len(images[0][0])}x{len(images[0])}"
        print("image dimensions: ", dimensions)

        args_mp4 = [
            ffmpeg_path, 
            "-v", "error", 
            "-f", "rawvideo", 
            "-pix_fmt", "rgb24",
            "-s", dimensions,
            "-r", str(frame_rate), 
            "-i", "-",
            "-crf", "20",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p"
        ]

        #filename = filename

        res = subprocess.run(args_mp4 + [filename], input=images.tobytes(), capture_output=True)
        print(res.stderr)

        return (filename,)

NODE_CLASS_MAPPINGS = {
    "Motion Vector Extractor":MotionVectorExtractor,
    "VideoCombineThenPath":VideoCombineThenPath,
}
