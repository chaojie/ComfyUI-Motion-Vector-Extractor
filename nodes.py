import sys
import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

import tempfile
import shutil
import subprocess
import json

class MotionVectorExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default":""}),
                "frame_length": ("INT", {"default": 2}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motionbrush",)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run_inference"
    CATEGORY = "Motion Brush"

    def run_inference(self,video_url,frame_length):
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
                for mv in np.split(motion_vectors, num_mvs):
                    trajs.append([[int(mv[0, 3]), int(mv[0, 4])],[int(mv[0, 5]), int(mv[0, 6])]])

                trajslist.append(json.dumps(trajs))

            # store motion vectors, frames, etc. in output directory


            step += 1
            if step==frame_length:
                break

        cap.release()

        return (trajslist,)

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

        tf = tempfile.NamedTemporaryFile()
        filename = tf.name

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

        filename = filename + '.mp4'

        res = subprocess.run(args_mp4 + [filename], input=images.tobytes(), capture_output=True)
        print(res.stderr)

        return (filename,)

NODE_CLASS_MAPPINGS = {
    "Motion Vector Extractor":MotionVectorExtractor,
    "VideoCombineThenPath":VideoCombineThenPath,
}
