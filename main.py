from pathlib import Path
import numpy as np
import argparse
import time
import os

import torch.backends.cudnn as cudnn
import torch
import cv2

from ffpyplayer.player import MediaPlayer
from emotion import detect_emotion, init

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import threading

# Create a flag to indicate whether the video has completed
video_completed = False

def getVideoSource(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# Define a function to display the video
def display_video():
    global video_completed
    filmpath = opt.filmpath

    camera = getVideoSource(filmpath, 720, 480)
    player = MediaPlayer(filmpath)

    while True:
        ret, frame = camera.read()
        audio_frame, val = player.get_frame()
        if not ret:
            video_completed = True
            break

        frame = cv2.resize(frame, (720, 480))
        cv2.imshow('Camera', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'): # Set speed of video from here
            break

    camera.release()
    # cv2.destroyAllWindows()

# Create a thread to display the video
video_thread = threading.Thread(target=display_video)

# Clear the previous data in 'file.txt'
with open('file.txt', 'w') as file:
    file.write('')

# Define a global variable to keep track of frame counts
frame_count = 0

def detect(opt):
    global frame_count
    source, livefootage, playfilm, view_img, imgsz, nosave, show_conf, save_path,  frame_skip = opt.source, opt.livefootage, opt.playfilm, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path,  opt.frame_skip

    # Directories
    create_folder(save_path)

    # Initialize
    set_logging()
    previous_minute=0
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if source == '0':
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = ((0, 52, 255), (121, 3, 195), (176, 34, 118), (87, 217, 255), (69, 199, 79), (233, 219, 155), (203, 139, 77), (214, 246, 255))

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    if source == '0' and playfilm == 1:
        if not video_thread.is_alive():  # Start the video thread only if it's not already running
            video_thread.start()

    for path, img, im0s, vid_cap in dataset:
        frame_count += 1  # Increment frame count
        if frame_count % frame_skip != 0:  # Skip frames according to frame_skip value
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 (Normalization)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if source == '0':  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Predict emotions for each face
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    face = im0[int(y1):int(y2), int(x1):int(x2)]
                    if face.size > 0:  # Ensure there's a valid face
                        emotions = detect_emotion([face], show_conf)
                        if emotions:
                            with open('file.txt', 'a') as file:
                                e = emotions[0][0]
                                emo = e.split(' (')
                                # calculate and display fps
                                current_minutes=int(((time.time()-t0)/15)+1)#divide by 60 for 1 min interval, divide by 15 for 15 sec interval
                                if previous_minute!=current_minutes:
                                    previous_minute=current_minutes
                                    file.write(f"{current_minutes}\n")

                                file.write(f"{emotions[0][0]}\n")
                    if view_img:  # Display the video frame with bounding boxes
                        label = emotions[0][0]
                        colour = colors[emotions[0][1]]
                        plot_one_box(xyxy, im0, label=label, color=colour, line_thickness=opt.line_thickness)

        # Stream results
        if view_img:
            if source == '0' and livefootage == 1 and playfilm == 0:  # Display the video frame for webcam only
                display_img = cv2.resize(im0, (400, 300))
                cv2.imshow("Emotion Detection", display_img)
                cv2.waitKey(1)  # 1 millisecond
            elif source == '0' and livefootage == 0 and playfilm == 1:  # Display the video frame for webcam only
                if video_completed:
                    video_thread.join()
                    return
            elif source == '0' and livefootage == 1 and playfilm == 1:
                display_img = cv2.resize(im0, (400, 300))
                cv2.imshow("Emotion Detection", display_img)
                cv2.waitKey(1)  # 1 millisecond
                if video_completed:
                    video_thread.join()
                    return

        if not nosave:
            ext = save_path.split(".")[-1]
            if ext in ["mp4", "avi"]:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
            elif ext in ["bmp", "pbm", "pgm", "ppm", "sr", "ras", "jpeg", "jpg", "jpe", "jp2", "tiff", "tif", "png"]:
                cv2.imwrite(save_path, im0)
            else:
                output_path = os.path.join(save_path, os.path.split(path)[1])
                create_folder(output_path)
                cv2.imwrite(output_path, im0)

        # if show_fps:
        #     print(f"FPS: {1 / (time.time() - t0):.2f}" + " " * 5, end="\r")
        #     t0 = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # Use '0' for webcam, or provide the path to your video file
    parser.add_argument('--livefootage', type=int, default=1, help='Display live footage to the user')  # use 1 if you want to show the user his video, 0 otherwise.
    parser.add_argument('--playfilm', type=int, default=1, help='Display film to the user')  # Use 1 if you want the user to see the film, 0 otherwise.
    parser.add_argument('--filmpath', type=str, default='video.mp4', help='Provide Path of Film here.')
    parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='face confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    save = parser.add_mutually_exclusive_group()
    save.add_argument('--output-path', default="output.mp4", help='save location')
    save.add_argument('--no-save', action='store_false', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=5, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    #parser.add_argument('--show-fps', default=False, action='store_true', help='print fps to console')
    parser.add_argument('--frame-skip', type=int, default=10, help='skip frames for video processing')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        detect(opt=opt)
