#!/usr/bin/env python
from __future__ import division
from flask import Flask, render_template, Response
import dlib
import cv2
import sys
import time
import numpy as np
from imutils import face_utils
import pandas as pd
import argparse
import imutils
from collections import OrderedDict
global checkcvipcamera,imagecp,checkcv
checkcvipcamera = 0
imagecp = 0
checkcv = 0

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1

def get_frame():
    camera_port=0
    ramp_frames=100
    # camera_port = "http://admin:0123456789@10.41.122.72/video/mjpg.cgi"
    name_video = 'video/ipainA2d_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.avi'
    camera = cv2.VideoCapture(camera_port) #this makes a web cam object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    out = cv2.VideoWriter(name_video,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    i=1

    while True:

        checkcv = 0
        retval, image = camera.read()
        retval1, im1 = camera.read()
        im = imutils.resize(image, width=500)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            name_jpg = 'images/ipainA2d_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg'
            # retval, im = camera.read()
            # retval1, im1 = camera.read()
            if retval == False:
                print('Failed to camera. Check camera index in cv2.VideoCapture(0)')
                break
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            (x, y, w, h) = rect_to_bb(rect)
            # cv2.rectangle(im, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
            cv2.rectangle(im, (x-10, y-2), ((x+w)+20, (y+h)+15), (0, 255, 0), 2)

            #------------------------------------------------------------------------------------------

            cropped = im[y-2:(y+h)+15,x-10:(x+w)+20]
            cropped1 = im[y-2:(y+h)+15,x-10:(x+w)+20]
            cropped = cv2.resize(cropped,(150,150),interpolation=cv2.INTER_AREA)
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            rects_cropped = detector(cropped_gray,1)
            for (icrop, rects_croppeds) in enumerate(rects_cropped):

                shape_cropped = predictor(cropped_gray, rects_croppeds)
                shape_cropped = shape_to_np(shape_cropped)
                (c_x, c_y, c_w, c_h) = rect_to_bb(rects_croppeds)
                for (c_x,c_y) in shape_cropped:
                    cv2.circle(cropped, (c_x,c_y),1,(0,0,255),-1)
            # cv2.imwrite(name_jpg,cropped)
            cv2.imwrite(name_jpg,cropped1)
            #------------------------------------------------------------------------------------------
            # show the face number
            cv2.putText(im, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(im, (x, y), 2, (0, 0, 255), -1)


        out.write(im1)
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        if checkcv == 1:
            checkcv = 0
            del(camera)
            del(image)
            camera.release()
            out.release()
            cv2.destroyAllWindows()

            break;
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        i+=1
        del(image)
    # while True:
    #     global checkcv
    #     checkcv = 0
    #     retval, im = camera.read()
    #     retval1, im1 = camera.read()
    #     if retval == False:
    #         print('Failed to camera. Check camera index in cv2.VideoCapture(0)')
    #         break
    #     im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     im_resized = resize(im_grey, width=150)
    #     dets = detector(im_resized, 1)
    #     if len(dets) > 0:
    #         for k, d in enumerate(dets):
    #             shape = predictor(im_resized, d)
    #             shape = shape_to_np(shape)
    #             for (x, y) in shape:
    #                 cv2.circle(im, (int(x/ratio), int(y/ratio)), 1, (255, 255, 255), -1)
    #             cv2.rectangle(im, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)
    #     out.write(im1)
    #
    #     imgencode=cv2.imencode('.jpg',im)[1]
    #     stringData=imgencode.tostring()
    #     if checkcv == 1:
    #         checkcv = 0
    #         camera.release()
    #         out.release()
    #         cv2.destroyAllWindows()
    #         break;
    #     yield (b'--frame\r\n'
    #         b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    #     i+=1
    # del(camera)

def get_frame1():
    # camera_port=0


    ramp_frames=100
    camera_port = "http://admin:0123456789@10.41.122.72/video/mjpg.cgi"
    name_video = 'video/ipainA2d_L_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.avi'
    camera = cv2.VideoCapture(camera_port) #this makes a web cam object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    out1 = cv2.VideoWriter(name_video,fourcc, 30, (frame_width,frame_height))
    i=1

    while True:
        imagecp = 0
        checkcvipcamera = 0
        retval, image = camera.read()
        retval1, im1 = camera.read()
        im = imutils.resize(image, width=500)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            name_jpg = 'images/ipainA2d_ipcamera_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg'
            # retval, im = camera.read()
            # retval1, im1 = camera.read()
            if retval == False:
                print('Failed to camera. Check camera index in cv2.VideoCapture(0)')
                break
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            (x, y, w, h) = rect_to_bb(rect)
            # cv2.rectangle(im, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
            cv2.rectangle(im, (x-10, y-2), ((x+w)+20, (y+h)+15), (0, 255, 0), 2)

            #------------------------------------------------------------------------------------------

            cropped = im[y-2:(y+h)+15,x-10:(x+w)+20]
            cropped1 = im[y-2:(y+h)+15,x-10:(x+w)+20]
            cropped = cv2.resize(cropped,(150,150),interpolation=cv2.INTER_AREA)
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            rects_cropped = detector(cropped_gray,1)
            for (icrop, rects_croppeds) in enumerate(rects_cropped):

                shape_cropped = predictor(cropped_gray, rects_croppeds)
                shape_cropped = shape_to_np(shape_cropped)
                (c_x, c_y, c_w, c_h) = rect_to_bb(rects_croppeds)
                for (c_x,c_y) in shape_cropped:
                    cv2.circle(cropped, (c_x,c_y),1,(0,0,255),-1)
            # cv2.imwrite(name_jpg,cropped)
            cv2.imwrite(name_jpg,cropped1)
            #------------------------------------------------------------------------------------------
            # show the face number
            cv2.putText(im, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
            if imagecp == 1:
                imagecp = 0
                print("imagecp")
                name_jpg_ = 'images/ipainA2d_C_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg'
                cv2.imwrite(name_jpg_,im1)


        out1.write(im1)

        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        if checkcvipcamera == 1:
            checkcvipcamera = 0
            del(camera)
            del(image)
            camera.release()
            out1.release()
            cv2.destroyAllWindows()

            break;

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        i+=1
        del(image)
    # while True:
    #     global checkcv
    #     checkcv = 0
    #     retval, im = camera.read()
    #     retval1, im1 = camera.read()
    #     # image = imutils.resize(im, width=500)
    #     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     rects = detector(gray, 1)
    #     for (i, rect) in enumerate(rects):
    #         name_video = 'ipainA2d_Ipcamera_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.avi'
    #         retval, im = camera.read()
    #         retval1, im1 = camera.read()
    #         if retval == False:
    #             print('Failed to camera. Check camera index in cv2.VideoCapture(0)')
    #             break
        #     shape = predictor(gray, rect)
        #     shape = shape_to_np(shape)
        #     (x, y, w, h) = rect_to_bb(rect)
        #     cv2.rectangle(image, (x-30, y), ((x+w)+50, (y+h)+70), (0, 255, 0), 2)
        #     #------------------------------------------------------------------------------------------
        #
        #     cropped = image[y:(y+h)+70,x-30:(x+w)+40]
        #     cropped = cv2.resize(cropped,(150,150),interpolation=cv2.INTER_AREA)
        #     cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        #     rects_cropped = detector(cropped_gray,1)
        #     for (icrop, rects_croppeds) in enumerate(rects_cropped):
        #
        #         shape_cropped = predictor(cropped_gray, rects_croppeds)
        #         shape_cropped = shape_to_np(shape_cropped)
        #         (c_x, c_y, c_w, c_h) = rect_to_bb(rects_croppeds)
        #         for (c_x,c_y) in shape_cropped:
        #             cv2.circle(cropped, (c_x,c_y),1,(0,0,255),-1)
        #     # dfr.to_csv('landmark.csv')
        #     # cv2.imshow("cropped",cropped)
        #     cv2.imwrite(image_path,cropped)
        #     #------------------------------------------------------------------------------------------
        #     # show the face number
        #     cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     # loop over the (x, y)-coordinates for the facial landmarks
        #     # and draw them on the image
        #     for (x, y) in shape:
        #         cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        #     out1.write(image)
        #
        #     imgencode=cv2.imencode('.jpg',image)[1]
        #     stringData=imgencode.tostring()
        #     if checkcvipcamera == 1:
        #         checkcvipcamera = 0
        #         camera.release()
        #         out1.release()
        #         cv2.destroyAllWindows()
        #         break;
        #     yield (b'--frame\r\n'
        #         b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        #     i+=1
        # del(camera)

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ip_camera')
def ip_camera():
    return Response(get_frame1(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    print("stop")
    checkcv = 1
    checkcvipcamera = 1
    return "stop"
    # yield (b'--frame\r\n'
    #     b'Content-Type: text/plain\r\n\r\n\r\n')

@app.route('/crop')
def image():
    print("image")
    imagecp = 1
    camera_port = "http://admin:0123456789@10.41.122.72/video/mjpg.cgi"
    camera = cv2.VideoCapture(camera_port) #this makes a web cam object
    retval, image_crop = camera.read()
    name_jpg_ = 'images/ipainA2d_Crop_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg'
    cv2.imwrite(name_jpg_,image_crop)
    return "image"
    # yield (b'--frame\r\n'
    #     b'Content-Type: text/plain\r\n\r\n\r\n')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
