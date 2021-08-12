# -*- coding: utf-8 -*-

try:
    import cv2
    import numpy as np
except Exception as e:
    print("Need to install third party modules by pip in DetectMotion!", e)

import ast
import threading
from datetime import datetime, timedelta

from utils.Utils import Utils
from utils.JsonManipulation import JsonManipulation
from utils.ClassifierClientCloud import ClassifierClientCloud
from utils.DigitalImageProcessing import DigitalImageProcessing

import configparser
configClass = configparser.RawConfigParser()
configClass.read('default_config/config_init.properties') 
CLASSIFICATION_CLOUD = bool(ast.literal_eval(configClass.get('DataClassifiers', 'CLASSIFICATION_CLOUD')))


class DetectMotion:
    """Class responsible for motion detection with classifiers"""
    def __init__(self, name_stream, name_analytical, setLight, blur, smallSize, largSize, objects, hashCamera):
        
        # init analytical parameters
        self.config_classifier = JsonManipulation().load_classifier_api()['detectMotion']
        self.accuracy = self.config_classifier['Accuracy']

        self.hashCamera= hashCamera
        self.box_alias = "box"
        self.name_stream = name_stream
        self.name_analytical = name_analytical
      
        self.setLight = setLight 
        self.blur = blur
      
        self.smallSize = smallSize 
        self.largSize = largSize 
       
        self.objects = objects
        
        self.utils = Utils()
        self.pdi = DigitalImageProcessing()
        self.client_cloud = ClassifierClientCloud(self.name_stream, self.name_analytical, "teste", self.objects, self.config_classifier)
        #-------------------------------
        
        # area analytical coordinates
        self.x, self.y, self.w, self.h = 0, 0, 0, 0
        self.pts = None
        self.control = True
        #-----------------------
      
        # attributes of the background subtractor
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # self.fgbg =cv2.createBackgroundSubtractorKNN()
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.firstFrame = None
        #-------------------------------------------
      
        # quanto mais proximo de 100%, mais o objeto detectado tem que ter de area em comum com a região desenhada
        # do analitico, ou seja, menos sensível a alarme.
        self.INTERSECTION_PERCENTAGE = 50
      
        # counter used to block the generation of events for a certain time
        self.tolerance = 8
        self.startTime = datetime.now()
        self.finalTime = datetime.now() + timedelta(seconds=self.tolerance)
        self.ctrl_obj_block = False
        self.crtl_detect = False
        #--------------------------------------------
        
        self.th_predict = None
        self.motion = ""       
        #--------------------

        self.step_flow = 0
        # --------------------

        self.bag = []

        # Alarm control

        self.last_alarm_state = []
        self.object_iou_threshold = 0.6
        self.memory_duration = 60 # tempo sem confirmacao em segundo

    def crop_polygonal_shape(self, roi):
        if self.control:                
            axis_x = self.pts[:, 0] - self.x
            axis_y = self.pts[:, 1] - self.y
            self.pts[:, 0] = axis_x
            self.pts[:, 1] = axis_y
            self.control = False
        
        mask = np.zeros(roi.shape, dtype=np.uint8)
        
        # cv2.imshow("mask", mask)
        roi_corners = [self.pts]
         
        # Fill in the ROI so it will not be deleted when the mask is applied
        channel_count = roi.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
       
        # aplly mask
        roi = cv2.bitwise_and(roi, mask)
        # cv2.imshow("roi", roi)
         
        return roi
    
    
    def set_coordinates(self, pts):
        self.pts = np.array(pts, np.int32)
        (self.x, self.y, self.w, self.h) = self.utils.retangle_points(self.pts)
        # print (self.x, self.y, self.w, self.h)
                
    def send_event(self, motion):
        # get_time = time.strftime("%H:%M:%S") + " " + time.strftime("%d/%m/%Y")
        get_time = datetime.now()
        if motion == "moveanyPolygon":
            motion = {'CameraId': int(self.name_stream), 'AnalyticId':int(self.name_analytical), 'value': str(motion), 'Date': get_time, 'EventTypeId': 0}


            try:
                step = self.step_flow
                event = motion.copy()
                timestamp = event['Date'].strftime('%Y-%m-%d %H:%M:%S')
                event.update({'Date': timestamp})

            except Exception as e:
                pass

            self.motion = ""

            self.update_set_flow()
        else:
            motion = ""
                        
        return motion

    def update_set_flow(self):
        self.step_flow += 1

    def thread_predict(self, roi_clas, motion_polygon):
        
        have_obj = 0
        self.ctrl_obj_block = True
        if CLASSIFICATION_CLOUD == True:
            try:
                step = self.step_flow

                self.bag = self.client_cloud.predict_objs(roi_clas, self.box_alias, self.name_stream, self.name_analytical, step)


                trans_pts = np.array([[self.x + p[0], self.y + p[1]] for p in self.pts])
                have_obj = self.utils.is_objects(self.objects, self.accuracy, self.INTERSECTION_PERCENTAGE, trans_pts, self.bag, self.name_stream,
                                                         self.name_analytical, step)
            except Exception as e:
                print("Error trying to classify image!", e)
        if have_obj >= 1:
            has_alarm = self.update_memory(self.bag)

            self.finalTime = datetime.now() + timedelta(seconds=self.tolerance)

            if has_alarm:
                print(self.last_alarm_state)
                self.motion = motion_polygon
            else:
                print("Blocking Alarm")
                self.motion = ""
        else:
            self.motion = ""
            self.ctrl_obj_block = False

    def update_memory(self, objects):
        current_time = datetime.now()
        deploy_alarm = False

        if self.last_alarm_state:

            temp_memory = []
            id_loop_control = []

            for obj in objects:

                new_alarm = True

                for idx, old_obj in enumerate(self.last_alarm_state):

                    if (current_time - old_obj["created"]).seconds >= self.memory_duration:
                        print("Removing bbox")
                        continue

                    if self.calculate_iou(obj["bb_o"], old_obj["bb_o"]) >= self.object_iou_threshold and obj["label"] == old_obj["label"]:
                        x, y, w, h = obj["bb_o"]
                        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        new_state_item = {
                            "bb_o": old_obj["bb_o"],
                            "label": old_obj["label"],
                            "created": current_time
                        }

                        old_obj["created"] = current_time
                        new_alarm = False
                        if idx not in id_loop_control:
                            temp_memory.append(new_state_item)
                            id_loop_control.append(idx)

                    else:
                        if idx not in id_loop_control:
                            temp_memory.append(old_obj)
                            id_loop_control.append(idx)

                if new_alarm:
                    temp_obj = {
                        "bb_o": obj["bb_o"],
                        "label": obj["label"],
                        "created": current_time
                    }
                    temp_memory.append(temp_obj)

                    deploy_alarm = True

            print(">> Switching state", len(objects), len(self.last_alarm_state), len(temp_memory))
            self.last_alarm_state = temp_memory

        else:
            deploy_alarm = True

            for obj in objects:

                temp_obj = {
                    "bb_o": obj["bb_o"],
                    "label": obj["label"],
                    "created": current_time
                }

                self.last_alarm_state.append(temp_obj)


        return deploy_alarm

    
    def apply_video_analytics(self, frame): 
        # print self.last_alarm_state
        motion_polygon = "moveanyPolygon"
        process_frame = frame.copy()
        self.frame = frame
        roi = self.frame[self.y:self.h, self.x:self.w]
        roi_draw = roi
        
        #-------------------------------------------------------------------------------------
        roi = self.crop_polygonal_shape(roi)
            
        roi_hsv = self.pdi.convert_hsv_brighten(roi)        
        gray = cv2.cvtColor(roi_hsv, cv2.COLOR_BGR2GRAY)
        gaus = self.pdi.apply_config(gray, self.setLight, self.blur)
        self.firstFrame, fgmask = self.pdi.appply_bg_subtractor(self.fgbg, self.firstFrame, gaus)
        fgmask = self.pdi.apply_morphological_op(fgmask)
        fgmask = self.pdi.remove_blob(fgmask)
        cv2.imshow("fgmask", fgmask)
        #-------------------------------------------------------------------------------------
        
        # control of the time to detect another object
        if self.ctrl_obj_block == True:
            self.startTime = datetime.now()
        if self.startTime > self.finalTime:
            self.ctrl_obj_block = False
        #-------------------------------------------------------------------------------------
        
        # find the outline of the image on thresh
        (_, cnts, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        # accumulates the breached areas            
        for c in cnts:    
            # when the bounding is small discard
            if not(cv2.contourArea(c) >= self.smallSize and cv2.contourArea(c) <= self.largSize):
                continue
            
            # calculates the bounding area of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # draw a rectangle around the calculated area
            cv2.rectangle(roi_draw, (x, y), (x + w, y + h), (000, 255, 204), 1)
            self.crtl_detect = True
            
        if self.crtl_detect == True:
            self.crtl_detect = False
            
            nothing_selected = self.utils.nothing_selected(self.objects)
            if nothing_selected == True:
                self.motion = motion_polygon
            else:
                if self.ctrl_obj_block == False:            
                    try:
                        th_status = self.th_predict.isAlive()
                    except:
                        th_status = False
                        
                    if th_status == False:
                        self.th_predict = threading.Thread(target=self.thread_predict, args=(process_frame, motion_polygon,))
                        self.th_predict.start()
                    #------------------------------------------------------------------------
    
        #-------------------------------------------------------------------------------------
        pts = self.pts.reshape((-1, 1, 2))
        cv2.polylines(roi_draw, [pts], False, (000, 000, 255), 2)
        self.frame[self.y:self.h, self.x:self.w] = roi_draw

        for class_obj in self.bag:
            x, y, w, h = class_obj["bb_o"]
            label = class_obj["label"]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(self.frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        for state_obj in self.last_alarm_state:
            x, y, w, h = state_obj["bb_o"]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return self.motion
    
    def set_vis(self, vis):
        self.frame = vis
    
    def get_vis(self):
        return self.frame

    def calculate_iou(self, bbox_A, bbox_B):

        xa, ya, wa, ha = bbox_A
        xb, yb, wb, hb = bbox_B

        area_a = wa * ha
        area_b = wb * hb

        xmin_a, ymin_a, xmax_a, ymax_a = xa, ya, xa + wa, ya + ha
        xmin_b, ymin_b, xmax_b, ymax_b = xb, yb, xb + wb, yb + hb

        inter_xmin = max(xmin_a, xmin_b)
        inter_ymin = max(ymin_a, ymin_b)

        inter_xmax = min(xmax_a, xmax_b)
        inter_ymax = min(ymax_a, ymax_b)

        inter_w = max(inter_xmax - inter_xmin, 0)
        inter_h = max(inter_ymax - inter_ymin, 0)

        area_inter = inter_w * inter_h

        area_union = area_a + area_b - area_inter

        iou = float(area_inter)/float(area_union)

        return iou