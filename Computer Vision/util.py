import cv2
import numpy as np

class RectDrawer(object):
    def __init__(self, img):
        self.img = img
        self.window_start = (30, 30)
        
        self.disp_size_min = 900
        w, h = self.img.shape[:2][::-1]
        if(h>w):
            disp_h = int(h/w*self.disp_size_min)
            disp_w = int(self.disp_size_min)
        else:
            disp_w = int(w/h*self.disp_size_min)
            disp_h = int(self.disp_size_min)
        self.disp_shape =(disp_w, disp_h)
        
        self.current = (0, 0)
        self.point = (0, 0)
        
        self.rects = []
        self.done = False
        
        self.RECT_COLOR = (0, 0, 255)
        self.WORKING_LINE_COLOR = (127, 127, 127)

    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done:
            return
        if (event == cv2.EVENT_MOUSEMOVE) and (self.point != (0,0)):
            self.current = (x, y)
            return
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.point == (0,0):
                self.point = (x, y)
                return
            else:
                width = x - self.point[0]
                height = y - self.point[1]
                self.rects.append([self.point[0], self.point[1], width, height])
                print("Adding rect #%d with position[(%d, %d),(%d, %d)]" % (len(self.rects), self.point[0], self.point[1], x, y))
                self.point=(0,0)
                self.current=(0,0)
            
    def set_rect(self):
        window_name = "Set Rect"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name,*self.disp_shape)
        cv2.moveWindow(window_name, *self.window_start)
        cv2.setMouseCallback(window_name, self.on_mouse)

        self.done =False
        self.current = (0, 0)
        self.point = (0, 0)
        self.rects = []
        while(not self.done):
            canvas = self.img.copy()
            if (len(self.rects) > 0):
                for x,y,w,h in self.rects: 
                    cv2.rectangle(canvas, (x,y), (x+w, y+h), (0,0,255), 2)
            if(self.point!=(0,0) and self.current !=(0,0)):
                cv2.line(canvas, self.point, self.current, self.WORKING_LINE_COLOR, 12)
                
            cv2.imshow(window_name, canvas)
            
            if cv2.waitKey(10) == ord('q'):
                print("set template with %d rectangle." % len(self.rects))
                self.done = True
        cv2.destroyWindow(window_name)
        
    def get_rect(self):
        return self.rects
    
    def get_template(self):
        templates=[]
        for x,y,w,h in self.rects:
            templates.append(self.img[y:y+h,x:x+w])
        return templates
    
def non_max_suppression_fast(boxes, probs, overlap_thresh=0.5, max_boxes=50):
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

#     np.testing.assert_array_less(x1, x2)
#     np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick)%500 == 0:
            print(len(pick))
        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs
