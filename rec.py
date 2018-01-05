import os
import cv2
import numpy
import copy

from sklearn.externals import joblib
from skimage.feature import hog

KNN_SQUARE_SIDE = 100  # Square 50 x 50 px.


def resize(cv_image, factor):
    new_size = tuple(map(lambda x: x * factor, cv_image.shape[::-1]))
    return cv2.resize(cv_image, new_size)


def crop(cv_image, box):
    x0, y0, x1, y1 = box
    return cv_image[y0:y1, x0:x1]


def draw_box(cv_image, box):
    x0, y0, x1, y1 = box
    cv2.rectangle(cv_image, (x0, y0), (x1, y1), (0, 0, 255), 2)


def draw_boxes_and_show(cv_image, boxes, title='N'):
    temp_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    for box in boxes:
        draw_box(temp_image, box)
    cv2.imshow(title, temp_image)
    cv2.waitKey(0)

def calc_overlap(box1, box2):
    bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax = box1
    bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax = box2

    if ((bbox2_xmin > bbox1_xmax )or (bbox2_xmax < bbox1_xmin) or
            (bbox2_ymin > bbox1_ymax )or (bbox2_ymax < bbox1_ymin)):
        overl_r =0
    else :
        intersect_xmin=(max(bbox1_xmin, bbox2_xmin))
        intersect_ymin=(max(bbox1_ymin, bbox2_ymin))
        intersect_xmax=(min(bbox1_xmax, bbox2_xmax))
        intersect_ymax=(min(bbox1_ymax, bbox2_ymax))

        intersect_size=(intersect_ymax-intersect_ymin) * (intersect_xmax-intersect_xmin)
        bbox1_size=(bbox1_ymax-bbox1_ymin) * (bbox1_xmax-bbox1_xmin)
        bbox2_size = (bbox2_ymax - bbox2_ymin) * (bbox2_xmax - bbox2_xmin)
        smaller_size = bbox1_size if bbox1_size < bbox2_size else bbox1_size
        #overl_r=float(intersect_size) / (bbox1_size + bbox2_size - intersect_size)
        overl_r=float(intersect_size) / (smaller_size)

    return overl_r

class BaseKnnMatcher(object):
    distance_threshold = 0

    def __init__(self, source_dir):
        self.model, self.label_map = self.get_model_and_label_map(source_dir)

        # prepare templates for rec
        self.lbl2temp = {}
        for label_idx, filename in enumerate(os.listdir(source_dir)):
            label = filename[:filename.index('.png')]
            image = cv2.imread(os.path.join(source_dir, filename), 0)
            #input_img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            suit_image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE, KNN_SQUARE_SIDE))
            self.lbl2temp[label] = suit_image_standard_size

    @staticmethod
    def get_model_and_label_map(source_dir):
        responses = []
        label_map = []
        samples = numpy.empty((0, KNN_SQUARE_SIDE * KNN_SQUARE_SIDE), numpy.float32)
        for label_idx, filename in enumerate(os.listdir(source_dir)):

            label = filename[:filename.index('.png')]
            label_map.append(label)
            responses.append(label_idx)

            image = cv2.imread(os.path.join(source_dir, filename), 0)

            suit_image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE, KNN_SQUARE_SIDE))
            sample = suit_image_standard_size.reshape((1, KNN_SQUARE_SIDE * KNN_SQUARE_SIDE))
            samples = numpy.append(samples, sample, 0)

        responses = numpy.array(responses, numpy.float32)
        responses = responses.reshape((responses.size, 1))
        model = cv2.KNearest()
        model.train(samples, responses)

        return model, label_map

    def predict(self, image):
        # with knn
        knn_result = None
        image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE, KNN_SQUARE_SIDE))
        image_standard_size = numpy.float32(image_standard_size.reshape((1, KNN_SQUARE_SIDE * KNN_SQUARE_SIDE)))
        closest_class, results, neigh_resp, distance = self.model.find_nearest(image_standard_size, k=1)

        if distance[0][0] > self.distance_threshold:
            knn_result = None

        knn_result = self.label_map[int(closest_class)]

        # with template matching 
        meth = 'cv2.TM_CCOEFF_NORMED'
        method = eval(meth)
        lbl2score = {}
        for lbl in self.lbl2temp.keys():
            template = self.lbl2temp[lbl]
            image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE, KNN_SQUARE_SIDE))
            res = cv2.matchTemplate(image_standard_size, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            lbl2score[lbl] = max_val
            #print(lbl + ' ' + str(max_val))

        # if knn_result == '8' and lbl2score[knn_result]<0.7:
        #     knn_result = '3'
        # if knn_result == '9' and lbl2score[knn_result]<0.8:
        #     knn_result = '7'

        return knn_result

class DigitKnnMatcher(BaseKnnMatcher):
    distance_threshold = 10 ** 10


class MeterValueReader(object):
    def __init__(self):
        self.digit_knn_matcher = DigitKnnMatcher(source_dir='templates')

    @classmethod
    def get_symbol_boxes(cls, cv_image):
        ofs = 0
        ret, thresh = cv2.threshold(cv_image.copy(), cv_image.mean(1).mean(0), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        symbol_boxes = []
        if len(contours) == 0:
            height, width = cv_image.shape
            symbol_boxes.append((0, 0, width, height))
            return symbol_boxes

        contours.pop(-1)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            # You can test here for box size, though not required in your example:
            # if cls.is_size_of_digit(width, height):
            #     symbol_boxes.append((x, y, x+width, y+height))

            symbol_boxes.append((x-ofs, y-ofs, x+width+ofs, y+height+ofs))          

        symbol_boxes_finetune = []
        symbol_boxes.reverse()
        for box in symbol_boxes:
            valid = True
            for k, finetuned_box in enumerate(symbol_boxes_finetune):
                if calc_overlap(box, finetuned_box) == 1:
                    if box[0] > finetuned_box[0]:
                        valid = False
                    else:
                        symbol_boxes_finetune.pop(k)
                    break

            if valid:
                symbol_boxes_finetune.append(box)

        # check for '7'
        symbol_boxes_finetune_normal = []
        symbol_boxes_finetune_abnormal = []
        for box in symbol_boxes_finetune:
            xmin, ymin, xmax, ymax = box
            if  ymax - ymin < 15:
                symbol_boxes_finetune_abnormal.append(box)
            else:
                symbol_boxes_finetune_normal.append(box)

        abnormal_groups = []
        for abnormal_box in symbol_boxes_finetune_abnormal:
            xmin, ymin, xmax, ymax = abnormal_box
            x_center = (xmin + xmax)/2
            added = False
            for grp in abnormal_groups:
                if len(grp) == 1:
                    xmin, ymin, xmax, ymax = grp[0]
                    x_center_ = (xmin + xmax)/2
                    if abs(x_center_ - x_center) < 4:
                        grp.append(abnormal_box)
                        added = True
            if not added:
                grp = []
                grp.append(abnormal_box)
                abnormal_groups.append(grp)

        for grp in abnormal_groups:
            box1 = grp[0]
            box2 = grp[0]
            if len(grp) == 2:
                box2 = grp[1]
            xmin1, ymin1, xmax1, ymax1 = box1
            xmin2, ymin2, xmax2, ymax2 = box2

            xmin = xmin1 if xmin1 < xmin2 else xmin2
            ymin = ymin1 if ymin1 < ymin2 else ymin2
            xmax = xmax1 if xmax1 > xmax2 else xmax2
            ymax = ymax1 if ymax1 > ymax2 else ymax2

            symbol_boxes_finetune_normal.append((xmin, ymin, xmax, ymax))


        return symbol_boxes_finetune_normal

    def get_value0(self, meter_cv2_image):
        symbol_boxes = self.get_symbol_boxes(meter_cv2_image)
        #draw_boxes_and_show(meter_cv2_image, symbol_boxes)
        symbol_boxes.sort()  # x is first in tuple
        symbols = []
        for box in symbol_boxes:
            roi = crop(meter_cv2_image, box)
            symbol = self.digit_knn_matcher.predict(roi)
            symbols.append(symbol)

            roi_gray = copy.deepcopy(roi)
            thres = roi_gray.mean(1).mean(0)
            roi_thresh = roi_gray[roi_gray>thres]
            #print(len(roi_thresh))
            #cv2.imshow('bw', roi_gray)
            #cv2.waitKey(0)

        return ''.join(symbols)

    def get_value1(self, meter_cv2_image):
        clf = joblib.load("digits_cls.pkl")
        symbol_boxes = self.get_symbol_boxes(meter_cv2_image)
        symbol_boxes.sort()  # x is first in tuple
        symbols = []
        for box in symbol_boxes:
            roi = crop(meter_cv2_image, box)
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(numpy.array([roi_hog_fd], 'float64'))
            symbols.append(str(nbr[0]))
        return ''.join(symbols)

if __name__ == '__main__':
    # If you want to see how boxes detection works, uncomment these:
    # img_bw = cv2.imread(os.path.join('/tmp/score.jpeg'), 0)
    # boxes = MeterValueReader.get_symbol_boxes(img_bw)
    # draw_boxes_and_show(img_bw, boxes)

    # Uncomment to generate templates from image
    # import random
    # TEMPLATE_DIR = 'templates'
    # img_bw = cv2.imread(os.path.join('./data/9.png'), 0)
    # boxes = MeterValueReader.get_symbol_boxes(img_bw)
    # for box in boxes:
    #     # You need to label templates manually after extraction
    #     cv2.imwrite(os.path.join(TEMPLATE_DIR, '%s.png' % random.randint(10, 1000)), crop(img_bw, box))

    img_bw = cv2.imread(os.path.join('/tmp/score.jpeg'), 0)
    vr = MeterValueReader()
    print vr.get_value0(img_bw)