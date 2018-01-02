import os
import cv2
import numpy
KNN_SQUARE_SIDE = 50  # Square 50 x 50 px.


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
        overl_r=float(intersect_size) / (bbox1_size + bbox2_size - intersect_size)

    return overl_r

class BaseKnnMatcher(object):
    distance_threshold = 0

    def __init__(self, source_dir):
        self.model, self.label_map = self.get_model_and_label_map(source_dir)

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
        image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE, KNN_SQUARE_SIDE))
        image_standard_size = numpy.float32(image_standard_size.reshape((1, KNN_SQUARE_SIDE * KNN_SQUARE_SIDE)))
        closest_class, results, neigh_resp, distance = self.model.find_nearest(image_standard_size, k=1)

        if distance[0][0] > self.distance_threshold:
            return None

        return self.label_map[int(closest_class)]


class DigitKnnMatcher(BaseKnnMatcher):
    distance_threshold = 10 ** 10


class MeterValueReader(object):
    def __init__(self):
        self.digit_knn_matcher = DigitKnnMatcher(source_dir='templates')

    @classmethod
    def get_symbol_boxes(cls, cv_image):
        ofs = 10
        ret, thresh = cv2.threshold(cv_image.copy(), 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        symbol_boxes = []
        contours.pop(-1)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            # You can test here for box size, though not required in your example:
            # if cls.is_size_of_digit(width, height):
            #     symbol_boxes.append((x, y, x+width, y+height))

            symbol_boxes.append((x-ofs, y-ofs, x+width+ofs, y+height+ofs))            

        symbol_boxes_finetune = []
        for box in symbol_boxes:
            valid = True
            for k, finetuned_box in enumerate(symbol_boxes_finetune):
                if calc_overlap(box, finetuned_box) > 0:
                    if box[0] > finetuned_box[0]:
                        valid = False
                    else:
                        symbol_boxes_finetune.pop(k)
                    break
            if valid:
                symbol_boxes_finetune.append(box)

        return symbol_boxes_finetune

    def get_value(self, meter_cv2_image):
        symbol_boxes = self.get_symbol_boxes(meter_cv2_image)
        symbol_boxes.sort()  # x is first in tuple
        symbols = []
        for box in symbol_boxes:
            symbol = self.digit_knn_matcher.predict(crop(meter_cv2_image, box))
            symbols.append(symbol)
        return ''.join(symbols)


if __name__ == '__main__':
    # If you want to see how boxes detection works, uncomment these:
    # img_bw = cv2.imread(os.path.join('./data/10.png'), 0)
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

    img_bw = cv2.imread(os.path.join('./data/9.png'), 0)
    vr = MeterValueReader()
    print vr.get_value(img_bw)