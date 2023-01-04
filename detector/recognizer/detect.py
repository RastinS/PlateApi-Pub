import sys
import time
from pathlib import Path
import numpy as np
import json
import torch
import os
from PlateAPI.settings import BASE_DIR

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy, clip_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.augmentations import letterbox
import os



def factorImage(image, imHeight=640):
    img = letterbox(image, imHeight)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)

    return img, image


def crop_one_box(xyxy, im, gain=1.02, pad=10):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes

    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::1]
    return crop


@torch.no_grad()
def initModels(weightS1, weightS2, device='cpu'):
    device = select_device(device)

    model1 = attempt_load(weightS1, map_location=device) 
    print('Loaded model 1')
    model2 = attempt_load(weightS2, map_location=device) 
    print('Loaded model 2')

    return model1, model2


@torch.no_grad()
def run(model, imageData, isStep1):

    imgsz=640  # inference size (pixels)
    conf_thres=0.50  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=100  # maximum detections per image
    device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu


    set_logging()
    device = select_device(device)

    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
 
    img, im0s = imageData
    img = torch.from_numpy(img).to(device)
    img = img.float()

    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]

    pred = model(img, augment=False, visualize=False)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    results = []
    for i, det in enumerate(pred):
        im0 = im0s.copy()

        imc = im0.copy()
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, _, cls in reversed(det):

                if isStep1:
                    results.append(crop_one_box(xyxy, imc))
                else:
                    xyxy = torch.tensor(xyxy).view(-1, 4)
                    b = xyxy2xywh(xyxy)  # boxes

                    b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                    xyxy = xywh2xyxy(b).long()
                    clip_coords(xyxy, im0.shape)
                    results.append((names[int(cls)], int(xyxy[0][0])))

    return results


def getWordDict():
    nameDict = {}

    nameDict['0'] = '۰'
    nameDict['1'] = '۱'
    nameDict['2'] = '۲'
    nameDict['3'] = '۳'
    nameDict['4'] = '۴'
    nameDict['5'] = '۵'
    nameDict['6'] = '۶'
    nameDict['7'] = '۷'
    nameDict['8'] = '۸'
    nameDict['9'] = '۹'

    nameDict['A'] = 'آ'
    nameDict['B'] = 'ب'
    nameDict['P'] = 'پ'
    nameDict['T'] = 'ت'
    nameDict['S'] = 'ث'
    nameDict['J'] = 'ج'
    nameDict['CH'] = 'چ'
    nameDict['H'] = 'ح'
    nameDict['KH'] = 'خ'
    nameDict['D'] = 'د'

    nameDict['ZAL'] = 'ذ'
    nameDict['R'] = 'ر'
    nameDict['Z'] = 'ز'
    nameDict['ZH'] = 'ژ'
    nameDict['SIN'] = 'س'
    nameDict['SH'] = 'ش'
    nameDict['SAD'] = 'ص'
    nameDict['ZAD'] = 'ض'
    nameDict['TA'] = 'ط'

    nameDict['ZA'] = 'ظ'
    nameDict['AIN'] = 'ع'
    nameDict['GHAIN'] = 'غ'
    nameDict['F'] = 'ف'
    nameDict['GHAF'] = 'ق'
    nameDict['K'] = 'ک'
    nameDict['G'] = 'گ'
    nameDict['L'] = 'ل'
    nameDict['M'] = 'م'
    nameDict['N'] = 'ن'
    nameDict['V'] = 'و'
    nameDict['H2'] = 'ه'
    nameDict['Y'] = 'ی'
    nameDict['PwD'] = 'معلول'

    return nameDict


def detectPhoto(photo, asJSON=False):
    weightS1 = os.path.join(BASE_DIR, 'detector/recognizer/weights/step1/best.pt')
    weightS2 = os.path.join(BASE_DIR, 'detector/recognizer/weights/step2/best.pt')

    model1, model2 = initModels(weightS1, weightS2)
    nameDict = getWordDict()
    out = []
    t0 = time.time()
    firstStepResults = run(model1, factorImage(photo), isStep1=True)

    i = 0
    for img in firstStepResults:
        t0 = time.time()
        res = run(model2, factorImage(img), isStep1=False)
        res.sort(key=lambda x: x[1])
        if len(res) != 8:
            continue
        
        out.append([nameDict[ch[0]] for ch in res])

    if asJSON:
        out = json.dumps(out, ensure_ascii=False)

    return out