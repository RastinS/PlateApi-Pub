import threading, time
from detect import *
import imutils
from collections import Counter

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())


RTSP_ADDR = ''
NORM_FRAMES_NUM = 5


def detectionThread(model1, model2):
    global detectedPlates, cameraCap, lastProcessedImg, detectedPlates, frameCap, vidDone

    plateImgs = []
    lastPlate = ''
    tempPlates = ['' for _ in range(NORM_FRAMES_NUM)]
    while not vidDone:
        if frameCap is None:
            continue
        # success, img = cameraCap.read()
        # if not success:
        #     print('bad frame')
        #     continue
        image = factorImage(frameCap)
        
        plateImgs = run(model1, image, isStep1=True)

        lastProcessedImg = image[0]

        
        for img in plateImgs:
            res = run(model2, factorImage(img), isStep1=False)
            res.sort(key=lambda x: x[1])
            outStr = ''
            if len(res) != 8:
                continue
            plate = outStr.join([ch[0] for ch in res])

            tempPlates.insert(0,plate)
            tempPlates.pop()
            # print(tempPlates)
            frequentPlate = Counter(tempPlates).most_common(1)[0][0]
            if frequentPlate != lastPlate:
                print(frequentPlate)
                lastPlate = frequentPlate
            # if not plate in detectedPlates:
            # print(plate)
            # detectedPlates.append(plate)
    exit()
        


def frameFetchThread():
    global cameraCap, frameCap, vidDone

    time.sleep(2)

    while(cameraCap.isOpened()):

        ret, frame = cameraCap.read()
        if ret == True:
            
            cv2.imshow('Frame', frame)
            frameCap = frame.copy()
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        else: 
            break

    cameraCap.release()
    cv2.destroyAllWindows()
    vidDone = True
    exit()

if __name__ == "__main__":

    weightS1 = './weights/step1/best.pt'
    weightS2 = './weights/step2/best.pt'

    model1, model2 = initModels(weightS1, weightS2)

    vidDone = False
    detectedPlates = []
    lastProcessedImg = np.array([None])
    cameraCap = None
    frameCap = None

    # cameraCap = cv2.VideoCapture(RTSP_ADDR)
    # cameraCap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cameraCap = cv2.VideoCapture('./imgData/vid5.mp4')


    fetchThread = threading.Thread(target=frameFetchThread)
    fetchThread.daemon = True
    fetchThread.start()

    detectThread = threading.Thread(target=detectionThread, args=[model1, model2])
    detectThread.daemon = True
    detectThread.start()

    while not vidDone:
        if lastProcessedImg.all() != None:
            pass
            # im0 = lastProcessedImg.copy()
            
            # h,w,c = img.shape
            # offset = 0
            # font = cv2.FONT_HERSHEY_SIMPLEX

            # for itr, word in enumerate(detectedPlates):
            #     offset += int(h / len(detectedPlates)) - 10
            #     cv2.putText(im0, word, (20, offset), font, 1, (0, 255, 0), 3)
            # print(detectedPlates)


            # cv2.imshow('a', imutils.resize(im0, height=700))
            # if (cv2.waitKey(1) & 0xFF) == ord('q'):
            #     break

        else:
            # print('no frames processed')
            pass