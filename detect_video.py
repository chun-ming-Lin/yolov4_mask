import tensorflow as tf
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input video path', required=True)
parser.add_argument('--output', type=str, help='output video path', default='./output.avi')
args = parser.parse_args()

model = tf.keras.models.load_model('yolov4_mask.h5')
classes = ['口罩沒戴好', '沒戴口罩', '有戴口罩']

def yolov4_mask_detect(model, image):
    # image preprocess
    img_original = image
    img = cv2.resize(img_original, (416, 416))
    img = img / 255
    img = img.reshape(1, 416, 416, 3)

    # get bounding box
    height, width, channels = img_original.shape
    preds = model.predict(img)
    scores = []

    for pred in preds:
        boxes = pred[:, 0:4]
        pred_conf = pred[:, 4:]
        for p in pred:
            scores.append(np.max(p[4:]))     # 取最大的分數
    scores = tf.convert_to_tensor(scores)    # 轉成tensor

    # nms
    selected_indices = tf.image.non_max_suppression(
        boxes=tf.reshape(boxes, (boxes.shape[0], 4)),
        scores=scores,
        max_output_size=50, iou_threshold=0.45
    )
    selected_boxes = tf.gather(boxes, selected_indices)

    # draw on image
    for i in range(len(selected_boxes.numpy())):
        y1 = int(selected_boxes.numpy()[i][0] * height)
        x1 = int(selected_boxes.numpy()[i][1] * width)
        y2 = int(selected_boxes.numpy()[i][2] * height)
        x2 = int(selected_boxes.numpy()[i][3] * width)
        pt1 = (x1, y1)
        pt2 = (x2, y2)

        # add label and confidence
        conf_ind = np.argmax(pred_conf[selected_indices.numpy()[i]])  # 用在取classes
        confidence = np.max(pred_conf[selected_indices.numpy()[i]])
        label = classes[conf_ind]  + ' :{:.2f}'.format(confidence)
        text_org = (x1, y1-5)
        green = (0, 255, 0)
        red = (0, 0, 255)
        pink = (191, 179, 255)
        colors = [pink, red, green]
        color = colors[conf_ind]
        # use freetype font
        fontpath = './fonts/GenSenRounded-EL.ttc'
        ft2 = cv2.freetype.createFreeType2()
        ft2.loadFontData(fontpath, id=0)
        fontHeight = 30

        # t_size, baseline = cv2.getTextSize(label, ft2, 0.5, thickness=1)
        t_size, baseLine = ft2.getTextSize(label, fontHeight=fontHeight, thickness=1)

        cv2.rectangle(img_original, pt1, pt2, color, thickness=2)
        label_pt1 = (pt1[0], pt1[1] - t_size[1] - 4)
        label_pt2 = (pt1[0] + t_size[0], pt1[1])
        cv2.rectangle(img_original, label_pt1, label_pt2, color, thickness=-1)  # fill
        # cv2.putText(img_original, label, text_org, font, 0.5, (255, 255, 255), thickness=1)

        ft2.putText(img_original, label, text_org, fontHeight=fontHeight, color=(0, 0, 0),
            thickness=1, line_type=cv2.LINE_AA, bottomLeftOrigin=True
        )
    return img_original

capture = cv2.VideoCapture(args.input)
fourcc = cv2.VideoWriter_fourcc(*'XVID') # *'DIVX', *'XVID', *'MJPG', *'MP4V'
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
print(width, height, 'fps:',fps)
print('開始偵測')
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret == True:
        frame_detected = yolov4_mask_detect(model, frame)

        out.write(frame_detected)
        cv2.imshow('frame', frame_detected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('結束')
        break

capture.release()
out.release()