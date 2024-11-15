import cv2
import mediapipe as mp
import argparse

def process_img(img,face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    print(out.detections)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 25, 0), 10)
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w],
                                                 (30, 30))  # converting specific part of image
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode",default="webcam")
args.add_argument("--filepath",default=None)

args = args.parse_args()
mp_face_detection = mp.solutions.face_detection



with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        img = cv2.imread(args.filepath)

        img = process_img(img,face_detection)



        cv2.imshow("img",img)
        cv2.waitKey(0)

    elif args.mode in ["video"]:
        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()
        while ret:
            ret, frame = cap.read()
            img = process_img(frame, face_detection)
            cv2.imshow("video",img)

        cap.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            img = process_img(frame, face_detection)
            cv2.imshow("webcam", img)
            cv2.waitKey(25)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()