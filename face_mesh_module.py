import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = 0.5

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_mode,
            max_num_faces=self.max_faces,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        self.results = None

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lm in self.results.multi_face_landmarks:
                face = []
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lm, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.draw_spec, self.draw_spec)

                    for idx, lm in enumerate(face_lm.landmark):
                        # print(lm)
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)  # Convert to pixel values
                        # cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                        # print(idx, x, y)
                        face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv.VideoCapture(0)
    p_time = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(len(faces))
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
