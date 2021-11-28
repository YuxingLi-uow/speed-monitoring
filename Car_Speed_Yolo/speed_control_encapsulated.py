import cv2
from yolo_v3 import *
import time

class SpeedDetect:
    def __init__(self, video_path, mask_path):
        self.video_path = video_path
        self.mask_path = mask_path
        self.classes = [2,7]
        self.video = None
        self.mask = None
        self.video_frame_duration = None
        self.previous_gray_frame = None
        self.TOO_FAR_Y = None
        self.pt_1 = None
        self.pt_2 = None
        self.M = None

    def initialize(self):
        self.video = cv2.VideoCapture(self.video_path)
        self.video_frame_duration = 1 / self.video.get(cv2.CAP_PROP_FPS)
        self.mask = cv2.imread(self.mask_path)

    def detection(self):
        self.initialize()

        while True:
            t_start_frame = time.time()
            ret, frame = self.video.read()

            if not ret:
                break

            # scale factor = 0.4, 288 x 512
            frame_resized = cv2.resize(frame, None, fx=0.4, fy=0.4)

            # apply mask and resize
            frame_masked = cv2.bitwise_and(frame, self.mask)
            frame_masked = cv2.resize(frame_masked, None, fx=0.4, fy=0.4)

            # save previous frame
            if self.previous_gray_frame is None:

                self.TOO_FAR_Y = int(143 * 0.4)  # detection zone back edge
                self.pt_1 = np.float32([[232 * 0.4, self.TOO_FAR_Y], [419 * 0.4, self.TOO_FAR_Y],
                                   [989 * 0.4, 639 * 0.4], [248 * 0.4, 574 * 0.4]])
                self.pt_2 = np.float32([[0, 0], [frame_resized.shape[1], 0],
                               [frame_resized.shape[1], frame_resized.shape[0]], [0, frame_resized.shape[0]]])
                self.M = cv2.getPerspectiveTransform(self.pt_1, self.pt_2)

                frame_gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

                self.previous_gray_frame = frame_gray
                continue

            frame_gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

            # calculate optical flow, 3 pyramids
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray_frame, frame_gray, None, 0.5, 3, 5, 3, 5, 1.1, 0)

            # calculate the vector magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            warped = cv2.warpPerspective(frame_resized, self.M, (frame_resized.shape[1], frame_resized.shape[0]))

            for x1, y1, x2, y2, cls in detect_cars(frame_masked):

                if y1 < self.TOO_FAR_Y:
                    break
                if cls not in self.classes:
                    break

                avg_mag = np.mean(mag[y1: y2, x1: x2])
                avg_ang = np.mean(ang[y1: y2, x1: x2])

                if np.isnan(avg_mag) or np.isnan(avg_ang):
                    break

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                delta_x = int(center_x + np.cos(avg_ang) * avg_mag * 15)
                delta_y = int(center_y + np.sin(avg_ang) * avg_mag * 15)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.arrowedLine(frame_resized, (center_x, center_y), (delta_x, delta_y), (0, 255, 0), 1)

                # speed = abs(np.cos(avg_ang)) * avg_mag * 920000 / (y1 ** 2)
                # speed = avg_mag * 7290 * 60 / (frame_resized.shape[0])
                speed_x = np.mean(flow[y1:y2, x1:x2, 0]) * 0.026 * 3.6 * 30
                speed_y = np.mean(flow[y1:y2, x1:x2, 1]) * 0.304 * 3.6 * 30
                speed = (speed_x ** 2 + speed_y ** 2) ** 0.5
                cv2.putText(frame_resized, str(int(speed)) + "Km/h", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # warped points
                a_center_x2, a_center_y2, a = np.matmul(self.M, [center_x, center_y, 1])
                cv2.circle(warped, (int(a_center_x2 / a), int(a_center_y2 / a)), 10, (0, 255, 0), -1)


            t_end_frame = time.time()
            frame_duration = t_end_frame - t_start_frame
            delay_need = int(self.video_frame_duration - frame_duration) * 1000
            if delay_need <= 0:
                delay_need = 1
            if cv2.waitKey(delay_need) == ord('q'):
                break

            self.previous_gray_frame = frame_gray
            fps = 1 / (time.time() - t_start_frame)
            cv2.putText(frame_resized, 'FPS:' + str(round(fps, 2)), (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.line(frame_resized, (0, self.TOO_FAR_Y), (frame_resized.shape[1], self.TOO_FAR_Y), (255, 0, 170), 1)
            cv2.imshow('Result', frame_resized)
            cv2.imshow('Warped', warped)

        cv2.destroyAllWindows()
        self.video.release()


detect = SpeedDetect('video_01.mp4', 'mask.jpg')
detect.detection()
