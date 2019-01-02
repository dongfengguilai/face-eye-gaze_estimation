import numpy as np
import cv2
import math
from mark_detector import MarkDetector,eyeDetector,gazeDetector
from stabilizer import Stabilizer
from pose_estimator import PoseEstimator
CNN_INPUT_SIZE = 128
import datetime
def get_face(detector, image):
    while True:
        box = detector.extract_cnn_facebox(image)
        return box
def convert_to_unit_vector(angle):
    x=-np.cos(angle[1])*np.sin(angle[0])
    y=-np.sin(angle[1])
    z=-np.cos(angle[0])*np.cos(angle[1])
    norm=np.sqrt(x**2+y**2+z**2)
    x/=norm
    y/=norm
    z/=norm
    return x,y,z
def get_peak_points(heatmaps):
    peak_points = []
    for j in range(68):
        a,b=cv2.threshold(heatmaps[:, :, j],50,255,0)
        yy,xx=np.where(b == 255)
        y=np.mean(yy,0)
        x=np.mean(xx,0)
        peak_points.append([x , y])
    all_peak_points = np.array(peak_points)
    return all_peak_points
def get_peak_points_eye(heatmaps):
    peak_points = []
    for j in range(5):
        a,b=cv2.threshold(heatmaps[:, :, j],50,255,0)
        yy,xx=np.where(b == 255)
        y=np.mean(yy,0)
        x=np.mean(xx,0)
        peak_points.append([x , y])
    all_peak_points = np.array(peak_points)
    return all_peak_points
def getpoint(img):
    point=[]
    for i in range(1):
        image=img[:,:,i]
        ret, g = cv2.threshold(image, 200, 255, 0)
        a, b, c = cv2.findContours(g, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
        d = np.array(b)[0, :, 0, :]
        for index in np.linspace(0, len(d) / 2, 2):
            point.append([int(d[int(index)][0]), int(d[int(index)][1])])
    return point
def get_gazebox2(marks):
    dian39=marks[39]
    dian36=marks[36]
    diancenter=(dian36+dian39)/2
    dis3639=np.sqrt(np.sum(np.square(dian36-dian39)))/2
    left=diancenter[0]-dis3639*1.5
    right=diancenter[0]+dis3639*1.5
    top=diancenter[1]-dis3639*2/3*1.5
    bottom=diancenter[1]+dis3639*2/3*1.5
    return [left,top,right,bottom]
def get_gazebox1(marks):
    dian45=marks[45]
    dian42=marks[42]
    diancenter=(dian45+dian42)/2
    dis4245=np.sqrt(np.sum(np.square(dian42-dian45)))/2
    left=diancenter[0]-dis4245*1.5
    right=diancenter[0]+dis4245*1.5
    top=diancenter[1]-dis4245*2/3*1.5
    bottom=diancenter[1]+dis4245*2/3*1.5
    return [left,top,right,bottom]
def convert_pose_L(vect):
    M, _ = cv2.Rodrigues(np.array(vect).astype(np.float32))
    vec = M[:, 2]
    yaw = np.arcsin(vec[0])
    pitch = -np.arctan2(-vec[1], vec[2])
    return np.array([yaw, pitch])

def main():
    video_src="myvedio7.avi"

    cam=cv2.VideoCapture(1)
    mark_detector=MarkDetector()
    eye_detector=eyeDetector()
    gaze_detector=gazeDetector()
    pose_estimator = PoseEstimator(img_size=(480, 640))
    pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(6)]
    gaze_stabilizers1 = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(2)]
    gaze_stabilizers2 = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(2)]
    radius_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(2)]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))
    while True:
        # Read frame, crop it, flip it, suits your needs.
        starttime=datetime.datetime.now()
        frame_got, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame_got is False:
            break
        facebox=get_face(mark_detector,frame)
        if facebox is not None:
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

            face_img=np.array(face_img).astype(np.float32)
            marks = mark_detector.detect_marks(face_img)



            img1 = marks[1][0]
            img1=np.clip(img1,0,255).astype(np.uint8)
            points = get_peak_points(img1)
            points=np.array(points).astype(np.float32)
            points=points/64
            points *= (facebox[2] - facebox[0])
            points[:, 0] += facebox[0]
            points[:, 1] += facebox[1]

            point_print=True
            for i in points:
                if math.isnan(i[0]) or math.isnan(i[1]):
                    point_print=False
                    continue
                else:
                    cv2.circle(frame, (int(i[0]), int(i[1])), 2, (255, 0, 0), -1, 8)
            if point_print:

                pose = pose_estimator.solve_pose_by_68_points(points)
                stabile_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    stabile_pose.append(ps_stb.state[0])
                stabile_pose = np.reshape(stabile_pose, (-1, 3))
                pose_gaze = convert_pose_L(stabile_pose[0])
                pose_estimator.draw_annotation_box(
                    frame, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))

                gaze_box1 = get_gazebox1(points)
                eyeimage1 = frame_gray[int(gaze_box1[1]): int(gaze_box1[3]),
                            int(gaze_box1[0]): int(gaze_box1[2])]
                eyeimage1 = cv2.resize(eyeimage1, (60, 40))
                eye_image1 = np.reshape(eyeimage1, (40, 60, 1))
                eye_data1 = eye_detector.detect_marks(eye_image1/255)
                eye_center1 = eye_data1[1, 0, :, :]
                eyepoints1 = get_peak_points_eye(eye_center1)
                eyepoints1 = np.array(eyepoints1).astype(np.float32)

                pupil_point = eyepoints1[4]
                eyepoints1[:, 0] = eyepoints1[:, 0] / 60* (gaze_box1[2] - gaze_box1[0])+ gaze_box1[0]
                eyepoints1[:, 1] = eyepoints1[:, 1] / 40* (gaze_box1[3] - gaze_box1[1])+ gaze_box1[1]
                look_vec1 = gaze_detector.detect_marks(eye_image1/255, pose_gaze)
                stabile_gaze1 = []
                for value1, ps_stb1 in zip(look_vec1[0], gaze_stabilizers1):
                    ps_stb1.update([value1])
                    stabile_gaze1.append(ps_stb1.state[0])
                stabile_gaze1 = np.reshape(stabile_gaze1, (-1, 2))
                look_vec1_ = convert_to_unit_vector(stabile_gaze1[0])
                look_vec1 = stabile_gaze1[0] * 180 / np.pi

                gaze_box2 = get_gazebox2(points)
                eyeimage2 = frame_gray[int(gaze_box2[1]): int(gaze_box2[3]),
                            int(gaze_box2[0]): int(gaze_box2[2])]
                eyeimage2 = eyeimage2[:, ::-1]
                eyeimage2 = cv2.resize(eyeimage2, (60, 40))

                eye_image2 = np.reshape(eyeimage2, (40, 60, 1))
                eye_data2 = eye_detector.detect_marks(eye_image2/255)
                eye_center2 = eye_data2[1, 0, :, :]
                eyepoints2 = get_peak_points_eye(eye_center2)
                eyepoints2 = np.array(eyepoints2).astype(np.float32)
                eyepoints2[:,0]=60-eyepoints2[:,0]
                eyepoints2[:, 0] = eyepoints2[:, 0] / 60*(gaze_box2[2] - gaze_box2[0])+gaze_box2[0]
                eyepoints2[:, 1] = eyepoints2[:, 1] / 40*(gaze_box2[3] - gaze_box2[1])+gaze_box2[1]
                pose_gaze[0]=-pose_gaze[0]
                look_vec2=gaze_detector.detect_marks(eye_image2/255,pose_gaze)
                look_vec2[0][0]=-look_vec2[0][0]
                stabile_gaze2 = []
                for value2, ps_stb2 in zip(look_vec2[0], gaze_stabilizers2):
                    ps_stb2.update([value2])
                    stabile_gaze2.append(ps_stb2.state[0])
                stabile_gaze2 = np.reshape(stabile_gaze2, (-1, 2))
                look_vec2_ = convert_to_unit_vector(stabile_gaze2[0])
                look_vec2 =stabile_gaze2[0] * 180 / np.pi



                eye_print = True
                for i in eyepoints1:
                    if math.isnan(i[0]) or math.isnan(i[1]):
                        eye_print = False
                        break
                for i in eyepoints2:
                    if math.isnan(i[0]) or math.isnan(i[1]):
                        eye_print = False
                        break
                if eye_print:
                    cv2.circle(frame, (int(eyepoints1[4, 0]), int(eyepoints1[4, 1])), 3, (0, 0, 255), -1, 8)
                    cv2.circle(frame, (int(eyepoints2[4, 0]), int(eyepoints2[4, 1])), 3, (0, 0, 255), -1, 8)
                    zuhe1=[]
                    for i in eyepoints1[0:3]:
                        dist=np.sqrt((i[0]-eyepoints1[4, 0])**2+(i[1]-eyepoints1[4, 1])**2)
                        zuhe1.append(dist)
                    radius1=np.mean(np.array(zuhe1),0)
                    zuhe2=[]
                    for i in eyepoints2[0:3]:
                        dist=np.sqrt((i[0]-eyepoints2[4, 0])**2+(i[1]-eyepoints2[4, 1])**2)
                        zuhe2.append(dist)
                    radius2=np.mean(np.array(zuhe2),0)
                    stabile_radius = []
                    for value, ps_stb in zip([radius1,radius2], radius_stabilizers):
                        ps_stb.update([value])
                        stabile_radius.append(ps_stb.state[0])
                    stabile_radius = np.reshape(stabile_radius, (-1, 2))
                    cv2.circle(frame, (int(eyepoints1[4, 0]), int(eyepoints1[4, 1])), int(stabile_radius[0][0]), (0, 0, 255), 2, 8)
                    cv2.circle(frame, (int(eyepoints2[4, 0]), int(eyepoints2[4, 1])), int(stabile_radius[0][1]),(0, 0, 255), 2, 8)

                    pupil1 = eyepoints1[4]
                    cv2.line(frame, (int(pupil1[0]), int(pupil1[1])),
                             (int(pupil1[0] + look_vec1_[0] * 80), int(pupil1[1] + look_vec1_[1] * 80)),
                             (0, 255, 0), 4)
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(frame, 'yam_gaze1:    ' + str(int(look_vec1[0])), (10, 50), font, 0.8, (255, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(frame, 'pitch_gaze1:    ' + str(int(look_vec1[1])), (10, 60), font, 0.8, (255, 0, 255),
                                1,
                                cv2.LINE_AA)

                    pupil2 = eyepoints2[4]
                    cv2.line(frame, (int(pupil2[0]), int(pupil2[1])),
                             (int(pupil2[0] + look_vec2_[0] * 80), int(pupil2[1] + look_vec2_[1] * 80)),
                             (0, 255, 0), 4)
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(frame, 'yam_gaze2:    ' + str(int(look_vec2[0])), (10, 80), font, 0.8, (255, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(frame, 'pitch_gaze2:    ' + str(int(look_vec2[1])), (10, 90), font, 0.8, (255, 0, 255),
                                1,
                                cv2.LINE_AA)
                    cv2.imwrite("frame.jpg", frame)
        out.write(frame)
        endtime = datetime.datetime.now()
        print(endtime - starttime)
        cv2.imshow("Preview", frame)
        if cv2.waitKey(30) == 27:
            break
if __name__ == '__main__':
    main()



