import datetime
import logging
import pathlib
import csv
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()
        self.log_file = None
        self.csv_writer = None
        self._setup_logging()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def _setup_logging(self):
        """Sets up the CSV file and writer for logging annotations."""
        if self.output_dir and (self.config.demo.use_camera or self.config.demo.video_path):
            if self.config.demo.use_camera:
                log_name = f'log_{self._create_timestamp()}.csv'
            elif self.config.demo.video_path:
                name = pathlib.Path(self.config.demo.video_path).stem
                log_name = f'log_{name}.csv'
            else:
                return
            
            log_path = self.output_dir / log_name
            logger.info(f'Logging annotations to: {log_path}')
            self.log_file = open(log_path, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            
            header = [
                'frame', 'face_id',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'head_rot_x', 'head_rot_y', 'head_rot_z',
                'head_pos_x', 'head_pos_y', 'head_pos_z',
                'gaze_vector_face_x', 'gaze_vector_face_y', 'gaze_vector_face_z',
                'gaze_pitch_face', 'gaze_yaw_face',
                'gaze_vector_reye_x', 'gaze_vector_reye_y', 'gaze_vector_reye_z',
                'gaze_pitch_reye', 'gaze_yaw_reye',
                'gaze_vector_leye_x', 'gaze_vector_leye_y', 'gaze_vector_leye_z',
                'gaze_pitch_leye', 'gaze_yaw_leye'
            ]
            self.csv_writer.writerow(header)

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            image = cv2.imread(self.config.demo.image_path)
            self._process_image(image, frame_number=0)
            if self.config.demo.display_on_screen:
                while True:
                    key_pressed = self._wait_key()
                    if self.stop:
                        break
                    if key_pressed:
                        self._process_image(image, frame_number=0)
                    cv2.imshow('image', self.visualizer.image)
            if self.config.demo.output_dir:
                name = pathlib.Path(self.config.demo.image_path).name
                output_path = pathlib.Path(self.config.demo.output_dir) / name
                cv2.imwrite(output_path.as_posix(), self.visualizer.image)
        else:
            raise ValueError

        if self.log_file:
            self.log_file.close()
            logger.info("Annotation log file closed.")

    def _run_on_video(self) -> None:
        frame_number = 0
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break

            self._process_image(frame, frame_number)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)

            frame_number += 1

        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image, frame_number: int) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face_index, face in enumerate(faces):
            self.gaze_estimator.estimate_gaze(undistorted, face)

            bbox = face.bbox.flatten().tolist() if face.bbox is not None else [None]*4
            head_rot_euler = face.head_pose_rot.as_euler('XYZ', degrees=True).tolist() if face.head_pose_rot else [None]*3
            head_pos = face.head_position.flatten().tolist() if face.head_position is not None else [None]*3
            gaze_vector_face = face.gaze_vector.tolist() if face.gaze_vector is not None else [None]*3
            pitch_yaw_face = np.rad2deg(face.vector_to_angle(face.gaze_vector)).tolist() if face.gaze_vector is not None else [None]*2

            if self.config.mode == 'MPIIGaze':
                reye_gaze_vector = face.reye.gaze_vector.tolist() if hasattr(face, 'reye') and face.reye.gaze_vector is not None else [None]*3
                reye_pitch_yaw = np.rad2deg(face.reye.vector_to_angle(face.reye.gaze_vector)).tolist() if hasattr(face, 'reye') and face.reye.gaze_vector is not None else [None]*2
                leye_gaze_vector = face.leye.gaze_vector.tolist() if hasattr(face, 'leye') and face.leye.gaze_vector is not None else [None]*3
                leye_pitch_yaw = np.rad2deg(face.leye.vector_to_angle(face.leye.gaze_vector)).tolist() if hasattr(face, 'leye') and face.leye.gaze_vector is not None else [None]*2
            else:
                reye_gaze_vector, reye_pitch_yaw = [None]*3, [None]*2
                leye_gaze_vector, leye_pitch_yaw = [None]*3, [None]*2

            if self.csv_writer:
                row = [
                    frame_number, face_index,
                    *bbox,
                    *head_rot_euler,
                    *head_pos,
                    *gaze_vector_face,
                    *pitch_yaw_face,
                    *reye_gaze_vector,
                    *reye_pitch_yaw,
                    *leye_gaze_vector,
                    *leye_pitch_yaw
                ]
                self.csv_writer.writerow(row)

            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError
