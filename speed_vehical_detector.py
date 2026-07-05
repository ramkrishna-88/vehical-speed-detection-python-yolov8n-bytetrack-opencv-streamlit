import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
source = np.array([
    [450, 170],
    [1450, 170],
    [1850, 1020],
    [100, 1020]
], dtype=np.int32)
TARGET_WIDTH=40
TARGET_HEIGHT=250
# Real-world length of the polygon in meters
REAL_ROAD_LENGTH = 60.0

# Conversion factor
PIXEL_TO_METER = REAL_ROAD_LENGTH / TARGET_HEIGHT
TARGET=np.array(
    [
        [0,0],
        [TARGET_WIDTH-1,0],
        [TARGET_WIDTH-1,TARGET_HEIGHT-1],
        [0,TARGET_HEIGHT-1],
    ],dtype=np.int32)
class ViewTransformer:
    def __init__(self, source:np.ndarray, target:np.ndarray):
        source=source.astype(np.float32)
        target=target.astype(np.float32)
        self.m=cv2.getPerspectiveTransform(source,target)
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)

        transformed_points = cv2.perspectiveTransform(
            reshaped_points,
            self.m
        )

        # print("Matrix:\n", self.m)
        # print("Input shape:", reshaped_points.shape)
        # print("Output:", transformed_points)

        return transformed_points.reshape(-1, 2)

def process_video(video_path, output_path):
        video_info=sv.VideoInfo.from_video_path(video_path)
        print(video_info.resolution_wh)
        model = YOLO("yolov8n.pt")
        byte_track=sv.ByteTrack(
            frame_rate=video_info.fps,
            track_activation_threshold=0.25,
            minimum_matching_threshold=0.8,
            lost_track_buffer=60,
        )
        thickness=sv.calculate_optimal_line_thickness(
            resolution_wh=video_info.resolution_wh)
        text_scale=sv.calculate_optimal_text_scale(
            resolution_wh=video_info.resolution_wh)
        bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_thickness=thickness,
            text_scale=text_scale,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK)
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps*2,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK)

        frame_generator=sv.get_video_frames_generator(video_path)
        polygon_zone = sv.PolygonZone(polygon=source)
        view_transform = ViewTransformer(source=source,target=TARGET)
        FPS = int(round(video_info.fps))
        coordinates=defaultdict(lambda:deque(maxlen=FPS))
        speed_history = defaultdict(lambda: deque(maxlen=10))
        final_speed = {}
        video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_info.fps,
        video_info.resolution_wh
        )
        for frame in frame_generator:
            results = model.predict(
                frame,
                imgsz=960,
                conf=0.25,
                verbose=False,
                classes=[2, 3, 5, 7],   # car, motorcycle, bus, truck
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections=detections[polygon_zone.trigger(detections)]
            detections=byte_track.update_with_detections(detections=detections)
            
            points=detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points=view_transform.transform_points(points).astype(int)
            
            labels=[]
            if detections.tracker_id is None:
                video_writer.write(frame)
                continue
            for tracker_id, [_,y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
                WINDOW = FPS // 2

                if tracker_id in final_speed:
                    labels.append(f"#{tracker_id} {final_speed[tracker_id]} km/h")

                elif len(coordinates[tracker_id]) >= WINDOW:

                    y_old = coordinates[tracker_id][-WINDOW]
                    y_new = coordinates[tracker_id][-1]

                    pixel_distance = abs(y_new - y_old)

                    distance_m = pixel_distance * PIXEL_TO_METER

                    time = WINDOW / FPS

                    speed = (distance_m / time) * 3.6 * 2.0

                    speed_history[tracker_id].append(speed)

                    avg_speed = int(np.mean(speed_history[tracker_id]))

                    # Lock the speed after enough samples
                    if len(speed_history[tracker_id]) >= 8:
                        final_speed[tracker_id] = avg_speed

                    labels.append(f"#{tracker_id} {avg_speed} km/h")

                else:
                    labels.append(f"#{tracker_id}")
                
                
            annotated_frame=frame.copy()
            annotated_frame=trace_annotator.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame=bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame=label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            video_writer.write(annotated_frame)

        video_writer.release()

        return output_path


