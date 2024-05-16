import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display
import tqdm
import concurrent.futures
from depth_optical import VideoToPointCloud, uvd2xyz
from annotation_generator import AnnotationGenerator
import json
import cv2


def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'
class DataProcessor:
    def __init__(self, datasets, output_dir, checkpoint_path):
        self.datasets = datasets
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.video_processor = VideoToPointCloud()
        self.annotation_generator = AnnotationGenerator(
            grounding_dino_config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounding_dino_checkpoint="./groundingdino_swint_ogc.pth",
            sam_encoder_version="vit_h",
            sam_checkpoint="./sam_vit_h_4b8939.pth",
            device='cuda'
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                self.processed_datasets = set(f.read().splitlines())
        else:
            self.processed_datasets = set()

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'w') as f:
            for dataset in self.processed_datasets:
                f.write(f"{dataset}\n")

    def process_video(self, dataset_name, b):
        try:
            ds = b.as_dataset(split='train[:10]').shuffle(10)
            episode = next(iter(ds))
            frames = [step['observation']['image'].numpy() for step in episode['steps']]
            camera_params = b.info.metadata['camera']

            depth_maps = self.video_processor.estimate_depth(frames)
            flows = self.video_processor.compute_optical_flow(frames)
            background_masks = self.video_processor.identify_background(flows)
            aligned_depth_maps = self.video_processor.align_depth_maps(depth_maps, background_masks)

            # 从数据集中获取相机参数
            poses = []
            for key in camera_params['view_params']:
                value = camera_params['view_params'][key]
                pose = np.array(value["extrinsic"]).reshape(4, 4)
                poses.append(pose)
            K = np.array(camera_params["intrinsic"])

            point_clouds = uvd2xyz(aligned_depth_maps, K, poses, 1000)

            # 生成3D注释
            text_instructions = "A running dog and a cat"  # 这里可以根据实际情况修改
            parsed_classes = self.annotation_generator.parse_instructions(text_instructions)
            annotations = self.annotation_generator.process_point_cloud_frame(frames[0], point_clouds, parsed_classes)

            # 返回处理后的结果
            return frames, point_clouds, annotations

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            return None

    def save_results(self, frames, point_clouds, annotations, dataset_name):
        try:
            dataset_output_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            # 保存视频帧
            for i, frame in enumerate(frames):
                frame_path = os.path.join(dataset_output_dir, f'frame_{i}.jpg')
                cv2.imwrite(frame_path, frame)

            # 保存点云数据
            point_cloud_path = os.path.join(dataset_output_dir, 'point_cloud.npy')
            np.save(point_cloud_path, point_clouds)

            # 保存注释数据
            annotations_path = os.path.join(dataset_output_dir, 'annotations.json')
            with open(annotations_path, 'w') as f:
                json.dump(annotations, f, indent=4)

            print(f"Saved results for dataset {dataset_name}")

        except Exception as e:
            print(f"Error saving results for dataset {dataset_name}: {e}")

    def process_datasets(self):
        for dataset in tqdm.tqdm(self.datasets):
            if dataset in self.processed_datasets:
                continue

            builder_dir = dataset2path(dataset)
            b = tfds.builder_from_directory(builder_dir)
            future = self.executor.submit(self.process_video, dataset, b)
            future.add_done_callback(lambda f: self.handle_result(f, dataset))

    def handle_result(self, future, dataset_name):
        result = future.result()
        if result:
            frames, point_clouds, annotations = result
            self.save_results(frames, point_clouds, annotations, dataset_name)
            self.processed_datasets.add(dataset_name)
            self.save_checkpoint()
        else:
            print(f"Failed to process dataset {dataset_name}")

# 初始化并处理数据集
datasets = [
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
]
output_dir = './processed_data'
checkpoint_path = './checkpoint.txt'
data_processor = DataProcessor(datasets, output_dir, checkpoint_path)
data_processor.process_datasets()
