import cv2
import numpy as np
import torch
import torchvision
import supervision as sv
import spacy
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

class GroundedSAM:
    """ 组合GroundingDINO和SAM模型进行目标检测和分割 """
    def __init__(self, dino_config, dino_checkpoint, sam_encoder_version, sam_checkpoint, device='cuda'):
        self.device = torch.device(device)
        
        # 初始化GroundingDINO模型
        self.dino_model = Model(model_config_path=dino_config, model_checkpoint_path=dino_checkpoint)
        
        # 初始化SAM模型
        self.sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        
    def detect_objects(self, image, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8):
        """ 使用GroundingDINO模型进行目标检测并应用NMS后处理 """
        detections = self.dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # NMS后处理
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()
        
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        
        return detections
        
    def segment(self, image, detections):
        """ 使用SAM模型根据GroundingDINO检测到的框进行分割 """
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.array(result_masks)
        return detections


class AnnotationGenerator:
    def __init__(self, grounding_dino_config, grounding_dino_checkpoint, sam_encoder_version, sam_checkpoint, device='cuda'):
        """ 初始化注释生成器，集成GroundingDINO和SAM模型 """
        self.device = torch.device(device)
        self.nlp = spacy.load("en_core_web_sm")
        self.grounded_sam = GroundedSAM(
            dino_config=grounding_dino_config,
            dino_checkpoint=grounding_dino_checkpoint,
            sam_encoder_version=sam_encoder_version,
            sam_checkpoint=sam_checkpoint,
            device=device
        )

    def parse_instructions(self, text):
        """ 使用 spaCy 解析指令中的名词短语 """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def extract_2d_masks(self, image, classes):
        """ 使用GroundedSAM模型提取2D遮罩 """
        detections = self.grounded_sam.detect_objects(image, classes)
        detections = self.grounded_sam.segment(image, detections)
        return detections

    def generate_3d_bounding_boxes(self, point_cloud, masks):
        """ 从点云和2D遮罩生成3D边界框 """
        bounding_boxes = []
        for mask in masks:
            points = np.argwhere(mask)
            selected_points = point_cloud[points[:, 0], points[:, 1], :]
            if len(selected_points) == 0:
                continue
            min_point = np.min(selected_points, axis=0)
            max_point = np.max(selected_points, axis=0)
            bounding_boxes.append((min_point, max_point))
        return bounding_boxes

    def process_point_cloud_frame(self, image, point_cloud, classes):
        """ 处理点云帧以生成3D注释 """
        detections = self.extract_2d_masks(image, classes)
        bounding_boxes = self.generate_3d_bounding_boxes(point_cloud, detections.mask)
        
        annotations = []
        for i, bbox in enumerate(bounding_boxes):
            annotation = {
                "class": classes[detections.class_id[i]],
                "confidence": detections.confidence[i],
                "mask": detections.mask[i],
                "3d_bounding_box": bbox
            }
            annotations.append(annotation)
        
        return annotations


if __name__ == '__main__':
    # 示例使用
    from depth_optical import VideoToPointCloud,uvd2xyz
    video_processor = VideoToPointCloud()
    frames = video_processor.load_video('path_to_video.mp4')
    depth_maps = video_processor.estimate_depth(frames)
    flows = video_processor.compute_optical_flow(frames)
    background_masks = video_processor.identify_background(flows)
    aligned_depth_maps = video_processor.align_depth_maps(depth_maps, background_masks)

    # 假设相机内参已知
    poses = []
    value = meta["view_params"][key]
    pose = np.array(value["extrinsic"]).reshape(4, 4)
    poses.append(pose)
    K = np.array(value["intrinsic"])
    point_clouds = uvd2xyz(aligned_depth_maps,K,poses,1000)

    # 初始化AnnotationGenerator
    annotation_generator = AnnotationGenerator(
        grounding_dino_config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="./groundingdino_swint_ogc.pth",
        sam_encoder_version="vit_h",
        sam_checkpoint="./sam_vit_h_4b8939.pth",
        device='cuda'
    )

    # 指令和类别解析
    text_instructions = "A running dog and a cat"
    parsed_classes = annotation_generator.parse_instructions(text_instructions)

    # 加载图像和点云数据
    image = cv2.imread("./assets/demo2.jpg")
    point_cloud = np.load("./assets/point_cloud_data.npy")  # 示例点云数据

    # 获取3D注释
    annotations = annotation_generator.process_point_cloud_frame(image, point_cloud, parsed_classes)

    # 打印详细的注释结果
    for annotation in annotations:
        print(annotation)
