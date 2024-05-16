import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
from RAFT_optical_flow.core.raft import RAFT
from RAFT_optical_flow.core.utils import InputPadder
repo = "isl-org/ZoeDepth"
model_path = "RAFT_optical_flow/models/raft-things.pth"
# Zoe_N
ZoeDepth = torch.hub.load(repo, "ZoeD_N", pretrained=True)
RAFT_model = torch.nn.DataParallel(RAFT({
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False
    }))
RAFT_model.load_state_dict(torch.load(model_path))

RAFT_model = RAFT_model.module
def uvd2xyz(depth, K, extrinsic, depth_trunc=np.inf):
    """
    depth: of shape H, W
    K: 3, 3
    extrinsic: 4, 4
    return points: of shape H, W, 3
    """
    depth[depth > depth_trunc] = 0
    mask = depth > 0
    H, W = depth.shape
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    x = np.arange(0, W) - cx
    y = np.arange(0, H) - cy
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    points = points * depth[..., None]
    points[..., 0] /= fx
    points[..., 1] /= fy
    points = points.reshape(-1, 3)
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    points = points @ np.linalg.inv(extrinsic).T
    points = points[:, :3].reshape(H, W, 3)
    points = points[mask]
    return points
class VideoToPointCloud:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.depth_model = ZoeDepth.to(self.device)
        self.optical_flow_model = RAFT_model.to(self.device)

    def load_video(self, video_path):
        """加载视频并将每帧转换为张量"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = ToTensor()(Image.fromarray(frame))
            frames.append(frame_tensor)
        cap.release()
        return frames

    def estimate_depth(self, frames):
        """估计每一帧的深度"""
        depth_maps = []
        for frame in frames:
            frame = frame.unsqueeze(0).to(self.device)
            depth_map = self.depth_model(frame)
            depth_maps.append(depth_map.squeeze(0).cpu().detach())
        return depth_maps

    def compute_optical_flow(self,frames):
        """计算光流"""
        flows = []

        with torch.no_grad():
            for img1, img2 in zip(frames[:-1], frames[1:]):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                _, flow_up = self.optical_flow_model(img1, img2)
                flows.append(flow_up)

        return flows
    def identify_background(self, flows, threshold=0.5):
        """根据光流估计背景区域
        Args:
            flows: 光流数据列表
            threshold: 用于判断是否为背景的阈值
        """
        background_masks = []
        for flow in flows:
            mag = torch.norm(flow, dim=0)
            mask = mag < threshold  # 背景区域判定
            background_masks.append(mask.numpy())
        return background_masks

    def align_depth_maps(self, depth_maps, background_masks):
        """对背景区域的深度图进行一致性调整"""
        aligned_depth_maps = []
        for depth_map, mask in zip(depth_maps[:-1], background_masks):
            depth_map_copy = depth_map.numpy()
            aligned_depth_map = depth_map_copy * mask
            aligned_depth_maps.append(aligned_depth_map)
        aligned_depth_maps.append(depth_maps[-1].numpy())  # 最后一帧的深度图保持原样
        return aligned_depth_maps

    
if __name__ == '__main__':
    # 示例使用
    video_processor = VideoToPointCloud()
    frames = video_processor.load_video('path_to_video.mp4')
    depth_maps = video_processor.estimate_depth(frames)
    flows = video_processor.compute_optical_flow(frames)
    background_masks = video_processor.identify_background(flows)
    aligned_depth_maps = video_processor.align_depth_maps(depth_maps, background_masks)

