import os
import cv2
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

if __name__ == "__main__":
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file("configs/1.yaml")
    cfg.MODEL.WEIGHTS = "/output/exp_5/modul_final.pth"
    model = DAobjTwoStagePseudoLabGeneralizedRCNN(cfg)
    model.eval()

    image_paths = [img] # 您的图像文件路径列表
    for image_path in image_paths:
        im = cv2.imread(image_path)
        predictions = model([{"image": im}])
        visualized_output = model.visualize_training(
            [{"image": im, "instances": predictions["instances"]}],
            predictions["proposals"],
            branch="supervised"
        )
        cv2.imshow("Visualized Output", visualized_output)
        cv2.waitKey(0)