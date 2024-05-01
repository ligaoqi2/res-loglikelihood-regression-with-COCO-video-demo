import torch
import torch.multiprocessing as mp
from rlepose.models import builder
from rlepose.opt import cfg, opt
from rlepose.utils.env import init_dist
from rlepose.utils.transforms import get_coord
from rlepose.utils.presets.simple_transform import SimpleTransform

import torchvision
import torch.backends.cudnn as cudnn

import cv2

from tqdm import tqdm

from utils import get_person_detection_boxes, get_pose_estimation_prediction, draw_pose, draw_bbox


def main():
    if opt.launcher in ['none', 'slurm']:
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    cudnn.benchmark = True

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    # device
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # box model
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    # pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f'Loading model from {opt.checkpoint}...')
    pose_model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)
    pose_model.to(CTX)
    pose_model = torch.nn.parallel.DistributedDataParallel(pose_model, device_ids=(0,))
    pose_model.eval()

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)
    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)

    # video config
    video_path = './your-video-path.mp4'

    video_name = video_path.split('/')[-1]
    cap = cv2.VideoCapture(video_path)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = '../output/' + 'RLE_' + video_name
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    COCODataset = builder.build_dataset(cfg.DATASET.VAL,
                                        preset_cfg=cfg.DATA_PRESET,
                                        train=False,
                                        heatmap2coord=cfg.TEST.HEATMAP2COORD)

    st = SimpleTransform(dataset=COCODataset,
                         scale_factor=0,
                         input_size=cfg.DATA_PRESET.IMAGE_SIZE,
                         output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
                         rot=0,
                         sigma=cfg.DATA_PRESET.SIGMA,
                         train=False
                         )

    for _ in tqdm(range(int(count)), desc="Processing: "):
        ret, image_bgr = cap.read()
        if ret:
            image = image_bgr[:, :, [2, 1, 0]]
            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
            input.append(img_tensor)

            # person detection
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

            # pose estimation
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    draw_bbox(box, image_bgr)
                    bbox = [box[0][0], box[0][1], box[1][0], box[1][1]]
                    image_pose = image.copy()
                    pose = get_pose_estimation_prediction(pose_model, image_pose, heatmap_to_coord, bbox, st)

                    if len(pose) >= 1:
                        for kpt in pose:
                            draw_pose(kpt, image_bgr)

            out.write(image_bgr)

    cap.release()
    out.release()
    print('video has been saved as {}'.format(save_path))


if __name__ == "__main__":
    # release first
    torch.cuda.empty_cache()

    # inference
    num_gpu = torch.cuda.device_count()
    if opt.world_size > num_gpu:
        print(f'Wrong world size. Changing it from {opt.world_size} to {num_gpu}.')
        opt.world_size = num_gpu
    main()
