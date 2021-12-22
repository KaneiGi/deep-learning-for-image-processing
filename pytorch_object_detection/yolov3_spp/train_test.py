import datetime
import argparse
import sys
from email.policy import default

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyperparameters):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device.type))

    wdir = 'weights' + os.sep
    best = wdir + 'best.pt'
    results_file = 'results{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)
    weights = opt.weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size
    multi_scale = opt.multi_scale

    gs = 32
    assert math.fmod(imgsz_test, gs) == 0, 'img-size %g must be a %g-multiple' % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print('Using multi_scale training,image range [{},{}]'.format(imgsz_min, imgsz_max))

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])
    hyp['cls'] *= nc / 80
    hyp['obj'] *= imgsz_test / 320

    for f in glob.glob(results_file):
        os.remove(f)

    model = Darknet(cfg).to(device)

    if opt.freeze_layers:

        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list)
                                if isinstance(module, YOLOLayer)]
        freeze_layer_indeces = [x for x in range(len(model.module_list))
                                if (x not in output_layer_indices)
                                and (x - 1 not in output_layer_indices)]

        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)
    else:

        darknet_end_layer = 74
        for idx in range(darknet_end_layer + 1):
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp['momentum'],
                          weight_decay=hyp['weight_decay'], nesterov=True)

    start_epoch = 0
    best_map = 0.0

    if weights.endswith('.pt') or weights.endswith('.pth'):
        checkpoint = torch.load(weights, map_location=device)

        try:
            checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                                   model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(checkpoint['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_map' in checkpoint.keys():
                best_map = checkpoint["best_map"]

        if checkpoint.get('training_results') is not None:
            with open(results_file, 'w', encoding='utf-8') as file:
                file.write(checkpoint['training_results'])

        start_epoch = checkpoint['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, checkpoint['epoch'], epochs))
            epochs += checkpoint['epoch']

        del checkpoint

    lf = lambda x: ((1 + math.cos(x / epochs * math.pi) / 2) * (1 - hyp['lrf']) + hyp['lrf'])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=True,
                                        hyp=hyp,
                                        rect=opt.rect,
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls)

    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0 if sys.platform == 'win32' else 8])  # number of workers

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    shuffle=not opt.rect,
                                                    pin_memory=True,
                                                    collate_fn=train_dataset.collate_fn)

    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0

    coco = get_coco_api_from_dataset(val_dataset)

    print('starting training for %g epochs...' % epochs)
    print("Using %g dataloader workers" % nw)

    for epoch in range(start_epoch, epochs):
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,
                                               img_size=imgsz_train,
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,
                                               grid_max=grid_max,
                                               gs=gs,
                                               print_freq=50,
                                               warmup=True)

        scheduler.step()

        if opt.no_test is False or epoch == epochs - 1:
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            with open(results_file, 'a') as f:
                result_info = [str(round(i, 4) for i in result_info + [mloss.tolist()[-1]] + [str(round(lr, 6))])]
                txt = 'epoch:{} {}'.format(epoch, ' '.join(result_info))
                f.write(txt + '\n')

            if coco_mAP > best_map:
                best_map = coco_mAP

            if opt.savebest is False:
                with open(results_file,'r') as f:
                    save_files ={
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'training_results':f.read(),
                        'epoch':epoch,
                        'best_map':best_map
                    }
                    torch.save(save_files,'./weights/yolov3spp-{}.pt'.format(epoch))

            else:
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map
                        }
                        torch.save(save_files, './weights/yolov3spp-{}.pt'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help='*.cfg.path')
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data.path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi_scale', type=bool, default=True,
                        help=r'adjust 67% - 150% img_size every 10 batch_size')
    parser.add_argument('--img_size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--save_best', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--no_test', action='store_true', help='only test final epoch')
    parser.add_argument('--cache_images_path', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    parser.add_argument('--name', type=str, default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', type=str, default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze_layers', type=bool, default=False, help='freeze non_output layers')
    opt = parser.parse_args()

    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
