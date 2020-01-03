import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(save_txt=False, save_img=False):
    opt_cfg = 'cfg/yolov3-tiny.cfg'
    opt_conf_thres = 0.3
    opt_device = ''
    opt_fourcc = 'mp4v'
    opt_half = False
    opt_img_size = 416
    opt_names = 'data/jy.names'
    opt_nms_thres = 0.5
    opt_output = 'yolo-result'
    opt_source = 'yolo-image'
    opt_view_img = False
    opt_weights = 'weights/backup490.pt'
    with torch.no_grad():
        img_size = (320, 192) if ONNX_EXPORT else opt_img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, half, view_img = opt_output, opt_source, opt_weights, opt_half, opt_view_img
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt_device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(opt_cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

            # Validate exported model
            import onnx
            model = onnx.load('weights/export.onnx')  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=img_size, half=half)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=img_size, half=half)

        # Get names and colors
        names = load_classes(opt_names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if opt_half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt_conf_thres, opt_nms_thres)

            # Apply
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt_fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))
