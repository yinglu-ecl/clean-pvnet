import os
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from lib.utils import data_utils
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
from lib.utils.tless import tless_config
import json
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.vsd import inout
from PIL import Image
from lib.csrc.nn import nn_utils
import yaml
if cfg.test.un_pnp:
    from lib.csrc.uncertainty_pnp import un_pnp_utils
    import scipy
if cfg.test.icp or cfg.test.vsd:
    from lib.utils.icp import icp_utils
    # from lib.utils.icp.icp_refiner.build import ext_
import torch
import cv2
from transforms3d.quaternions import mat2quat, quat2mat
import tqdm
import time


class Evaluator:

    def __init__(self, result_dir):
        # from IPython import embed; embed()
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        if cfg.test.det_gt:
            timestamp = 'gt_'+timestamp
        if cfg.training:
            self.result_dir = os.path.join(result_dir, 'train_'+timestamp)
        else:
            self.result_dir = os.path.join(result_dir, 'test_'+timestamp)
        os.makedirs(self.result_dir)
        self.log_file = os.path.join(self.result_dir, 'log.txt')
        self.log = open(self.log_file, 'w')
        args = DatasetCatalog.get(cfg.test.dataset)
        self.log_print('args: {}'.format(args))
        self.ann_file = args['ann_file']
        self.obj_id = int(args['obj_id'])
        self.coco = coco.COCO(self.ann_file)
        message = 'annotations loaded from: {}'.format(self.ann_file)
        self.log_print(message)
        self.gt_img_ids = self.coco.getImgIds(catIds=[self.obj_id])
        message = 'length of gt_img_ids: {}'.format(len(self.gt_img_ids))
        self.log_print(message)

        model_dir = 'data/tless/models_cad'
        obj_path = os.path.join(model_dir, 'obj_{:02d}.ply'.format(self.obj_id))
        self.model = inout.load_ply(obj_path)
        self.model_pts = self.model['pts'] / 1000.

        model_info = yaml.load(open(os.path.join(model_dir, 'models_info.yml')))
        self.diameter = model_info[self.obj_id]['diameter'] / 1000.

        self.vsd = []
        self.adi = []
        self.cmd5 = []

        self.icp_vsd = []
        self.icp_adi = []
        self.icp_cmd5 = []

        self.pose_per_id = []
        self.pose_icp_per_id = []
        self.img_ids = []
        if cfg.test.eval_scene:
            self.scene_ids = []
        if cfg.test.eval_vis:
            self.vis_percents = []

        self.height = 540
        self.width = 720

        if cfg.test.icp or cfg.test.vsd:
            self.icp_refiner = icp_utils.ICPRefiner(self.model, (self.width, self.height))
            # model_path = os.path.join(model_dir, 'colobj_{:02d}.ply'.format(self.obj_id))
            # self.icp_refiner = ext_.Synthesizer(os.path.realpath(model_path))
            # self.icp_refiner.setup(self.width, self.height)

    def log_print(self, message):
        print(message)
        self.log.write(message)
        self.log.write('\n')

    def log_finish(self):
        self.log.close()

    def vsd_metric(self, pose_pred, pose_gt, K, depth_path, icp=False):
        from lib.utils.vsd import vsd_utils

        depth = inout.load_depth(depth_path) * 0.1
        im_size = (depth.shape[1], depth.shape[0])
        dist_test = vsd_utils.misc.depth_im_to_dist_im(depth, K)

        delta = tless_config.vsd_delta
        tau = tless_config.vsd_tau
        cost_type = tless_config.vsd_cost
        error_thresh = tless_config.error_thresh_vsd

        depth_gt = {}
        dist_gt = {}
        visib_gt = {}

        for pose_pred_ in pose_pred:
            R_est = pose_pred_[:, :3]
            t_est = pose_pred_[:, 3:] * 1000
            depth_est = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_est, t_est)
            # depth_est = self.opengl.render(im_size, 100, 10000, K, R_est, t_est)
            dist_est = vsd_utils.misc.depth_im_to_dist_im(depth_est, K)

            for gt_id, pose_gt_ in enumerate(pose_gt):
                R_gt = pose_gt_[:, :3]
                t_gt = pose_gt_[:, 3:] * 1000
                if gt_id not in visib_gt:
                    depth_gt_ = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_gt, t_gt)
                    # depth_gt_ = self.opengl.render(im_size, 100, 10000, K, R_gt, t_gt)
                    dist_gt_ = vsd_utils.misc.depth_im_to_dist_im(depth_gt_, K)
                    dist_gt[gt_id] = dist_gt_
                    visib_gt[gt_id] = vsd_utils.visibility.estimate_visib_mask_gt(
                        dist_test, dist_gt_, delta)

                e = vsd_utils.vsd(dist_est, dist_gt[gt_id], dist_test, visib_gt[gt_id],
                                  delta, tau, cost_type)
                if e < error_thresh:
                    return 1

        return 0

    def adi_metric(self, pose_pred, pose_gt, percentage=0.1):
        diameter = self.diameter * percentage
        for pose_pred_ in pose_pred:
            for pose_gt_ in pose_gt:
                model_pred = np.dot(self.model_pts, pose_pred_[:, :3].T) + pose_pred_[:, 3]
                model_targets = np.dot(self.model_pts, pose_gt_[:, :3].T) + pose_gt_[:, 3]
                idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
                mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
                if mean_dist < diameter:
                    return 1
        return 0

    def adi_correct(self, pose_pred, pose_gt, percentage=0.1):
        rslt = [0 for i in range(len(pose_gt))]
        rslt2 = [0 for i in range(len(pose_pred))]
        diameter = self.diameter * percentage
        for i in range(len(pose_gt)):
            pose_gt_ = pose_gt[i]
            for j in range(len(pose_pred)):
                pose_pred_ = pose_pred[j]
                model_pred = np.dot(self.model_pts, pose_pred_[:, :3].T) + pose_pred_[:, 3]
                model_targets = np.dot(self.model_pts, pose_gt_[:, :3].T) + pose_gt_[:, 3]
                idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
                mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
                if mean_dist < diameter:
                    rslt[i] = 1
                    rslt2[j] = 1
        return rslt, rslt2

    def cm_degree_5_metric(self, pose_pred, pose_gt):
        for pose_pred_ in pose_pred:
            for pose_gt_ in pose_gt:
                trans_distance, ang_distance = pvnet_pose_utils.cm_degree_5(pose_pred_, pose_gt_)
                if trans_distance < 5 and ang_distance < 5:
                    return 1
        return 0

    def cm_degree_5_correct(self, pose_pred, pose_gt):
        rslt = [0 for i in range(len(pose_gt))]
        for i in range(len(pose_gt)):
            pose_gt_ = pose_gt[i]
            for pose_pred_ in pose_pred:
                trans_distance, ang_distance = pvnet_pose_utils.cm_degree_5(pose_pred_, pose_gt_)
                if trans_distance < 5 and ang_distance < 5:
                    rslt[i] = 1
        return rslt

    def uncertainty_pnp(self, kpt_3d, kpt_2d, var, K):
        cov_invs = []
        for vi in range(var.shape[0]):
            if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
            else:
                cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
                cov_invs.append(cov_inv)

        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d, K)

        return pose_pred

    def icp_refine(self, pose_pred, depth_path, mask, K):
        depth = inout.load_depth(depth_path).astype(np.int32) / 10.
        mask = mask.astype(np.int32)
        pose = pose_pred.astype(np.float32)

        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def icp_refine_(self, pose, depth_path, mask, K):
        depth = inout.load_depth(depth_path).astype(np.uint16)
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        box = cv2.boundingRect(mask.astype(np.uint8))
        x, y = box[0] + box[2] / 2., box[1] + box[3] / 2.
        z = np.mean(depth[mask != 0] / 10000.)
        x = ((x - K[0, 2]) * z) / float(K[0, 0])
        y = ((y - K[1, 2]) * z) / float(K[1, 1])
        center = [x, y, z]
        pose[:, 3] = center

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        zfar = 6.0
        znear = 0.25
        factor = 10000.
        error_threshold = 0.01

        rois = np.zeros([1, 6], dtype=np.float32)
        rois[:, :] = 1

        self.icp_refiner.solveICP(mask, depth,
                                  self.height, self.width,
                                  fx, fy, px, py,
                                  znear, zfar,
                                  factor,
                                  rois.shape[0], rois,
                                  poses, poses_new, poses_icp,
                                  error_threshold
                                  )

        pose_icp = np.zeros([3, 4], dtype=np.float32)
        pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
        pose_icp[:, 3] = poses_icp[0, 4:]

        return pose_icp

    def evaluate(self, output, batch, debug=''):
        if len(debug)>0:
            debugout = debug
            debug = True
        else:
            debug = False

        # batch is a dictionary with keys: 'inp', 'meta'
        # batch['inp'] is a tensor of size [3, 3, 128, 128]
        # batch['meta'] is a dictionary with keys: 'center'(a list of tensor with size 1*2), 
        # 'scale'(a list of tensors with size 1), 
        # 'box'(a list of tensors with size 1*4), 
        # 'img_id' (tensor of size 1), 'pose_test' (a list with an empty string)
        # output is a dictionary with keys: 'seg', 'vertex', 'mask', 'kpt_2d'
        img_id = int(batch['meta']['img_id'])
        self.img_ids.append(img_id)
        img_data = self.coco.loadImgs(int(img_id))[0]
        # img_data is a dictionary, an example:
        # {'file_name': 'data/tless/test_primesense/17/rgb/0081.png',
        # 'depth_path': 'data/tless/test_primesense/17/depth/0081.png',
        # 'height': 540,
        # 'width': 720,
        # 'id': 8192}
        if cfg.test.eval_scene:
            scene_id = int(img_data['file_name'].split('/')[-3])

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.obj_id)
        # ann_ids is a list of annotation ids in this image, for example [48861, 48862, 48863]
        annos = self.coco.loadAnns(ann_ids)
        # annos is a list of annotation dictionaries, each dictionary contains keys:
        # 'corner_3d', 'corner_2d', 'center_3d', 'center_2d', 'fps_3d', 'fps_2d', 'K', 'pose', 'category_id', 'image_id', 'id'
        # a new key: 'vis_percent'
        if cfg.test.eval_scene:
            for i in range(len(annos)):
                self.scene_ids.append(scene_id)
        if cfg.test.eval_vis:
            vis_percent_list = [ann['vis_percent'] for ann in annos]
            self.vis_percents += vis_percent_list
        kpt_3d = np.concatenate([annos[0]['fps_3d'], [annos[0]['center_3d']]], axis=0) # same for all annos
        corner_3d = np.array(annos[0]['corner_3d']) # same for all annos
        K = np.array(annos[0]['K']) # same for all annos
        pose_gt = [np.array(anno['pose']) for anno in annos] # a list of poses each correspond to one anno

        kpt_2d = output['kpt_2d'].detach().cpu().numpy()
        centers = batch['meta']['center']
        scales = batch['meta']['scale']
        boxes = batch['meta']['box']
        h, w = batch['inp'].size(2), batch['inp'].size(3)

        pose_preds = []
        pose_preds_icp = []
        # from IPython import embed; embed()
        # assert len(vis_percent_list)==len(centers)
        for i in range(len(centers)): # loop for each annotation in meta
            # why the length of output equals that of meta?
            center = centers[i].detach().cpu().numpy()
            scale = scales[i].detach().cpu().numpy()
            kpt_2d_ = kpt_2d[i]
            trans_inv = data_utils.get_affine_transform(center[0], scale[0], 0, [w, h], inv=1)
            kpt_2d_ = data_utils.affine_transform(kpt_2d_, trans_inv)
            if cfg.test.un_pnp:
                var = output['var'][i].detach().cpu().numpy()
                pose_pred = self.uncertainty_pnp(kpt_3d, kpt_2d_, var, K)
            else:
                pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_, K)
            pose_preds.append(pose_pred)

            if cfg.test.icp:
                depth_path = img_data['depth_path']
                seg = torch.argmax(output['seg'][i], dim=0).detach().cpu().numpy()
                seg = seg.astype(np.uint8)
                seg = cv2.warpAffine(seg, trans_inv, (self.width, self.height), flags=cv2.INTER_NEAREST)
                pose_pred_icp = self.icp_refine(pose_pred.copy(), depth_path, seg.copy(), K.copy())
                pose_preds_icp.append(pose_pred_icp)

        if cfg.test.icp:
            self.icp_adi.append(self.adi_metric(pose_preds_icp, pose_gt))
            self.icp_cmd5.append(self.cm_degree_5_metric(pose_preds_icp, pose_gt))
            self.pose_icp_per_id.append(pose_preds_icp)

        self.log_print('image: {}, obj: {}, length pred: {}, length gt: {}'.format(img_data['file_name'], annos[0]['category_id'], len(pose_preds), len(pose_gt)))
        # from IPython import embed; embed()
        # self.adi.append(self.adi_metric(pose_preds, pose_gt))
        tempo_adi, tempo_adi2 = self.adi_correct(pose_preds, pose_gt)
        self.adi += tempo_adi
        # self.cmd5.append(self.cm_degree_5_metric(pose_preds, pose_gt))
        tempo_cmd5 = self.cm_degree_5_correct(pose_preds, pose_gt)
        self.cmd5 += tempo_cmd5
        self.log_print('adi: {}, cmd5: {}'.format(tempo_adi, tempo_cmd5))
        self.pose_per_id.append(pose_preds)
        # print('image: {}, length img_ids: {}, current annos: {}'.format(img_id, len(self.img_ids), len(annos)))

        if debug:
            from PIL import Image
            import imageio, cv2
            img_color = np.array(Image.open(img_data['file_name']))
            for pose_gt_ in pose_gt:
                corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt_)
                cv2.polylines(img_color, pts=np.int32([corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]]]), isClosed=True, color=(255,0,0), thickness=1)
                cv2.polylines(img_color, pts=np.int32([corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]]]), isClosed=True, color=(255,0,0), thickness=1)
            im_inputs = batch['inp'].detach().cpu().numpy()
            for i in range(len(centers)):
                im_in = np.transpose(im_inputs[i,:,:,:],(1,2,0))
                seg = torch.argmax(output['seg'][i], dim=0).detach().cpu().numpy()
                seg = 255 * np.expand_dims(seg.astype('uint8'), axis=2)
                zeroseg = np.zeros(seg.shape, dtype='uint8')
                seg = np.concatenate((zeroseg, seg, zeroseg), axis=2)
                t_img = seg.copy()
                kpt_2d_ = kpt_2d[i]
                for j in range(kpt_2d_.shape[0]):
                    point = (kpt_2d_[j,0], kpt_2d_[j,1])
                    cv2.circle(t_img, point, 1, (255,0,0), 2)
                gt_kpt_2d = np.array(annos[0]['fps_2d'])
                for j in range(gt_kpt_2d.shape[0]):
                    point = (int(gt_kpt_2d[j,0]), int(gt_kpt_2d[j,1]))
                    cv2.circle(t_img, point, 1, (0,0,255), 1)
                im_in = im_in - np.min(im_in)
                if np.max(im_in)>0:
                    im_in = im_in/np.max(im_in)
                im_in = im_in * 255
                im_in = im_in.astype('uint8')
                added_img = cv2.addWeighted(im_in, 0.7, t_img, 0.3, 0)
                if cfg.test.eval_scene:
                    image_name = int(img_data['file_name'].split('/')[-1].split('.')[0])
                    savename = os.path.join(debugout, 'ADI{:01d}_scene{:02d}_img{:04d}_org_seg_kpts{:02d}.png'.format(tempo_adi2[i], scene_id, image_name, i))
                else:
                    image_name = img_data['file_name'].split('/')[-2]
                    savename = os.path.join(debugout, 'ADI{}_img{}_ann{}.png'.format(tempo_adi2[i], image_name, i))
                imageio.imwrite(savename, added_img)
                ###
                pose_pred = pose_preds[i]
                corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
                cv2.polylines(img_color, pts=np.int32([corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]]]), isClosed=True, color=(0,0,255), thickness=1)
                cv2.polylines(img_color, pts=np.int32([corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]]]), isClosed=True, color=(0,0,255), thickness=1)
            if cfg.test.eval_scene:
                savename = os.path.join(debugout, 'ADI{:01d}-{:01d}_scene{:02d}_img{:04d}_pred_vs_gt.png'.format(sum(tempo_adi2), len(tempo_adi2), scene_id, image_name))
            else:
                savename = os.path.join(debugout, 'ADI{}_img{}_pred_vs_gt.png'.format(tempo_adi2[i], image_name))
            imageio.imwrite(savename, img_color)
    def summarize_vsd(self, pose_preds, img_ids, vsd):
        for pose_pred, img_id in tqdm.tqdm(zip(pose_preds, img_ids)):
            img_data = self.coco.loadImgs(int(img_id))[0]
            depth_path = img_data['depth_path']

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.obj_id)
            annos = self.coco.loadAnns(ann_ids)
            K = np.array(annos[0]['K'])
            pose_gt = [np.array(anno['pose']) for anno in annos]
            vsd.append(self.vsd_metric(pose_pred, pose_gt, K, depth_path))

    def summarize(self):
        self.log_print('##########')
        self.log_print('cfg: {}'.format(cfg))
        self.log_print('##########')
        if cfg.test.vsd:
            self.log_print('summarizing vsd...')
            from lib.utils.vsd import vsd_utils
            # self.opengl = vsd_utils.renderer.DepthRender(self.model, (720, 540))
            self.summarize_vsd(self.pose_per_id, self.img_ids, self.vsd)
            if cfg.test.icp:
                self.summarize_vsd(self.pose_icp_per_id, self.img_ids, self.icp_vsd)
            self.pose_per_id = []
            self.pose_icp_per_id = []
            self.img_ids = []

        # from IPython import embed; embed()
        # here is strange, len(self.img_ids) = len(self.gt_img_ids) = len(self.cmd5) = 2927
        # but len(self.gt_img_ids) = 3014
        '''vsd = np.sum(self.vsd) / len(self.gt_img_ids)
        adi = np.sum(self.adi) / len(self.gt_img_ids)
        cmd5 = np.sum(self.cmd5) / len(self.gt_img_ids)
        icp_vsd = np.sum(self.icp_vsd) / len(self.gt_img_ids)
        icp_adi = np.sum(self.icp_adi) / len(self.gt_img_ids)
        icp_cmd5 = np.sum(self.icp_cmd5) / len(self.gt_img_ids)'''
        # the following 3 lines are corrected
        # vsd = np.sum(self.vsd) / len(self.img_ids)
        # adi = np.sum(self.adi) / len(self.img_ids)
        # cmd5 = np.sum(self.cmd5) / len(self.img_ids)
        vsd = np.sum(self.vsd) / len(self.vsd)
        adi = np.sum(self.adi) / len(self.adi)
        cmd5 = np.sum(self.cmd5) / len(self.cmd5)

        self.log_print('vsd metric: {}/{}, result: {}'.format(np.sum(self.vsd), len(self.vsd), vsd))
        self.log_print('adi metric: {}/{}, result: {}'.format(np.sum(self.adi), len(self.adi), adi))
        self.log_print('5 cm 5 degree metric: {}/{}, result: {}'.format(np.sum(self.cmd5), len(self.cmd5), cmd5))

        ### result v.s. visibility
        if cfg.test.eval_vis:
            np.save(os.path.join(self.result_dir, 'adi_results.npy'), self.adi)
            np.save(os.path.join(self.result_dir, 'cmd5_results.npy'), self.cmd5)
            np.save(os.path.join(self.result_dir, 'visibilities.npy'), self.vis_percents)

        ### result per scene
        # from IPython import embed; embed()
        if cfg.test.eval_scene:
            self.log_print('length scene ids: {}, length adi: {}'.format(len(self.scene_ids), len(self.adi)))
            scenes = np.unique(self.scene_ids)
            for sce in scenes:
                # inds = [i for i, x in enumerate(self.scene_ids) if x==sce]
                # vsd = np.sum([self.vsd[i] for i in inds])/len(inds)
                # adi = np.sum([self.adi[i] for i in inds])/len(inds)
                # cmd5 = np.sum([self.cmd5[i] for i in inds])/len(inds)
                inds = np.array(self.scene_ids) == sce
                metric_results = []
                metric_names = ['vsd', 'adi', 'cmd5']
                for metric in [self.vsd, self.adi, self.cmd5]:
                    if len(metric)==len(inds):
                        metric_rslt = np.sum(np.array(metric)[inds]) / np.sum(inds)
                    else:
                        metric_rslt = 0
                        self.log_print('metric {}, length not correct {}/{}, return 0'.format(metric_names[len(metric_results)],len(metric),len(inds)))
                    metric_results.append(metric_rslt)
                vsd, adi, cmd5 = metric_results
                self.log_print('scene: {}\n vsd metric: {}\n adi metric: {}\n 5 cm 5 degree metric: {}'.format(sce, vsd, adi, cmd5))
        ###
        if cfg.test.icp:
            # the following 3 lines are corrected
            icp_vsd = np.sum(self.icp_vsd) / len(self.img_ids)
            icp_adi = np.sum(self.icp_adi) / len(self.img_ids)
            icp_cmd5 = np.sum(self.icp_cmd5) / len(self.img_ids)
            self.log_print('vsd metric after icp: {}'.format(icp_vsd))
            self.log_print('adi metric after icp: {}'.format(icp_adi))
            self.log_print('5 cm 5 degree metric after icp: {}'.format(icp_cmd5))

        self.vsd = []
        self.adi = []
        self.cmd5 = []
        self.icp_vsd = []
        self.icp_adi = []
        self.icp_cmd5 = []
        # self.log_finish()

        return {'vsd': vsd, 'adi': adi, 'cmd5': cmd5}
