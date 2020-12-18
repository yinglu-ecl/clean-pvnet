# clpvnet_evaluate_single.py
# ying 2020/07/08

from lib.config import cfg, args
import numpy as np
import os
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.utils.tless import tless_test_utils
import cv2
import yaml
import json

def read_image(img_path):
	img = cv2.imread(img_path)
	bboxes = [[0.0, 0.0, img.shape[0], img.shape[1]]]
	return img, bboxes

def get_bboxes(bb_path, obj_name):
	bboxes = []
	rets = json.load(open(bb_path,'r'))
	for i in range(len(rets['bboxes'])):
		if rets['cat_ids'][i] == obj_name:
			bboxes.append(rets['bboxes'][i])
	return bboxes

def expand_bbox(img_size, bbox, scale):
	print('original bbox {}'.format(bbox))
	bbox_w = bbox[3] - bbox[1]
	bbox_h = bbox[2] - bbox[0]
	new_w = bbox_w * scale
	new_h = bbox_h * scale
	bbox[0] = max(int(bbox[0] - (new_h - bbox_h)/2), 0)
	bbox[1] = max(int(bbox[1] - (new_w - bbox_w)/2), 0)
	bbox[2] = min(int(bbox[0] + new_h), img_size[0])
	bbox[3] = min(int(bbox[1] + new_w), img_size[1])
	print('expanded bbox {}'.format(bbox))
	return bbox

def evaluate_single_image(img_path, K, fps_3d, corner_3d, bboxes=None, pose_gts=None, save_dir=None, expbb_scale=1):
	network = make_network(cfg).cuda()
	load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
	network.eval()
	visualizer = make_visualizer(cfg)
	if bboxes is None:
		img, bboxes = read_image(img_path)
	else:
		img = cv2.imread(img_path)
	data_list = [tless_test_utils.pvnet_transform(img, box) for box in bboxes]
	orig_imgs, inps, centers, scales = [list(d) for d in zip(*data_list)]
	inps = torch.from_numpy(np.array(inps)).cuda()
	centers = [torch.from_numpy(np.expand_dims(c,axis=0)) for c in centers]
	scales = [torch.from_numpy(np.expand_dims(s,axis=0)) for s in scales]

	ret = {'inp': inps}
	if not expbb_scale==1:
		bboxes = [expand_bbox(img.shape, bbox, expbb_scale) for bbox in bboxes]
	bboxes = [np.array(box) for box in bboxes]
	bboxes = [torch.from_numpy(np.expand_dims(box,axis=0)) for box in bboxes]
	meta = {'center': centers, 'scale': scales, 'box': bboxes, 'img_path': img_path, 'pose_test': ''}
	ret.update({'meta': meta})

	# from IPython import embed; embed()
	output = network(ret['inp'])
	# if pose_gts is None:
	# 	visualizer.visualize_ext(output, ret, K, fps_3d, corner_3d)
	# else:

	visualizer.visualize_ext(output, ret, K, fps_3d, corner_3d, pose_gts=pose_gts, save_dir=save_dir)

def compute_projection(points_3D, transformation, internal_calibration):
	print('points_3D shape', points_3D.shape)
	print('transformation shape', transformation.shape)
	print('internal_calibration shape', internal_calibration.shape)
	projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
	camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
	projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
	projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
	return projections_2d

if __name__=='__main__':
	
	fps_3d = np.array([[0.06891841125488281, -0.03479923629760742, -0.0009470702409744262],
                       [-0.06889827728271485, -0.034754451751708985, 0.0022315025329589845],
                       [-0.04479666137695312, 0.00125, -0.02536104393005371],
                       [0.04402099609375, 0.00125, 0.025576007843017577],
                       [0.04336452865600586, 0.0009661999940872193, -0.025466848373413085],
                       [-0.04319840621948242, 0.0011616064310073853, 0.026025938034057616],
                       [-0.02130895805358887, -0.03675, -0.016588050842285155],
                       [0.020614850997924806, -0.03675, 0.01703841209411621],
                       [0.0, 0.0, 0.0]])
	corner_3d = np.array([[-0.06897419691085815, -0.03674999997019768, -0.026032600551843643],
                          [-0.06897419691085815, -0.03674999997019768, 0.026032600551843643],
                          [-0.06897419691085815, 0.03674999997019768, -0.026032600551843643],
                          [-0.06897419691085815, 0.03674999997019768, 0.026032600551843643],
                          [0.06897419691085815, -0.03674999997019768, -0.026032600551843643],
                          [0.06897419691085815, -0.03674999997019768, 0.026032600551843643],
                          [0.06897419691085815, 0.03674999997019768, -0.026032600551843643],
                          [0.06897419691085815, 0.03674999997019768, 0.026032600551843643]])
	
	with open('/home/ying/Documents/Bitbucket/pikaflex/temp_data/real_test_sample01/info.yml', 'r') as f:
		cam_infos = yaml.load(f, Loader=yaml.CLoader)
		cam_info = cam_infos[1] # camera2
	K = np.array(cam_info['cam_K']).reshape((3,3))
	img_path_mask = '/media/ying/DATA/Pikaflex/real_photos/20200703Images/rgbs/rgb{:02d}.png'
	bb_path_mask = '/media/ying/DATA/ubuntu/GitData/output/20200703Images_CN_dets_epoch110_withbb/rgb{:02d}_annos_over0.50.json'
	save_dir_mask = '/media/ying/DATA/ubuntu/GitData/output/20200703Images_CN_dets_epoch110_expbb1-2/rgb{:02d}_3dbb_over0.50.png'
	for i in range(1,12):
		img_path = img_path_mask.format(i)
		bb_path = bb_path_mask.format(i)
		bboxes = get_bboxes(bb_path, 23)
		save_dir = save_dir_mask.format(i)
		evaluate_single_image(img_path, K, fps_3d, corner_3d, bboxes=bboxes, save_dir=save_dir, expbb_scale=1)

	# img_path = '/media/ying/DATA/Pikaflex/real_photos/20200703Images/rgbs/rgb05.png'
	# evaluate_single_image(img_path, K, fps_3d, corner_3d)
	# '''
	# K = np.array([1075.65091572, 0.0, 222.06888344, 0.0, 1073.90347929, 175.72159802, 0.0, 0.0, 1.0]).reshape((3,3))
	# bboxes = [np.array([80, 139, 236, 130])]
	# R = np.array([0.99872571, 0.05014835, 0.00559502, 0.05044692, -0.98983481, -0.13297146, -0.00113016, 0.13308457, -0.99110434]).reshape((3,3))
	# R = np.dot(R, np.eye(3).astype('float32') * 1000)
	# t = np.array([-13.04494187, 14.76332966, 639.79415515]).reshape((3,1))
	# pose_gts = [np.concatenate((R,t),axis=1)]
	# img_path = '/media/ying/DATA/Pikaflex/DATA/T-LESS/t-less_v2/train_primesense/23/rgb/0000.png'
	# evaluate_single_image(img_path, K, fps_3d, corner_3d, pose_gts=pose_gts)
	# # evaluate_single_image(img_path, K, fps_3d, corner_3d, bboxes=bboxes, pose_gts=pose_gts)
	
	'''
	# These lines are for testing old pvnet trained models on clpvnet
	# but the prediction does not work, the ground-truth is correctly shown
	K = np.array([1075.65091572, 0.0, 224.06888344, 0.0, 1073.90347929, 167.72159802, 0.0, 0.0, 1.0]).reshape((3,3))
	R = np.array([0.99968707, 0.02496792, -0.00160221, 0.02462464, -0.99322556, -0.11356043, -0.00442673, 0.11348608, -0.99352966]).reshape((3,3))
	R = np.dot(R, np.eye(3).astype('float32') * 1000)
	t = np.array([-14.09421585, 19.09785011, 634.54525338]).reshape((3,1))
	img_path = '/media/ying/DATA/Pikaflex/DATA/T-LESS/t-less_v2/train_primesense/01/rgb/0000.png'
	# pose_gts = [np.concatenate((R,t),axis=1)]
	
	from lib.utils.old_data_utils import TlessModelDB
	modeldb = TlessModelDB()
	obj_name = 1
	fps_3d = modeldb.get_farthest_3d(obj_name)
	center_3d = modeldb.get_centers_3d(obj_name).reshape((1,3))
	corner_3d = modeldb.get_corners_3d(obj_name)
	fps_3d = np.concatenate((fps_3d,center_3d), axis=0)
	# from IPython import embed; embed()
	rectify = np.array([[1,0,0],[0,0,-1],[0,1,0]]).astype('float32')
	R = np.dot(R,rectify)
	pose_gts = [np.concatenate((R,t),axis=1)]
	evaluate_single_image(img_path, K, fps_3d, corner_3d, pose_gts=pose_gts)
	'''

	'''
	# The following lines are for testing visualization on synthetic images
	img_path = '/media/ying/DATA/Pikaflex/Output_tless/first_generate/mixed_01/5/RGB.png'
	K = np.array([1910.81005859375, 0.0, 512.0, 0.0, 1910.81005859375, 512.0, 0.0, 0.0, 1.0]).reshape((3,3))
	R = np.array([0.0008381366496905684, -0.0002287440438522026, 0.0004951799637638032, -0.0002857915973891074, -0.0009573942491580122, 4.146814832741916e-05, 0.00046459681325115274, -0.0001762742544984472, -0.0008678002859065174]).reshape((3,3))
	rectify = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype('float32')
	R = np.dot(R,rectify) * 1000.0
	t = np.array([0.08531713113188744, 0.07507159847136635, 1.3287777908245788]).reshape((3,1))
	pose_gts = [np.concatenate((R,t),axis=1)]
	bboxes = [np.array([550, 563, 550+192, 563+118])]

	# ###check image and pose ## unfinished
	# from PIL import Image
	# import sys, cv2, imageio
	# sys.path.append('/home/ying/Documents/Bitbucket/pikaflex/data_processing/ssd_6d')
	# from rendering.model import Model3D
	# img_color = np.array(Image.open(img_path))
	# model = Model3D()
	# model.load('/media/ying/DATA/Pikaflex/DATA/T-LESS/t-less_v2/models_cad/obj_23.ply', scale=1.0/1000.0)
	# transform = np.concatenate((R, t), axis=1)
	# bb_ssp = model.bb_ssp
	# bb_ssp4 = np.concatenate((bb_ssp,np.ones((bb_ssp.shape[0],1),dtype='float32')),axis=1)
	# bb_2d_points = compute_projection(bb_ssp4.T, transform, K)
	# cv2.rectangle(img_color, (bboxes[0][0],bboxes[0][1]), (bboxes[0][2],bboxes[0][3]), (255,0,0), 2)
	# img_color = draw_3dbb_ssp(img_color, bb_2d_points, 3, 0.5)
	# ###

	evaluate_single_image(img_path, K, fps_3d, corner_3d, bboxes=bboxes, pose_gts=pose_gts)
	'''