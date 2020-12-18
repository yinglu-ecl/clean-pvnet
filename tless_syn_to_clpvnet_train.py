# tless_syn_to_clpvnet_train.py
# ying 2020/07/15
# based on clpvnet/tools/handle_custom_dataset.py

import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
import imageio
from lib.utils import base_utils
import json
import yaml

data_root = '/home/ying/Documents/Bitbucket/pikaflex/temp_data/SYN_TLESS_MIXED_01'
RGB_mask = os.path.join(data_root, 'mixed_01', '{}', 'RGB.png')
MASK_mask = os.path.join(data_root, 'mixed_01', '{}', 'mask.png')
new_mask_dir = os.path.join(data_root, 'single_object_masks')
GT_file = os.path.join(data_root, 'gt.yml')
K = np.array([1910.81005859375, 0.0, 512.0, 0.0, 1910.81005859375, 512.0, 0.0, 0.0, 1.0]).reshape((3,3))
OCCLUSION_file = os.path.join(data_root, 'occlusion_truncated_list.txt')
VIS_THRES = 0.01 # visibility threshold
RECTIFY = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype('float32')

def load_gt(path):
	with open(path, 'r') as f:
		gts = yaml.load(f, Loader=yaml.CLoader)
		for im_id, gts_im in gts.items():
			for gt in gts_im:
				if 'cam_R_m2c' in gt.keys():
					gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
				if 'cam_t_m2c' in gt.keys():
					gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
	return gts

def occ_list_2_dict(occ_list):
	occ_dict = {}
	for line in occ_list:
		items = line.strip().split()
		if int(items[0]) in occ_dict.keys():
			if int(items[1]) in occ_dict[int(items[0])].keys():
				print('already exists', occ_dict[int(items[0])][int(items[1])], float(items[4]))
				continue
		else:
			occ_dict[int(items[0])] = {}
		occ_dict[int(items[0])][int(items[1])] = float(items[4])
	return occ_dict

def find_nearest(in_list, in_num):
	for num in in_list:
		if abs(num-in_num) <= 2:
			out_num = num
	return out_num

def get_model_corners(model):
	min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
	min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
	min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
	corners_3d = np.array([
		[min_x, min_y, min_z],
		[min_x, min_y, max_z],
		[min_x, max_y, min_z],
		[min_x, max_y, max_z],
		[max_x, min_y, min_z],
		[max_x, min_y, max_z],
		[max_x, max_y, min_z],
		[max_x, max_y, max_z],
	])
	return corners_3d

def tless_syn_to_clpvnet(obj_name, split, save_path, debug=False):
	model_path = '/media/ying/DATA/ubuntu/GitData/clpvnet/tless/models_cad/colobj_{:02d}.ply'.format(obj_name)
	renderer = OpenGLRenderer(model_path)
	model = renderer.model['pts'] / 1000
	corner_3d = get_model_corners(model)
	center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
	fps_3d = np.loadtxt('/media/ying/DATA/ubuntu/GitData/clpvnet/tless/farthest/farthest_{:02d}.txt'.format(obj_name))

	ann_id = 0
	images = []
	annotations = []

	if split == 'train':
		image_set = [i for i in range(1, 10001)]
	elif split == 'val':
		image_set = [i for i in range(10001, 11001)]
	elif split == 'test':
		image_set = [i for i in range(11001, 15001)]

	with open(OCCLUSION_file, 'r') as fid:
		occlusion_dict = occ_list_2_dict(fid.readlines())
	print('occlusion dictionary loaded from {}, size {}'.format(OCCLUSION_file, len(occlusion_dict)))

	gts = load_gt(GT_file)
	print('ground truth file loaded from {}'.format(GT_file))

	for im_name in image_set:
		print('{}/{}'.format(im_name, max(image_set)))
		rgb_path = RGB_mask.format(im_name)
		rgb = Image.open(rgb_path)
		img_size = rgb.size
		info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': im_name}
		images.append(info)
		obj_num = len(gts[im_name])
		for obj_ind in range(obj_num):
			if gts[im_name][obj_ind]['obj_id'] == obj_name:
				vis_percent = 1.0
				if im_name in occlusion_dict.keys():
					if obj_ind in occlusion_dict[im_name].keys():
						vis_percent = occlusion_dict[im_name][obj_ind]
				if vis_percent >= VIS_THRES:

					R = gts[im_name][obj_ind]['cam_R_m2c']
					R = np.dot(R,RECTIFY) * 1000.0
					t = gts[im_name][obj_ind]['cam_t_m2c']
					pose = np.concatenate((R,t), axis=1)

					corner_2d = base_utils.project(corner_3d, K, pose)
					center_2d = base_utils.project(center_3d[None], K, pose)[0]
					fps_2d = base_utils.project(fps_3d, K, pose)

					old_mask_path = MASK_mask.format(im_name)
					old_mask = np.array(Image.open(old_mask_path))
					elements = np.unique(old_mask)
					elements = [ele for ele in elements if ele>0]
					obj_mask = find_nearest(elements, (obj_ind+1)*255/obj_num)
					new_mask = np.ma.getmaskarray(np.ma.masked_equal(old_mask, obj_mask)) * 255
					new_mask = new_mask.astype('uint8')
					new_mask_path = os.path.join(new_mask_dir, '{:08d}_{:03d}.png'.format(im_name, obj_ind))
					imageio.imwrite(new_mask_path, new_mask)

					ann_id += 1
					anno = {'mask_path': new_mask_path, 
							'image_id': im_name, 
							'category_id': 1, 
							'id': ann_id, 
							'corner_3d': corner_3d.tolist(), 
							'corner_2d': corner_2d.tolist(), 
							'center_3d': center_3d.tolist(), 
							'center_2d': center_2d.tolist(), 
							'fps_3d': fps_3d.tolist(), 
							'fps_2d': fps_2d.tolist(), 
							'K': K.tolist(), 
							'pose': pose.tolist(), 
							'data_root': '', 
							'type': 'syn_photo_real', 
							'cls': obj_name}
					annotations.append(anno)

					if debug:
						import matplotlib.pyplot as plt
						import matplotlib.patches as patches
						_, ax = plt.subplots(1)
						ax.imshow(np.array(rgb))
						ax.add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
						ax.add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
						for p in range(fps_2d.shape[0]):
							point = fps_2d[p,:]
							ax.add_patch(patches.Circle(point, 1, color='r'))
						plt.show()


	categories = [{'supercategory': 'none', 
				   'id': 1, 
				   'name': obj_name}]

	instance = {'images': images, 
				'annotations': annotations, 
				'categories': categories}

	with open(save_path, 'w') as f:
		json.dump(instance, f)

def tless_syn_to_clpvnet_test(obj_name, split, save_ann_path, save_det_path, debug=False):
	model_path = '/media/ying/DATA/ubuntu/GitData/clpvnet/tless/models_cad/colobj_{:02d}.ply'.format(obj_name)
	renderer = OpenGLRenderer(model_path)
	model = renderer.model['pts'] / 1000
	corner_3d = get_model_corners(model)
	center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
	fps_3d = np.loadtxt('/media/ying/DATA/ubuntu/GitData/clpvnet/tless/farthest/farthest_{:02d}.txt'.format(obj_name))

	ann_id = 0
	images = []
	images_det = []
	annotations = []
	detections = []

	if split == 'train':
		image_set = [i for i in range(1, 10001)]
	elif split == 'val':
		image_set = [i for i in range(10001, 11001)]
	elif split == 'test':
		image_set = [i for i in range(11001, 15001)]

	with open(OCCLUSION_file, 'r') as fid:
		occlusion_dict = occ_list_2_dict(fid.readlines())
	print('occlusion dictionary loaded from {}, size {}'.format(OCCLUSION_file, len(occlusion_dict)))

	gts = load_gt(GT_file)
	print('ground truth file loaded from {}'.format(GT_file))

	for im_name in image_set:
		print('{}/{}'.format(im_name, max(image_set)))
		rgb_path = RGB_mask.format(im_name)
		rgb = Image.open(rgb_path)
		img_size = rgb.size
		info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': im_name}
		images.append(info)
		info2 = {'rgb_path': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': im_name}
		images_det.append(info2)
		obj_num = len(gts[im_name])
		for obj_ind in range(obj_num):
			if gts[im_name][obj_ind]['obj_id'] == obj_name:
				vis_percent = 1.0
				if im_name in occlusion_dict.keys():
					if obj_ind in occlusion_dict[im_name].keys():
						vis_percent = occlusion_dict[im_name][obj_ind]
				if vis_percent >= VIS_THRES:
					R = gts[im_name][obj_ind]['cam_R_m2c']
					R = np.dot(R,RECTIFY) * 1000.0
					t = gts[im_name][obj_ind]['cam_t_m2c']
					pose = np.concatenate((R,t), axis=1)

					corner_2d = base_utils.project(corner_3d, K, pose)
					center_2d = base_utils.project(center_3d[None], K, pose)[0]
					fps_2d = base_utils.project(fps_3d, K, pose)

					old_mask_path = MASK_mask.format(im_name)
					old_mask = np.array(Image.open(old_mask_path))
					elements = np.unique(old_mask)
					elements = [ele for ele in elements if ele>0]
					obj_mask = find_nearest(elements, (obj_ind+1)*255/obj_num)
					new_mask = np.ma.getmaskarray(np.ma.masked_equal(old_mask, obj_mask)) * 255
					new_mask = new_mask.astype('uint8')
					new_mask_path = os.path.join(new_mask_dir, '{:08d}_{:03d}.png'.format(im_name, obj_ind))
					imageio.imwrite(new_mask_path, new_mask)

					ann_id += 1
					anno = {'mask_path': new_mask_path, 
							'image_id': im_name, 
							'category_id': obj_name, #1, 
							'id': ann_id, 
							'corner_3d': corner_3d.tolist(), 
							'corner_2d': corner_2d.tolist(), 
							'center_3d': center_3d.tolist(), 
							'center_2d': center_2d.tolist(), 
							'fps_3d': fps_3d.tolist(), 
							'fps_2d': fps_2d.tolist(), 
							'K': K.tolist(), 
							'pose': pose.tolist(), 
							'data_root': '', 
							'type': 'syn_photo_real', 
							'cls': obj_name}
					annotations.append(anno)

					bbox = gts[im_name][obj_ind]['obj_bb']
					det = {'area': bbox[2] * bbox[3],
						   'image_id': im_name,
						   'bbox': bbox,
						   'iscrowd': 0,
						   'category_id': obj_name, #1,
						   'id': ann_id}
					detections.append(det)

					if debug:
						import matplotlib.pyplot as plt
						import matplotlib.patches as patches
						_, ax = plt.subplots(1)
						ax.imshow(np.array(rgb))
						ax.add_patch(patches.Polygon(xy=corner_2d[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
						ax.add_patch(patches.Polygon(xy=corner_2d[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
						for p in range(fps_2d.shape[0]):
							point = fps_2d[p,:]
							ax.add_patch(patches.Circle(point, 1, color='r'))
						point = np.array([bbox[0], bbox[1]])
						ax.add_patch(patches.Rectangle(point, bbox[2], bbox[3], color='b', fill=False))
						plt.show()


	categories = [{'supercategory': 'none', 
				   'id': obj_name, #1, 
				   'name': obj_name}]

	instance = {'images': images, 
				'annotations': annotations, 
				'categories': categories}

	with open(save_ann_path, 'w') as f:
		json.dump(instance, f)

	# categories2 = [{'supercategory': 'none', 
	# 			   'id': obj_name, 
	# 			   'name': obj_name}]

	instance2 = {'images': images_det,
				 'annotations': detections,
				 'categories': categories}

	with open(save_det_path, 'w') as f:
		json.dump(instance2, f)

if __name__=='__main__':
	# for split in ['test','val']:
	# 	for obj_name in range(1,31):
	# 		save_ann_path = '/media/ying/DATA/ubuntu/GitData/clpvnet/cache/tless_syn_photo_real_annos/{:s}_{:02d}.json'.format(split, obj_name)
	# 		save_det_path = '/media/ying/DATA/ubuntu/GitData/clpvnet/cache/tless_syn_photo_real_gt_dets/{:s}_{:02d}.json'.format(split, obj_name)
	# 		# tless_syn_to_clpvnet(obj_name, split, save_path, debug=False)
	# 		tless_syn_to_clpvnet_test(obj_name, split, save_ann_path, save_det_path, debug=False)

	split = 'train'
	for obj_name in range(1, 23):
		save_path = '/media/ying/DATA/ubuntu/GitData/clpvnet/cache/tless_syn_photo_real_annos/{:s}_{:02d}.json'.format(split, obj_name)
		tless_syn_to_clpvnet(obj_name, split, save_path, debug=False)