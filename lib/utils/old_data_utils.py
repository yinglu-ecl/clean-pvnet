# old_data_utils.py
# ying 2020/07/13
# some old data utils from pvnet_old
# for compatibility with tless data
import os
import numpy as np
from plyfile import PlyData
import yaml

TLESS_SYN = '/home/ying/Documents/Bitbucket/pikaflex/temp_data/SYN_TLESS_MIXED_01'
TLESS_REAL = '/media/ying/DATA/Pikaflex/DATA/T-LESS/t-less_v2'

class TlessModelDB(object):
    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {'8': {}, '4': {}, '12': {}, '16': {}, '20': {}}
    small_bbox_corners = {}

    def __init__(self):
        self.ply_pattern = os.path.join(TLESS_SYN, 'ply_files', 'obj_{:02d}.ply')
        self.farthest_pattern = os.path.join(TLESS_SYN, 'fps_points_m', 'obj_m_{:02d}_farthest{}.txt')
        self.diameter_file = os.path.join(TLESS_SYN, 'ply_files', 'models_info.yml')
        # get diameters
        with open(self.diameter_file,'r') as f:
            model_info = yaml.load(f, Loader=yaml.CLoader)
            for obj_ind in range(30):
                self.diameters[obj_ind+1] = model_info[obj_ind+1]['diameter']

    def get_corners_3d(self, obj_name):
        if obj_name in self.corners_3d:
            return self.corners_3d[obj_name]

        corner_pth = os.path.join(TLESS_SYN, 'markers', 'obj_{:02d}'.format(obj_name), 'corners.txt')
        if not os.path.exists(corner_pth):
            self.get_ply_model(obj_name)

        self.corners_3d[obj_name] = np.loadtxt(corner_pth)
        return self.corners_3d[obj_name]

    def get_small_bbox(self, obj_name):
        if obj_name in self.small_bbox_corners:
            return self.small_bbox_corners[obj_name]

        corners = self.get_corners_3d(obj_name)
        center = np.mean(corners, 0)
        small_bbox_corners = (corners - center[None, :])*2.0/3.0 + center[None, :]
        self.small_bbox_corners[obj_name] = small_bbox_corners

        return small_bbox_corners

    def get_ply_model(self, obj_name):
        if obj_name in self.models:
            return self.models[obj_name]

        ply = PlyData.read(self.ply_pattern.format(obj_name))
        data = ply.elements[0].data
        x = data['x'] / 1000.0
        y = data['y'] / 1000.0
        z = data['z'] / 1000.0
        model = np.stack([x,y,z], axis=-1)
        self.models[obj_name] = model # / 1000.0
        # get corners
        corner_pth = os.path.join(TLESS_SYN, 'markers', 'obj_{:02d}'.format(obj_name))
        if not os.path.exists(corner_pth):
            os.makedirs(corner_pth)
        corner_pth = os.path.join(corner_pth, 'corners.txt')
        if not os.path.exists(corner_pth):
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            min_z, max_z = np.min(z), np.max(z)
            corners_3d = np.array([
                [min_x, min_y, min_z],
                [min_x, min_y, max_z],
                [min_x, max_y, min_z],
                [min_x, max_y, max_z],
                [max_x, min_y, min_z],
                [max_x, min_y, max_z],
                [max_x, max_y, min_z],
                [max_x, max_y, max_z],])
            self.corners_3d[obj_name] = corners_3d
            np.savetxt(corner_pth, corners_3d)

        return model

    def get_diameter(self, obj_name):
        return self.diameters[obj_name]

    def get_centers_3d(self, obj_name):
        if obj_name in self.centers_3d:
            return self.centers_3d[obj_name]

        c3d = self.get_corners_3d(obj_name)
        self.centers_3d[obj_name] = (np.max(c3d,0) + np.min(c3d,0))/2
        return self.centers_3d[obj_name]

    def get_farthest_3d(self, obj_name, num=8):
        if obj_name in self.farthest_3d[str(num)]:
            return self.farthest_3d[str(num)][obj_name]

        farthest_path = self.farthest_pattern.format(obj_name, num)
        farthest_pts = np.loadtxt(farthest_path)
        self.farthest_3d[str(num)][obj_name] = farthest_pts
        return farthest_pts

    def get_ply_mesh(self, obj_name):
        ply = PlyData.read(self.ply_pattern.format(obj_name))
        vert = np.asarray([ply['vertex'].data['x'],ply['vertex'].data['y'],ply['vertex'].data['z']]).transpose()
        vert_id = [id for id in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id,np.int64)

        return vert, vert_id