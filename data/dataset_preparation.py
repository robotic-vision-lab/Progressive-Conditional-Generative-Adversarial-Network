#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu> 

Script to extract point cloud from mesh surface of ShapeNetCore dataset. It will 
iterate through all the sub-directories of a given class and generate a pcd file
with sub-directory name.

#TODO: Use appropriate texture value instead of a single pixel value

"""
import cv2
import pandas as pd
import re
import numpy as np
import os

def read_file(path):
    """
    Read files and gather points and color information.
    """
    # store vertex points
    v = []
    # store vertex normals
    vn = []
    # store face points
    f = []
    # store material names
    mtl_name = []
    color = {}

    obj_file = path+'/model.obj'
    mtl_file = path+'/model.mtl'
    curr_mtl = ''

    # read material file
    with open(mtl_file) as mtl:
        for line in mtl:

            # get mtl name from line
            if line.startswith('newmtl'):
                curr_mtl = line.strip().split()[1]

            # get mtl color from line
            elif line.startswith('Kd'):
                color[curr_mtl] = line.strip().split()[1:]

            # collect texture value from image if provided
            elif line.startswith('map_Kd'):
                img_path = path + line.strip().split()[1][1:]
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # storing rgb value from mid pixel of texture image
                    color[curr_mtl] = img[img.shape[0]//2,img.shape[1]//2]/255

    data = {'color' : color}
    curr_mtl = ''

    with open(obj_file) as obj:
        for line in obj:

            # store current mtl name to a list of appropriate length before next
            #   object.
            if line.startswith('o ') and curr_mtl:
                mtl_name += [curr_mtl]*(len(f)-len(mtl_name))
                curr_mtl = ''
            elif line.startswith('v '):
                v.append(line.strip()[1:].split())

            elif line.startswith('vn'):
                vn.append(line.strip()[2:].split())

            # face values combined with vertex values define the surface
            # face format - vertex/../..; saving only vertex values.
            elif line.startswith('f'):
                temp = [re.split(r'\D+', x) for x in [line.strip()[1:].lstrip()]][0]
                if len(temp)==3: f.append(temp)
                if len(temp)==6: f.append([temp[0],temp[2],temp[4]])
                elif len(temp)==9: f.append([temp[0],temp[3],temp[6]])

            # disregard past mtl. keep track of latest ones only.
            elif line.startswith('usemtl'):
                curr_mtl = line.strip().split()[1]

    # store last mtl name to a list of appropriate length.
    mtl_name += [curr_mtl]*(len(f)-len(mtl_name))

    # storing the vertex points.
    points = pd.DataFrame(v, dtype='f4', columns=["x", "y", "z", "w"][:len(v[0])])

    # storing vertex normals.
    if len(vn) > 0:
        points = points.join(pd.DataFrame(vn, dtype='f4', columns=['nx', 'ny', 'nz']))

    data["points"] = points

    if len(f) < 1:
        return data

    # storing mesh faces.
    mesh_columns = ['v1','v2','v3']
    mesh = pd.DataFrame(f, dtype='i4', columns=mesh_columns).astype('i4')

    # because index starts from 1 in obj file
    mesh -= 1
    mesh['mtl'] = mtl_name

    data["mesh"] = mesh

    return data

def triangle_area_multi(v1, v2, v3):
    """
    v1, v2, v3 are (N,3) arrays. Each one represents the vertices
    such as v1[i], v2[i], v3[i] represent the ith triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

def compute_points(data, n=2048):
    """
    data: pyntcloud object representing points and mesh; may or may not hold vertex
    texture/rgb values.
    rgb: boolean flag;
    """

    points_xyz = data['points'][['x', 'y', 'z']].values
    color = np.array([data['color'][m] for m in data['mesh']['mtl']], dtype='f4')

    v1 = points_xyz[data['mesh']["v1"].values]
    v2 = points_xyz[data['mesh']["v2"].values]
    v3 = points_xyz[data['mesh']["v3"].values]

    v1_xyz, v2_xyz, v3_xyz = v1[:, :3], v2[:, :3], v3[:, :3]

    areas = triangle_area_multi(v1_xyz, v2_xyz, v3_xyz)
    probabilities = areas / np.sum(areas)
    random_idx = np.random.choice(
        np.arange(len(areas)), size=n, p=probabilities)

    v1_xyz = v1_xyz[random_idx]
    v2_xyz = v2_xyz[random_idx]
    v3_xyz = v3_xyz[random_idx]

    # (n, 1) the 1 is for broadcasting
    u = np.random.uniform(low=0., high=1., size=(n, 1))
    v = np.random.uniform(low=0., high=1-u, size=(n, 1))

    result = pd.DataFrame()

    result_xyz = (v1_xyz * u) + (v2_xyz * v) + ((1 - (u + v)) * v3_xyz)
    result_xyz = result_xyz.astype(np.float32)



    result["x"] = result_xyz[:, 0]
    result["y"] = result_xyz[:, 1]
    result["z"] = result_xyz[:, 2]
    result['r'] = color[:,0][random_idx]
    result['g'] = color[:,1][random_idx]
    result['b'] = color[:,2][random_idx]


    return result

def create_pcd(mesh_points, filename='', verbose=True):
    """
    will create point cloud from mesh points and write in disk and display
    """
    if verbose:
        print(filename)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(mesh_points[['x', 'y', 'z']].values)
    pcd.colors = o3d.utility.Vector3dVector(mesh_points[['r', 'g', 'b']].values)
    o3d.io.write_point_cloud(filename, pcd)

        # o3d.visualization.draw_geometries([pcd], height=750, width=750)


def create_npz(mesh_points, filename='', verbose=True):
    if not filename.split('.')[-1] == 'npz': filename += '.npz'
    if verbose:
        print(filename)

    np.savez(filename, mesh_points)

def extract_all_from_top_dir(file_top_dir='./', n=10240, save_path=''):
    """
    given a base directory ex: ShapeNetCore.v1, this function will create point cloud for all the mesh files in sub-directory i.e all class.
    """
#    saving_dir = 'ShapeNetCore.v1_pclouds'
    if not os.path.isdir(save_path): os.mkdir(save_path)
    sub_dirs = [d for d in os.listdir(file_top_dir) if os.path.isdir(os.path.join(file_top_dir,d))]
    for sd in sub_dirs: #iterating through all the sub-directories
        saving_sub_dir = os.path.join(save_path,sd)
        if not os.path.isdir(saving_sub_dir): os.mkdir(saving_sub_dir)
        file_sub_dir = os.path.join(file_top_dir,sd)
        extract_all_in_dir(file_sub_dir,n,saving_sub_dir)

def extract_all_in_dir(file_sub_dir,n,saving_sub_dir=''):
    """
    given a class directory this function will create and save point cloud for all the mesh files for that class.
    """
    if not os.path.isdir(saving_sub_dir): os.mkdir(saving_sub_dir)
    file_dirs = [d for d in os.listdir(file_sub_dir) if os.path.isdir(os.path.join(file_sub_dir,d))]
    for cd in file_dirs:    #iterating through all the directories
        saving_path = os.path.join(saving_sub_dir, cd)
        file_path = os.path.join(file_sub_dir,cd)
        extract_from_single_file(file_path, n, saving_path)

def extract_from_single_file(path, n, save_path=''):
    """
    given a directory of OBJ and MTL files this function will create and save point cloud.
    """
    obj = read_file(path)
    mesh_points = compute_points(obj, n)
    if save_path.split('.')[-1] == 'pcd':
        create_pcd(mesh_points, filename=save_path)
    else:
        create_npz(mesh_points, filename=save_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--choice", type=int, default=1,
                        help="extract pcd from one file(0)/one class(1)/all class(2).")
    parser.add_argument("-f", "--obj_path", help="path of OBJ file directory")
    parser.add_argument("-n", default=2048, type=int, help="number of points to be extracted; default=2048")
    parser.add_argument("-t", "--save_path", help="path to save the pcd file.")
    args = parser.parse_args()

    if args.choice==2:
        extract_all_from_top_dir(args.obj_path, args.n, args.save_path)
    elif args.choice==1:
        extract_all_in_dir(args.obj_path, args.n, args.save_path)
    else:
        extract_from_single_file(args.obj_path, args.n, args.save_path)



if __name__ == '__main__':
    main()
