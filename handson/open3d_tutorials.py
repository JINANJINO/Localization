import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# 1. load and visualize a point cloud
def show_point_cloud():
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("./fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
        zoom = 0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0474, 1.532],
        up=[-0.0694, -0.9768, 0.2024])
    return None

# 2. voxel downsampling
def voxel_downsample():
    print("Downsample the point cloud with a voxel of 0.05")
    pcd = o3d.io.read_point_cloud("./fragment.ply")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd],
        zoom = 0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0474, 1.532],
        up=[-0.0694, -0.9768, 0.2024])
    return None

# 3. estimate normals
def estimate_normals():
    print("Recompute the normal of the downsampled point cloud")
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd],
        zoom = 0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0474, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
        point_show_normal=True)
    return None
    
# 4. crop a point cloud
def crop_point_cloud1():
    print("Crop the point cloud")
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    print("Displaying original point cloud ...")
    o3d.visualization.draw_geometries([pcd],
        zoom = 0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0474, 1.532],
        up=[-0.0694, -0.9768, 0.2024])
    print("Please pick at least three points using[shift + left click]")
    print("After picking points, press 'q' to close the window")
    vis=o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()   # user picks points
    vis.destroy_window()
    print("The picked pints are")
    print(vis.get_picked_points())
    
    pcd_crop=pcd.select_by_index(vis.get_picked_points())
    print("Displaying cropped point cloud ... ")
    o3d.visualization.draw_geometries([pcd_crop],
        zoom = 0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0474, 1.532],
        up=[-0.0694, -0.9768, 0.2024])
    return None

def crop_point_cloud2():
    print("Crop the point using a polygon voluem")
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    print("Load a polygon volume and use it to crop the original point cloud")
    vol=o3d.visualization.read_selection_polygon_volume("./cropped.json")
    chair=vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair],
        zoom = 0.7,
        front=[0.5439, -0.2333, -0.8060],
        lookat=[2.4615, 2.1331, 1.338],
        up=[-0.1781, -0.9708, 0.1608])
    return None
    
# 5. paint a point cloud
def paint_point_cloud():
    print("Paint a point cloud")
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    vol=o3d.visualization.read_selection_polygon_volume("./cropped.json")
    chair=vol.crop_point_cloud(pcd)
    chair.paint_uniform_color([1, 0.706,0]) 
    o3d.visualization.draw_geometries([chair],
        zoom = 0.7,
        front=[0.5439, -0.2333, -0.8060],
        lookat=[2.4615, 2.1331, 1.338],
        up=[-0.1781, -0.9708, 0.1608])
    return None
    
# 6. point cloud distance
def point_cloud_distance():
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    vol=o3d.visualization.read_selection_polygon_volume("./cropped.json")
    chair = vol.crop_point_cloud(pcd)
    
    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0] # if point's distance is short to 0.01, point will remove
    pcd_without_chair = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_without_chair],
                                      zoom = 0.3412,
                                      front = [0.4257, -0.2125, -0.8795],
                                      lookat = [2.6172, 2.0475, 1.532],
                                      up = [-0.0694, -0.9768, 0.2024])
    return None

# 7. convex hull
def convex_hull():
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    vol=o3d.visualization.read_selection_polygon_volume("./cropped.json")
    chair = vol.crop_point_cloud(pcd)
    
    hull, _ = chair.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([chair, hull_ls])
    return None

# 8. clustering
def clustering():
    pcd=o3d.io.read_point_cloud("./fragment.ply")
    
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd],
                                      zoom = 0.455,
                                      front = [-0.4999, -0.1659, -0.8499],
                                      lookat = [2.1813, 2.0619, 2.0999],
                                      up = [0.1204, -0.9852, 0.1215])
    return None
    
# 9. plane segmentation
def plane_segmentation():
    pcd = o3d.io.read_point_cloud("./fragment.ply")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert = True)
    
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom = 0.3412,
                                      front = [0.4257, -0.2125, -0.8795],
                                      lookat = [2.6172, 2.0475, 1.532],
                                      up = [-0.0694, -0.9768, 0.2024])
    
    return None


#show_point_cloud()
#voxel_downsample()
#estimate_normals()
#crop_point_cloud1()
#crop_point_cloud2()
#paint_point_cloud()
#point_cloud_distance()
#convex_hull()
#clustering()
plane_segmentation()