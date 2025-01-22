import os
import random
import numpy as np
import open3d as o3d
import torch
import trimesh
import time

def get_files_with_string(root_dir, search_string):
    file_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if search_string in file:
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    
    return file_paths

start_time = time.time()
# Define folder paths and search string
raw_obj_file_dir = r'H:\moni\OBJ_file' # Path to the original OBJ files
output_obj_file_dir = r'H:\moni\test_pcd_obj' # Output path for all intermediate OBJ and PCD files
output_ply_file_dir = output_obj_file_dir # Output path for all PLY files, currently unused
dataset_pcd_file_dir = r'H:\moni\test_dataset'  # Folder for the final dataset, containing occluded and non-occluded point clouds
combined_mesh_file_path = r'H:\moni\merge_obj' # Path for the combined mesh file
# combined_pcd_file_path = r'H:\moni\merge_ply' # Path for the combined mesh file
search_string = '.obj'

input_file_name_list = get_files_with_string(raw_obj_file_dir, search_string)

obj_num = len(input_file_name_list)

def extract_random_files(file_list, num_files):
    random_files = random.sample(file_list, num_files)
    return random_files

# Extract random OBJ files
random_file_list = extract_random_files(input_file_name_list, 4)

# Create translation vectors for each OBJ file
np_arrays = [
    np.array([0.2, 0, 0.2]),
    np.array([0.2, 0, -0.2]),
    np.array([-0.2, 0, 0.2]),
    np.array([-0.2, 0, -0.2]),
    # Add more np arrays
]

# Combine file paths and translation vectors using zip
files_and_translations = list(zip(random_file_list, np_arrays))

def move_mesh(file_path, target_position, output_dir):
    # Read the OBJ file
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Get the vertex array
    vertices = np.asarray(mesh.vertices)
    
    # Calculate the current projection center
    current_projection_center_xz = np.mean(vertices[:, [0, 2]], axis=0)
    current_projection_center_y = np.min(vertices[:, 1])
    current_projection_center = np.array([current_projection_center_xz[0], current_projection_center_y, current_projection_center_xz[1]])
        
    # Calculate the translation vector
    translation_vector = target_position - current_projection_center
        
    # Apply the translation vector
    vertices += translation_vector
        
    # Update the mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
    # Construct the new file name and path
    directory = output_dir
    base_name = os.path.basename(file_path)
    new_file_name = os.path.splitext(base_name)[0] + '_moved.obj'
    new_file_path = os.path.join(directory, new_file_name)
        
    # Save the modified mesh to a new OBJ file
    o3d.io.write_triangle_mesh(new_file_path, mesh)
    
    return new_file_path

# Move each OBJ file
moved_obj_files_list = []
for file_path, target_position in files_and_translations:
    moved_file_path = move_mesh(file_path, target_position, output_obj_file_dir)
    moved_obj_files_list.append(moved_file_path)

# Convert OBJ files to PLY files using Open3D (directly using the vertices of the triangle mesh)
def convert_obj_to_ply_with_features(obj_file_path, ply_file_path):
    # Load the OBJ file
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    
    # Ensure the mesh contains color and normal information
    if not mesh.has_vertex_colors() or not mesh.has_vertex_normals():
        print("Warning: OBJ file lacks color or normal information.")
    
    # Create a point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices
    point_cloud.colors = mesh.vertex_colors
    point_cloud.normals = mesh.vertex_normals

    # Save as a PLY file
    o3d.io.write_point_cloud(ply_file_path, point_cloud, write_ascii=False)
    print(f"Point cloud saved to {ply_file_path}")

# Convert each OBJ file to a PLY file, resulting in a new list of PLY files
moved_pcd_files_list = []
for obj_filename in moved_obj_files_list:
    ply_filename = os.path.splitext(obj_filename)[0] + '.ply'
    convert_obj_to_ply_with_features(obj_filename, ply_filename)
    moved_pcd_files_list.append(ply_filename)

def iterative_farthest_point_sampling_torch(points, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(points).float().to(device)
    num_points = points.shape[0]
    sampled_idxs = torch.zeros(k, dtype=torch.long).to(device)
    sampled_idxs[0] = torch.randint(0, num_points, (1,)).item()
    dists = torch.norm(points - points[sampled_idxs[0]], dim=1)
    
    for i in range(1, k):
        farthest_point = torch.argmax(dists).item()
        sampled_idxs[i] = farthest_point
        new_dists = torch.norm(points - points[farthest_point], dim=1)
        dists = torch.min(dists, new_dists)
        
    return sampled_idxs.cpu().numpy()

def perform_ifps_then_remove_points(original_file_path, output_file_path, num_points):
    # Read the original point cloud
    original_pcd = o3d.io.read_point_cloud(original_file_path)
    points = np.asarray(original_pcd.points)
    normals = np.asarray(original_pcd.normals) if original_pcd.has_normals() else None
    colors = np.asarray(original_pcd.colors) if original_pcd.has_colors() else None

    # First IFPS, denoise point count, can set different point counts
    denoise_num_points = 10000
    sampled_indices = iterative_farthest_point_sampling_torch(points, denoise_num_points)
    remaining_indices = np.setdiff1d(np.arange(len(points)), sampled_indices)

    # Create the remaining point cloud
    remaining_pcd = original_pcd.select_by_index(remaining_indices)
    
    # Second IFPS
    remaining_points = np.asarray(remaining_pcd.points)
    final_sampled_indices = iterative_farthest_point_sampling_torch(remaining_points, num_points)

    # Create a new point cloud based on the final indices
    final_pcd = remaining_pcd.select_by_index(final_sampled_indices)
    
    # Save the new point cloud file
    o3d.io.write_point_cloud(output_file_path, final_pcd)

# Downsample point clouds, for each point cloud file, perform IFPS and remove points, then perform IFPS again and save the result, resulting in a new list of point cloud files
down_pcd_files_list = []
num_points = 4096
for original_file_path in moved_pcd_files_list:
    output_file_path = original_file_path.replace('.ply', '_down.ply')
    perform_ifps_then_remove_points(original_file_path, output_file_path, num_points)
    down_pcd_files_list.append(output_file_path)

def merge_obj_files(file_paths):
    meshes = [trimesh.load(file_path) for file_path in file_paths]
    combined_mesh = trimesh.util.concatenate(meshes)
    return combined_mesh

# Merge multiple OBJ files
combined_mesh = merge_obj_files(moved_obj_files_list)

# Get the number of files in a directory
def count_files_in_directory(directory):
    # Get all file and folder names in the directory
    names = os.listdir(directory)
    
    # Count the number of files
    file_count = sum(os.path.isfile(os.path.join(directory, name)) for name in names)
    
    return file_count

obj_file_count = count_files_in_directory(combined_mesh_file_path)

combined_mesh_file_name = combined_mesh_file_path + '\\' + 'merge_' + str(obj_file_count+1) + '.obj'

# Save the combined mesh to a new OBJ file
combined_mesh.export(combined_mesh_file_name)

def merge_ply_files(ply_file_paths, output_file_path):
    """
    Merge multiple PLY files into a new PLY file.
    
    :param ply_file_paths: List of PLY file paths.
    :param output_file_path: Path for the merged PLY file.
    """
    # Initialize an empty point cloud object for accumulation
    merged_pcd = o3d.geometry.PointCloud()
    
    for file_path in ply_file_paths:
        # Read each PLY file into a point cloud object
        current_pcd = o3d.io.read_point_cloud(file_path)
        # Merge the current point cloud object into the accumulated point cloud object
        merged_pcd += current_pcd
    
    # Save the merged point cloud object as a new PLY file
    o3d.io.write_point_cloud(output_file_path, merged_pcd)

# dateset_counted = count_files_in_directory(dataset_pcd_file_dir)
dateset_counted = obj_file_count
dataset_pcd_file_path = dataset_pcd_file_dir + '\\' + 'merge_pcd_' + str(dateset_counted+1) + '.ply'

merge_ply_files(down_pcd_files_list, dataset_pcd_file_path)

# Load point cloud file
def load_point_cloud(ply_path):
    return o3d.io.read_point_cloud(ply_path)

# Load mesh file
def load_mesh(obj_path):
    return trimesh.load(obj_path)

# Read ray directions from a CSV file
def load_directions(csv_path):
    return np.loadtxt(csv_path, delimiter=",")

# Perform ray queries for each point and each direction, and calculate the intersection points between rays and the mesh
def ray_mesh_intersection(mesh, point, directions):
    counts = []
    for direction in directions:
        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=[point], ray_directions=[direction])
        counts.append(len(locations))
    return np.array(counts)

# Classify and save point clouds based on the number of ray intersections
def classify_and_save_points(pcd, mesh, directions, output_path_less_equal_1, output_path_greater_1):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)

    points_less_equal_1 = []
    points_greater_1 = []

    for i, point in enumerate(points):
        counts = ray_mesh_intersection(mesh, point, directions)
        if np.any(counts < 1):
            points_less_equal_1.append((point, colors[i], normals[i]))
        else:
            points_greater_1.append((point, colors[i], normals[i]))

    # Save point clouds
    for points, path in [(points_less_equal_1, output_path_less_equal_1), (points_greater_1, output_path_greater_1)]:
        pcd_new = o3d.geometry.PointCloud()
        if points:
            pcd_new.points = o3d.utility.Vector3dVector([p[0] for p in points])
            pcd_new.colors = o3d.utility.Vector3dVector([p[1] for p in points])
            pcd_new.normals = o3d.utility.Vector3dVector([p[2] for p in points])
            o3d.io.write_point_cloud(path, pcd_new)

# Define file paths
ply_path = dataset_pcd_file_path
obj_path = combined_mesh_file_name
csv_path = r'D:\PhD_Study\camera_xyz.csv'

output_path_less_equal_1 = ply_path.replace('.ply', '_covered.ply')  # Output file path
output_path_greater_1 = ply_path.replace('.ply', '_nocovered.ply')
# Load files
pcd = load_point_cloud(ply_path)
mesh = load_mesh(obj_path)
directions = load_directions(csv_path)
# Classify and save point clouds
classify_and_save_points(pcd, mesh, directions, output_path_less_equal_1, output_path_greater_1)

end_time = time.time()
print(f"Total processing time: {end_time - start_time} seconds.")
