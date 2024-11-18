import cv2
import numpy as np
import os

# def readVideo(filename):
#     cap = cv2.VideoCapture(filename)

#     # Check if camera opened successfully
#     if (cap.isOpened()== False): 
#       print("Error opening video stream or file")
#     frames=[]
#     # Read until video is completed
#     while(cap.isOpened()):
#       # Capture frame-by-frame
#       ret, frame = cap.read()
#       if ret == True:
#         frames.append(frame)

#       # Break the loop
#       else: 
#         break
#     cap.release()
#     return frames

# rgb_to_depth_affine_transforms = dict(
#    C001=np.array([[3.45638311e-01,  2.79844266e-03, -8.22281898e+01],
#                   [-1.37185375e-03, 3.46949734e-01,  1.30882644e+01],
#                   [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),
 
#    C002=np.array([[3.42938209e-01,  8.72629655e-04, -7.28786114e+01],
#                   [3.43287830e-04,  3.43578203e-01,  1.75767495e+01],
#                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

#    C003=np.array([[3.45121348e-01,  8.53232038e-04, -7.33328852e+01],
#                   [1.51167845e-03,  3.45115132e-01,  2.22178592e+01],
#                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
# )

depth_to_rgb_affine_transforms = dict(
    C001=np.array([[2.89310518e+00, -2.33353370e-02,  2.38200221e+02],
                   [1.14394588e-02,  2.88216964e+00, -3.67819523e+01],
                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

    C002=np.array([[2.90778446e+00, -1.04633946e-02,  2.15505801e+02],
                   [-3.43830682e-03,  2.91094100e+00, -5.13416831e+01],
                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),

    C003=np.array([[2.89756295e+00, -7.16367761e-03,  2.12645813e+02],
                   [-1.26919485e-02,  2.89761514e+00, -6.53095423e+01],
                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
)

def process_folder(foldername):
    # Extract the C value from foldername
    c_value = foldername[4:8]  # Extract 'C001', 'C002', etc.
    print(c_value)
    # Choose the affine matrix based on the extracted 'C' value
    affine_transform = depth_to_rgb_affine_transforms.get(c_value)
    if affine_transform is None:
        print(f"Skipping {foldername} because no affine transformation is available for {c_value}")
        return
    
    # Define paths
    depth_folder = f'./data/nturgbd/nturgb+d_depth_masked/train/{foldername}'
    output_folder = f'./data/nturgbd/depth_to_rgb/train/{foldername}/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all depth images in the folder
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
    for i, depth_file in enumerate(depth_files):
        frame_depth = cv2.imread(os.path.join(depth_folder, depth_file), cv2.IMREAD_GRAYSCALE)
        
        if frame_depth is None:
            print(f"Error loading depth image {depth_file} in {foldername}")
            continue
        
        # Apply affine transformation
        frame_depth_transformed = cv2.warpAffine(frame_depth, affine_transform[:2, :], (1920, 1080))
        
        # Save the transformed depth image
        output_filename = os.path.join(output_folder, f"depth_to_rgb_{i+1}.png")
        cv2.imwrite(output_filename, frame_depth_transformed)
        print(f"Saved transformed depth image {i+1} in {foldername}")

# Get a list of all folder names (e.g., S001C001P001R001A001)
folder_names = [f for f in os.listdir('./data/nturgbd/nturgb+d_depth_masked/train') if f.startswith('S')]

# Process each folder
for foldername in folder_names:
    process_folder(foldername)

print("All folders processed.")

# foldername = 'S001C001P001R001A001'
# video_file = './data/nturgbd/nturgb+d_rgb/train/'+foldername+'_rgb.avi'
# frames = readVideo(video_file)
# frame_depth = cv2.imread('./data/nturgbd/nturgb+d_depth/S001C001P001R001A001/Depth-00000001.png',cv2.IMREAD_GRAYSCALE)
# # affine_transform = rgb_to_depth_affine_transforms["C002"]
# # frame_rgb = cv2.warpAffine(frames[0], affine_transform[:2, :], (512, 424))

# affine_transform = depth_to_rgb_affine_transforms["C001"]
# frame_depth = cv2.warpAffine(frame_depth, affine_transform[:2, :], (1920, 1080))

# output_folder = './data/nturgbd/depth_to_rgb.jpg'  # Set the path and file name where the image will be saved
# cv2.imwrite(output_folder, frame_depth)  # Save the result image
