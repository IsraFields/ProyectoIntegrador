


from scipy.interpolate import griddata
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import transforms
import numpy as np



def read(filename):
    """
    Reads a text file with node coordinates and stress values, scales the coordinates to integers,
    creates a 2D NumPy array with stress values at their respective (X, Y) positions,
    and fills gaps using interpolation.
    
    Parameters:
        filename (str): Path to the input text file.
    
    Returns:
        np.ndarray: 2D array with gaps filled by interpolation.
    """
    x_list, y_list, stresses = [], [], []
    
    with open(filename, 'r') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # Extract and clean X, Y values (handle trailing dots)
            x = float(parts[1].rstrip('.'))
            y = float(parts[2].rstrip('.'))
            stress = float(parts[4])
            x_list.append(x)
            y_list.append(y)
            stresses.append(stress)
    

    scaling = 4
    
    # Scale coordinates to integers
    scaled_x = [int(round(x * scaling)) for x in x_list]
    scaled_y = [int(round(y * scaling)) for y in y_list]
    
    # Create mappings from unique scaled coordinates to indices
    unique_x = sorted(set(scaled_x))
    unique_y = sorted(set(scaled_y))
    x_to_idx = {x: idx for idx, x in enumerate(unique_x)}
    y_to_idx = {y: idx for idx, y in enumerate(unique_y)}
    
    # Initialize matrix with NaNs
    matrix = np.full((len(unique_y), len(unique_x)), np.nan)
    
    # Populate the matrix with stress values
    for x, y, stress in zip(scaled_x, scaled_y, stresses):
        xi = x_to_idx[x]
        yi = y_to_idx[y]
        if matrix[yi, xi] is np.nan:
            matrix[yi, xi] = stress
        else:
            matrix[yi, xi]= (matrix[yi, xi] + stress) / 2
            
    # Prepare grid coordinates for interpolation
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)
    points = np.array([[x, y] for x, y in zip(scaled_x, scaled_y)])
    values = np.array(stresses)
    
    # Interpolate missing values
    filled_matrix = griddata(points, values, (grid_x, grid_y), method='linear')
    flipped_matrix = filled_matrix[::-1, :].copy()


    return flipped_matrix



class StressDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        """
        Args:
            root_folder (str): Path to the root folder containing subfolders named from 1 to X.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.root_folder = root_folder
        self.transform = transform

        # Collect folder paths and labels from each subfolder
        self.data = []
        #for folder_name in sorted(os.listdir(root_folder), key=lambda x: int(x)):
        for sub_folder_name in os.listdir(root_folder):
            for folder_name in os.listdir(os.path.join(root_folder, sub_folder_name)):

                folder_path = os.path.join(root_folder, sub_folder_name, folder_name)
                if os.path.isdir(folder_path):
                    print(folder_path)
                    # Image and text file in the subfolder
                    image_path = os.path.join(folder_path, "image.png")
                    label_path = os.path.join(folder_path, "data2.txt")

                    # Ensure both files exist
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        # Use the provided `read` function to process the label
                        stress_array = read(label_path)
                        self.data.append((image_path, stress_array))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path and stress grid
        image_path, stress_grid = self.data[idx]

        # Open the image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        #skl = skeletonize(np.array(image)) #for later
        image=np.array(image)
        #image=image+skl

        
        #min max scaling for image
        image = (image - image.min()) / (image.max() - image.min())
        
        image = Image.fromarray(image)
        #convert stress grid np to pil obj
        stress_grid = stress_grid
        stress_grid = Image.fromarray(stress_grid)
        


        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            stress_grid=self.transform(stress_grid)
        else:
            image = transforms.ToTensor()(image)  # Default transformation to tensor
            stress_grid=transforms.ToTensor()(stress_grid) 


        return image, stress_grid
