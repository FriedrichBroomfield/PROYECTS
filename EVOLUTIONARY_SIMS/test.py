import numpy as np
import matplotlib.pyplot as plt

def create_circular_grid_mask(radius, center_x, center_y, grid_size):
    """
    Generates a boolean mask for a circular area on a 2D grid.
    
    Args:
        radius (float): The radius of the circular area.
        center_x (float): The x-coordinate of the circle's center.
        center_y (float): The y-coordinate of the circle's center.
        grid_size (int): The number of points along each dimension of the square grid.
        
    Returns:
        numpy.ndarray: A 2D boolean array where True indicates points inside the circle.
    """
    # Create x and y coordinate vectors
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    
    # Create 2D coordinate matrices (XX, YY)
    XX, YY = np.meshgrid(x, y)
    
    # Calculate the distance of each point from the center
    distances = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)
    
    # Create a boolean mask for points within the radius
    mask = distances == radius
    
    return mask, XX, YY

# Example Usage:
grid_size = 100
radius = 40
center_x, center_y = 50, 50

# Generate the mask and coordinates
circle_mask, XX, YY = create_circular_grid_mask(radius, center_x, center_y, grid_size)

# Use the mask to set values or filter points for your simulation
# For example, count the number of points inside the circle
points_inside = np.sum(circle_mask)
print(f"Total grid points inside the circle: {points_inside}")

# Visualize the circular grid points
plt.figure(figsize=(6, 6))
plt.scatter(XX[circle_mask], YY[circle_mask], color='blue', s=5)
plt.title("Circular Grid Points for Simulation")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()