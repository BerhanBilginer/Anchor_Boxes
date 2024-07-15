import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import math

class ColorMapper:
    def __init__(self):
        self.color_dict = {}

    def get_color(self, class_name):
        if class_name not in self.color_dict:
            self.color_dict[class_name] = np.random.rand(3,)
        return self.color_dict[class_name]

def parse_xml_with_class(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('.//object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        width = xmax - xmin
        height = ymax - ymin
        class_name = obj.find('name').text
        annotations.append({'width': width, 'height': height, 'class': class_name})
    
    return annotations

def generate_anchor_boxes(xml_files, num_clusters=3):
    all_annotations = []
    
    for xml_file in xml_files:
        annotations = parse_xml_with_class(xml_file)
        all_annotations.extend(annotations)
    
    all_annotations = np.array([(anno['width'], anno['height'], anno['class']) for anno in all_annotations])
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_annotations[:, :2])
    
    # Get cluster centers as anchor box sizes
    anchor_box_sizes = kmeans.cluster_centers_
    
    # Calculate anchor box scales
    anchor_scales = calculate_anchor_scales(anchor_box_sizes)
    
    # Calculate anchor box ratios
    anchor_ratios = calculate_anchor_ratios(anchor_box_sizes)
    
    return anchor_box_sizes, anchor_scales, anchor_ratios, all_annotations

def calculate_anchor_scales(anchor_box_sizes, scale_factors=[1, 2, 4]):
    mean_width = np.mean(anchor_box_sizes[:, 0])
    mean_height = np.mean(anchor_box_sizes[:, 1])
    
    scales = [mean_width * scale_factor for scale_factor in scale_factors]
    
    return scales

def calculate_anchor_ratios(anchor_box_sizes):
    # Calculate anchor box ratios based on the mean aspect ratio
    mean_aspect_ratio = np.mean(anchor_box_sizes[:, 0] / anchor_box_sizes[:, 1])
    
    # Use the mean aspect ratio to define anchor box ratios
    ratios = [[1, 1], [1, mean_aspect_ratio], [mean_aspect_ratio, 1]]
    
    return ratios

def plot_dataset_and_anchor_boxes(xml_files, anchor_box_sizes, anchor_scales, anchor_ratios, output_path):
    color_mapper = ColorMapper()
    anchor_colors = plt.cm.tab10(np.arange(len(anchor_box_sizes)))  # Use a color map for anchor boxes

    # Plot dataset annotations without showing class name for each annotation
    for xml_file in xml_files:
        annotations = parse_xml_with_class(xml_file)
        for anno in annotations:
            color = color_mapper.get_color(anno['class'])
            plt.scatter(anno['width'], anno['height'], marker='o', alpha=0.5, color=color, label=anno['class'], s=100000)

    # Plot generated anchor boxes
    for i, anchor_box in enumerate(anchor_box_sizes):
        color = anchor_colors[i]
        plt.scatter(anchor_box[0], anchor_box[1], marker='x', color=color, label=f'Anchor Box {i+1}', s=50000)
        
        # Annotate anchor box values, scales, and ratios
        plt.annotate(f'({anchor_box[0]:.2f}, {anchor_box[1]:.2f})\nScale: {anchor_scales[i]:.2f}\nRatio: {anchor_ratios[i]}', 
                     (anchor_box[0], anchor_box[1]), 
                     textcoords="offset points", 
                     xytext=(-15,5), 
                     ha='center', fontsize=8)

    # Create a custom legend without duplicating class names
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    legend_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(legend_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('Distribution of Dataset Annotations and Generated Anchor Boxes')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')  # Save the plot as a PNG with tight layout
    plt.show()

# Directory containing XML files
xml_directory = '/media/berhan/ÖZGÜN/PPE_DETECTION_DATA/FasterRCNN_v2_vest_helmet_noVest_noHelmet_dataset/train/labels'

# List all XML files in the directory
xml_files = [os.path.join(xml_directory, file) for file in os.listdir(xml_directory) if file.endswith('.xml')]

# Example usage:
output_path = '/home/berhan/Desktop/Development-Berhan/Test_Workspace/Faster_R-CNN/AnchorBoxesGenerator/fasterRCNN_vest_helmet_noVest_noHelmet_anchors.png'  # Specify the path where you want to save the PNG file
anchor_box_sizes, anchor_scales, anchor_ratios, all_annotations = generate_anchor_boxes(xml_files, num_clusters=3)

print("Generated Anchor Boxes:")
print(anchor_box_sizes)
print("Calculated Anchor Box Scales:")
print(anchor_scales)
print("Calculated Anchor Box Ratios:")
print(anchor_ratios)

# Plot the dataset annotations and anchor boxes, and save the plot as a PNG
plot_dataset_and_anchor_boxes(xml_files, anchor_box_sizes, anchor_scales, anchor_ratios, output_path)
