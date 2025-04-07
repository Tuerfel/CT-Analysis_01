"""
Funktionen für den CT-Daten Analyse Workflow
==========================================

Dieses Modul enthält alle Hilfsfunktionen für den CT-Daten Analyse Workflow.
Die Funktionen sind in logische Gruppen unterteilt und decken den gesamten
Analyseprozess von der Vorverarbeitung bis zur Visualisierung ab.

Funktionsgruppen:
----------------
1. Verzeichnis- und Dateimanagement
   - create_main_output_directory: Erstellt Hauptausgabeordner mit Zeitstempel
   - select_input_directory: Ermöglicht Auswahl zwischen vordefinierten Eingabeordnern
   - cleanup_temporary_files: Löscht temporäre Dateien und Ordner

2. Bildstapel-Verarbeitung
   - load_image_stack: Lädt Bildstapel verschiedener Formate
   - average_pooling_3d: Führt 3D Average Pooling durch
   - finde_threshold: Bestimmt optimalen Schwellenwert für Binarisierung
   - binarize_stack: Binarisiert den Bildstapel
   - save_processed_stack: Speichert verarbeiteten Bildstapel

3. Visualisierung
   - visualize_processing_steps: Zeigt alle Verarbeitungsschritte
   - visualize_projections: Erstellt 2D-Projektionen des Stapels
   - visualize_skeleton_3d: Visualisiert 3D-Skelett
   - visualize_unit_cell: Visualisiert Einheitszellen
   - visualize_volume_and_surface: Visualisiert Volumen und Oberfläche
   - visualize_density_3d_with_sphere: Visualisiert Punktdichte mit Referenz-Sphere
   - visualize_clustering: Visualisiert Clustering-Ergebnisse

4. Skelettierung und Punktwolken
   - skeletonize_stack: Erzeugt 3D-Skelett
   - skeleton_to_point_cloud: Konvertiert Skelett in Punktwolke
   - save_skeleton_ply: Speichert Skelett als PLY-Datei
   - extract_surface: Extrahiert Oberfläche aus binärem Volumen

5. Punktdichteanalyse
   - analyze_point_density: Berechnet lokale Punktdichte
   - filter_by_density: Filtert Punkte basierend auf Dichtewerten
   - create_sphere_mesh: Erstellt Sphere-Mesh für Visualisierung
   - save_density_ply: Speichert Punktwolke mit Dichtewerten

6. Clustering und Gitteranalyse
   - perform_clustering: Führt K-Means Clustering durch
   - analyze_cluster_directions: Analysiert Hauptrichtungen der Cluster
   - create_ideal_grid: Erstellt ideales Gitter
   - interactive_density_filtering: Interaktive Dichtefilterung

7. Hilfsfunktionen
   - _set_plotly_layout: Konsistente Plotly-Layout-Einstellungen

Datenformate:
------------
- Bildstapel: numpy arrays (uint8 oder uint16)
- Punktwolken: numpy arrays (float32)
- Skelette: numpy arrays (uint8)
- Parameter: Textdateien und NPZ-Archive

Verarbeitungsschritte:
--------------------
1. Bildstapel laden und vorverarbeiten
2. Average Pooling anwenden
3. Gauss-Filterung durchführen
4. Binarisierung mit automatischem Threshold
5. Morphologische Operationen (Opening/Closing)
6. Skelettierung
7. Punktdichteanalyse
8. Clustering und Gitteranalyse
9. Einheitszellen-Visualisierung

Visualisierungsoptionen:
----------------------
- Verarbeitungsschritte (Original, Pooling, Gauss, Binarisierung, Morphologie)
- 3D-Skelett mit interaktivem Viewer
- Punktdichte mit Referenz-Sphere
- Clustering-Ergebnisse mit Hauptrichtungen
- Einheitszellen mit Oberflächenvisualisierung

Bildstapel-Verarbeitung:
-----------------------
create_main_output_directory():
    Erstellt einen Hauptausgabeordner mit Zeitstempel

select_input_directory():
    Ermöglicht Auswahl zwischen vordefinierten Eingabeordnern
    - BCC_gesamt
    - Variation_Winkel

load_image_stack(image_path):
    Lädt Bildstapel verschiedener Formate
    - Unterstützt: BMP, TIFF, PNG, JPEG
    - Automatische 8/16-bit Erkennung
    - Fortschrittsanzeige
    - Fehlerbehandlung

average_pooling_3d(bildstapel, kernel_size):
    Führt 3D Average Pooling durch
    - Optimierte NumPy-Implementation
    - Konfigurierbare Kernel-Größe

finde_threshold(bildstapel, output_dir):
    Findet optimalen Threshold
    - Histogramm-Analyse
    - Peak-Detektion
    - Interaktive Auswahl
    - Plotly-Visualisierung

Visualisierung:
-------------
visualize_processing_steps(original_stack, gaussian, pooled, binary, cleaned, output_dir, sigma, pooling_kernel, morph_kernel):
    Zeigt Verarbeitungsschritte
    - Original
    - Nach Average Pooling
    - Nach Gauss-Filterung
    - Nach Binarisierung
    - Nach Morphologie

visualize_projections(stack, output_dir, sigma, pooling_kernel, morph_kernel):
    Erstellt Projektionen
    - Z-Projektion (von oben)
    - Y-Projektion (von vorne)
    - X-Projektion (von der Seite)

visualize_skeleton_3d(points, output_dir, pooling_kernel, morph_kernel):
    3D-Visualisierung des Skeletts
    - Interaktive Plotly-Darstellung
    - Orthographische Projektion

Punktdichte-Analyse:
------------------
analyze_point_density():
    Berechnet lokale Punktdichte
    - KD-Tree für Nachbarschaftssuche
    - Konfigurierbare Sphere-Größe
    - Visualisierung mit Referenz-Sphere

interactive_density_filtering():
    Ermöglicht Filterung nach Dichte
    - Interaktive Threshold-Wahl
    - Visualisierung der gefilterten Punkte
    - Speicherung der Ergebnisse

Clustering und Gitteranalyse:
---------------------------
perform_clustering():
    Führt K-Means Clustering durch
    - Automatische Bestimmung der Cluster-Anzahl
    - Elbow-Methode
    - Manuelle Übersteuerung möglich
    - Visualisierung der Cluster-Center

analyze_cluster_directions():
    Analysiert Hauptrichtungen
    - Bestimmung der Gitterstruktur
    - Berechnung der Einheitszellengröße
    - Erstellung des idealen Gitters
    - Finale Visualisierung

"""

import os
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import ndimage
import glob
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import trimesh
from skimage import measure
from plotly.subplots import make_subplots
from scipy.ndimage import label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from skimage.measure import marching_cubes
import shutil

def create_main_output_directory():
    """Creates a main output directory for all processing steps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"complete_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nHauptausgabeordner erstellt: {output_dir}")
    return output_dir

def select_input_directory():
    """Allows selection between predefined input directories"""
    paths = {
        "1": {
            "name": "BCC_gesamt",
            "path": "/Users/haftbefelix/Desktop/CT Daten/BCC_groß/BCC_Bildstapel_gesamt_bmp"
        },
        "2": {
            "name": "Variation_Winkel",
            "path": "/Users/haftbefelix/Desktop/CT Daten/Variation Winkel/kompletter Scan 50-600"
        },
        "3": {
            "name": "bmp_stack_perfect",
            "path": "/Users/haftbefelix/Desktop/Python/Python Codes/cursor_02_stl_to_volume/bmp_stack_perfect"
        }
    }
    
    print("\nVerfügbare Bildstapel:")
    for key, value in paths.items():
        print(f"{key}: {value['name']}")
    
    while True:
        choice = input("\nWählen Sie den Bildstapel (1/2/3): ").strip()
        if choice in paths:
            selected_path = paths[choice]["path"]
            print(f"\nGewählter Bildstapel: {paths[choice]['name']}")
            print(f"Pfad: {selected_path}")
            return selected_path
        else:
            print("Ungültige Auswahl! Bitte 1, 2 oder 3 eingeben.")

def load_image_stack(image_path):
    """Loads all supported image files from the specified directory"""
    print(f"\nLoading image stack from: {image_path}")
    
    # Define supported formats
    supported_formats = ['.bmp', '.tiff', '.tif', '.png', '.jpg', '.jpeg']
    
    # Get list of all image files
    image_files = []
    for format in supported_formats:
        image_files.extend(glob.glob(os.path.join(image_path, f"*{format}")))
        image_files.extend(glob.glob(os.path.join(image_path, f"*{format.upper()}")))
    
    if not image_files:
        raise FileNotFoundError(
            f"No supported image files found in directory.\n"
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    # Sort files to ensure correct order
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")
    
    # Load first image to get dimensions and check if 16-bit
    test_img = np.array(Image.open(image_files[0]))
    height, width = test_img.shape
    is_16bit = test_img.dtype == np.uint16
    
    # Create 3D array
    stack = np.zeros((len(image_files), height, width), dtype=np.uint8)
    
    # Load all images with optimized 16-bit handling
    total_files = len(image_files)
    if is_16bit:
        # For 16-bit images, find global min/max from 100 random images
        print("16-bit Bilder erkannt, berechne Normalisierung...")
        sample_size = min(100, total_files)
        random_indices = np.random.choice(total_files, sample_size, replace=False)
        sample_min = float('inf')
        sample_max = float('-inf')
        
        for idx in random_indices:
            img = np.array(Image.open(image_files[idx]))
            sample_min = min(sample_min, img.min())
            sample_max = max(sample_max, img.max())
        
        print(f"Normalisierungsbereich aus {sample_size} zufälligen Bildern: [{sample_min}, {sample_max}]")
        scale_factor = 255.0 / (sample_max - sample_min)
        
        # Process all images with calculated scaling
        for i, file in enumerate(image_files):
            if i % 100 == 0:
                print(f"Loading image {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
            img = np.array(Image.open(file))
            stack[i] = np.clip((img - sample_min) * scale_factor, 0, 255).astype(np.uint8)
    else:
        # For 8-bit images, direct loading
        for i, file in enumerate(image_files):
            if i % 100 == 0:
                print(f"Loading image {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
            stack[i] = np.array(Image.open(file))
    
    print("\nImage stack loaded successfully!")
    print(f"Stack dimensions: {stack.shape}")
    print(f"Value range: [{np.min(stack)}, {np.max(stack)}]")
    
    return stack

def average_pooling_3d(bildstapel, kernel_size=4):
    """Performs 3D average pooling with numpy optimization"""
    shape = np.array(bildstapel.shape)
    new_shape = shape // kernel_size
    
    print(f"\nFühre Average Pooling durch...")
    print(f"Eingabegröße: {shape}")
    print(f"Ausgabegröße: {new_shape}")
    
    # Reshape for efficient pooling
    reshaped = bildstapel[:new_shape[0]*kernel_size, 
                         :new_shape[1]*kernel_size, 
                         :new_shape[2]*kernel_size]
    
    # Perform pooling using reshape and mean
    pooled = reshaped.reshape(new_shape[0], kernel_size, 
                            new_shape[1], kernel_size, 
                            new_shape[2], kernel_size).mean(axis=(1,3,5))
    
    return pooled

def finde_threshold(stack, sample_fraction=0.01, image_step=10):

    # Wähle nur jedes n-te Bild
    selected_images = stack[::image_step]
    
    # Zufällige Auswahl von Voxeln aus den Bildern
    total_voxels = selected_images.size
    sample_size = int(total_voxels * sample_fraction)
    flat_indices = np.random.choice(total_voxels, sample_size, replace=False)
    sampled_values = selected_images.flat[flat_indices]
    
    print(f"Verwende {len(selected_images)} von {len(stack)} Bildern")
    print(f"Analysiere {sample_size} von {stack.size} Voxeln ({sample_size/stack.size*100:.3f}%)")
    
    # Berechne Histogramm
    hist, bin_edges = np.histogram(sampled_values, bins=256, range=(0, 255))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_smooth = ndimage.gaussian_filter1d(hist, sigma=2)  # Reduzierte Glättung
    
    # Finde Peaks (optimierte Version)
    peaks = []
    hist_len = len(hist_smooth)
    for i in range(1, hist_len - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            peaks.append((bin_centers[i], hist_smooth[i], i))
    
    # Sortiere Peaks nach Höhe
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Erstelle Plotly Figure
    fig = go.Figure()
    
    # Füge Histogramm und Kurve in einem Trace hinzu
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=hist_smooth,
        name='Histogramm',
        line=dict(color='blue'),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.3)'
    ))
    
    # Füge Peaks hinzu
    if peaks:
        peak_x = [p[0] for p in peaks[:2]]  # Nur die zwei höchsten Peaks
        peak_y = [p[1] for p in peaks[:2]]
        fig.add_trace(go.Scatter(
            x=peak_x,
            y=peak_y,
            mode='markers+text',
            marker=dict(color='red', size=10),
            text=[f'Peak {i+1}: {x:.1f}' for i, x in enumerate(peak_x)],
            textposition="top right",
            name='Peaks'
        ))
    
    # Bestimme und zeige Threshold
    proposed_threshold = None
    if len(peaks) >= 2:
        peak1, peak2 = peaks[0], peaks[1]
        if peak1[2] > peak2[2]:
            peak1, peak2 = peak2, peak1
        
        proposed_threshold = (peak1[0] + peak2[0]) / 2
        fig.add_vline(
            x=proposed_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f'Vorgeschlagener Threshold: {proposed_threshold:.1f}'
        )
        
        print(f"\nZwei Peaks gefunden bei {peak1[0]:.1f} und {peak2[0]:.1f}")
        print(f"Vorgeschlagener Threshold: {proposed_threshold:.1f}")
    else:
        proposed_threshold = 110
        fig.add_vline(
            x=proposed_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f'Standard Threshold: {proposed_threshold}'
        )
        
        if len(peaks) == 0:
            print("\nKeine Peaks gefunden.")
        else:
            print(f"\nNur ein Peak gefunden bei {peaks[0][0]:.1f}")
        print(f"Standard Threshold: {proposed_threshold}")
    
    # Optimiertes Layout
    fig.update_layout(
        title='Histogramm mit Peaks und Threshold',
        xaxis_title='Grauwert',
        yaxis_title='Häufigkeit',
        width=800,
        height=600,
        showlegend=True,
        template='plotly_white'  # Schnelleres Template
    )
    
    fig.show()
    
    # Interaktive Threshold-Auswahl
    while True:
        print("\nMöchten Sie den vorgeschlagenen Threshold verwenden? (j/n)")
        choice = input().lower().strip()
        
        if choice == 'j':
            return proposed_threshold
        elif choice == 'n':
            while True:
                try:
                    custom_threshold = float(input("Bitte geben Sie einen eigenen Threshold-Wert ein (0-255): "))
                    if 0 <= custom_threshold <= 255:
                        return custom_threshold
                    else:
                        print("Wert muss zwischen 0 und 255 liegen!")
                except ValueError:
                    print("Bitte geben Sie eine gültige Zahl ein!")
        else:
            print("Bitte antworten Sie mit 'j' oder 'n'")


def visualize_processing_steps(original_stack, gaussian, pooled, binary, cleaned, output_dir, sigma, pooling_kernel, morph_kernel):
    """Visualizes the processing steps using Plotly with a slider for image navigation"""
    # Anzahl der anzuzeigenden Bilder
    num_images = min(10, original_stack.shape[0])
    step = original_stack.shape[0] // num_images
    original_indices = list(range(0, original_stack.shape[0], step))[:num_images]
    
    # Berechne die entsprechenden Indizes für die anderen Stacks
    gaussian_indices = [int(idx * (gaussian.shape[0] / original_stack.shape[0])) for idx in original_indices]
    pooled_indices = [int(idx * (pooled.shape[0] / original_stack.shape[0])) for idx in original_indices]
    
    # Erstelle Figure mit Subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Original",
            f"Nach Average Pooling (Kernel: {pooling_kernel})",
            f"Nach Gauss-Filterung (σ={sigma})",
            "Nach Binarisierung",
            f"Nach Erosion + Dilatation (Kernel: {morph_kernel})",
            ""  # Leerer Subplot für bessere Anordnung
        )
    )
    
    # Füge die Bilder für jeden Schritt hinzu
    traces = []
    for orig_idx, gauss_idx, pool_idx in zip(original_indices, gaussian_indices, pooled_indices):
        # Original
        traces.append(go.Heatmap(
            z=original_stack[orig_idx],
            colorscale='gray',
            showscale=False,
            visible=(orig_idx == original_indices[0]),
            name=f'Original {orig_idx}'
        ))
        
        # Nach Average Pooling
        traces.append(go.Heatmap(
            z=pooled[pool_idx],
            colorscale='gray',
            showscale=False,
            visible=(orig_idx == original_indices[0]),
            name=f'Pooled {pool_idx}'
        ))
        
        # Nach Gauss-Filterung
        traces.append(go.Heatmap(
            z=gaussian[gauss_idx],
            colorscale='gray',
            showscale=False,
            visible=(orig_idx == original_indices[0]),
            name=f'Gaussian {gauss_idx}'
        ))
        
        # Nach Binarisierung
        traces.append(go.Heatmap(
            z=binary[pool_idx],
            colorscale='gray',
            showscale=False,
            visible=(orig_idx == original_indices[0]),
            name=f'Binary {pool_idx}'
        ))
        
        # Nach Erosion + Dilatation
        traces.append(go.Heatmap(
            z=cleaned[pool_idx],
            colorscale='gray',
            showscale=False,
            visible=(orig_idx == original_indices[0]),
            name=f'Cleaned {pool_idx}'
        ))
    
    # Füge alle Traces zu den entsprechenden Subplots hinzu
    for i, trace in enumerate(traces):
        if i % 5 == 0:  # Original
            row = 1
            col = 1
        elif i % 5 == 1:  # Pooled
            row = 1
            col = 2
        elif i % 5 == 2:  # Gaussian
            row = 1
            col = 3
        elif i % 5 == 3:  # Binary
            row = 2
            col = 1
        else:  # Cleaned
            row = 2
            col = 2
        fig.add_trace(trace, row=row, col=col)
    
    # Erstelle Slider
    steps = []
    for i in range(num_images):
        # Erstelle eine Liste mit False für alle Spuren
        visibility = [False] * len(traces)
        
        # Setze die fünf Spuren für das aktuelle Bild auf sichtbar
        start_idx = i * 5
        for j in range(5):
            visibility[start_idx + j] = True
        
        step = dict(
            method="update",
            args=[{"visible": visibility}],
            label=f"Bild {original_indices[i]}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Bild: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update Layout
    fig.update_layout(
        width=1500,  # Breiteres Layout
        height=1000,  # Höheres Layout
        title=f"Verarbeitungsschritte (σ={sigma}, Pooling Kernel: {pooling_kernel}, Morph Kernel: {morph_kernel})",
        showlegend=False,
        sliders=sliders
    )
    
    # Setze gleiches Seitenverhältnis für jeden Subplot
    for i in range(1, 3):
        for j in range(1, 4):
            if not (i == 2 and j == 3):  # Überspringe den leeren Subplot
                fig.update_xaxes(scaleanchor=f"y{i}{j}", scaleratio=1, row=i, col=j)
    
    # Speichere die Visualisierung
    fig.write_html(os.path.join(output_dir, 'processing_steps.html'))
    fig.show()

def visualize_projections(stack, output_dir, sigma, pooling_kernel, morph_kernel):
    """Creates and saves projections using Plotly in a 1x3 grid"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Z-Projektion (von oben)",
            "Y-Projektion (von vorne)",
            "X-Projektion (von der Seite)"
        )
    )
    
    # Projektionen
    fig.add_trace(
        go.Heatmap(
            z=np.mean(stack, axis=0),
            colorscale='gray',
            showscale=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=np.mean(stack, axis=1),
            colorscale='gray',
            showscale=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=np.mean(stack, axis=2),
            colorscale='gray',
            showscale=False
        ),
        row=1, col=3
    )
    
    # Update Layout
    fig.update_layout(
        width=1000,
        height=400,
        title=f"Projektionen (Pooling Kernel: {pooling_kernel}, Morph Kernel: {morph_kernel})",
        showlegend=False
    )
    
    # Setze gleiches Seitenverhältnis für jeden Subplot
    for i in range(1, 2):
        for j in range(1, 4):
            fig.update_xaxes(scaleanchor=f"y{i}{j}", scaleratio=1, row=i, col=j)
    
    fig.write_html(os.path.join(output_dir, 'projections.html'))
    fig.show()


def save_skeleton_ply(points, output_dir):
    """Saves skeleton points as PLY file"""
    vertex = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(os.path.join(output_dir, 'skeleton_pointcloud.ply'))

def _set_plotly_layout(fig, title='', is_3d=False):
    """Helper function for consistent Plotly layout"""
    base_layout = {
        'width': 1000,
        'height': 800,
        'title': title,
        'showlegend': True
    }
    
    if is_3d:
        base_layout['scene'] = {
            'aspectmode': 'data',
            'camera': {
                'projection': {'type': 'orthographic'},
                'eye': {'x': 1, 'y': 1, 'z': 1},
                'up': {'x': 0, 'y': 0, 'z': 1}
            }
        }
    
    fig.update_layout(**base_layout)

def visualize_skeleton_3d(points, sigma, pooling_kernel, morph_kernel):
    """Creates 3D visualization of skeleton using Plotly"""
    if points.shape[1] != 3:
        raise ValueError(f"Punktwolke hat falsche Dimension: {points.shape}")
        
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.8)
    )])
    
    _set_plotly_layout(fig, f'Skelett-Visualisierung (σ={sigma}, Pooling Kernel: {pooling_kernel}, Morph Kernel: {morph_kernel})', is_3d=True)
    
    fig.show()

def analyze_point_density(points, output_dir):
    """Analyzes point density and visualizes results"""
    print("\n=== Punktdichteanalyse ===")
    
    density_dir = os.path.join(output_dir, "3_density_analysis")
    os.makedirs(density_dir, exist_ok=True)
    
    while True:
        # Wähle Sphere Diameter
        while True:
            try:
                sphere_diameter = float(input("\nGeben Sie den Sphere Diameter ein (5-20): "))
                if 5 <= sphere_diameter <= 20:
                    break
                print("Wert muss zwischen 5 und 20 liegen!")
            except ValueError:
                print("Bitte geben Sie eine gültige Zahl ein!")
        
        # Calculate densities
        print("\nBerechne Punktdichten...")
        tree = cKDTree(points)
        densities = np.zeros(len(points))
        
        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, sphere_diameter/2, p=2)
            densities[i] = len(neighbors) - 1
        
        # Visualisiere mit Sphere
        min_point_idx = np.lexsort((points[:,2], points[:,1], points[:,0]))[0]
        min_point = points[min_point_idx]
        visualize_density_3d_with_sphere(points, densities, min_point, sphere_diameter/2, density_dir)
        
        # Frage nach Zufriedenheit
        print("\nSind Sie mit der Sphere-Größe zufrieden?")
        choice = input("Geben Sie 'j' für ja ein, beliebige andere Taste für neue Sphere-Größe: ").lower()
        
        if choice == 'j':
            return points, densities
        else:
            print("\nBitte wählen Sie eine neue Sphere-Größe.")

def create_sphere_mesh(center, radius, resolution=20):
    """Creates a sphere mesh for visualization"""
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(-np.pi/2, np.pi/2, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.cos(theta) * np.cos(phi)
    y = center[1] + radius * np.cos(theta) * np.sin(phi)
    z = center[2] + radius * np.sin(theta)
    
    return x, y, z

def visualize_density_3d_with_sphere(points, density_values, sphere_center, sphere_radius, output_dir):
    """Creates 3D visualization of points colored by density with reference sphere"""
    print("Erstelle 3D-Visualisierung der Punktdichten...")
    
    # Finde tatsächliche min und max Dichte
    min_density = np.min(density_values)
    max_density = np.max(density_values)
    
    # Erstelle die 3D-Visualisierung
    fig = go.Figure()
    
    # Füge Punktwolke hinzu
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=density_values,
            opacity=0.8,
            colorbar=dict(
                title='Punktdichte'
            ),
            cmin=min_density,
            cmax=max_density
        ),
        name='Punktwolke',
        hovertext=[f'Dichte: {d:.2f}' for d in density_values],
        hoverinfo='text'
    ))
    
    # Füge Sphere hinzu
    x_sphere, y_sphere, z_sphere = create_sphere_mesh(sphere_center, sphere_radius)
    fig.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        opacity=0.3,
        showscale=False,
        name='Referenz Sphere'
    ))
    
    # Füge Sphere-Zentrum hinzu
    fig.add_trace(go.Scatter3d(
        x=[sphere_center[0]],
        y=[sphere_center[1]],
        z=[sphere_center[2]],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            symbol='x'
        ),
        name='Sphere Zentrum'
    ))
    
    _set_plotly_layout(fig, f'Punktwolke mit Dichtewerten (Sphere Diameter: {sphere_radius*2:.1f})', is_3d=True)
    
    fig.show()

def save_density_ply(points, densities, output_dir):
    """Saves points with density values as PLY"""
    vertex = np.zeros(len(points), 
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('density', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['density'] = densities
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(os.path.join(output_dir, 'skeleton_with_density.ply')) 

def filter_points_by_density(points, density_values, threshold, output_dir):
    """Visualizes and filters points based on density threshold"""
    high_density_mask = density_values >= threshold
    
    fig = go.Figure()
    
    # Ausgeschlossene Punkte
    fig.add_trace(go.Scatter3d(
        x=points[~high_density_mask, 0],
        y=points[~high_density_mask, 1],
        z=points[~high_density_mask, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.3
        ),
        name=f'Punkte unter Threshold ({np.sum(~high_density_mask)})'
    ))
    
    # Ausgewählte Punkte
    fig.add_trace(go.Scatter3d(
        x=points[high_density_mask, 0],
        y=points[high_density_mask, 1],
        z=points[high_density_mask, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        ),
        name=f'Punkte über Threshold ({np.sum(high_density_mask)})'
    ))
    
    _set_plotly_layout(fig, f'Gefilterte Punktwolke (Threshold: {threshold:.1f})', is_3d=True)
    
    fig.show()
    
    return points[high_density_mask]

def interactive_density_filtering(points, density_values, output_dir):
    """Interactive density threshold selection and filtering"""
    print("\n=== Interaktive Dichtefilterung ===")
    
    # Verwende den bestehenden density_analysis Ordner
    density_dir = os.path.join(output_dir, "3_density_analysis")
    
    while True:
        # Berechne min und max Dichte
        min_density = np.min(density_values)
        max_density = np.max(density_values)
        
        print(f"\nDichtebereich: {min_density:.1f} - {max_density:.1f}")
        
        # Wähle Threshold
        while True:
            try:
                threshold = float(input(f"\nGeben Sie einen Threshold für die Dichte ein ({min_density:.1f} - {max_density:.1f}): "))
                if min_density <= threshold <= max_density:
                    break
                print(f"Wert muss zwischen {min_density:.1f} und {max_density:.1f} liegen!")
            except ValueError:
                print("Bitte geben Sie eine gültige Zahl ein!")
        
        # Filtere und visualisiere Punkte
        filtered_points = filter_points_by_density(points, density_values, threshold, density_dir)
        
        # Frage nach Zufriedenheit
        print("\nSind Sie mit dem Threshold zufrieden?")
        choice = input("Geben Sie 'j' für ja ein, beliebige andere Taste für neuen Threshold: ").lower()
        
        if choice == 'j':
            # Speichere gefilterte Punkte als npz
            output_file = os.path.join(density_dir, f'skeleton_centers_threshold_{threshold:.1f}.npz')
            np.savez_compressed(output_file, 
                              filtered_points=filtered_points,
                              threshold=threshold,
                              total_points=len(points),
                              kept_points=len(filtered_points),
                              percentage_kept=len(filtered_points)/len(points)*100)
            print(f"\nGefilterte Punkte gespeichert als: {output_file}")
            
            # Füge Filterparameter zur Hauptparameterdatei hinzu
            param_file = os.path.join(output_dir, 'processing_parameters.txt')
            with open(param_file, 'a') as f:
                f.write('\n=== Dichtefilterung ===\n')
                f.write(f'Density Threshold: {threshold:.1f}\n')
                f.write(f'Anzahl gefilterter Punkte: {len(filtered_points)}\n')
                f.write(f'Ursprüngliche Punktanzahl: {len(points)}\n')
                f.write(f'Beibehaltener Anteil: {len(filtered_points)/len(points)*100:.1f}%\n')
            
            # Führe Clustering durch
            print("\n=== Schritt 4: Clustering und Richtungsanalyse ===")
            cluster_centers, analysis_results = perform_clustering(filtered_points, points, output_dir)
            
            break
    
    return filtered_points, cluster_centers, analysis_results

def perform_clustering(points, skeleton_points, output_dir):
    """
    Führt K-Means Clustering auf den gefilterten Punkten durch.
    
    Args:
        points: Gefilterte Punktwolke aus der Dichteanalyse
        skeleton_points: Das komplette Skelett
        output_dir: Ausgabeverzeichnis
    """
    print("\n=== K-Means Clustering ===")
    
    cluster_dir = os.path.join(output_dir, "4_clustering")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Prepare data for K-means
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Calculate inertia for different numbers of clusters
    inertias = []
    K = range(150, 250)  # Testing from 150 to 250 clusters
    
    print("\nBerechne optimale Cluster-Anzahl...")
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(points_scaled)
        inertias.append(kmeans.inertia_)
    
    # Find the elbow point using KneeLocator
    knee_finder = KneeLocator(K, inertias, curve='convex', direction='decreasing')
    elbow_k = knee_finder.elbow
    
    # Create elbow plot
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=list(K),
        y=inertias,
        mode='lines+markers',
        name='Inertia'
    ))
    
    # Add vertical line at the elbow point
    if elbow_k:
        elbow_fig.add_vline(
            x=elbow_k, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Vorgeschlagene Cluster-Anzahl: k={elbow_k}",
            annotation_position="top right"
        )
    
    elbow_fig.update_layout(
        title='Elbow Plot für K-means Clustering',
        xaxis_title='Anzahl Cluster (k)',
        yaxis_title='Inertia',
        width=800,
        height=600
    )
    
    elbow_fig.show()
    
    # Ask user for confirmation or new cluster number
    if elbow_k:
        print(f"\nVorgeschlagene Anzahl Cluster: {elbow_k}")
        while True:
            user_input = input("Möchten Sie diese Anzahl verwenden? (j/n): ").lower()
            
            if user_input == 'j':
                chosen_k = elbow_k
                break
            elif user_input == 'n':
                while True:
                    try:
                        chosen_k = int(input("\nGeben Sie die gewünschte Anzahl Cluster ein (150-250): "))
                        if 150 <= chosen_k <= 250:
                            break
                        print("Die Anzahl muss zwischen 150 und 250 liegen!")
                    except ValueError:
                        print("Bitte geben Sie eine ganze Zahl ein!")
                break
            else:
                print("Bitte antworten Sie mit 'j' oder 'n'")
    else:
        print("Kein klarer Elbow-Punkt gefunden.")
        while True:
            try:
                chosen_k = int(input("\nGeben Sie die gewünschte Anzahl Cluster ein (150-250): "))
                if 150 <= chosen_k <= 250:
                    break
                print("Die Anzahl muss zwischen 150 und 250 liegen!")
            except ValueError:
                print("Bitte geben Sie eine ganze Zahl ein!")
    
    # Perform K-means clustering with the chosen k
    print(f"\nFühre Clustering mit {chosen_k} Clustern durch...")
    kmeans = KMeans(n_clusters=chosen_k, random_state=42)
    kmeans.fit(points_scaled)
    
    # Get cluster centers and transform back to original scale
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create visualization
    cluster_fig = go.Figure()
    
    # Add original points in light gray
    cluster_fig.add_trace(go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='lightgray',
            opacity=0.5
        ),
        name='Originalpunkte'
    ))
    
    # Add cluster centers in red
    cluster_fig.add_trace(go.Scatter3d(
        x=centers_original[:, 0],
        y=centers_original[:, 1],
        z=centers_original[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=1
        ),
        name='Cluster-Center'
    ))
    
    # Update layout
    cluster_fig.update_layout(
        title=f'K-means Clustering Ergebnisse (k={chosen_k})',
        scene=dict(
            camera=dict(
                projection=dict(
                    type='orthographic'
                )
            ),
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    cluster_fig.show()
    
    # Save results
    cluster_file = os.path.join(cluster_dir, 'clustered_centers.npy')
    np.save(cluster_file, centers_original)
    print(f"\nCluster-Center gespeichert als: {cluster_file}")
    
    # Führe Richtungsanalyse durch
    print("\n=== Schritt 5: Richtungsanalyse ===")
    analysis_results = analyze_cluster_directions(centers_original, skeleton_points)
    
    if analysis_results is not None:
        # Speichere die Ergebnisse
        results_file = os.path.join(cluster_dir, 'analysis_results.npz')
        np.savez(results_file, **analysis_results)
        print(f"\nAnalyse-Ergebnisse gespeichert als: {results_file}")
    
    return centers_original, analysis_results

def analyze_cluster_directions(centers, skeleton_points):
    """
    Analysiert die Richtungen zwischen Cluster-Zentren und erstellt ein Gitter.
    
    Args:
        centers: Die gefundenen Cluster-Center aus perform_clustering
        skeleton_points: Das komplette Skelett
    """
    print("\n=== Analyse der Cluster-Richtungen ===")
    
    # Create KDTree
    tree = cKDTree(centers)
    
    # Find nearest neighbors for each center
    k = 14
    distances, indices = tree.query(centers, k=k+1)  # k+1 weil der erste Punkt der Punkt selbst ist
    
    # Remove self-distances (first column)
    distances = distances[:, 1:]
    
    # Flatten distances for histogram 
    all_distances = distances.flatten()
    
    # Create histogram data
    hist, bin_edges = np.histogram(all_distances, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create histogram plot
    hist_fig = go.Figure()
    
    # Add histogram
    hist_fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        name='Histogram',
        opacity=0.3,
        marker_color='gray'
    ))
    
    # Update histogram layout
    hist_fig.update_layout(
        title='Verteilung der Abstände zwischen Cluster-Zentren',
        xaxis_title='Abstand',
        yaxis_title='Häufigkeit',
        width=1000,
        height=800
    )
    
    hist_fig.show()
    
    # User input for thresholds
    while True:
        try:
            threshold1 = float(input("\nGeben Sie den ersten Threshold ein: "))
            threshold2 = float(input("Geben Sie den zweiten Threshold ein: "))
            if threshold1 < threshold2:
                break
            print("Der erste Threshold muss kleiner als der zweite sein!")
        except ValueError:
            print("Bitte geben Sie gültige Zahlen ein!")
    
    # Calculate means for different ranges
    filtered_distances_1 = all_distances[(all_distances >= 0) & (all_distances <= threshold1)]
    mean1 = np.mean(filtered_distances_1)
    num_points_mean1 = len(filtered_distances_1)
    
    filtered_distances_2 = all_distances[(all_distances > threshold1) & (all_distances <= threshold2)]
    mean2 = np.mean(filtered_distances_2)
    num_points_mean2 = len(filtered_distances_2)
    
    # Calculate unit cell width
    unit_cell_width = mean1 * 2 / np.sqrt(3)
    
    # Collect all directions
    directions = []
    for i in range(len(centers)):
        for j in range(1, k+1):  # Skip first distance (self)
            if threshold1 < distances[i, j-1] <= threshold2:
                point1 = centers[i]
                point2 = centers[indices[i, j]]
                direction = point2 - point1
                direction = direction / np.linalg.norm(direction)
                directions.append(direction)
    
    directions = np.array(directions)
    
    if len(directions) == 0:
        print(f"\nKeine Verbindungen zwischen {threshold1} und {threshold2} gefunden!")
        return None
    
    # Perform k-means clustering with k=6
    kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans.fit(directions)
    main_directions = kmeans.cluster_centers_
    normalized_directions = main_directions / np.linalg.norm(main_directions, axis=1)[:, np.newaxis]
    
    # Combine opposite directions to get 3 axes
    axes = []
    used = set()
    coordinate_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    for coord_axis in coordinate_axes:
        best_alignment = -1
        best_direction = None
        best_idx = None
        
        for i in range(len(normalized_directions)):
            if i in used:
                continue
            
            alignment = abs(np.dot(normalized_directions[i], coord_axis))
            if alignment > best_alignment:
                best_alignment = alignment
                best_idx = i
                if np.dot(normalized_directions[i], coord_axis) > 0:
                    best_direction = normalized_directions[i]
                else:
                    best_direction = -normalized_directions[i]
        
        if best_idx is not None:
            dots = [np.dot(best_direction, other_dir) for other_dir in normalized_directions]
            opposite_idx = np.argmin(dots)
            opposite_direction = normalized_directions[opposite_idx]
            
            if np.dot(opposite_direction, coord_axis) < 0:
                opposite_direction = -opposite_direction
            
            mean_direction = (best_direction + opposite_direction) / 2
            mean_direction = mean_direction / np.linalg.norm(mean_direction)
            
            used.add(best_idx)
            used.add(opposite_idx)
            axes.append(mean_direction)
    
    normalized_directions = np.array(axes)
    
    # Calculate coordinates in main axis directions
    skewed_coords = np.zeros_like(centers)
    for i in range(3):
        skewed_coords[:, i] = np.dot(centers, normalized_directions[i])
    
    # Calculate center point
    center_point = np.mean(skewed_coords, axis=0)
    original_center = sum(center_point[i] * normalized_directions[i] for i in range(3))
    
    # Create 5x5x5 grid
    grid_size = 5
    grid_points = []
    grid_range = range(-(grid_size//2), grid_size//2 + 1)
    
    for x in grid_range:
        for y in grid_range:
            for z in grid_range:
                point = original_center + (
                    x * unit_cell_width * normalized_directions[0] +
                    y * unit_cell_width * normalized_directions[1] +
                    z * unit_cell_width * normalized_directions[2]
                )
                grid_points.append(point)
    
    grid_points = np.array(grid_points)
    
    # Visualize results
    connections_fig = go.Figure()
    
    # Komplettes Skelett (grau)
    connections_fig.add_trace(go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.5
        ),
        name='Komplettes Skelett'
    ))
    
    # Grid points (rot)
    connections_fig.add_trace(go.Scatter3d(
        x=grid_points[:, 0],
        y=grid_points[:, 1],
        z=grid_points[:, 2],
        mode='markers',
        marker=dict(
            size=4,  # Größere Punkte für bessere Sichtbarkeit
            color='red',
            opacity=1
        ),
        name='Ideales Gitter'
    ))
    
    # Koordinatenachsen vom Zentrum
    scale_factor = mean2 * 2
    colors = ['red', 'green', 'blue']
    axis_names = ['X-Achse', 'Y-Achse', 'Z-Achse']
    
    for i, direction in enumerate(normalized_directions):
        end_point = original_center + direction * scale_factor
        start_point = original_center 
        
        connections_fig.add_trace(go.Scatter3d(
            x=[start_point[0], end_point[0]],
            y=[start_point[1], end_point[1]],
            z=[start_point[2], end_point[2]],
            mode='lines',
            line=dict(
                color=colors[i],
                width=5
            ),
            name=axis_names[i]
        ))
    
    # Update layout
    connections_fig.update_layout(
        title='Skelett mit idealem Gitter und Hauptachsen',
        scene=dict(
            camera=dict(
                projection=dict(
                    type='orthographic'
                ),
                eye=dict(x=1.5, y=1.5, z=1.5)  # Angepasste Kameraposition
            ),
            aspectmode='data',
            xaxis=dict(showspikes=False),  # Entferne Achsen-Spikes
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        ),
        width=1000,  
        height=800,
        showlegend=True,
        legend=dict(
            x=0.01,  # Position der Legende
            y=0.99,
            bordercolor="black",
            borderwidth=1
        )
    )
    
    connections_fig.show()
    
    # Print results
    print(f"\nMittelwert 0 bis {threshold1}: {mean1:.6f} (berechnet aus {num_points_mean1} Punkten)")
    print(f"Mittelwert {threshold1} bis {threshold2}: {mean2:.6f} (berechnet aus {num_points_mean2} Punkten)")
    print(f"\nEinheitszellen-Breite (2/√3 * {mean1:.6f}): {unit_cell_width:.6f}")
    
    print("\nHauptrichtungsvektoren (normiert):")
    for i, direction in enumerate(normalized_directions, 1):
        print(f"Richtung {i}: [{direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f}]")
    
    print("\nZentrumskoordinaten im ursprünglichen System:")
    print(f"[{original_center[0]:.6f}, {original_center[1]:.6f}, {original_center[2]:.6f}]")
    
    return {
        'mean1': mean1,
        'mean2': mean2,
        'directions': normalized_directions,
        'grid_points': grid_points,
        'grid_center': original_center,
        'unit_cell_width': unit_cell_width
    }

def visualize_unit_cell(original_binary_stack, grid_points, unit_cell_width, pooling_kernel, output_dir, directions):
    """
    Visualisiert eine ausgewählte Einheitszelle aus dem ursprünglichen binarisierten Stack.
    
    Args:
        original_binary_stack: Der ursprüngliche binarisierte Stack
        grid_points: Die Koordinaten der Gitterpunkte
        unit_cell_width: Die Breite der Einheitszelle
        pooling_kernel: Die Größe des verwendeten Pooling-Kernels
        output_dir: Das Ausgabeverzeichnis
        directions: Die Hauptrichtungsvektoren des Gitters
    """
    # Erstelle Verzeichnis für die Oberflächen
    surfaces_dir = os.path.join(output_dir, "5_surfaces_unitcells")
    os.makedirs(surfaces_dir, exist_ok=True)
    
    # Berechne Rotationsmatrix
    # Finde den Vektor, der am nächsten an der Z-Achse liegt
    z_axis = np.array([0, 0, 1])
    dot_products = np.abs(np.dot(directions, z_axis))
    z_idx = np.argmax(dot_products)
    z_direction = directions[z_idx]
    
    # Berechne die Rotationsmatrix
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[2] = z_direction  # Z-Achse
    
    # Finde X-Achse (Kreuzprodukt mit Y-Achse)
    y_direction = np.cross(z_direction, np.array([1, 0, 0]))
    if np.allclose(y_direction, 0):
        y_direction = np.cross(z_direction, np.array([0, 1, 0]))
    y_direction = y_direction / np.linalg.norm(y_direction)
    rotation_matrix[1] = y_direction
    
    # X-Achse ist Kreuzprodukt von Y und Z
    x_direction = np.cross(y_direction, z_direction)
    x_direction = x_direction / np.linalg.norm(x_direction)
    rotation_matrix[0] = x_direction
    
    while True:
        # Frage nach den Indizes der Einheitszelle
        print("\n=== Einheitszellen-Visualisierung ===")
        print("Geben Sie die Indizes der zu visualisierenden Einheitszelle ein (1-5 für jede Achse):")
        
        while True:
            try:
                x_idx = int(input("X-Index (1-5): ")) - 1
                y_idx = int(input("Y-Index (1-5): ")) - 1
                z_idx = int(input("Z-Index (1-5): ")) - 1
                
                if all(0 <= idx < 5 for idx in [x_idx, y_idx, z_idx]):
                    break
                print("Bitte geben Sie Indizes zwischen 1 und 5 ein!")
            except ValueError:
                print("Bitte geben Sie gültige Zahlen ein!")
        
        # Berechne den Index im grid_points Array
        cell_idx = x_idx + y_idx * 5 + z_idx * 25
        center_point = grid_points[cell_idx]
        
        # Berechne die Grenzen der Einheitszelle mit 10% Vergrößerung
        half_width = unit_cell_width * 0.55  # 50% + 10% = 55% der Breite
        x_min = int((center_point[0] - half_width) * pooling_kernel)
        x_max = int((center_point[0] + half_width) * pooling_kernel)
        y_min = int((center_point[1] - half_width) * pooling_kernel)
        y_max = int((center_point[1] + half_width) * pooling_kernel)
        z_min = int((center_point[2] - half_width) * pooling_kernel)
        z_max = int((center_point[2] + half_width) * pooling_kernel)
        
        # Stelle sicher, dass die Grenzen innerhalb des Stapels liegen
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(original_binary_stack.shape[2], x_max)
        y_max = min(original_binary_stack.shape[1], y_max)
        z_max = min(original_binary_stack.shape[0], z_max)
        
        # Extrahiere das Teilvolumen
        volume_section = original_binary_stack[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Erstelle Punktwolke aus den Voxeln
        points = np.argwhere(volume_section > 0)
        
        if len(points) == 0:
            print("Keine Punkte im ausgewählten Volumen gefunden!")
            continue
        
        # Extrahiere Oberflächenpunkte
        surface_points = extract_surface(volume_section)
        
        if surface_points is None:
            print("Konnte keine Oberfläche extrahieren!")
            continue
        
        # Maximale Anzahl von Punkten für die Visualisierung
        MAX_POINTS = 100000
        
        # Reduziere die Anzahl der Punkte wenn nötig
        if len(points) > MAX_POINTS:
            print(f"\nReduziere Volumenpunkte von {len(points)} auf {MAX_POINTS}")
            indices = np.random.choice(len(points), MAX_POINTS, replace=False)
            points = points[indices]
        
        if len(surface_points) > MAX_POINTS:
            print(f"Reduziere Oberflächenpunkte von {len(surface_points)} auf {MAX_POINTS}")
            indices = np.random.choice(len(surface_points), MAX_POINTS, replace=False)
            surface_points = surface_points[indices]
        
        print(f"\nAnzahl der visualisierten Punkte:")
        print(f"Volumen: {len(points)} Punkte")
        print(f"Oberfläche: {len(surface_points)} Punkte")
        
        # Rotiere die Punkte
        points_rotated = np.dot(points, rotation_matrix.T)
        surface_points_rotated = np.dot(surface_points, rotation_matrix.T)
        
        # Erstelle zwei Subplots nebeneinander
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Volumen', 'Oberfläche')
        )
        
        # Füge Volumenpunktwolke hinzu
        fig.add_trace(
            go.Scatter3d(
                x=points_rotated[:, 0] + x_min,
                y=points_rotated[:, 1] + y_min,
                z=points_rotated[:, 2] + z_min,
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.8
                ),
                name='Volumen'
            ),
            row=1, col=1
        )
        
        # Füge Oberflächenpunkte hinzu
        fig.add_trace(
            go.Scatter3d(
                x=surface_points_rotated[:, 0] + x_min,
                y=surface_points_rotated[:, 1] + y_min,
                z=surface_points_rotated[:, 2] + z_min,
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.8
                ),
                name='Oberfläche'
            ),
            row=1, col=2
        )
        
        # Update Layout für beide Subplots
        for i in [1, 2]:
            fig.update_scenes(
                camera=dict(
                    projection=dict(type='orthographic')
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(
                    range=[x_min, x_max],
                    title='X'
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    title='Y'
                ),
                zaxis=dict(
                    range=[z_min, z_max],
                    title='Z'
                ),
                row=1, col=i
            )
        
        # Gesamtlayout aktualisieren
        fig.update_layout(
            width=1400,  # Doppelte Breite für zwei Plots
            height=800,
            title=f'Volumenausschnitt und Oberfläche (Einheitszelle {x_idx+1},{y_idx+1},{z_idx+1})',
            showlegend=True
        )
        
        fig.show()
        
        # Speichere die rotierten Oberflächenpunkte
        output_file = os.path.join(surfaces_dir, f'surface_x{x_idx+1}_y{y_idx+1}_z{z_idx+1}.npz')
        np.savez_compressed(output_file, 
                          surface_points=surface_points_rotated,
                          original_coordinates=surface_points,
                          rotation_matrix=rotation_matrix)
        print(f"\nOberflächenpunkte gespeichert in: {output_file}")
        
        # Frage nach weiterer Visualisierung
        print("\nMöchten Sie eine andere Einheitszelle visualisieren? (j/n)")
        if input().lower() != 'j':
            break

def extract_surface(volume):
    """
    Extrahiert die Oberfläche aus einem binären Volumen mittels Marching Cubes.
    
    Args:
        volume: Binäres Volumen
    
    Returns:
        vertices: Koordinaten der Oberflächenpunkte
    """
    try:
        # Marching Cubes Algorithmus
        vertices, faces, normals, values = marching_cubes(volume)
        return vertices
    except:
        return None

def visualize_volume_and_surface(volume_section, output_dir):
    """
    Visualisiert einen Ausschnitt des binären Volumens und dessen Oberfläche.
    
    Args:
        volume_section: Binärer Volumenausschnitt
        output_dir: Ausgabeverzeichnis
    """
    # Erstelle Punktwolke aus den Voxeln
    points = np.argwhere(volume_section > 0)
    
    if len(points) == 0:
        print("Keine Punkte im ausgewählten Volumen gefunden!")
        return
    
    # Extrahiere Oberflächenpunkte
    surface_points = extract_surface(volume_section)
    
    if surface_points is None:
        print("Konnte keine Oberfläche extrahieren!")
        return
    
    # Maximale Anzahl von Punkten für die Visualisierung
    MAX_POINTS = 100000
    
    # Reduziere die Anzahl der Punkte wenn nötig
    if len(points) > MAX_POINTS:
        print(f"\nReduziere Volumenpunkte von {len(points)} auf {MAX_POINTS}")
        indices = np.random.choice(len(points), MAX_POINTS, replace=False)
        points = points[indices]
    
    if len(surface_points) > MAX_POINTS:
        print(f"Reduziere Oberflächenpunkte von {len(surface_points)} auf {MAX_POINTS}")
        indices = np.random.choice(len(surface_points), MAX_POINTS, replace=False)
        surface_points = surface_points[indices]
    
    print(f"\nAnzahl der visualisierten Punkte:")
    print(f"Volumen: {len(points)} Punkte")
    print(f"Oberfläche: {len(surface_points)} Punkte")
    
    # Erstelle zwei Subplots nebeneinander
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Volumen', 'Oberfläche')
    )
    
    # Füge Volumenpunktwolke hinzu
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.8
            ),
            name='Volumen'
        ),
        row=1, col=1
    )
    
    # Füge Oberflächenpunkte hinzu
    fig.add_trace(
        go.Scatter3d(
            x=surface_points[:, 0],
            y=surface_points[:, 1],
            z=surface_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.8
            ),
            name='Oberfläche'
        ),
        row=1, col=2
    )
    
    # Update Layout für beide Subplots
    for i in [1, 2]:
        fig.update_scenes(
            camera=dict(
                projection=dict(type='orthographic')
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(
                range=[0, volume_section.shape[2]],
                title='X'
            ),
            yaxis=dict(
                range=[0, volume_section.shape[1]],
                title='Y'
            ),
            zaxis=dict(
                range=[0, volume_section.shape[0]],
                title='Z'
            ),
            row=1, col=i
        )
    
    # Gesamtlayout aktualisieren
    fig.update_layout(
        width=1400,  # Doppelte Breite für zwei Plots
        height=800,
        title='Volumenausschnitt und Oberfläche',
        showlegend=True
    )
    
    fig.show()

def cleanup_temporary_files(output_dir):
    """
    Löscht temporäre Dateien und Ordner.
    
    Args:
        output_dir: Das Hauptausgabeverzeichnis
    """
    # Lösche temporäre Threshold-Ordner
    temp_dirs = glob.glob(os.path.join(output_dir, "temp_*"))
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporärer Ordner gelöscht: {temp_dir}")
        except Exception as e:
            print(f"Fehler beim Löschen von {temp_dir}: {e}")
    
    # Lösche __pycache__ Ordner
    pycache_dirs = glob.glob(os.path.join(output_dir, "**", "__pycache__"), recursive=True)
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"__pycache__ Ordner gelöscht: {pycache_dir}")
        except Exception as e:
            print(f"Fehler beim Löschen von {pycache_dir}: {e}")


