"""
CT-Daten Analyse Workflow
========================

Dieses Skript führt einen vollständigen Workflow zur Analyse von CT-Daten durch.
Der Workflow umfasst die Vorverarbeitung, Skelettierung, Punktdichteanalyse und
Gitteranalyse von CT-Bildstapeln.

Workflow-Schritte:
-----------------
1. Bildstapel-Vorverarbeitung
   - Laden des Bildstapels (BMP, TIFF, PNG, JPEG)
   - Average Pooling zur Dimensionsreduktion (optional)
   - Gauss-Filterung zur Glättung (optional)
   - Binarisierung mit automatischer Threshold-Findung
   - Morphologische Operationen (Opening/Closing, optional)

2. 3D-Skelettierung
   - Erzeugung des 3D-Skeletts
   - Konvertierung in Punktwolke
   - Visualisierung des Skeletts
   - Speicherung als PLY-Datei

3. Punktdichteanalyse
   - Berechnung der lokalen Punktdichte
   - Interaktive Filterung basierend auf Dichtewerten
   - Visualisierung der Dichtewerte mit Referenz-Sphere
   - Speicherung der gefilterten Punktwolke

4. Clustering und Gitteranalyse
   - K-Means Clustering der gefilterten Punkte
   - Automatische oder manuelle Auswahl der Cluster-Anzahl
   - Bestimmung der Hauptrichtungen
   - Erstellung des idealen Gitters
   - Finale Visualisierung mit Skelett, Gitter und Hauptachsen

5. Einheitszellen-Visualisierung
   - Interaktive Auswahl von Einheitszellen
   - Extraktion und Visualisierung der Oberfläche
   - Rotierte Darstellung für bessere Orientierung
   - Speicherung der Oberflächenpunkte

Ausgabestruktur:
---------------
complete_analysis_YYYYMMDD_HHMMSS/
├── processing_parameters.txt
    ├── Verarbeitungsparameter
    │   ├── Input Directory
    │   ├── Pooling Kernel Size
    │   ├── Gaussian Sigma
    │   ├── Threshold
    │   └── Morphological Kernel Size
    ├── Dichtefilterung
    │   ├── Density Threshold
    │   ├── Anzahl gefilterter Punkte
    │   ├── Ursprüngliche Punktanzahl
    │   └── Beibehaltener Anteil
    └── Clustering und Richtungsanalyse
        ├── Anzahl Cluster
        ├── Einheitszellen-Breite
        ├── Mittlere Abstände
        ├── Hauptrichtungsvektoren
        └── Gitterzentrum
        
├── 1_processed_stack/
│   └── processed_stack.npz
        ├── binarisiert: Verarbeiteter Bildstapel
        └── threshold: Verwendeter Schwellenwert

├── 2_skeleton/
│   └── skeleton.npz
        ├── skeleton: 3D Skelett (uint8)
        └── threshold: Verwendeter Schwellenwert

├── 3_density_analysis/
│   └── skeleton_centers_threshold_XX.npz
        ├── filtered_points: Gefilterte Punktwolke
        ├── threshold: Verwendeter Dichte-Schwellenwert
        ├── total_points: Gesamtanzahl Punkte
        ├── kept_points: Anzahl behaltener Punkte
        └── percentage_kept: Prozentsatz behaltener Punkte

├── 4_clustering/
│   └── clustering_results.npz
        ├── cluster_centers: K-Means Cluster-Zentren
        ├── mean1: Mittlerer Abstand der ersten Gruppe
        ├── mean2: Mittlerer Abstand der zweiten Gruppe
        ├── directions: Hauptrichtungsvektoren
        ├── grid_points: Koordinaten der idealen Gitterpunkte
        └── grid_center: Zentrum des berechneten Gitters

└── 5_surfaces_unitcells/
    └── surface_xX_yY_zZ.npz
        ├── surface_points: Rotierte Oberflächenpunkte
        ├── original_coordinates: Ursprüngliche Koordinaten
        └── rotation_matrix: Angewandte Rotationsmatrix

Verarbeitungsparameter:
---------------------
- Pooling Kernel: 1-8 (1 = überspringen)
- Gaussian Sigma: 0-2 (0 = überspringen)
- Morphological Kernel: 1,3,5,7 (1 = überspringen)
- Threshold: Automatisch bestimmt oder manuell angepasst

"""

from workflow_functions import (
    create_main_output_directory,
    select_input_directory,
    load_image_stack,
    analyze_point_density,
    finde_threshold,
    visualize_skeleton_3d,
    save_skeleton_ply,
    interactive_density_filtering,
    average_pooling_3d,
    visualize_processing_steps,
    visualize_projections,
    visualize_unit_cell,
    cleanup_temporary_files
)

import numpy as np
import os
from skimage.morphology import skeletonize
from scipy import ndimage

def main():
    try:
        # Erstelle Hauptausgabeordner
        main_dir = create_main_output_directory()
        
        # Wähle Eingabeordner
        input_dir = select_input_directory()
        
        # Lade Bildstapel
        print("\n=== Schritt 1: Bildstapel laden ===")
        original_stack = load_image_stack(input_dir)
        
        # Finde initialen Threshold basierend auf komplettem Bildstapel
        print("\n=== Berechne initialen Threshold ===")
        threshold = finde_threshold(original_stack, sample_fraction=0.01, image_step=10)
        print(f"\nGewählter Threshold: {threshold}")
        
        # Binarisiere den ursprünglichen Stack (für spätere Schritte)
        print("\n=== Binarisierung des ursprünglichen Stapels ===")
        original_binary_stack = (original_stack > threshold).astype(np.uint8) * 255
        print("Binarisierung abgeschlossen")

        # Initialisiere Verarbeitungsparameter
        sigma = 0
        pooling_kernel = 1
        morph_kernel = 0

        while True:  # Hauptschleife für die Verarbeitung
            # 1. Frage nach Pooling Kernel
            while True:
                try:
                    pooling_kernel = int(input("\nGeben Sie die Kernel-Größe für das Pooling ein (1-8, 1 = überspringen): "))
                    if 1 <= pooling_kernel <= 8:
                        break
                    print("Bitte geben Sie eine Zahl zwischen 1 und 8 ein!")
                except ValueError:
                    print("Bitte geben Sie eine ganze Zahl ein!")
            
            # 2. Frage nach Sigma für Gauss-Filter
            while True:
                try:
                    sigma = float(input("\nGeben Sie den Sigma-Wert für den Gauss-Filter ein (0-2, 0 = überspringen): "))
                    if 0 <= sigma <= 2:
                        break
                    print("Bitte geben Sie einen Wert zwischen 0 und 2 ein!")
                except ValueError:
                    print("Bitte geben Sie eine gültige Zahl ein!")
            
            # 3. Frage nach Kernel für morphologische Operationen
            while True:
                try:
                    morph_kernel = int(input("\nGeben Sie die Kernel-Größe für Opening/Closing ein (1,3,5,7, 1 = überspringen): "))
                    if morph_kernel in [1, 3, 5, 7]:
                        break
                    print("Bitte geben Sie 1, 3, 5 oder 7 ein!")
                except ValueError:
                    print("Bitte geben Sie eine ganze Zahl ein!")
            
            # Führe Verarbeitungsschritte durch
            if pooling_kernel > 1:
                print(f"\n=== Verarbeitung mit Pooling-Kernel {pooling_kernel} ===")
                pooled_stack = average_pooling_3d(original_stack, kernel_size=pooling_kernel)
            else:
                print("\n=== Pooling wird übersprungen ===")
                pooled_stack = original_stack
            
            if sigma > 0:
                print(f"\n=== Anwendung des Gauss-Filters (σ={sigma}) ===")
                gaussian_stack = ndimage.gaussian_filter(pooled_stack, sigma=sigma)
            else:
                print("\n=== Gauss-Filterung wird übersprungen ===")
                gaussian_stack = pooled_stack
            
            binary_stack = (gaussian_stack > threshold).astype(np.uint8) * 255
            
            if morph_kernel > 1:
                print(f"\n=== Führe morphologische Operationen durch (Kernel: {morph_kernel}) ===")
                structure = np.ones((morph_kernel, morph_kernel, morph_kernel), dtype=np.uint8)
                opened = ndimage.binary_opening(binary_stack, structure=structure).astype(np.uint8)
                cleaned_stack = ndimage.binary_closing(opened, structure=structure).astype(np.uint8) * 255
            else:
                print("\n=== Morphologische Operationen werden übersprungen ===")
                cleaned_stack = binary_stack

            # Temporäres Verzeichnis für aktuelle Iteration
            temp_dir = os.path.join(main_dir, f"temp_sigma_{sigma}_pooling_{pooling_kernel}_morph_{morph_kernel}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Visualisierungen
            visualize_processing_steps(original_stack, gaussian_stack, pooled_stack, binary_stack, cleaned_stack, 
                                     temp_dir, sigma, pooling_kernel, morph_kernel)
            
            # Projektionen vorerst nicht anzeigen
            
            # visualize_projections(cleaned_stack, temp_dir, sigma, pooling_kernel, morph_kernel)
            
            # Skelettierung für Vorschau
            print("\n=== Vorschau der Skelettierung ===")
            skeleton = skeletonize(cleaned_stack > 0)
            skeleton_points = np.argwhere(skeleton)[:, [1, 2, 0]]
            visualize_skeleton_3d(skeleton_points, sigma, pooling_kernel, morph_kernel)
            
            # Frage nach Zufriedenheit oder Änderung einzelner Parameter
            print("\nMöchten Sie einen Parameter ändern?")
            print("1. Pooling Kernel")
            print("2. Gauss Sigma")
            print("3. Morphologischer Kernel")
            print("4. Nein, alle Parameter sind in Ordnung")
            
            choice = input("\nWählen Sie eine Option (1-4): ")
            
            if choice == '4':
                # Speichere finale Ergebnisse
                final_processed_dir = os.path.join(main_dir, "1_processed_stack")
                final_skeleton_dir = os.path.join(main_dir, "2_skeleton")
                
                os.makedirs(final_processed_dir, exist_ok=True)
                os.makedirs(final_skeleton_dir, exist_ok=True)
                
                # Speichere Parameter
                with open(os.path.join(main_dir, 'processing_parameters.txt'), 'w') as f:
                    f.write(f'Input Directory: {input_dir}\n')
                    f.write(f'Gaussian Sigma: {sigma}\n')
                    f.write(f'Pooling Kernel Size: {pooling_kernel}\n')
                    f.write(f'Threshold: {threshold}\n')
                    f.write(f'Morphological Kernel Size: {morph_kernel}\n')
                
                # Speichere Ergebnisse
                np.savez_compressed(os.path.join(final_processed_dir, "processed_stack.npz"), 
                                  binarisiert=cleaned_stack,
                                  threshold=threshold)
                
                np.savez_compressed(os.path.join(final_skeleton_dir, 'skeleton.npz'), 
                                  skeleton=skeleton.astype(np.uint8),
                                  threshold=threshold)
                
                # Führe Punktdichteanalyse durch
                print("\n=== Schritt 3: Punktdichteanalyse ===")
                density_points, density_values = analyze_point_density(skeleton_points, main_dir)
                
                # Interaktive Dichtefilterung und Clustering
                filtered_points, cluster_centers, analysis_results = interactive_density_filtering(
                    density_points, density_values, main_dir)
                
                # Visualisiere Einheitszellen
                print("\n=== Schritt 5: Einheitszellen-Visualisierung ===")
                visualize_unit_cell(
                    original_binary_stack,
                    analysis_results['grid_points'],
                    analysis_results['unit_cell_width'],
                    pooling_kernel,
                    main_dir,
                    analysis_results['directions']
                )
                
                # Aufräumen
                import shutil
                shutil.rmtree(temp_dir)
                
                # Lösche temporäre Dateien
                cleanup_temporary_files(main_dir)
                
                break  # Verlasse die Hauptschleife
            
            # Aufräumen des temporären Verzeichnisses
            import shutil
            shutil.rmtree(temp_dir)
            
            # Wenn der Benutzer einen Parameter ändern möchte, wird die Schleife fortgesetzt
            # und nur der entsprechende Parameter wird neu abgefragt
            
        print(f"\nAnalyse abgeschlossen. Alle Ergebnisse wurden in {main_dir} gespeichert.")
        
    except Exception as e:
        print(f"\nFehler aufgetreten: {str(e)}")
        raise

if __name__ == "__main__":
    main() 