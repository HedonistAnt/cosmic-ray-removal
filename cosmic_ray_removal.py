import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

def remove_cosmic_rays(intensity, kernel_size=5, threshold=5):
    intensity = intensity.astype(float)
    smoothed = medfilt(intensity, kernel_size=kernel_size)
    difference = np.abs(intensity - smoothed)
    std_dev = np.std(difference)
    spikes = difference > (threshold * std_dev)
    cleaned = intensity.copy()
    cleaned[spikes] = smoothed[spikes]
    return cleaned, spikes

def process_spectrum(file_path, save_cleaned=True, show_plot=False, kernel_size=5, threshold=5, cleaned_path="None", plot_path = "None",save_plot = False):
    spectrum = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["wavelength", "intensity"])
    cleaned, spikes = remove_cosmic_rays(spectrum["intensity"].values, kernel_size, threshold)
    spectrum["cleaned_intensity"] = cleaned

    if save_cleaned:
        #out_file = file_path.replace(".txt", "_cleaned.txt")
        spectrum.to_csv(cleaned_path, sep="\t", index=False)
        print(f"📝 Saving cleaned file to: {cleaned_path}")
        
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(spectrum["wavelength"], spectrum["cleaned_intensity"], label="Cleaned", linewidth=2)
        plt.plot(spectrum["wavelength"], spectrum["intensity"], label="Original", alpha=0.5)
        plt.scatter(spectrum["wavelength"][spikes], spectrum["intensity"][spikes], color='red', s=10, label="Spikes")
        plt.title(os.path.basename(file_path))
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if save_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(spectrum["wavelength"], spectrum["cleaned_intensity"], label="Cleaned", linewidth=2)
        plt.plot(spectrum["wavelength"], spectrum["intensity"], label="Original", alpha=0.5)
        plt.scatter(spectrum["wavelength"][spikes], spectrum["intensity"][spikes], color='red', s=10, label="Spikes")
        plt.title(os.path.basename(file_path))
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"🖼️ Saving plot to: {plot_path}")
    
def process_folder(folder_path, output_dir, kernel_size=5, threshold=5, show_plot=False, save_plot = False,plot_path = "None"):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".asc"):
            full_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}")
            base, ext = os.path.splitext(filename)
            cleaned_path = os.path.join(output_dir, base + "_cleaned" + ext)
            plot_path = os.path.join(output_dir, base + "_plot.png")
            process_spectrum(full_path, save_cleaned=True, show_plot=show_plot,
                             kernel_size=kernel_size, threshold=threshold,cleaned_path=cleaned_path, plot_path=plot_path, save_plot=save_plot)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove cosmic rays from PL spectra in .txt format.")
    parser.add_argument("folder", type=str, help="Folder containing spectra (.txt files)")
    parser.add_argument("output_folder", type=str, help="Where to save cleaned spectra")
    parser.add_argument("--kernel", type=int, default=5, help="Kernel size for median filter (default: 5)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Threshold multiplier for spike detection (default: 5.0)")
    parser.add_argument("--plot", action="store_true", help="Show plots of each processed spectrum")
    parser.add_argument("--saveplot", action="store_true", help="Save plot of each processed spectrum")

    args = parser.parse_args()
    process_folder(args.folder, args.output_folder, kernel_size=args.kernel,
                    threshold=args.threshold, show_plot=args.plot, save_plot=args.saveplot)
    print("Processing complete.")