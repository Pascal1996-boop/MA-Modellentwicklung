import os
import warnings
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import autokeras as ak
import tempfile
import shutil
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from skopt import forest_minimize
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import binned_statistic
import openpyxl
import time
from scipy.stats import linregress
from tqdm import tqdm
from skopt.space import Categorical
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from plotting.plot_init import setupPGF, Mode, ikv_common_colors, ikv_colors

setupPGF([6.299213, 2.5234375])
from sklearn.inspection import PartialDependenceDisplay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial
from numpy.linalg import norm
from math import acos, degrees

print(openpyxl.__version__)
print("TensorFlow-Version:", tf.__version__)
print("Verfügbare GPUs:", tf.config.list_physical_devices('GPU'))
print("AutoKeras-Version:", ak.__version__)
print("📂 Aktuelles Arbeitsverzeichnis:", os.getcwd())
warnings.filterwarnings("ignore")

##############################DATEN EINLESEN#########################################
df = pd.read_csv(
    r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Spektren\Fertige ML CSVs\Iteration_10Spektren_1639_Max_global.csv",
    sep=";")

#########################Wellenlänge kalibrieren#######################################################################

# 1. Lade dein Spektrum (z.B. ID=2125) aus der Excel
spektrum_row = df[df["id"] == 4496].iloc[0]
pixel_cols = [col for col in df.columns if col.startswith("pixel_") and "_mean" in col]
spektrum = spektrum_row[pixel_cols].values

# 2. (Optional) Initiale Wellenlänge mit linespace (wie bisher)
pixel_indices = np.arange(len(spektrum))
wavelength_init = np.linspace(200, 1200, len(spektrum))

# 3. Automatische Peakfindung im gewünschten Bereich (z.B. 600–900 nm)
#     O15 bei 777.4 nm, O16 bei 844.8 nm, Halpha bei 656.3 nm (siehe Literaturwerte: Ueberwachung von Plasmaprozessen durch OES)
literature_peaks = {
    "OH": 306,    # Mittelpunkt der OH-Bande, Literatur z.B. 313–314 nm prominent
    "Hb":486,
    "Hα": 656.3,
    "O¹⁵": 777.4,
    "O¹⁶": 844.8
}
peak_search_ranges = {
    "OH": (340, 375),
    "Hb": (525, 540),
    "Hα": (675, 700),
    "O¹⁵": (790, 820),
    "O¹⁶": (850, 880)
}

pixel_to_wavelength = lambda pixel: 200 + pixel * (1000 / (len(spektrum) - 1))
found_pixels = []

for label, (min_wl, max_wl) in peak_search_ranges.items():
    min_pixel = np.argmin(np.abs(wavelength_init - min_wl))
    max_pixel = np.argmin(np.abs(wavelength_init - max_wl))
    sub_spec = spektrum[min_pixel:max_pixel]
    # Peak suchen (maximaler Wert im Bereich)
    peak_pixel_rel = np.argmax(sub_spec)
    peak_pixel = min_pixel + peak_pixel_rel
    found_pixels.append(peak_pixel)

    print(f"{label}: Literatur {literature_peaks[label]} nm, gefunden bei Pixel {peak_pixel}")

# 4. Fit: Pixel → Wellenlänge (Polynom 2. Ordnung)
measured_pixels = np.array(found_pixels)
literature_wavelengths = np.array([literature_peaks[k] for k in literature_peaks])
poly_fit = np.polyfit(measured_pixels, literature_wavelengths, deg=3)
poly_func = np.poly1d(poly_fit)

# 5. Korrigierte Wellenlängen berechnen
wavelength_corrected = poly_func(pixel_indices)

########################## Liste charakteristischer Peaks (aus der Kalibrierung / Literatur)############################

peaks= [
        {"wavelength": 234.4, "label": "SiO¹"},
        {"wavelength": 251.6, "label": "Si²"},
        {"wavelength": 306.0, "label": "OH"},
        {"wavelength": 391.0, "label": "CH³"},
        {"wavelength": 414.7, "label": "SiH⁴"},
        {"wavelength": 427.1, "label": "SiO⁵"},
        {"wavelength": 431.3, "label": "CH⁶"},
        {"wavelength": 451.1, "label": "CO⁷"},
        {"wavelength": 483.5, "label": "CO⁸"},
        {"wavelength": 486.22, "label": "Hβ⁹"},
        {"wavelength": 519.8, "label": "CO¹⁰"},
        {"wavelength": 561.0, "label": "CO¹¹"},
        {"wavelength": 608.0, "label": "CO¹²"},
        {"wavelength": 656.3, "label": "Hα¹³"},
        {"wavelength": 662.0, "label": "CO¹⁴"},
        {"wavelength": 777.4, "label": "O¹⁵"},
        {"wavelength": 844.8, "label": "O¹⁶"},
        {"wavelength": 516.0, "label": "C₂¹⁷"},
    ]

# Gibt alle Literaturpeaks im Wellenlängenbereich [start_nm, end_nm] ± tolerance zurück.
#Für die Zuordnung in der Sensitivitätsanalyse später

def find_peaks_in_range(start_nm, end_nm, tolerance=1):
    peaks = [
        {"wavelength": 234.4, "label": "SiO¹"},
        {"wavelength": 251.6, "label": "Si²"},
        {"wavelength": 306.0, "label": "OH"},
        {"wavelength": 391.0, "label": "CH³"},
        {"wavelength": 414.7, "label": "SiH⁴"},
        {"wavelength": 427.1, "label": "SiO⁵"},
        {"wavelength": 431.3, "label": "CH⁶"},
        {"wavelength": 451.1, "label": "CO⁷"},
        {"wavelength": 483.5, "label": "CO⁸"},
        {"wavelength": 486.22, "label": "Hβ⁹"},
        {"wavelength": 519.8, "label": "CO¹⁰"},
        {"wavelength": 561.0, "label": "CO¹¹"},
        {"wavelength": 608.0, "label": "CO¹²"},
        {"wavelength": 656.3, "label": "Hα¹³"},
        {"wavelength": 662.0, "label": "CO¹⁴"},
        {"wavelength": 777.4, "label": "O¹⁵"},
        {"wavelength": 844.8, "label": "O¹⁶"},
        {"wavelength": 516.0, "label": "C₂¹⁷"},
    ]
    found = []
    for peak in peaks:
        pos = peak["wavelength"]
        label = peak["label"]
        if (start_nm - tolerance) <= pos <= (end_nm + tolerance):
            found.append(label)
    return found

def find_peak_name_for_pixel(wl_nm, tolerance=0.5, peaks=peaks):
    for peak in peaks:
        if abs(wl_nm - peak["wavelength"]) <= tolerance:
            return peak["label"]
    return None

############################################################################################################################

# Originaldaten laden (unnormierte Spektren)
df_original = pd.read_csv(
    r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Spektren\Fertige ML CSVs\Iteration_10Spektren_1639_ohne_norm.csv",
    sep=";")

#####################Ordnerstruktur festlegen#######################################################
# Fester Speicherpfad für Ergebnisse auf dem Netzlaufwerk
BASE_OUTPUT_DIR = r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Ergbenisse"

# Input-Datei und -Name ohne Endung
input_file = r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Spektren\Fertige ML CSVs\Iteration_10Spektren_1639_Max_global.csv"
input_basename = os.path.splitext(os.path.basename(input_file))[0]  # z. B. "Spectra_maxnorm"

# Basis-Unterverzeichnisse innerhalb des festen Pfads
plot_dir = os.path.join(BASE_OUTPUT_DIR, "plots_p0.5_T0.1", input_basename)
doc_dir = os.path.join(BASE_OUTPUT_DIR, "dokumentation_p0.5_T0.1", input_basename)
result_dir = os.path.join(BASE_OUTPUT_DIR, "ergebnisse_p0.5_T0.1", input_basename)
model_dir = os.path.join(BASE_OUTPUT_DIR, "modelle_p0.5_T0.1", input_basename)
cv_plot_dir = os.path.join(BASE_OUTPUT_DIR, "cv_plots_p0.5_T0.1", input_basename)
interpretation_dir = os.path.join(BASE_OUTPUT_DIR, "interpretation_p0.5_T0.1", input_basename)

# Ordner anlegen
for d in [plot_dir, doc_dir, result_dir, model_dir, cv_plot_dir, interpretation_dir]:
    os.makedirs(d, exist_ok=True)

# Unterordner für PCA-Plots und -Excels
pca_plot_dir = os.path.join(interpretation_dir, "pca_plots")
pca_excel_dir = os.path.join(interpretation_dir, "pca_excel")
sam_plot_dir = os.path.join(interpretation_dir, "sam_plots")
sam_excel_dir = os.path.join(interpretation_dir, "sam_excel")
shap_dir = os.path.join(interpretation_dir, "sam_excel")
PDP_excel_dir = os.path.join(interpretation_dir, "PDP_excel")
PDP_einzel = os.path.join(interpretation_dir, "PDP_einzel")
PDP_alle = os.path.join(interpretation_dir, "PDP_alle")
PDP_global_einzel= os.path.join(interpretation_dir, "PDP_global_einzel")
PDP_global_einzel_pixel= os.path.join(interpretation_dir, "PDP_global_einzel_pixel")
PDP_2d = os.path.join(interpretation_dir, "PDP_2d")
save_dir = os.path.join(plot_dir, "plots/top10")
sens_dir = os.path.join(interpretation_dir, "PDP_Pixel_Einzel")

os.makedirs(pca_plot_dir, exist_ok=True)
os.makedirs(pca_excel_dir, exist_ok=True)
os.makedirs(sam_plot_dir, exist_ok=True)
os.makedirs(sam_excel_dir, exist_ok=True)
os.makedirs(PDP_excel_dir, exist_ok=True)
os.makedirs(PDP_einzel, exist_ok=True)
os.makedirs(PDP_alle, exist_ok=True)
os.makedirs(PDP_2d, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(sens_dir, exist_ok=True)
os.makedirs(PDP_global_einzel, exist_ok=True)
os.makedirs(PDP_global_einzel_pixel, exist_ok=True)

df_meta = df[["id", "exposure"]].copy()  # ID und Belichtungszeit sichern

###################MIN-MAX Global berechnen #############################################
y_cols = [col for col in df_original.columns if col.startswith("pixel_") and col.endswith("_mean")]
global_min = df_original[y_cols].values.min()
global_max = df_original[y_cols].values.max()

print("Globales Minimum:", global_min)
print("Globales Maximum:", global_max)

###############Datenaufteilung in Targets und Features + Splitting in Trainings und Testdaten#############

# Separierung der Targets und Features
exclude_cols = ["id", "t_coat", "t_purge", "exposure"]
std_cols = [col for col in df.columns if col.startswith("pixel_") and col.endswith("_std")]
y_cols = [col for col in df.columns if col.startswith("pixel_") and col.endswith("_mean")]

# Split in Trainings- und Testdaten (90/10)
trainval_df, test_df = train_test_split(df, test_size=0.10, random_state=42)
X_trainval = trainval_df.drop(columns=exclude_cols + y_cols + std_cols)
y_trainval = trainval_df[y_cols]
X_test = test_df.drop(columns=exclude_cols + y_cols + std_cols)
y_test = test_df[y_cols]

# Umwandlung in Numpy-Arrays für Modellinput
X_trainval_np = X_trainval.values
X_test_np = X_test.values

############################################################################################################

# Zielgrößen: Mittelwerte der Spektren
y = df[y_cols].copy()

# Feature-Matrix: alle Parameter ohne ID, Zeit, Targets und std
X = df.drop(columns=exclude_cols + y_cols + std_cols)
X_np = X.values

# === Funktionen zur Modellerkennung, Training, Auswertung, Dokumentation ===
df_trainval = df_meta.loc[X_trainval.index].copy()
df_test = df_meta.loc[X_test.index].copy()


###########################################################################################################

########################Sicherheitscheck auf ungültige Daten##################################
def find_non_floats(df, name=""):
    failed = df.applymap(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
    if failed.any().any():
        bad_entries = df[failed.any(axis=1)]
        raise ValueError(f"\u274c Nicht-numerische Werte in {name} gefunden:\n{bad_entries}")


find_non_floats(X, name="Eingabematrix X")
find_non_floats(y, name="Zielmatrix y")
if not np.isfinite(X.values).all():
    raise ValueError("\u274c X enthält ungültige Werte!")
if not np.isfinite(y.values).all():
    raise ValueError("\u274c y enthält ungültige Werte!")


##############################################Plot für Auswirkung Belichtungszeit kleiner ton###########################


def plot_extreme_spectrum_from_excel_and_model(
    csv_path, best_model, feature_cols, wavelength_axis, plot_dir, ikv_common_colors):

    df = pd.read_csv(csv_path, sep=";")

    # --------- 1. Dein bisheriger Spezialfall ---------
    mask = (
        (df["exposure"] < df["t_on"]) &
        (df["O2"] == df["O2"].max()) &
        (df["P_MW"] == df["P_MW"].max()) &
        (df["t_on"] == df["t_on"].max()) &
        (df["HMDSO"] == df["HMDSO"].min())
    )
    if not mask.any():
        print("❌ Kein passender Parametersatz gefunden!")
    else:
        row = df[mask].iloc[0]
        pixel_cols = [col for col in df.columns if col.startswith("pixel_") and col.endswith("mean")]
        std_cols = [col for col in df.columns if col.endswith("std")]

        spectrum = row[pixel_cols].values.astype(float)
        stds = row[std_cols].values.astype(float)
        std_up = spectrum + stds
        std_lo = spectrum - stds

        X_input = row[feature_cols].values.astype(float).reshape(1, -1)
        pred_spectrum = best_model.predict(X_input)[0]
        pred_spectrum = inverse_max(pred_spectrum, global_max)

        id_str = row["id"] if "id" in row else f"idx={int(row.name)}"
        param_str = (
            f"O₂={row['O2']}, Power={row['P_MW']}, ton={row['t_on']}, "
            f"HMDSO={row['HMDSO']}, toff={row['t_off']}, p={row['p']}, "
            f"exposure={row['exposure']}"
        )
        title = f"id: {id_str} | {param_str}"

        plt.figure(figsize=(11,5))
        plt.plot(wavelength_axis, spectrum, label="Echtes Spektrum", color=ikv_common_colors["g"], lw=1)
        plt.plot(wavelength_axis, std_up, label="+1σ", color=ikv_common_colors["b"], linestyle="--")
        plt.plot(wavelength_axis, std_lo, label="-1σ", color=ikv_common_colors["b"], linestyle="--")
        plt.plot(wavelength_axis, pred_spectrum, label="Vorhersage", color=ikv_common_colors["d"], linestyle="--")
        plt.title(title, fontsize=11)
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.legend()
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"Spektrum_mit_STDW_{id_str}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ Plot gespeichert: {save_path}")

    # --------- 2. SPEKTRUM MIT MAXIMALER EINZEL-STDW ---------
    # Finde die Zeile und Pixel mit dem absolut größten Einzelwert bei allen STDW-Spalten
    std_cols = [col for col in df.columns if col.endswith("std")]
    std_array = df[std_cols].values  # shape: (n_samples, n_pixel)
    max_std_idx_flat = np.argmax(np.abs(std_array))
    row_idx, pixel_idx = np.unravel_index(max_std_idx_flat, std_array.shape)
    row_stdmax = df.iloc[row_idx]
    pixel_col = std_cols[pixel_idx]
    pixel_nr = pixel_col.replace("pixel_", "").replace("_mean", "").replace("_std", "")

    spectrum_maxstd = row_stdmax[pixel_cols].values.astype(float)
    stds_maxstd = row_stdmax[std_cols].values.astype(float)
    std_up_maxstd = spectrum_maxstd + stds_maxstd
    std_lo_maxstd = spectrum_maxstd - stds_maxstd

    X_input_maxstd = row_stdmax[feature_cols].values.astype(float).reshape(1, -1)
    pred_spectrum_maxstd = best_model.predict(X_input_maxstd)[0]
    pred_spectrum_maxstd = inverse_max(pred_spectrum_maxstd, global_max)

    id_str_maxstd = row_stdmax["id"] if "id" in row_stdmax else f"idx={int(row_stdmax.name)}"
    param_str_maxstd = (
        f"O₂={row_stdmax['O2']}, Power={row_stdmax['P_MW']}, ton={row_stdmax['t_on']}, "
        f"HMDSO={row_stdmax['HMDSO']}, toff={row_stdmax['t_off']}, p={row_stdmax['p']}, "
        f"exposure={row_stdmax['exposure']}"
    )
    title_maxstd = (
        f"id: {id_str_maxstd} | {param_str_maxstd}\n"
        f"(Max Einzel-STDW bei Pixel {pixel_nr}, Wert: {std_array[row_idx, pixel_idx]:.2f})"
    )

    plt.figure(figsize=(11,5))
    plt.plot(wavelength_axis, spectrum_maxstd, label="Echtes Spektrum", color=ikv_common_colors["g"], lw=1)
    plt.plot(wavelength_axis, std_up_maxstd, label="+1σ", color=ikv_common_colors["b"], linestyle="--")
    plt.plot(wavelength_axis, std_lo_maxstd, label="-1σ", color=ikv_common_colors["b"], linestyle="--")
    plt.plot(wavelength_axis, pred_spectrum_maxstd, label="Vorhersage", color=ikv_common_colors["d"], linestyle="--")
    plt.title(title_maxstd, fontsize=11)
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.legend()
    plt.tight_layout()
    save_path_maxstd = os.path.join(plot_dir, f"Spektrum_mit_MaxEinzelSTDW_{id_str_maxstd}_pix{pixel_nr}.png")
    plt.savefig(save_path_maxstd)
    plt.close()
    print(f"🖼️ Plot mit max. Einzel-STDW gespeichert: {save_path_maxstd}")

    return

def plot_top_n_highest_std_spectra_from_excel(
    csv_path, best_model, feature_cols, wavelength_axis, plot_dir, ikv_common_colors, inverse_max, global_max, n=10):

    df = pd.read_csv(csv_path, sep=";")
    pixel_cols = [col for col in df.columns if col.startswith("pixel_") and col.endswith("mean")]
    std_cols = [col for col in df.columns if col.endswith("std")]

    # Alle Einzel-STDWs als Matrix (n_samples, n_pixel)
    std_array = np.abs(df[std_cols].values)
    # Finde die n größten Einzelwerte (Positionen im Array)
    flat_indices = np.argpartition(std_array.flatten(), -n)[-n:]
    sorted_flat_indices = flat_indices[np.argsort(-std_array.flatten()[flat_indices])]  # Absteigend sortiert

    # Zu jedem gefundenen Index: Zeile/Spektrum bestimmen, aber nur einzigartige Parameterkombis
    selected_rows = set()
    spectra_to_plot = []

    for idx in sorted_flat_indices:
        row_idx, pixel_idx = np.unravel_index(idx, std_array.shape)
        if row_idx not in selected_rows:
            selected_rows.add(row_idx)
            row = df.iloc[row_idx]
            pixel_col = std_cols[pixel_idx]
            pixel_nr = pixel_col.replace("pixel_", "").replace("_mean", "").replace("_std", "")

            spectrum = row[pixel_cols].values.astype(float)
            stds = row[std_cols].values.astype(float)
            std_up = spectrum + stds
            std_lo = spectrum - stds

            X_input = row[feature_cols].values.astype(float).reshape(1, -1)
            pred_spectrum = best_model.predict(X_input)[0]
            pred_spectrum = inverse_max(pred_spectrum, global_max)

            id_str = row["id"] if "id" in row else f"idx={int(row.name)}"
            param_str = (
                f"O₂={row['O2']}, Power={row['P_MW']}, ton={row['t_on']}, "
                f"HMDSO={row['HMDSO']}, toff={row['t_off']}, p={row['p']}, "
                f"exposure={row['exposure']}"
            )
            title = (
                f"id: {id_str} | {param_str}\n"
                f"(Einzel-STDW bei Pixel {pixel_nr}, Wert: {std_array[row_idx, pixel_idx]:.2f})"
            )

            plt.figure(figsize=(11, 5))
            plt.plot(wavelength_axis, spectrum, label="Echtes Spektrum", color=ikv_common_colors["g"], lw=1)
            plt.plot(wavelength_axis, std_up, label="+1σ", color=ikv_common_colors["b"], linestyle="--")
            plt.plot(wavelength_axis, std_lo, label="-1σ", color=ikv_common_colors["b"], linestyle="--")
            plt.plot(wavelength_axis, pred_spectrum, label="Vorhersage", color=ikv_common_colors["d"], linestyle="--")
            plt.title(title, fontsize=11)
            plt.xlabel("Wellenlänge [nm]")
            plt.ylabel("Intensität [a.u.]")
            plt.legend()
            plt.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"Top10_Spektrum_MaxEinzelSTDW_{id_str}_pix{pixel_nr}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"🖼️ Top10-Plot gespeichert: {save_path}")

            spectra_to_plot.append((row_idx, pixel_idx, std_array[row_idx, pixel_idx]))
            if len(spectra_to_plot) >= n:
                break

    return spectra_to_plot  # Optional, für weitere Auswertung/Tabellen




######################################################################################################


#####################################SAM-Hilfsfunktionen#########################

def compute_sam_angle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Berechnet den Spectral Angle (in Grad) zwischen einem echten Spektrum y_true
    # und einer Vorhersage y_pred.

    # Formel (pro Spektrum):
    #  θ = arccos( (y_true · y_pred) / (‖y_true‖ * ‖y_pred‖) )

    # - y_true · y_pred  = Σ_{k=1..P} (y_true[k] * y_pred[k])           # Skalarprodukt
    # - ‖y_true‖         = sqrt(Σ_{k=1..P} y_true[k]²)                 # euklidische Norm
    # - ‖y_pred‖         = sqrt(Σ_{k=1..P} y_pred[k]²)
    # - θ in Radiant = arccos( (y_true · y_pred) / (‖y_true‖ * ‖y_pred‖) )
    # - θ in Grad    = θ_rad * (180 / π)

    # 1) Skalarprodukt berechnen: y_true · y_pred
    dot = np.dot(y_true, y_pred)

    # 2) Normen berechnen: ‖y_true‖ und ‖y_pred‖
    n_true = norm(y_true)
    n_pred = norm(y_pred)

    # 3) cos(θ) = (y_true·y_pred) / (‖y_true‖ * ‖y_pred‖)
    cos_sim = dot / (n_true * n_pred)

    # 4) Numerische Stabilität: Wert auf [-1, +1] beschränken
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # 5) θ_rad = arccos(cos_sim)
    angle_rad = acos(cos_sim)

    # 6) Umrechnung in Grad: θ_deg = θ_rad * (180/π)
    return degrees(angle_rad)

############################SAM ANgleS für viele Vektoren##############################################

def compute_sam_angles(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # y_true und y_pred: 2D-Array, z.B. (n_samples, n_pixel)
    dot = np.sum(y_true * y_pred, axis=1)
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-12
    cosine = np.clip(dot / denom, -1.0, 1.0)
    angles = np.arccos(cosine) * 180 / np.pi
    return angles  # Shape: (n_samples,)


def compute_mean_sam_batch(y_true_batch: np.ndarray, y_pred_batch: np.ndarray, return_std: bool = False):
    # Berechnet den Mean SAM (und optional die Standardabweichung) für ein Batch von Spektren.
    # Args:
    #   y_true_batch: Array (N × P) mit echten Spektren
    #  y_pred_batch: Array (N × P) mit vorhergesagten Spektren
    #  return_std: Wenn True, wird zusätzlich die Standardabweichung zurückgegeben

    # Returns:
    #    float (mean SAM), oder tuple (mean SAM, std SAM) wenn return_std=True
    #

    sam_vals = [compute_sam_angle(y_true_batch[i], y_pred_batch[i]) for i in range(y_true_batch.shape[0])]
    if return_std:
        return float(np.mean(sam_vals)), float(np.std(sam_vals))
    return float(np.mean(sam_vals))


def evaluate_model_max_with_sam(model, X, y_normiert, name="Datensatz"):
    y_pred_normiert = model.predict(X, verbose=0)
    y_true_normiert = y_normiert.values

    # --- Normierte Werte
    mse_norm = mean_squared_error(y_true_normiert, y_pred_normiert)
    rmse_norm = np.sqrt(mse_norm)
    mean_sam_norm, std_sam_norm = compute_mean_sam_batch(y_true_normiert, y_pred_normiert, return_std=True)

    # --- Rückskalierte Werte
    y_pred_global = y_pred_normiert * global_max
    y_true_global = y_true_normiert * global_max

    mse_global = mean_squared_error(y_true_global, y_pred_global)
    rmse_global = np.sqrt(mse_global)

    # ✅ RMSE pro Sample + Standardabweichung berechnen
    rmse_per_sample = np.sqrt(np.mean((y_true_global - y_pred_global) ** 2, axis=1))
    rmse_std_global = float(np.std(rmse_per_sample))

    r2_global = r2_score(y_true_global, y_pred_global)
    mean_sam_global, std_sam_global = compute_mean_sam_batch(y_true_global, y_pred_global, return_std=True)

    # --- Konsolenausgabe: Tabellarisch
    print(f"\n===[ {name} ]===")
    print(f"{'':16} | {'Normiert':>12} | {'Rückskaliert':>12}")
    print("-" * 50)
    print(f"{'MSE':16} | {mse_norm:12.6f} | {mse_global:12.2f}")
    print(f"{'RMSE':16} | {rmse_norm:12.6f} | {rmse_global:12.2f} ± {rmse_std_global:.2f}")
    print(f"{'R²':16}   | {'':12} | {r2_global:12.4f}")
    print(f"{'Mean SAM [°][°]':16} | {mean_sam_norm:12.2f} | {mean_sam_global:12.2f} ± {std_sam_global:.2f}")

    # --- Rückgabe aller Metriken
    return (
        mse_norm, rmse_norm, mean_sam_norm, std_sam_norm,
        mse_global, rmse_global, rmse_std_global, r2_global,
        mean_sam_global, std_sam_global
    )


def inverse_max(spectra_max, spec_max):
    return spectra_max * spec_max


def evaluate_and_plot_sam(y_df, y_pred_arr, df_meta, plot_dir, doc_dir, prefix="Test"):
    # 1) True-Werte als NumPy
    y_true_arr = y_df.values  # Shape: (N, P)
    y_pred_arr = np.asarray(y_pred_arr)  # Shape: (N, P)

    # 2) Berechne SAM-Winkel für jedes Sample
    sam_vals = []
    for i in range(y_true_arr.shape[0]):
        sam_vals.append(compute_sam_angle(y_true_arr[i], y_pred_arr[i]))
    sam_values = np.array(sam_vals)  # Array der Länge N

    # 3) Mean SAM und STDW Sam global
    mean_sam = float(np.mean(sam_values))
    stdw_sam = float(np.std(sam_values))

    # 4) Ausgabe in Konsole
    print(f"\n📊 {prefix} Mean SAM: {mean_sam:.2f}°")
    for idx, angle in enumerate(sam_values):
        meta_str = get_id_exposure_string(df_meta, y_df.index[idx])
        print(f"{prefix} {idx + 1:03d} ({meta_str}): SAM = {angle:.2f}°")

    # 5) Erstelle Verzeichnis, falls nicht vorhanden
    os.makedirs(plot_dir, exist_ok=True)

    # 6) Histogramm der SAM-Verteilung mit Mittelwert-Linie
    plt.figure(figsize=(6, 4))
    plt.hist(sam_values, bins=30, edgecolor=ikv_common_colors["g"], alpha=0.7, label="SAM-Werte")
    plt.axvline(mean_sam, color=ikv_common_colors["d"], linestyle="--", lw=2, label=f"Mittelwert: {mean_sam:.2f}°")
    plt.xlabel("SAM-Winkel [°]")
    plt.ylabel("Anzahl Spektren")
    plt.title(f"{prefix}-Set: Verteilung des SAM")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(plot_dir, f"sam_histogramm_{prefix.lower()}_clean.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"🖼️ Histogramm gespeichert: {hist_path}")

    # 7) Extrembeispiele: minimaler / maximaler Winkel
    idx_min = np.argmin(sam_values)
    idx_max = np.argmax(sam_values)

    # 7.1) True & Pred für Extremfälle
    y_min_true = y_true_arr[idx_min]
    y_min_pred = y_pred_arr[idx_min]
    y_max_true = y_true_arr[idx_max]
    y_max_pred = y_pred_arr[idx_max]

    # ⏱ R² berechnen für die beiden Extremfälle
    r2_min = r2_score(y_min_true, y_min_pred)
    r2_max = r2_score(y_max_true, y_max_pred)

    pixels = np.arange(y_true_arr.shape[1])  # Pixel-Indizes 0..P-1

    plt.figure(figsize=(10, 4))

    # 7.2) Plot mit minimalem SAM-Wert
    plt.subplot(1, 2, 1)
    plt.plot(wavelength_corrected, y_min_true, label="Reales Spektrum",color=ikv_common_colors["g"], linewidth=1)
    plt.plot(wavelength_corrected, y_min_pred, label="Vorhersage",color=ikv_common_colors["d"], linestyle="--")
    title_min = (
        f"{prefix} Min SAM: {sam_values[idx_min]:.2f}°\n"
        f"{get_id_exposure_string(df_meta, y_df.index[idx_min])}\n"
        f"R² = {r2_min:.3f}"
    )
    plt.title(title_min)
    plt.ylim(bottom=0)
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.legend()

    # 7.3) Plot mit maximalem SAM-Wert
    plt.subplot(1, 2, 2)
    plt.plot(wavelength_corrected, y_max_true, label="Reales Spektrum", color=ikv_common_colors["g"], linewidth=1)
    plt.plot(wavelength_corrected, y_max_pred, label="Vorhersage",color=ikv_common_colors["d"], linestyle="--")
    title_max = (
        f"{prefix} Max SAM: {sam_values[idx_max]:.2f}°\n"
        f"{get_id_exposure_string(df_meta, y_df.index[idx_max])}\n"
        f"R² = {r2_max:.3f}"
    )
    plt.title(title_max)
    plt.ylim(bottom=0)
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.legend()

    plt.tight_layout()
    ext_path = os.path.join(plot_dir, f"sam_extrembeispiele_{prefix.lower()}.png")
    plt.savefig(ext_path)
    plt.close()
    print(f"🖼️ Extrembeispiel-Plots gespeichert: {ext_path}")

    # 8) Rückgabe von Mean SAM und allen SAM-Werten
    # 9) SAM-Werte als Excel speichern
    meta_sub = df_meta.loc[y_df.index]  # exakt dieselben Indizes wie y_df!
    sam_df = pd.DataFrame({
        "Index": y_df.index,
        "ID": meta_sub["id"].values,
        "Exposure": meta_sub["exposure"].values,
        "SAM [°]": sam_values
    })
    sam_path = os.path.join(doc_dir, f"sam_winkel_{prefix.lower()}.xlsx")
    sam_df.to_excel(sam_path, index=False)
    print(f"📄 SAM-Winkel exportiert: {sam_path}")

    return mean_sam, stdw_sam, sam_values

######################Um Spektren zu Plotten bei verschiedenen SAM werten##############################################

def plot_sam_examples(y_true_arr, y_pred_arr, sam_values, wavelength_axis, target_sams=(1,5,10,15), plot_dir="./plots"):
    """Plottet Beispiele für verschiedene SAM-Winkel (z.B. ca. 1°, 5°, 10°, 15°)."""
    os.makedirs(plot_dir, exist_ok=True)
    for target in target_sams:
        idx = np.argmin(np.abs(sam_values - target))  # Index des Spektrums, das dem Zielwert am nächsten ist
        true_spec = y_true_arr[idx]
        pred_spec = y_pred_arr[idx]
        actual_sam = sam_values[idx]
        plt.figure(figsize=(10,4))
        plt.plot(wavelength_axis, true_spec, label="Reales Spektrum", color=ikv_common_colors["g"], lw=1)
        plt.plot(wavelength_axis, pred_spec, label="Vorhersage", color=ikv_common_colors["d"], linestyle="--")
        plt.title(f"SAM ≈ {target}° (tatsächlich: {actual_sam:.2f}°), Index {idx}")
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"sam_beispiel_{target}deg_idx{idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ SAM-Beispiel für {target}° gespeichert: {save_path}")

##################################################################################################################

def plot_vertical_abstand_parameterkombinationen(y_true, y_pred, plot_dir, prefix="Test"):
    """
    Paired Dotplot und Residuenplot für signed Abweichungen.
    - Grüner Punkt: Real (signed Abweichung)
    - Oranger Punkt: Predicted (signed Abweichung)
    - Unten: Differenzbalken (Residual: Predicted - Real)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    y_true_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mean_spectrum = np.mean(y_true_arr, axis=0)

    abstand_real = np.linalg.norm(y_true_arr - mean_spectrum, axis=1) * np.sign(
        np.sum(y_true_arr - mean_spectrum, axis=1))
    abstand_pred = np.linalg.norm(y_pred_arr - mean_spectrum, axis=1) * np.sign(
        np.sum(y_pred_arr - mean_spectrum, axis=1))

    n = len(abstand_real)
    x = np.arange(n)

    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Oben: Paired-Dotplot signed
    axs[0].axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    axs[0].scatter(x, abstand_real, color=ikv_common_colors["g"], label='Real', s=36, alpha=0.85, zorder=3)
    axs[0].scatter(x, abstand_pred, color=ikv_common_colors["d"], label='Predicted', s=36, alpha=0.85, zorder=3)
    for i in range(n):
        axs[0].plot([x[i], x[i]], [abstand_real[i], abstand_pred[i]], color="gray", alpha=0.25)
    axs[0].set_ylabel("Signed Abweichung zum Mittelwert")
    axs[0].set_title(f"Real (grün) vs. Predicted (orange) je Parameterkombination ({prefix}-Set)")
    axs[0].legend()

    # Unten: Fehlerplot (Residuals)
    axs[1].bar(x, abstand_pred - abstand_real, color=ikv_common_colors["d"], alpha=0.7)
    axs[1].axhline(0, color=ikv_common_colors["g"], linestyle="--", alpha=0.7)
    axs[1].set_xlabel("Parameterkombination / Spektrum-Index")
    axs[1].set_ylabel("Vorhersage - Real (signed)")
    axs[1].set_title("Residuen: Vorhersage minus Real")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"vertical_symmetry_residuals_{prefix.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Symmetrischer Residuenplot gespeichert: {plot_path}")


################# Charakteristische Linien der HMDSO/02 Spektren für das Clustering##############################################

def plot_characteristic_lines_numbered():
    ax = plt.gca()
    gruen = ikv_common_colors["g"]
    dunkelblau = ikv_common_colors["d"]
    blassgrau = "#dddddd"
    label_fontsize = 7
    grey = ikv_common_colors["grey"]

    # Für bestimmte Peaks einen x-Offset für die Labels vergeben
    label_x_offsets = {
        "CH³": -10,  # CH-Label etwa 18 nm nach links schieben, damit es bei 389 nicht die 400 verdeckt
        "SiO⁵": 0,  # Beispiel: nach rechts
        "SiH⁴": -3,  # ggf. weitere Anpassungen
        "CO⁷": 3,  # ggf. weitere Anpassungen
    }

    peaks = [
        {"wavelength": 234.4, "label": "SiO¹"},
        {"wavelength": 251.6, "label": "Si²"},
        {"wavelength": 306.0, "label": "OH"},
        {"wavelength": 391.0, "label": "CH³"},
        {"wavelength": 414.7, "label": "SiH⁴"},
        {"wavelength": 427.1, "label": "SiO⁵"},
        {"wavelength": 431.3, "label": "CH⁶"},
        {"wavelength": 451.1, "label": "CO⁷"},
        {"wavelength": 483.5, "label": "CO⁸"},
        {"wavelength": 486.22, "label": "Hβ⁹"},
        {"wavelength": 519.8, "label": "CO¹⁰"},
        {"wavelength": 561.0, "label": "CO¹¹"},
        {"wavelength": 608.0, "label": "CO¹²"},
        {"wavelength": 656.3, "label": "Hα¹³"},
        {"wavelength": 662.0, "label": "CO¹⁴"},
        {"wavelength": 777.4, "label": "O¹⁵"},
        {"wavelength": 844.8, "label": "O¹⁶"},
        {"wavelength": 516.0, "label": "C₂¹⁷"},
    ]

    areas = [
        {"start": 306, "end": 321, "label": "OH-Bande"},
        {"start": 453, "end": 464, "label": "H₂ Fulcher-α"},
        {"start": 570, "end": 650, "label": "H₂ Fulcher-α"},
    ]

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin, ymax + 0.05 * (ymax - ymin))  # 5% mehr Platz oben

    # Bereiche (horizontal, dezent)
    for area in areas:
        ax.hlines(y=ymin + 0.04 * y_range, xmin=area["start"], xmax=area["end"],
                  color=blassgrau, linewidth=1.2, alpha=0.5, zorder=0)
        ax.annotate(area["label"],
                    xy=((area["start"] + area["end"]) / 2, ymin + (0.03) * y_range),
                    xycoords='data',
                    ha="center", va="top", fontsize=label_fontsize + 1, color=grey, alpha=0.94, zorder=0,
                    annotation_clip=False)

    colors = [gruen, dunkelblau]
    for i, peak in enumerate(peaks):
        color = colors[i % 2]
        xpos = peak["wavelength"]
        ax.axvline(xpos, color=color, linestyle="--", alpha=0.7, linewidth=0.25, zorder=1)
        offset = label_x_offsets.get(peak["label"], 0)
        # Unten
        if i % 2 == 0:
            ax.annotate(peak["label"],
                        xy=(xpos, 0), xycoords=('data', 'axes fraction'),
                        xytext=(offset, -7), textcoords='offset points',
                        ha="center", va="top", fontsize=label_fontsize, color=color, alpha=0.98, rotation=0,
                        annotation_clip=False)
        # Oben
        else:
            ax.annotate(peak["label"],
                        xy=(xpos, 1), xycoords=('data', 'axes fraction'),
                        xytext=(offset, -10), textcoords='offset points',
                        ha="center", va="bottom", fontsize=label_fontsize, color=color, alpha=0.88, rotation=0,
                        annotation_clip=False)

    ax.tick_params(axis='x', pad=1)  # 8 Punkte nach oben (default: 4)


######### SAM CLUSTER#######################
def cluster_by_sam_angle(y_pred, angle_threshold_deg=5):
    y_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    similarity = cosine_similarity(y_norm)
    sam_angles = np.arccos(np.clip(similarity, -1, 1))
    np.fill_diagonal(sam_angles, 0.0)
    sam_degrees = np.degrees(sam_angles)

    cluster_labels = -np.ones(len(y_pred), dtype=int)
    cluster_id = 0
    for i in range(len(y_pred)):
        if cluster_labels[i] != -1:
            continue
        close_indices = np.where(sam_degrees[i] < angle_threshold_deg)[0]
        new_members = [idx for idx in close_indices if cluster_labels[idx] == -1]
        cluster_labels[new_members] = cluster_id
        cluster_labels[i] = cluster_id
        cluster_id += 1
    return cluster_labels


def get_representative_spectrum_and_parameters(y_cluster, X_cluster):
    if isinstance(y_cluster, pd.DataFrame):
        y_cluster = y_cluster.values
    if not isinstance(X_cluster, pd.DataFrame):
        X_cluster = pd.DataFrame(X_cluster)

    X_cluster = X_cluster.reset_index(drop=True)
    centroid = np.mean(y_cluster, axis=0)
    distances = np.linalg.norm(y_cluster - centroid, axis=1)
    idx = int(np.argmin(distances))
    return y_cluster[idx], X_cluster.iloc[idx]


def get_representative_spectrum(y_cluster):
    if isinstance(y_cluster, pd.DataFrame):
        y_cluster = y_cluster.values
    centroid = np.mean(y_cluster, axis=0)
    distances = np.linalg.norm(y_cluster - centroid, axis=1)
    idx = int(np.argmin(distances))
    return y_cluster[idx]


def validate_clusters_with_real_sam(y_real, cluster_labels):
    results = {}
    for cid in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cid)[0]
        if len(indices) < 2:
            continue
        if isinstance(y_real, pd.DataFrame):
            cluster_specs = y_real.iloc[indices]
        else:
            cluster_specs = y_real[indices]
        normed = cluster_specs / np.linalg.norm(cluster_specs, axis=1, keepdims=True)
        sim = cosine_similarity(normed)
        sam_deg = np.degrees(np.arccos(np.clip(sim, -1, 1)))
        upper = sam_deg[np.triu_indices_from(sam_deg, k=1)]
        results[cid] = np.mean(upper)
    return results


def plot_cluster_spectra(y_pred, cluster_labels, outdir):
    os.makedirs(outdir, exist_ok=True)
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    for cid in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cid)[0]
        plt.figure(figsize=(10, 5))
        for idx in indices:
            plt.plot(wavelength_corrected, y_pred[idx], alpha=0.3)

        plot_characteristic_lines_numbered()
        plt.title(f" Cluster {cid}")
        plt.ylim(bottom=0)
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(outdir, f"cluster_{cid}_spectra.png"))
        plt.close()


def parameter_analysis(parameter_df, cluster_labels, outdir, suffix=""):
    os.makedirs(outdir, exist_ok=True)
    df = parameter_df.copy()
    df["Cluster_ID"] = cluster_labels
    suffix_str = f"_{suffix}" if suffix else ""

    for col in df.columns:
        if col == "Cluster_ID":
            continue
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="Cluster_ID", y=col, data=df, palette="Set3")
        plt.title(f"Verteilung von {col} über Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"boxplot_{col}{suffix_str}.png"), dpi=300)
        plt.close()

    representative_params = []
    for cid in sorted(df["Cluster_ID"].unique()):
        cluster_data = df[df["Cluster_ID"] == cid].copy()
        y_cluster = cluster_data.drop(columns=["Cluster_ID"]).values
        _, representative_param = get_representative_spectrum_and_parameters(y_cluster, cluster_data)
        representative_param = representative_param.copy()
        representative_param["Cluster_ID"] = cid
        representative_param["N_Spectra_in_Cluster"] = len(cluster_data)  # <--- HINZUGEFÜGT
        representative_params.append(representative_param)

    representative_df = pd.DataFrame(representative_params)
    representative_df.to_excel(os.path.join(outdir, f"cluster_parameter_representative{suffix_str}.xlsx"), index=False)

    scatter_pairs = [("O2", "HMDSO"), ("P_MW", "t_on"), ("t_on", "t_off"), ("p", "HMDSO")]

    for x_param, y_param in scatter_pairs:
        if x_param in df.columns and y_param in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[x_param], y=df[y_param], hue=df["Cluster_ID"], palette="tab20", s=100,
                            edgecolor=ikv_common_colors["g"], linewidth=0.3)
            plt.title(f"{x_param} vs. {y_param} nach Cluster")
            plt.xlabel(x_param)
            plt.ylabel(y_param)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"scatter_{x_param}_{y_param}{suffix_str}.png"), dpi=300,
                        bbox_inches='tight')
            plt.close()

    df.to_excel(os.path.join(outdir, f"parameter_cluster_mapping{suffix_str}.xlsx"), index=False)


def compute_cluster_closest_to_middle_and_sam(y_real, y_pred, cluster_labels, parameter_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []
    cluster_mapping = parameter_df.copy()
    cluster_mapping["Cluster_ID"] = cluster_labels

    representative_params = []

    for cid in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cid)[0]
        if len(indices) < 1:
            continue

        real_cluster = y_real.iloc[indices] if isinstance(y_real, pd.DataFrame) else y_real[indices]
        pred_cluster = y_pred.iloc[indices] if isinstance(y_pred, pd.DataFrame) else y_pred[indices]

        if len(indices) >= 2:
            center_real, param_real = get_representative_spectrum_and_parameters(real_cluster,
                                                                                 parameter_df.iloc[indices].reset_index(
                                                                                     drop=True))
            center_pred, _ = get_representative_spectrum_and_parameters(pred_cluster, parameter_df.iloc[indices])

            a_norm = center_real / np.linalg.norm(center_real)
            b_norm = center_pred / np.linalg.norm(center_pred)
            sam_rad = np.arccos(np.clip(np.dot(a_norm, b_norm), -1, 1))
            sam_deg = np.degrees(sam_rad)

            param_row = param_real.copy()
            param_row["Cluster_ID"] = cid
            param_row["SAM_Mittigste_to_Mittigste"] = sam_deg
            representative_params.append(param_row)
        else:
            results.append({
                "Cluster_ID": cid,
                "N_Members": len(indices),
                "SAM_Mittigste_to_Mittigste": np.nan
            })

    result_df = pd.DataFrame(representative_params)
    result_df.to_excel(os.path.join(outdir, "cluster_sam_Mittigste_shift.xlsx"), index=False)
    cluster_mapping.to_excel(os.path.join(outdir, "parameter_cluster_mapping.xlsx"), index=False)

    return result_df


def analyze_sam_cluster_similarity(cluster_centers, name="SAM_Cluster_Similarity", output_dir="cluster_output"):
    # Berechnet eine Ähnlichkeitsmatrix (Cosinus-Similarität) zwischen Clusterzentren (z. B. aus SAM),
    # visualisiert sie als Heatmap und gibt die Matrix zurück.
    # Cosinus-Ähnlichkeit (je höher, desto ähnlicher)
    similarity_matrix = cosine_similarity(cluster_centers)

    # Optional: Cosinus-Winkel (arccos), falls gewünscht
    # angle_matrix = np.arccos(np.clip(similarity_matrix, -1.0, 1.0)) * 180 / np.pi

    # Heatmap anzeigen und speichern
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Cosinus-Similarität zwischen Clusterzentren")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_similarity_matrix.png")
    plt.close()

    return similarity_matrix


def sam_angle(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    sim = np.clip(np.dot(a_norm, b_norm), -1, 1)
    return np.degrees(np.arccos(sim))


def match_predicted_to_real_clusters(y_real, y_pred, labels_real, labels_pred, parameter_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []

    central_real = {
        cid: get_representative_spectrum_and_parameters(
            y_real[labels_real == cid], parameter_df[labels_real == cid]
        )[0]
        for cid in np.unique(labels_real)
        if np.sum(labels_real == cid) >= 2
    }

    central_pred = {
        cid: get_representative_spectrum_and_parameters(
            y_pred[labels_pred == cid], parameter_df[labels_pred == cid]
        )[0]
        for cid in np.unique(labels_pred)
        if np.sum(labels_pred == cid) >= 2
    }

    for pred_cid, pred_center in central_pred.items():
        best_match = None
        best_sam = np.inf
        for real_cid, real_center in central_real.items():
            angle = sam_angle(pred_center, real_center)
            if angle < best_sam:
                best_sam = angle
                best_match = real_cid

        results.append({
            "Pred_Cluster": pred_cid,
            "Best_Real_Cluster": best_match,
            "SAM_Center_to_Center": best_sam
        })

    result_df = pd.DataFrame(results)
    result_path = os.path.join(outdir, "matched_clusters_sam.xlsx")
    result_df.to_excel(result_path, index=False)
    return result_df


########SAM

def prediction_consistency_by_real_clusters(X, y_pred, y_real_labels, outdir, suffix=""):
    #
    #  Für jeden realen Spektrencluster (aus y_real_labels) werden:
    # - die zugehörigen Eingabeparameter X extrahiert
    #  - deren Modellvorhersagen y_pred analysiert (SAM-Winkel + Linienplots)
    #  - alle relevanten Informationen dokumentiert und gespeichert

    #  Args:
    #     X (pd.DataFrame): Eingabeparameter
    #     y_pred (np.ndarray): Modellvorhersagen (Spektren)
    #     y_real_labels (np.ndarray): Clusterzuordnung der realen Spektren (z. B. durch SAM)
    #     outdir (str): Speicherort für Ergebnisdateien
    #     suffix (str): Optionaler Suffix zur Unterscheidung
    #
    os.makedirs(outdir, exist_ok=True)
    suffix_str = f"_{suffix}" if suffix else ""

    df = X.copy()
    df["Cluster_ID"] = y_real_labels

    summary_rows = []

    for cluster_id in sorted(df["Cluster_ID"].unique()):
        indices = df[df["Cluster_ID"] == cluster_id].index.to_numpy()
        cluster_params = X.loc[indices]
        cluster_preds = y_pred[indices]

        param_outdir = os.path.join(outdir, f"cluster_{cluster_id}")
        os.makedirs(param_outdir, exist_ok=True)

        cluster_params.to_excel(
            os.path.join(param_outdir, f"parameter_mapping{suffix_str}.xlsx"), index=False
        )

        # === Repräsentativer Parametersatz zum mittigsten Spektrum ===
        center_spec, rep_param = get_representative_spectrum_and_parameters(cluster_preds, cluster_params)
        pd.DataFrame([rep_param]).to_excel(
            os.path.join(param_outdir, f"cluster_parameter_representative{suffix_str}.xlsx"), index=False
        )

        if len(cluster_preds) >= 2:
            normed = cluster_preds / np.linalg.norm(cluster_preds, axis=1, keepdims=True)
            cos_sim = cosine_similarity(normed)
            sam_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            upper_tri = sam_deg[np.triu_indices_from(sam_deg, k=1)]

            mean_sam = np.mean(upper_tri)
            std_sam = np.std(upper_tri)

            plt.figure()
            plt.hist(upper_tri, bins=30, edgecolor=ikv_common_colors["g"])
            plt.title(f"Cluster {cluster_id}: SAM-Winkel zwischen Vorhersagen")
            plt.xlabel("SAM [°]")
            plt.ylabel("Häufigkeit")
            plt.tight_layout()
            plt.savefig(os.path.join(param_outdir, f"sam_hist{suffix_str}.png"), dpi=300)
            plt.close()
        else:
            mean_sam = np.nan
            std_sam = np.nan

        plt.figure(figsize=(10, 5))
        for spec in cluster_preds:
            plt.plot(wavelength_corrected, spec, alpha=0.3)
        plot_characteristic_lines_numbered()
        plt.title(f"Vorhergesagte Spektren – Cluster {cluster_id}")
        plt.ylim(bottom=0)
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.tight_layout()
        plt.savefig(os.path.join(param_outdir, f"predicted_spectra_plot{suffix_str}.png"), dpi=300)
        plt.close()

        summary_rows.append({
            "Cluster_ID": cluster_id,
            "N_Samples": len(indices),
            "Mean_SAM": mean_sam,
            "Std_SAM": std_sam
        })

    pd.DataFrame(summary_rows).to_excel(
        os.path.join(outdir, f"prediction_variation_summary{suffix_str}.xlsx"), index=False
    )


def cluster_parameter_sam_excel(
        X, cluster_labels, spectra, angle_threshold=10,
        parameter_names=None, output_dir="winkelanalyse"):
    os.makedirs(output_dir, exist_ok=True)
    cluster_ids = sorted(np.unique(cluster_labels))

    medoid_indices = []
    for cid in cluster_ids:
        inds = np.where(cluster_labels == cid)[0]
        X_cluster = X.iloc[inds]
        centroid = np.mean(X_cluster.values, axis=0)
        dists = np.linalg.norm(X_cluster.values - centroid, axis=1)
        medoid_idx_in_cluster = np.argmin(dists)
        medoid_indices.append(inds[medoid_idx_in_cluster])  # speichert Index im Original-DataFrame

    # Jetzt holen wir die Spektren der Medoids (statt die Parameter)
    medoid_spectra = pd.DataFrame(spectra[medoid_indices])

    n = len(cluster_ids)
    angle_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                angle_matrix[i, j] = 0
            elif i < j:
                # *** Winkelberechnung im SPEKTRALRAUM ***
                angle = sam_angle(medoid_spectra.iloc[i], medoid_spectra.iloc[j])
                angle_matrix[i, j] = angle
                angle_matrix[j, i] = angle

    # 3. Excel: Medoid-Spektren + Winkel-Matrix in EINER Datei
    param_df = pd.DataFrame(medoid_spectra)
    param_df.insert(0, "Cluster_ID", cluster_ids)
    angle_df = pd.DataFrame(angle_matrix, columns=[f"Winkel_zu_{cid}" for cid in cluster_ids])
    combined_df = pd.concat([param_df, angle_df], axis=1)
    combined_df.to_excel(os.path.join(output_dir, "cluster_spectrum_angle_matrix_combined.xlsx"), index=False)

    # 4. Maximales Winkelpaar plotten (diesmal: Spektren, nicht Parameter)
    triu = np.triu(angle_matrix, k=1)
    max_idx = np.unravel_index(np.argmax(triu), triu.shape)
    max_angle = angle_matrix[max_idx]
    print(
        f"Größter SAM-Winkel: {max_angle:.2f}° zwischen Cluster {cluster_ids[max_idx[0]]} und {cluster_ids[max_idx[1]]}")

    plt.figure(figsize=(12, 6))
    plt.plot(wavelength_corrected, medoid_spectra.iloc[max_idx[0]], label=f"Cluster {cluster_ids[max_idx[0]]}")
    plot_characteristic_lines_numbered()
    plt.plot(wavelength_corrected, medoid_spectra.iloc[max_idx[1]], label=f"Cluster {cluster_ids[max_idx[1]]}")
    plt.title(f"Medoid-Spektren der Cluster mit größtem SAM-Winkel ({max_angle:.2f}°)")
    plt.ylim(bottom=0)
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_sam_angle_spectra.png"))
    plt.close()

    # 5. Merge-Kandidaten finden (Winkel < threshold)
    merge_pairs = []
    member_counts = [np.sum(cluster_labels == cid) for cid in cluster_ids]
    for i in range(n):
        for j in range(i + 1, n):
            if angle_matrix[i, j] < angle_threshold:
                merge_pairs.append(
                    (cluster_ids[i], cluster_ids[j], angle_matrix[i, j], member_counts[i], member_counts[j]))

    merge_df = pd.DataFrame(merge_pairs, columns=["Cluster_1", "Cluster_2", "SAM_angle_deg", "N_1", "N_2"])
    merge_df.to_excel(os.path.join(output_dir, f"cluster_merge_candidates_{angle_threshold}deg.xlsx"), index=False)

    # 6. Spektrenvergleichsplot für Merge-Kandidaten (wie gehabt)
    for c1, c2, _, n1, n2 in merge_pairs:
        idx1 = np.where(cluster_labels == c1)[0]
        idx2 = np.where(cluster_labels == c2)[0]
        plt.figure(figsize=(12, 5))
        for i in idx1:
            plt.plot(wavelength_corrected, spectra[i], color=ikv_common_colors["b"], alpha=0.3)
        for i in idx2:
            plt.plot(wavelength_corrected, spectra[i], color=ikv_common_colors["grey"], alpha=0.3)
        plot_characteristic_lines_numbered()
        plt.title(f"Spektrenvergleich: Cluster {c1} (blau, n={n1}) & Cluster {c2} (rot, n={n2})")
        plt.ylim(bottom=0)
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"merge_candidate_{c1}_{c2}.png"))
        plt.close()

    return combined_df, merge_df


#########SAM Interpretation ENDE#######################################

###############PCA Anfang#######################################
# === PCA UND VISUALISIERUNGSMODULE ===

def plot_pca_variance_analysis(ev_ratio, threshold=0.01, cumulative_threshold=0.90, plot_dir=".", name="Modell"):
    cumulative_var = np.cumsum(ev_ratio)
    num_components = len(ev_ratio)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(1, num_components + 1), ev_ratio, label='Erklärte Varianz pro PC')
    plt.plot(range(1, num_components + 1), cumulative_var, color='black', marker='o', label='Kumulative Varianz')

    for i, bar in enumerate(bars):
        if ev_ratio[i] < threshold:
            bar.set_color('red')
        elif cumulative_var[i] >= cumulative_threshold and ev_ratio[i] >= threshold:
            bar.set_color('orange')
    plt.axhline(threshold, color='red', linestyle='--', label=f'{int(threshold * 100)}% Schwelle')
    plt.axhline(cumulative_threshold, color='gray', linestyle=':',
                label=f'{int(cumulative_threshold * 100)}% kumulativ')
    plt.xlabel("PC Index")
    plt.ylabel("Erklärte Varianz")
    plt.title("PCA: Erklärte und kumulative Varianz")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{name}_variance_explained.png")
    plt.savefig(save_path)
    plt.close()  # 🔍 wichtig: schließt das Plot-Fenster und gibt Speicher frei
    print(f"✅ Plot gespeichert: {save_path}")


def plot_pca_loadings(pca, pixel_axis=None, n_components=5, plot_dir=".", name="Modell"):
    import os
    for i in range(n_components):
        plt.figure(figsize=(10, 3))
        x_axis = pixel_axis if pixel_axis is not None else np.arange(pca.components_.shape[1])
        plt.plot(wavelength_corrected, pca.components_[i])
        plot_characteristic_lines_numbered()
        plt.title(f"PCA-Komponente {i + 1} – Loadings über Pixel")
        plt.xlabel("Pixel (Index oder Wellenlänge)")
        plt.ylabel("Gewichtung")
        plt.grid(True)
        plt.tight_layout()
        # Jetzt: Jede Komponente als separate Datei
        plt.savefig(os.path.join(plot_dir, f"{name}_loadings_PC{i + 1}.png"))
        plt.close()


def plot_pc_scatter(pcs, name="Modell", plot_dir="."):
    plt.figure(figsize=(6, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Scatterplot (PC1 vs PC2) – {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_scatter_PC1_PC2.png"))

    if pcs.shape[1] >= 3:
        plt.figure(figsize=(6, 6))
        plt.scatter(pcs[:, 0], pcs[:, 2], alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.title(f"PCA Scatterplot (PC1 vs PC3) – {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name}_scatter_PC1_PC3.png"))


def plot_pc_correlations(pcs, parameter_df, plot_dir=".", name="Modell"):
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
    corr_matrix = pd.concat([parameter_df.reset_index(drop=True), pcs_df], axis=1).corr()
    pc_cols = pcs_df.columns
    param_cols = parameter_df.columns
    corr_subset = corr_matrix.loc[param_cols, pc_cols]

    plt.figure(figsize=(12, 6))
    import seaborn as sns
    sns.heatmap(corr_subset, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Korrelation zwischen Eingabeparametern und PCs")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_correlations.png"))


######################PCA HAUPTFUNKTION #########################

def full_pca_analysis_with_all_visuals(y_real, parameter_df=None, pixel_axis=None, n_components=30, name="Modell",
                                       plot_dir=None, excel_dir=None):
    """
    Führt vollständige PCA-Analyse inkl.:
    - Scree-Plot
    - Loadings (PC1–PC5)
    - PC1 vs. PC2 / PC1 vs. PC3 Scatter
    - Korrelation PCx ↔ Eingabeparameter (falls gegeben)
    """
    if plot_dir is None:
        plot_dir = "."
    if excel_dir is None:
        excel_dir = "."
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    print(f"🔎 Starte PCA-Analyse für: {name}")

from sklearn.decomposition import PCA

# Initialisierung der PCA mit 30 Komponenten
pca = PCA(n_components=30)
pca.fit(y_real)

# Transformation der Spektren in den Raum der Hauptkomponenten
pcs = pca.transform(y_real)

    plot_pca_variance_analysis(pca.explained_variance_ratio_, plot_dir=plot_dir, name=name)
    plot_pca_loadings(pca, pixel_axis=pixel_axis, n_components=min(5, n_components), plot_dir=plot_dir, name=name)
    plot_pc_scatter(pcs, name=name, plot_dir=plot_dir)

    if parameter_df is not None:
        print("📈 Berechne Korrelationen zwischen PCs und Eingabeparametern…")
        plot_pc_correlations(pcs, parameter_df, plot_dir=plot_dir, name=name)

    # 1. Explained Variance
    pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained Variance Ratio": pca.explained_variance_ratio_
    }).to_excel(os.path.join(excel_dir, f"{name}_explained_variance.xlsx"), index=False)

    # 2. Principal Components (Scores)
    pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])]).to_excel(
        os.path.join(excel_dir, f"{name}_principal_components.xlsx"), index=False
    )

    # 3. Loadings
    pixel_index = pixel_axis if pixel_axis is not None else np.arange(pca.components_.shape[1])
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])])
    loadings_df.insert(0, "Pixel", pixel_index)
    loadings_df.to_excel(os.path.join(excel_dir, f"{name}_loadings.xlsx"), index=False)

    print(f"📄 PCA-Daten exportiert nach: {excel_dir}")

    return pca, pcs


###############PCA ENDE#######################################

def plot_spectra_per_region(
        df_bereichszuordnung,
        pixel_cols=None,
        wavelength_axis=None,
        save_dir=".",
        show_reference=False,
        reference_id=None,
        show_grenzspektren=True
):
    """
    Plottet für jeden SAM-Bereich ("nah", "mittel", "weit") alle zugehörigen Spektren
    in einem überlagerten Plot. Optional werden das Referenzspektrum und die Grenzspektren
    farblich hervorgehoben.

    df_bereichszuordnung : DataFrame mit Spalten wie "SAM_Bereich", "SAM_angle", Parameter, "id", pixel_xxx...
    pixel_cols           : Liste der Spaltennamen, die zum Spektrum gehören (z.B. ["pixel_0_mean", ...])
    wavelength_axis      : 1D-Array mit Wellenlängen (optional, sonst x=Pixelnummer)
    save_dir             : Zielordner für Plots
    show_reference       : Wenn True, Referenzspektrum grün hervorheben (benötigt reference_id)
    reference_id         : ID des Referenzspektrums (für Highlighting)
    show_grenzspektren   : Wenn True, Grenzspektren (max/min SAM je Bereich) rot hervorheben
    """
    if pixel_cols is None:
        pixel_cols = [col for col in df_bereichszuordnung.columns if "pixel_" in col]

    region_names = ["nah", "mittel", "weit"]

    for region in region_names:
        sub = df_bereichszuordnung[df_bereichszuordnung["SAM_Bereich"] == region]
        X = sub[pixel_cols].values
        n = len(sub)
        plt.figure(figsize=(12, 5))

        # (Optional) Grenzspektren hervorheben (mit max/min SAM im Bereich)
        if show_grenzspektren and n > 2:
            sam_vals = sub["SAM_angle"].values
            idx_min = np.argmin(sam_vals)
            idx_max = np.argmax(sam_vals)
        else:
            idx_min, idx_max = None, None

        for i in range(n):
            color = "grey"
            linewidth = 1
            alpha = 0.45
            label = None
            if show_grenzspektren and (i == idx_min or i == idx_max):
                color=ikv_common_colors["g"]
                linewidth = 2.5
                alpha = 0.9
                label = "Grenzspektrum" if i == idx_min else None
            if show_reference and (reference_id is not None) and (sub.iloc[i]["id"] == reference_id):
                color=ikv_common_colors["d"]
                linewidth = 3.5
                alpha = 1.0
                label = "Referenz"
            x_axis = wavelength_axis if wavelength_axis is not None else np.arange(X.shape[1])
            plt.plot(x_axis, X[i], color=color, linewidth=linewidth, alpha=alpha, label=label if i == 0 else None)

        plt.xlabel("Wellenlänge [nm]" if wavelength_axis is not None else "Pixel")
        plt.ylabel("Intensität [counts]")
        plt.title(f"Alle Spektren im Bereich '{region}' (n={n})")
        plt.grid(True, alpha=0.3)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        save_path = f"{save_dir}/SAM_Spektren_{region}.png"
        plt.savefig(os.path.join(plot_dir, save_path), dpi=300)
        plt.close()
        print(f"✅ Plot für Bereich '{region}' gespeichert: {save_path}")


#############################Extrahiere die Indies der signifikanten Wellenlängenbereiche für die gezielte PDP#########

def get_significant_pixel_ranges(component, threshold_abs=0.05):
    abs_weights = np.abs(component)
    significant = abs_weights > threshold_abs
    ranges = []
    in_range = False
    for i, val in enumerate(significant):
        if val and not in_range:
            start = i
            in_range = True
        elif not val and in_range:
            end = i - 1
            in_range = False
            ranges.append((start, end))
    if in_range:
        ranges.append((start, len(significant) - 1))
    return ranges


################################### Eigene PDP/Sensitivitätsanalyse##################################################

def pdp_for_keras_model_auto_grid(
        model,
        X,
        parameter,
        pixel_range,
        grid_resolution=10,
        plot_path=None,
        title=None
):
    # 1D-PDP: Für einen Parameter und Pixelbereich
    unique_vals = np.unique(X[parameter])  # Alle unterschiedlichen Werte des Parameters im Datensatz
    if (X[parameter].dtype == "O") or (len(unique_vals) <= grid_resolution):
        param_values = unique_vals  # Wenn kategorisch ODER nur wenige Ausprägungen: nutze nur diese Werte
    else:
        param_min, param_max = X[parameter].min(), X[parameter].max()  # Minimum und Maximum des Parameters bestimmen
        param_values = np.linspace(param_min, param_max,
                                   grid_resolution)  # Gleichmäßig verteilte Werte im Bereich [min, max]

    mean_intensities = []  # Hier werden die Mittelwerte der Intensitäten gespeichert
    X_mod = X.copy()  # Es wird eine Kopie der Eingabematrix erstellt, um Parameter zu variieren
    start, end = pixel_range  # Start- und Endindex für den Pixel-/Wellenlängenbereich

    for val in param_values:  # Für jeden Wert, den der zu untersuchende Parameter annimmt:
        X_mod[
            parameter] = val  # Setze alle Werte dieser Spalte im DataFrame auf den aktuellen Wert (alle anderen Features bleiben gleich!)
        preds = model.predict(X_mod.values,verbose=0)  # Mache Vorhersagen für alle modifizierten Datensätze (N x Pixels)
        mean_intensities.append(preds[:, start:end + 1].mean())  # Berechne den Mittelwert:
        #  - erst über den gewünschten Pixelbereich (start:end+1)
        #  - dann gemittelt über alle Samples (also über alle Reihen des DataFrames)
    if plot_path or title:
        plt.figure(figsize=(7, 4))
        plt.plot(param_values, mean_intensities, marker="o")
        plt.xlabel(parameter)
        plt.ylabel(f"Intensität [a.u.]Pixel {start}-{end}")
        plt.title(title if title else f"PDP: {parameter} → Mittelwert Pixel {start}-{end}")
        plt.grid(True)
        if plot_path:
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            print(f"PDP gespeichert: {plot_path}")
        plt.close()
    return param_values, mean_intensities


def pdp_for_keras_model_2d(
        model,
        X,
        parameter1,
        parameter2,
        pixel_range,
        pixel_source_range=None,       # <--- NEU
        grid_resolution=10,
        plot_path=None,
        title=None
):
    param1_min, param1_max = X[parameter1].min(), X[parameter1].max()
    param2_min, param2_max = X[parameter2].min(), X[parameter2].max()
    param1_values = np.linspace(param1_min, param1_max, grid_resolution)
    param2_values = np.linspace(param2_min, param2_max, grid_resolution)

    mean_intensities = np.zeros((grid_resolution, grid_resolution))
    X_mod = X.copy()
    start, end = pixel_range

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            X_mod[parameter1] = val1
            X_mod[parameter2] = val2
            preds = model.predict(X_mod.values, verbose=0)
            mean_intensities[i, j] = preds[:, start:end + 1].mean()

    # Automatisch Peakname bestimmen (nur EIN Pixel bei Peakpixel!)
    wl_peak = wavelength_corrected[start]  # start == end für Peakpixel!
    peak_name = find_peak_name_for_pixel(wl_peak, tolerance=1)
    peak_str = f" ({peak_name})" if peak_name else ""

    # Bereich für den Titel extrahieren
    if pixel_source_range is not None:
        src_start, src_end = pixel_source_range
        wl_start = wavelength_corrected[src_start]
        wl_end = wavelength_corrected[src_end]
        bereich_str = f"(Peakpixel aus Bereich: {wl_start:.1f}-{wl_end:.1f} nm)"
    else:
        bereich_str = "(Peakpixel aus Bereich:)"

    # Plot-Titel setzen
    plot_title = (
        f"2D: {parameter1} & {parameter2} | λ={wl_peak:.1f}nm {peak_str}\n"
        f"{bereich_str}"
    )

    if plot_path or title:
        plt.figure(figsize=(7, 5))
        X_grid, Y_grid = np.meshgrid(param2_values, param1_values)
        cp = plt.contourf(X_grid, Y_grid, mean_intensities, levels=20, cmap="viridis")
        plt.xlabel(f"{parameter2} [sccm]")
        plt.ylabel(f"{parameter1} [sccm]")
        plt.title(plot_title)
        plt.colorbar(cp, label="Mittlere Intensität [a.u.]")
        if plot_path:
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            print(f"2D PDP gespeichert: {plot_path}")
        plt.close()
    return param1_values, param2_values, mean_intensities

#################### Mehrere Pixelbereiche für einen Parameter in EINEM Plot#########################################

def plot_pdp_multiple_pixelranges(
        model,
        X,
        parameter,
        pixel_ranges,
        grid_resolution=10,
        plot_path=None,
        title=None,
        poly_func=None
):

    plt.figure(figsize=(8, 5))
    for (start, end) in pixel_ranges:
        param_values, mean_intensities = pdp_for_keras_model_auto_grid(
            model=model,
            X=X,
            parameter=parameter,
            pixel_range=(start, end),
            grid_resolution=grid_resolution,
            plot_path=None,
            title=None
        )
        # Umrechnung der Pixelgrenzen in Wellenlängen (korrekte Kalibrierung!)
        wl_start = float(poly_func(start)) if poly_func else start
        wl_end = float(poly_func(end)) if poly_func else end
        # Peaks in diesem Bereich suchen
        peaks_in_range = find_peaks_in_range(wl_start, wl_end,tolerance=1 )
        if peaks_in_range:
            label = f"{wl_start:.1f}–{wl_end:.1f} nm ({', '.join(peaks_in_range)})"
        else:
            label = f"{wl_start:.1f}–{wl_end:.1f} nm"
        plt.plot(param_values, mean_intensities, marker="o", label=label)
    plt.xlabel(parameter)
    plt.ylabel("Mittlere Intensität [a.u.]")
    plt.title(title if title else f"Mittlere Intensität für {parameter} auf verschiedenen Pixelbereichen")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=300)
        print(f"Sammel-PDP gespeichert: {plot_path}")
    plt.close()

############################Autokeras_Bezug###################################################

def detect_architecture_style(model):
    layer_names = [layer.__class__.__name__ for layer in model.layers]
    if any("Conv" in name for name in layer_names):
        return "CNN"
    elif any("LSTM" in name or "GRU" in name or "RNN" in name for name in layer_names):
        return "RNN"
    elif any("Dense" in name for name in layer_names):
        return "MLP (Dense)"
    else:
        return "Unbekannt / Custom"


# damit später bei den plots die Belichtungszeit mit angegeben wird
def get_id_exposure_string(df, index):
    row = df.loc[index]
    return f"ID: {row['id']}, Exposure: {row['exposure']} ms"


########################### Training mehrerer AutoKeras-Modelle ###########################
def train_autokeras_models(X_trainval_np, y_trainval, X_test_np, y_test, n_models=10):
    models, metrics = [], []
    histories = []
    durations = []
    optimizers_classes, optimizers_configs = [], []

    for i in range(n_models):
        print(f"\nStarte AutoKeras-Modell {i + 1}/{n_models}...")

        try:
            # Manuelles Splitting der Trainingsdaten in Trainings- und Validierungsdaten
            X_train_np, X_val_np, y_train_df, y_val_df = train_test_split(
                X_trainval_np, y_trainval, test_size=0.15, random_state=42 + i)

            # Initialisierung des AutoKeras StructuredDataRegressor
            reg = ak.StructuredDataRegressor(max_trials=50, seed=42, overwrite=True)

            # Start Zeitmessung für das Training
            start_time = time.time()

            # Training des Modells mit Validierungsdaten
            history = reg.fit(X_train_np, y_train_df, validation_data=(X_val_np, y_val_df), epochs=300, verbose=0)

            # Ende der Zeitmessung
            end_time = time.time()

            # Angenommen reg ist dein AutoKeras-Model nach fit()
            for trial in reg.tuner.oracle.trials.values():
                val_loss_history = trial.metrics.get_history('val_loss')
                num_epochs = len(val_loss_history)
                print(f"Trial {trial.trial_id}: {num_epochs} Epochen")

            # Export des trainierten Modells
            model = reg.export_model()

            # Ergebnisse und Verlauf speichern
            duration = end_time - start_time
            models.append(model)
            durations.append(duration)

            if history is not None:
                # Zugriff auf history.history["loss"], "val_loss", etc.
                print(f"  ✅ Epochen: {len(history.history['loss'])}")
            else:
                print("⚠️ Kein History-Objekt zurückgegeben! (Möglicherweise nach Export oder Fehler beim Fit.)")

            histories.append(history.history)
            optimizer_class = model.optimizer.__class__
            optimizer_config = model.optimizer.get_config()
            optimizers_classes.append(optimizer_class)
            optimizers_configs.append(optimizer_config)

            # Modell abspeichern
            model.save(f"{model_dir}/modell{i + 1}", save_format="tf")

            #################################Metriken auf die Train-Daten der  Modelle bestimmen####################################
            (train_mse_norm, train_rmse_norm, train_mean_sam_norm, train_std_sam_norm,
             train_mse_global, train_rmse_global, train_rmse_std_global, train_r2_global,
             train_mean_sam_global, train_std_sam_global) = evaluate_model_max_with_sam(
                model,
                X_trainval_np,
                y_trainval,
                name="Trainvaldaten"
            )
            #################################Metriken auf die Test-Daten der  Modelle bestimmen####################################
            (test_mse_norm, test_rmse_norm, test_mean_sam_norm, test_std_sam_norm,
             test_mse_global, test_rmse_global, test_rmse_std_global, test_r2_global,
             test_mean_sam_global, test_std_sam_global) = evaluate_model_max_with_sam(
                model,
                X_test_np,
                y_test,
                name="Testdaten"
            )

            metrics.append({
                "Modell": f"AutoKeras_{i + 1}",

                # --- Normierte Metriken
                "Train_RMSE_norm": train_rmse_norm,
                "Train_Mean_SAM_norm": train_mean_sam_norm,
                "Train_Std_SAM_norm": train_std_sam_norm,

                "Test_RMSE_norm": test_rmse_norm,
                "Test_Mean_SAM_norm": test_mean_sam_norm,
                "Test_Std_SAM_norm": test_std_sam_norm,

                # --- Rückskalierte Metriken
                "Train_RMSE_global": train_rmse_global,
                "Train_Std_RMSE_global": train_rmse_std_global,
                "Train_R2_global": train_r2_global,
                "Train_Mean_SAM_global": train_mean_sam_global,
                "Train_Std_SAM_global": train_std_sam_global,

                "Test_RMSE_global": test_rmse_global,
                "Test_Std_RMSE_global": test_rmse_std_global,
                "Test_R2_global": test_r2_global,
                "Test_Mean_SAM_global": test_mean_sam_global,
                "Test_Std_SAM_global": test_std_sam_global
            })
        except Exception as e:
            print(f"❌ Fehler bei Modell {i + 1}: {e}")
            history = None
            continue

    # Nur Modelle mit Test-R² > 0 behalten
    filtered = [
        (model, metric, history, duration, optimizer_class, optimizer_config)
        for model, metric, history, duration, optimizer_class, optimizer_config in zip(
            models, metrics, histories, durations, optimizers_classes, optimizers_configs
        )
        if metric["Test_R2_global"] > 0
    ]

    print(f"Gefundene Modelle mit Test-R² > 0: {len(filtered)}")
    if not filtered:
        print("❌ Keine gültigen Modelle mit Test-R² > 0 gefunden.")
        return [], [], [], [], [], []

    # Sortieren nach Test-MEAN-SAM (aufsteigend = besser)
    sorted_filtered = sorted(filtered, key=lambda x: x[1]["Test_Mean_SAM_global"])
    print("Sortierte Modelle (aufsteigend nach Test_Mean_SAM_global):")
    for i, (model, metric, history, duration, opt_class, opt_config) in enumerate(sorted_filtered):
        print(f"{i + 1:2d}. Modell: {metric.get('Modell', '<Kein Name>')}, "
              f"Test_Mean_SAM_global: {metric['Test_Mean_SAM_global']:.3f}, "
              f"Test_R2_global: {metric['Test_R2_global']:.3f}")

    # Top 3 auswählen (aufsteigend = beste SAM-Werte)
    top = sorted_filtered[:3]
    print("\nTop 3 Modelle nach Test_Mean_SAM_global:")
    for i, (model, metric, history, duration, opt_class, opt_config) in enumerate(top):
        print(f"Top {i + 1}: {metric.get('Modell', '<Kein Name>')}, "
              f"Test_Mean_SAM_global: {metric['Test_Mean_SAM_global']:.3f}, "
              f"Test_R2_global: {metric['Test_R2_global']:.3f}")

    # bestes Modell nach höchstem SAM auf Testdaten (eigentlich niedrigster SAM besser? Überprüfe ggf. hier!)
    best_idx = min(range(len(metrics)), key=lambda i: metrics[i]["Test_Mean_SAM_global"])
    best_model = models[best_idx]
    print(f"\nBestes Einzelmodell (niedrigster Test_Mean_SAM_global): Modell {best_idx}, "
          f"Test_Mean_SAM_global: {metrics[best_idx]['Test_Mean_SAM_global']:.3f}")

    #  Abspeichern als "best_model"
    best_model_path = os.path.join(model_dir, "best_model")
    best_model.save(best_model_path, save_format="tf")
    print(f"✅ Bestes Modell gespeichert unter: {best_model_path}")

    # Entpacken
    models, metrics, histories, durations, optimizers_classes, optimizers_configs = zip(*top)

    for i, opt in enumerate(optimizers_classes):
        print(f"optimizers_classes[{i}] = {opt}, type: {type(opt)}")

    # In Listen umwandeln
    return list(models), list(metrics), list(histories), list(durations), list(optimizers_classes), list(
        optimizers_configs)


###################################Training der besten 3 Autokeras Modelle mit CV####################################

def cross_validate_models(base_models, X, y, X_test_np, y_test, optimizers_classes=None, optimizers_configs=None):
    # Initialisierung der Kreuzvalidierung mit 5 Folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ergebnislisten zur Sammlung
    cv_models, cv_metrics, cv_histories = [], [], []
    sam_mean_cv_list, sam_std_cv_list = [], []  # ggf. SAM-spezifische Metriken

    # Fortschrittsanzeige für alle Kombinationen (Modelle × Folds)
    total = len(base_models) * kf.get_n_splits()
    progress = tqdm(total=total, desc="Cross-Validation")

    # Schleife über alle Modellarchitekturen
    for idx, base_model in enumerate(base_models):

        # Schleife über alle Folds der Kreuzvalidierung
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            progress.set_description(f"Architektur {idx + 1} – Fold {fold + 1}")

            # Aufteilung der Daten in Trainings- und Validierungsanteile für diesen Fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Neuladen der Kerasmodelle +Temporäres abspeichern
            temp_dir = tempfile.mkdtemp()
            base_model.save(temp_dir, save_format="tf")  # als vollständiges SavedModel speichern
            model = tf.keras.models.load_model(temp_dir)  # reload für neuen Trainingslauf

            # Optimierer für diesen Fold rekonstruieren

            optimizer_class = optimizers_classes[idx]
            optimizer_config = optimizers_configs[idx]
            optimizer = optimizer_class.from_config(optimizer_config)
            print(f"Verwende Optimizer {optimizer_class} für Modell {idx}")

            # Kompilierung des Modells mit MSE-Verlustfunktion
            model.compile(optimizer=optimizer, loss="mse")

            # Training des Modells mit Early Stopping
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                                verbose=0)  # Training im Hintergrund (kein Terminal-Output)

            # Modell speichern + Metriken sammeln
            cv_models.append(model)
            cv_histories.append(history.history)

            # --- Train
            (train_mse_norm, train_rmse_norm, train_mean_sam_norm, train_std_sam_norm,
             train_mse_global, train_rmse_global, train_rmse_std_global, train_r2_global,
             train_mean_sam_global, train_std_sam_global) = evaluate_model_max_with_sam(
                model, X_train, y_train, name="Train"
            )

            # --- Validation
            (val_mse_norm, val_rmse_norm, val_mean_sam_norm, val_std_sam_norm,
             val_mse_global, val_rmse_global, val_rmse_std_global, val_r2_global,
             val_mean_sam_global, val_std_sam_global) = evaluate_model_max_with_sam(
                model, X_val, y_val, name="Val"
            )

            # --- Test
            (test_mse_norm, test_rmse_norm, test_mean_sam_norm, test_std_sam_norm,
             test_mse_global, test_rmse_global, test_rmse_std_global, test_r2_global,
             test_mean_sam_global, test_std_sam_global) = evaluate_model_max_with_sam(
                model, X_test_np, y_test, name="Test"
            )

            cv_metrics.append({
                "Modell": f"Arch{idx + 1}_Fold{fold + 1}",

                # --- Train
                "Train_MSE_norm": train_mse_norm,
                "Train_RMSE_norm": train_rmse_norm,
                "Train_Mean_SAM_norm": train_mean_sam_norm,
                "Train_Std_SAM_norm": train_std_sam_norm,

                "Train_MSE_global": train_mse_global,
                "Train_RMSE_global": train_rmse_global,
                "Train_Std_RMSE_global": train_rmse_std_global,
                "Train_R2_global": train_r2_global,
                "Train_Mean_SAM_global": train_mean_sam_global,
                "Train_Std_SAM_global": train_std_sam_global,

                # --- Validation
                "Val_MSE_norm": val_mse_norm,
                "Val_RMSE_norm": val_rmse_norm,
                "Val_Mean_SAM_norm": val_mean_sam_norm,
                "Val_Std_SAM_norm": val_std_sam_norm,

                "Val_MSE_global": val_mse_global,
                "Val_RMSE_global": val_rmse_global,
                "Val_Std_RMSE_global": val_rmse_std_global,
                "Val_R2_global": val_r2_global,
                "Val_Mean_SAM_global": val_mean_sam_global,
                "Val_Std_SAM_global": val_std_sam_global,

                # --- Test
                "Test_MSE_norm": test_mse_norm,
                "Test_RMSE_norm": test_rmse_norm,
                "Test_Mean_SAM_norm": test_mean_sam_norm,
                "Test_Std_SAM_norm": test_std_sam_norm,

                "Test_MSE_global": test_mse_global,
                "Test_RMSE_global": test_rmse_global,
                "Test_Std_RMSE_global": test_rmse_std_global,
                "Test_R2_global": test_r2_global,
                "Test_Mean_SAM_global": test_mean_sam_global,
                "Test_Std_SAM_global": test_std_sam_global,
            })

            cv_df = pd.DataFrame(cv_metrics)
            n_arch = len(base_models)
            for i in range(n_arch):
                arch_df = cv_df[cv_df["Modell"].str.startswith(f"Arch{i + 1}_")]
                # hier wird Pro Modell 1 Wer abgespeichert über die Folds
                mean_sam_vals = arch_df["Val_Mean_SAM_global"].values
                sam_mean_cv_list.append(np.mean(mean_sam_vals))
                sam_std_cv_list.append(np.std(mean_sam_vals))

            # Verlustkurve plotten
            h = history.history
            plt.figure(figsize=(8, 4))
            plt.plot(h["loss"], label="Train Loss")
            if "val_loss" in h:
                plt.plot(h["val_loss"], label="Val Loss")
            plt.title(f"Architektur {idx + 1} – Fold {fold + 1}")
            plt.xlabel("Epoche")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cv_plot_dir, f"Arch{idx + 1}_Fold{fold + 1}_loss.png"))
            plt.close()

            # Temp-Verzeichnis aufräumen
            shutil.rmtree(temp_dir, ignore_errors=True)
            progress.update(1)

    progress.close()
    return cv_models, cv_metrics, cv_histories, sam_mean_cv_list, sam_std_cv_list


def mc_model_from_model(model, dropout_rate=0.5):
    x = model.input
    dropout_count = 0

    dense_count = 0
    for i, layer in enumerate(model.layers[1:-1]):  # Skip Input and Output layer
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Dense Layer gefunden bei Index {i}: Name={layer.name}, Units={layer.units}")
            dense_count += 1
            x = tf.keras.layers.Dropout(
                dropout_rate, name=f"mc_dropout_{dropout_count}"
            )(x, training=True)
            dropout_count += 1

    print(f"Anzahl der Dense-Layer im Modell (ohne Input/Output): {dense_count}")

    output = model.layers[-1](x)
    return tf.keras.Model(inputs=model.input, outputs=output)


import re


def aggregate_metric_per_architecture(cv_metrics, metric_key):
    """
    Aggregiert eine Metrik (z.B. 'Test_RMSE_global') über alle Folds pro Architektur.
    cv_metrics: Liste von Dicts mit Keys wie 'Modell': 'Arch1_Fold1' und z.B. 'Test_RMSE_global'
    metric_key: Name des Schlüssels in cv_metrics für die gewünschte Metrik
    Rückgabe:
        arch_names:    Liste aller Architekturen, z.B. ['Arch1', 'Arch2', ...]
        metric_means:  Liste der Mittelwerte pro Architektur (in Reihenfolge arch_names)
        metric_stds:   Liste der Standardabweichungen pro Architektur (in Reihenfolge arch_names)
    """
    arch_dict = {}
    for m in cv_metrics:
        # Extrahiere den Architekturnamen (z.B. "Arch1" aus "Arch1_Fold2")
        match = re.match(r"(Arch\d+)_Fold\d+", m["Modell"])
        if match:
            arch = match.group(1)
            if arch not in arch_dict:
                arch_dict[arch] = []
            arch_dict[arch].append(m[metric_key])
    arch_names = []
    metric_means = []
    metric_stds = []
    for arch, values in sorted(arch_dict.items()):
        arch_names.append(arch)
        metric_means.append(np.mean(values))
        metric_stds.append(np.std(values))
    return arch_names, metric_means, metric_stds


def predict_with_mc_dropout(
        models,
        X,
        y_true=None,
        repeats=100,
        plot_dir=None,
        label="Test",
        model_names=None,
        doc_dir=None
    ):
    """
    Führt MC Dropout für eine Liste von Modellen aus, erstellt gemeinsame Plots der Hauptmetriken.
    Gibt alle MC Dropout-Metriken pro Modell als Listen zurück.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    n_models = len(models)
    if model_names is None:
        model_names = [f"Modell {i+1}" for i in range(n_models)]
    x = np.arange(n_models)
    bar_width = 0.6 if n_models <= 5 else 0.35

    # Ergebnisse sammeln
    all_means, all_stds = [], []
    mc_rmse_list, mc_rmse_std_list = [], []
    mc_r2_list = []
    mc_sam_mean_list, mc_sam_std_list = [], []

    for i, model in enumerate(models):
        preds = []
        for j in range(repeats):
            pred = model.predict(X, verbose=0)
            preds.append(pred)
        preds = np.array(preds)  # [repeats, samples, spektral_pixel]
        mean_pred = preds.mean(axis=0)
        std_pred  = preds.std(axis=0)
        all_means.append(mean_pred)
        all_stds.append(std_pred)

        if y_true is not None:
            class DummyModel:
                def predict(self, _X, verbose=0):
                    return mean_pred
            dummy_model = DummyModel()


            (mse_norm, rmse_norm, mean_sam_norm, std_sam_norm,
             mse_global, rmse_global, rmse_std_global, r2_global,
             mean_sam_global, std_sam_global) = evaluate_model_max_with_sam(
                dummy_model,
                X=None,
                y_normiert=y_true,
                name=f"{label} (MC Dropout)"
            )
            mc_rmse_list.append(rmse_global)
            mc_rmse_std_list.append(rmse_std_global)
            mc_r2_list.append(r2_global)
            mc_sam_mean_list.append(mean_sam_global)
            mc_sam_std_list.append(std_sam_global)
        else:
            mc_rmse_list.append(np.nan)
            mc_rmse_std_list.append(0)
            mc_r2_list.append(np.nan)
            mc_sam_mean_list.append(np.nan)
            mc_sam_std_list.append(0)

        # Optional: Excel-Speicherung pro Modell
        if doc_dir:
            df_mc_single = pd.DataFrame({
                "MC_RMSE_Global": [rmse_global],
                "MC_RMSE_Std": [rmse_std_global],
                "MC_R2": [r2_global],
                "MC_SAM": [mean_sam_global],
                "MC_SAM_Std": [std_sam_global],
            })
            df_mc_single.to_excel(os.path.join(doc_dir, f"mc_dropout_metrics_{label.lower()}_{model_names[i]}.xlsx"))

    # === Gemeinsamer Balkenplot für alle Modelle ===
    fig, axes = plt.subplots(1, 3, figsize=(max(12, n_models * 1.4), 5))

    # RMSE
    axes[0].bar(x, mc_rmse_list, yerr=mc_rmse_std_list, capsize=5, color=ikv_common_colors["g"], width=bar_width)
    axes[0].set_title("RMSE (MC Dropout)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE [counts]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].grid(True, axis='y')

    # R²
    axes[1].bar(x, mc_r2_list, color=ikv_common_colors["g"], width=bar_width)
    axes[1].set_title("R² (MC Dropout)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].grid(True, axis='y')

    # SAM
    axes[2].bar(x, mc_sam_mean_list, yerr=mc_sam_std_list, capsize=5, color=ikv_common_colors["g"], width=bar_width)
    axes[2].set_title("SAM (MC Dropout)")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("SAM-Winkel [°]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=45)
    axes[2].grid(True, axis='y')

    fig.suptitle("MC Dropout Metriken – Top 3 Modelle")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"mc_dropout_metrics_{label.lower()}.png"))
    plt.close()

    # Optionale Gesamtausgabe als Excel (alle Modelle nebeneinander)
    if doc_dir:
        df_all = pd.DataFrame({
            "MC_RMSE_Global": mc_rmse_list,
            "MC_RMSE_Std": mc_rmse_std_list,
            "MC_R2": mc_r2_list,
            "MC_SAM": mc_sam_mean_list,
            "MC_SAM_Std": mc_sam_std_list,
        }, index=model_names)
        df_all.to_excel(os.path.join(doc_dir, f"mc_dropout_metrics_{label.lower()}_all.xlsx"))

    # Rückgabe: alle wichtigen Metriken je Modell
    return (all_means, all_stds, mc_rmse_list, mc_rmse_std_list, mc_r2_list, mc_sam_mean_list, mc_sam_std_list)

##########################Kalibrierungsplots für MC Dropouts, um die Unsicherheiten freizugeben
from scipy.stats import spearmanr


def plot_uncertainty_vs_error_both_sets(
        y_true_train, mean_pred_train, unsicher_train,
        y_true_test, mean_pred_test, unsicher_test,
        plot_dir, ikv_common_colors, prefix="Train_Test"
):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, linregress

    # Fehler pro Spektrum
    rmse_train = [np.sqrt(np.mean((mean_pred_train[i] - y_true_train.iloc[i].values) ** 2))
                  for i in range(len(unsicher_train))]
    rmse_test = [np.sqrt(np.mean((mean_pred_test[i] - y_true_test.iloc[i].values) ** 2))
                 for i in range(len(unsicher_test))]

    # SAM pro Spektrum
    def compute_sam_angle(true, pred):
        num = np.dot(true, pred)
        denom = np.linalg.norm(true) * np.linalg.norm(pred) + 1e-12
        angle = np.arccos(np.clip(num / denom, -1.0, 1.0))
        return np.degrees(angle)

    sam_train = [compute_sam_angle(y_true_train.iloc[i].values, mean_pred_train[i]) for i in range(len(unsicher_train))]
    sam_test = [compute_sam_angle(y_true_test.iloc[i].values, mean_pred_test[i]) for i in range(len(unsicher_test))]

    # --- Plot: Unsicherheit vs. RMSE mit Regressionslinien und Spearman-Koeffizienten ---
    plt.figure(figsize=(6, 5))
    plt.scatter(unsicher_train, rmse_train, alpha=0.5, s=18, label="Train", color=ikv_common_colors["g"])
    plt.scatter(unsicher_test, rmse_test, alpha=0.5, s=18, label="Test", color=ikv_common_colors["d"])

    # Lineare Regression für Train
    slope_train, intercept_train, _, _, _ = linregress(unsicher_train, rmse_train)
    x_train_range = np.linspace(np.min(unsicher_train), np.max(unsicher_train), 200)
    plt.plot(x_train_range, slope_train * x_train_range + intercept_train, color=ikv_common_colors["g"], linestyle='--',
             label="Regression Train")

    # Lineare Regression für Test
    slope_test, intercept_test, _, _, _ = linregress(unsicher_test, rmse_test)
    x_test_range = np.linspace(np.min(unsicher_test), np.max(unsicher_test), 200)
    plt.plot(x_test_range, slope_test * x_test_range + intercept_test, color=ikv_common_colors["d"], linestyle='--',
             label="Regression Test")

    # Spearman-Koeffizienten berechnen
    rho_train, pval_train = spearmanr(unsicher_train, rmse_train)
    rho_test, pval_test = spearmanr(unsicher_test, rmse_test)
    txt = (
        f"Train: ρ = {rho_train:.2f} (p={pval_train:.1g})\n"
        f"Test:  ρ = {rho_test:.2f} (p={pval_test:.1g})"
    )
    plt.gca().text(
        0.97, 0.03, txt,
        transform=plt.gca().transAxes,
        ha='right', va='bottom',
        fontsize=11, color=ikv_common_colors["g"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )

    plt.xlabel("MC Dropout Unsicherheit (STD pro Spektrum)")
    plt.ylabel("RMSE pro Spektrum")
    plt.title("Unsicherheit vs. RMSE (Train/Test)")
    plt.legend()
    plt.tight_layout()
    path_rmse = os.path.join(plot_dir, f"unsicherheit_vs_rmse_{prefix.lower()}.png")
    plt.savefig(path_rmse)
    plt.close()
    print(f"📊 Plot gespeichert: {path_rmse}")

    # --- Plot: Unsicherheit vs. SAM mit Regressionslinien und Spearman-Koeffizienten ---
    plt.figure(figsize=(6, 5))
    plt.scatter(unsicher_train, sam_train, alpha=0.5, s=18, label="Train", color=ikv_common_colors["g"])
    plt.scatter(unsicher_test, sam_test, alpha=0.5, s=18, label="Test", color=ikv_common_colors["d"])

    # Regression Train
    slope_train, intercept_train, _, _, _ = linregress(unsicher_train, sam_train)
    x_train_range = np.linspace(np.min(unsicher_train), np.max(unsicher_train), 200)
    plt.plot(x_train_range, slope_train * x_train_range + intercept_train, color=ikv_common_colors["g"], linestyle='--',
             label="Regression Train")

    # Regression Test
    slope_test, intercept_test, _, _, _ = linregress(unsicher_test, sam_test)
    x_test_range = np.linspace(np.min(unsicher_test), np.max(unsicher_test), 200)
    plt.plot(x_test_range, slope_test * x_test_range + intercept_test, color=ikv_common_colors["d"], linestyle='--',
             label="Regression Test")

    plt.xlabel("MC Dropout Unsicherheit (STD pro Spektrum)")
    plt.ylabel("SAM-Winkel pro Spektrum [°]")
    plt.title("Unsicherheit vs. SAM (Train/Test)")
    plt.legend()
    plt.tight_layout()
    path_sam = os.path.join(plot_dir, f"unsicherheit_vs_sam_{prefix.lower()}.png")
    plt.savefig(path_sam)
    plt.close()
    print(f"📊 Plot gespeichert: {path_sam}")

    # Rückgabe jetzt inklusive der Spearman-Koeffizienten!
    return rmse_train, rmse_test, sam_train, sam_test, rho_train, rho_test



#####################Kalibrierungsmetrik Spearmen###################################################

def compute_spearman_uncertainty_vs_error(unsicher, rmse, set_name="Test", plot_dir=".",
                                          filename="unsicherheit_vs_rmse"):
    # Berechnet den Spearman-Rangkorrelationskoeffizienten (ρ) zwischen MC Dropout-Unsicherheiten und den pro-Spektrum-RMSE-Fehlern. Gibt ρ und den p-Wert aus.

    # Args:
    # unsicher : 1D-Array/List mit Unsicherheitswerten (z. B. mittlere STD je Spektrum)
    # rmse     : 1D-Array/List mit RMSE pro Spektrum (gleiche Reihenfolge!)
    # set_name : Label für Print-Ausgabe (z. B. "Test" oder "Train")
    ##Returns:
    # rho_spear: Spearman-Rangkorrelationskoeffizient
    # pval     : p-Wert zur Signifikanz

    rho_spear, pval = spearmanr(unsicher, rmse)
    print(f"{set_name}: Spearman ρ = {rho_spear:.3f}, p = {pval:.2g}")

    # Scatterplot
    plt.figure(figsize=(6, 5))
    plt.scatter(unsicher, rmse, alpha=0.5, s=18, color=ikv_common_colors["b"])
    plt.xlabel("MC Dropout Unsicherheit pro Spektrum)")
    plt.ylabel("RMSE pro Spektrum")
    plt.title(f"{set_name}: Unsicherheit vs. RMSE")
    # Textbox mit Statistik unten rechts
    txt = f"Spearman ρ = {rho_spear:.2f}\np = {pval:.1g}"
    plt.gca().text(
        0.97, 0.03, txt,
        transform=plt.gca().transAxes,
        ha='right', va='bottom',
        fontsize=11, color=ikv_common_colors["g"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    plt.tight_layout()
    path = os.path.join(plot_dir, f"{filename}_{set_name.lower()}.png")
    plt.savefig(path)
    plt.close()
    print(f"📊 Plot gespeichert: {path}")
    return rho_spear, pval


def visualize_metrics(metrics, title, filename_prefix, plot_dir, doc_dir):

    df = pd.DataFrame(metrics)
    model_names = df["Modell"].tolist()

    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns)
    print("Model names:", model_names)

    short_names = [f"Modell {i + 1}" for i in range(len(model_names))]
    n = len(model_names)
    bar_width = 0.35
    x = np.arange(n)

    # Spaltennamen
    cols = {
        "R2":      ("Train_R2_global",      "Test_R2_global"),
        "RMSE":    ("Train_RMSE_global",    "Test_RMSE_global"),
        "RMSEstd": ("Train_Std_RMSE_global","Test_Std_RMSE_global"),
        "SAM":     ("Train_Mean_SAM_global","Test_Mean_SAM_global"),
        "SAMstd":  ("Train_Std_SAM_global", "Test_Std_SAM_global"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(max(12, n * 1.4), 5))

    # 1. RMSE
    axes[0].bar(x - bar_width/2, df[cols["RMSE"][0]], yerr=df[cols["RMSEstd"][0]], capsize=5,
                width=bar_width, color=ikv_common_colors["g"], label="Train")
    axes[0].bar(x + bar_width/2, df[cols["RMSE"][1]], yerr=df[cols["RMSEstd"][1]], capsize=5,
                width=bar_width, color=ikv_common_colors["d"], label="Test")
    axes[0].set_title("RMSE")
    axes[0].set_ylabel("RMSE [counts]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=45)
    axes[0].grid(True, axis='y')

    # 2. R²
    axes[1].bar(x - bar_width/2, df[cols["R2"][0]], width=bar_width, color=ikv_common_colors["g"], label="Train")
    axes[1].bar(x + bar_width/2, df[cols["R2"][1]], width=bar_width, color=ikv_common_colors["d"], label="Test")
    axes[1].set_title("R²")
    axes[1].set_ylabel("R²")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=45)
    axes[1].grid(True, axis='y')

    # 3. SAM
    axes[2].bar(x - bar_width/2, df[cols["SAM"][0]], yerr=df[cols["SAMstd"][0]], capsize=5,
                width=bar_width, color=ikv_common_colors["g"], label="Train")
    axes[2].bar(x + bar_width/2, df[cols["SAM"][1]], yerr=df[cols["SAMstd"][1]], capsize=5,
                width=bar_width, color=ikv_common_colors["d"], label="Test")
    axes[2].set_title("SAM")
    axes[2].set_ylabel("SAM-Winkel [°]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=45)
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    axes[2].grid(True, axis='y')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"{filename_prefix}_rmse_r2_sam.png")
    plt.savefig(save_path)
    plt.close()

    # Optional: Metriken als Excel speichern
    metrics_path = os.path.join(doc_dir, f"{filename_prefix}_metriken.xlsx")
    df.to_excel(metrics_path, index=False)
    print(f"📄 Visualisierte Metriken gespeichert: {metrics_path}")



from sklearn.inspection import permutation_importance


def compute_and_plot_permutation_importance(
        models, X, y, feature_names=None, plot_dir=".", prefix="AutoKeras", n_repeats=20,
        scoring='neg_root_mean_squared_error'
):
    """
    Plotte Feature Importance für beliebig viele Modelle + Ensemble.
    """
    os.makedirs(plot_dir, exist_ok=True)
    # Liste draus machen, falls Einzelmodell
    if not isinstance(models, list):
        models = [models]

    # Feature-Namen übernehmen
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"F{i}" for i in range(X.shape[1])]

    importances_list = []
    for i, model in enumerate(models):
        result = permutation_importance(
            estimator=model,
            X=X,
            y=y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring
        )
        importances = result.importances_mean
        stds = result.importances_std
        # Normieren (optional, für besser vergleichbare Balkenlängen)
        if importances.sum() != 0:
            importances = importances / importances.sum()
        importances_list.append(importances)

        # Einzelplot
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(np.array(feature_names)[sorted_idx], importances[sorted_idx], yerr=stds[sorted_idx])
        plt.xticks(rotation=45)
        plt.ylabel("Permutation Importance (Summe = 1, ±Std)")
        plt.title(f"Feature Importance ({prefix} {i + 1})")
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"feature_importance_{prefix.lower()}_{i + 1}.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Einzelplot gespeichert: {filename}")

    # ENSEMBLE-PLOT
    if len(importances_list) > 1:
        importances_array = np.stack(importances_list)
        mean_importance = importances_array.mean(axis=0)
        std_importance = importances_array.std(axis=0)
        # Wieder normieren (optional)
        if mean_importance.sum() != 0:
            mean_importance = mean_importance / mean_importance.sum()
        sorted_idx = np.argsort(mean_importance)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(np.array(feature_names)[sorted_idx], mean_importance[sorted_idx], yerr=std_importance[sorted_idx])
        plt.xticks(rotation=45)
        plt.ylabel("Mean Permutation Importance (Summe = 1, ±Std)")
        plt.title(f"Feature Importance ({prefix} Ensemble)")
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"feature_importance_{prefix.lower()}_ensemble.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Ensemble-Plot gespeichert: {filename}")

        # 🔍 Importance als Excel speichern
        imp_df = pd.DataFrame(importances_list, columns=feature_names)
        imp_df.to_excel(os.path.join(doc_dir, f"feature_importance_{prefix.lower()}.xlsx"), index=False)
        print(f"📄 Feature Importances gespeichert: {prefix}")

    return importances_list


def compute_and_plot_permutation_importance_cv(
        cv_models, X, y, feature_names=None, plot_dir=".", prefix="CV", n_repeats=20, scoring='neg_mean_squared_error'
):
    """
    Plotte Feature Importance für alle CV-Modelle und Ensemble.
    """
    os.makedirs(plot_dir, exist_ok=True)
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"F{i}" for i in range(X.shape[1])]

    importances_list = []
    for i, model in enumerate(cv_models):
        result = permutation_importance(
            estimator=model,
            X=X,
            y=y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring
        )
        importances = result.importances_mean
        stds = result.importances_std
        if importances.sum() != 0:
            importances = importances / importances.sum()
        importances_list.append(importances)

        # Einzelplot
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(np.array(feature_names)[sorted_idx], importances[sorted_idx], yerr=stds[sorted_idx])
        plt.xticks(rotation=45)
        plt.ylabel("Permutation Importance (Summe = 1, ±Std)")
        plt.title(f"Feature Importance ({prefix} {i + 1})")
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"feature_importance_{prefix.lower()}_{i + 1}.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Einzelplot gespeichert: {filename}")

    # Ensemble-Plot
    if len(importances_list) > 1:
        importances_array = np.stack(importances_list)
        mean_importance = importances_array.mean(axis=0)
        std_importance = importances_array.std(axis=0)
        if mean_importance.sum() != 0:
            mean_importance = mean_importance / mean_importance.sum()

        sorted_idx = np.argsort(mean_importance)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(np.array(feature_names)[sorted_idx], mean_importance[sorted_idx], yerr=std_importance[sorted_idx])
        plt.xticks(rotation=45)
        plt.ylabel("Mean Permutation Importance (Summe = 1, ±Std)")
        plt.title(f"Feature Importance ({prefix} Ensemble)")
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"feature_importance_{prefix.lower()}_ensemble.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Ensemble-Plot gespeichert: {filename}")

        # 🔍 Importance als Excel speichern
        imp_df = pd.DataFrame(importances_list, columns=feature_names)
        imp_df.to_excel(os.path.join(doc_dir, f"feature_importance_{prefix.lower()}.xlsx"), index=False)
        print(f"📄 Feature Importances gespeichert: {prefix}")

    return importances_list


def document_models(models, histories, durations, doc_dir):
    infos = []

    # Nimm nur so viele, wie in ALLEN drei Listen vorhanden sind
    n = min(len(models), len(histories), len(durations))

    for i in range(n):
        model = models[i]
        history = histories[i]
        training_time = durations[i]
        name = f"AutoKeras_{i + 1}"

        arch = detect_architecture_style(model)
        layer_classes = [layer.__class__.__name__ for layer in model.layers]
        num_layers = len(layer_classes)
        num_params = model.count_params()
        num_trainable = np.sum([np.prod(w.shape) for w in model.trainable_weights])
        dense_info = []
        for j, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                dense_info.append(f"{j}: {layer.name} – {layer.units} Neuronen")

        try:
            optimizer_class = model.optimizer.__class__.__name__
            optimizer_config = model.optimizer.get_config()
        except Exception:
            optimizer_class = "Unbekannt"
            optimizer_config = {}

        try:
            loss_name = model.loss if hasattr(model, "loss") else "Unbekannt"
        except:
            loss_name = "Fehler beim Abruf"

        try:
            learning_rate = float(tf.keras.backend.eval(model.optimizer.learning_rate))
        except:
            learning_rate = "Nicht abrufbar"

        # 🔍 Verlustkurve plotten
        if "loss" in history:
            plt.figure(figsize=(8, 4))
            plt.plot(history["loss"], label="Train Loss")
            if "val_loss" in history:
                plt.plot(history["val_loss"], label="Val Loss")
            plt.title(f"Loss-Verlauf: {name}")
            plt.xlabel("Epoche")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{doc_dir}/AutoKeras_{i + 1}_lossverlauf.png")
            plt.close()

        infos.append({
            "Modell": name,
            "Architekturstil": arch,
            "Anzahl_Layer": num_layers,
            "Layer_Typen": ", ".join(layer_classes),
            "Dense_Layer_Info": "; ".join(dense_info),
            "Parameter_gesamt": num_params,
            "Trainierbare_Parameter": num_trainable,
            "Loss-Funktion": loss_name,
            "Optimizer": optimizer_class,
            "Optimizer_Config": str(optimizer_config),
            "Lernrate": learning_rate,
            "Trainingsdauer [s]": f"{training_time:.1f}"
        })

    df_info = pd.DataFrame(infos)
    excel_path = f"{doc_dir}/AutoKeras_Modellübersicht.xlsx"
    df_info.to_excel(excel_path, index=False)
    print("📘 Modellübersicht inkl. Trainingszeit & Losskurve gespeichert.")


def save_metrics_to_excel(autokeras_metrics, cv_metrics, result_dir):
    df_ak = pd.DataFrame(autokeras_metrics)
    # DataFrame aus den Metriken (falls noch nicht geschehen)
    df_cv = pd.DataFrame(cv_metrics)

    # Nur numerische Spalten mitteln (z. B. RMSE, R², SAM, MSE usw.)
    mean_values = df_cv.select_dtypes(include='number').mean()

    # Eine neue Zeile daraus machen
    df_avg = pd.DataFrame([{
        **{"Modell": "Durchschnitt (CV)"},
        **mean_values.to_dict()
    }])

    # Einzeln speichern mit Input-Dateinamen im Verzeichnis
    df_ak.to_excel(os.path.join(result_dir, f"autokeras_{len(df_ak)}modelle.xlsx"), index=False)
    df_cv.to_excel(os.path.join(result_dir, f"cv_{len(df_cv)}modelle.xlsx"), index=False)
    df_avg.to_excel(os.path.join(result_dir, "durchschnitt_cv.xlsx"), index=False)
    # Gesamtdatei mit allen zusammengefasst
    combined_path = os.path.join(result_dir, "alle_modelle_metriken.xlsx")
    pd.concat([df_ak, df_cv, df_avg], ignore_index=True).to_excel(combined_path, index=False)
    print("📄 Metriken als xlsx gespeichert.")


# === Komplettstruktur zur Unsicherheitsbasierten Analyse und Parametergenerierung ===

def export_plot_rohdaten_to_excel(
        y_trainval, mean_pred_train, unsicher_train, train_thresh,
        y_test, mean_pred_test, unsicher_test, test_thresh,
        df_trainval, df_test, doc_dir
):
    sicher_train_idx = np.argmin(unsicher_train[unsicher_train <= train_thresh])
    sicher_test_idx = np.argmin(unsicher_test[unsicher_test <= test_thresh])

    unsichere_train_inds = np.where(unsicher_train > train_thresh)[0]
    unsichere_test_inds = np.where(unsicher_test > test_thresh)[0]

    max_unsicher_train_idx = unsichere_train_inds[np.argmax(unsicher_train[unsichere_train_inds])] if len(
        unsichere_train_inds) else None
    min_unsicher_train_idx = unsichere_train_inds[np.argmin(unsicher_train[unsichere_train_inds])] if len(
        unsichere_train_inds) else None

    max_unsicher_test_idx = unsichere_test_inds[np.argmax(unsicher_test[unsichere_test_inds])] if len(
        unsichere_test_inds) else None
    min_unsicher_test_idx = unsichere_test_inds[np.argmin(unsicher_test[unsichere_test_inds])] if len(
        unsichere_test_inds) else None

    def make_df(real, pred, idx, meta_df):
        df = pd.DataFrame({
            "Pixel": np.arange(len(real)),
            "Real": real,
            "Prediction": pred
        })
        meta = meta_df.loc[meta_df.index[idx]]
        for col in meta.index:
            df[col] = meta[col]
        return df

    data_frames = {}

    # Sicherster Punkt
    data_frames["sicherster_Train"] = make_df(
        y_trainval.iloc[sicher_train_idx].values,
        mean_pred_train[sicher_train_idx],
        sicher_train_idx,
        df_trainval
    )
    data_frames["sicherster_Test"] = make_df(
        y_test.iloc[sicher_test_idx].values,
        mean_pred_test[sicher_test_idx],
        sicher_test_idx,
        df_test
    )

    # Extrempunkte
    if min_unsicher_train_idx is not None:
        data_frames["Train_min_unsicher"] = make_df(
            y_trainval.iloc[min_unsicher_train_idx].values,
            mean_pred_train[min_unsicher_train_idx],
            min_unsicher_train_idx,
            df_trainval
        )
    if max_unsicher_train_idx is not None:
        data_frames["Train_max_unsicher"] = make_df(
            y_trainval.iloc[max_unsicher_train_idx].values,
            mean_pred_train[max_unsicher_train_idx],
            max_unsicher_train_idx,
            df_trainval
        )
    if min_unsicher_test_idx is not None:
        data_frames["Test_min_unsicher"] = make_df(
            y_test.iloc[min_unsicher_test_idx].values,
            mean_pred_test[min_unsicher_test_idx],
            min_unsicher_test_idx,
            df_test
        )
    if max_unsicher_test_idx is not None:
        data_frames["Test_max_unsicher"] = make_df(
            y_test.iloc[max_unsicher_test_idx].values,
            mean_pred_test[max_unsicher_test_idx],
            max_unsicher_test_idx,
            df_test
        )

    # Excel schreiben
    export_path = os.path.join(doc_dir, "plot_rohdaten.xlsx")
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        for sheet_name, df in data_frames.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"📁 Rohdaten der Plots exportiert: {export_path}")


def plot_most_certain_uncertain_spectra(
        y_true, mean_pred, std_pred, meta_df, plot_dir, label="Train"
):
    """
    Plottet das sicherste und unsicherste Spektrum nach MC Dropout.
    Titel enthält Unsicherheitswert (MC STD) und SAM.
    """
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    # Index mit minimaler/maximaler Unsicherheit finden
    sicher_idx = np.argmin(std_pred)
    unsicher_idx = np.argmax(std_pred)

    # Echte und vorhergesagte Spektren
    y_true_sicher = y_true.iloc[sicher_idx].values
    y_pred_sicher = mean_pred[sicher_idx]
    std_sicher = std_pred[sicher_idx]
    sam_sicher = compute_sam_angle(y_true_sicher, y_pred_sicher)

    y_true_unsicher = y_true.iloc[unsicher_idx].values
    y_pred_unsicher = mean_pred[unsicher_idx]
    std_unsicher = std_pred[unsicher_idx]
    sam_unsicher = compute_sam_angle(y_true_unsicher, y_pred_unsicher)

    # Plotten
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, y_r, y_p, std, sam, idx, title in [
        (axes[0], y_true_sicher, y_pred_sicher, std_sicher, sam_sicher, sicher_idx, "Sicherstes"),
        (axes[1], y_true_unsicher, y_pred_unsicher, std_unsicher, sam_unsicher, unsicher_idx, "Unsicherstes"),
    ]:
        ax.plot(wavelength_corrected, y_r, color=ikv_common_colors["g"], label="Reales Spektrum", linewidth=1)
        ax.plot(wavelength_corrected, y_p, color=ikv_common_colors["d"], label="Vorhersage", linestyle="--")
        meta = meta_df.loc[y_true.index[idx]]
        meta_str = f"ID: {meta['id']}, Exposure: {meta['exposure']} ms"
        ax.set_title(
            f"{title} Sample\n"
            f"Unsicherheit: {std:.4f} | SAM: {sam:.2f}°\n"
            f"{meta_str}"
        )
        ax.set_xlabel("Wellenlänge [nm]")
        ax.legend()
    axes[0].set_ylabel("Intensität [a.u.]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{label.lower()}_most_certain_uncertain.png"))
    plt.close()
    print(f"📊 Plot gespeichert: {plot_dir}/{label.lower()}_most_certain_uncertain.png")


def plot_extreme_sam_spectra_with_uncertainty(
        y_true, y_pred, std_pred, meta_df, plot_dir, label="BestModel"
):
    """
    Plottet Spektrum mit größtem und kleinstem SAM-Winkel, Titel enthält MC Dropout-Unsicherheit.
    """
    # SAM pro Sample berechnen
    sam_vals = [compute_sam_angle(y_true.iloc[i].values, y_pred[i]) for i in range(len(y_pred))]
    sam_vals = np.array(sam_vals)

    min_sam_idx = np.argmin(sam_vals)
    max_sam_idx = np.argmax(sam_vals)

    # Daten
    for idx, desc in zip([min_sam_idx, max_sam_idx], ["Kleinster SAM", "Größter SAM"]):
        real = y_true.iloc[idx].values
        pred = y_pred[idx]
        std = std_pred[idx]
        sam = sam_vals[idx]
        meta = meta_df.loc[y_true.index[idx]]
        meta_str = f"ID: {meta['id']}, Exposure: {meta['exposure']} ms"
        plt.figure(figsize=(8, 4))
        plt.plot(wavelength_corrected, real, color=ikv_common_colors["g"], label="Reales Spektrum", linewidth=1)
        plot_characteristic_lines_numbered()
        plt.plot(wavelength_corrected, pred, color=ikv_common_colors["d"], label="Vorhersage", linestyle="--")
        plt.title(f"{desc}: SAM={sam:.2f}°, Unsicherheit={std:.4f}\n{meta_str}")
        plt.ylim(bottom=0)
        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{label.lower()}_{desc.replace(' ', '_').lower()}.png"))
        plt.close()
        print(f"📊 Plot gespeichert: {plot_dir}/{label.lower()}_{desc.replace(' ', '_').lower()}.png")


#################Plotte die 10 spektren mit dem größten RMSE, Sam und unsicherheit###############################

def plot_top_n_spectra(
        y_true,
        y_pred,
        metric_arr,
        metric_name,
        meta_df=None,
        wavelength_axis=None,
        n=10,
        save_dir=".",
        rmse_arr=None,
        sam_arr=None,
        unsicher_arr=None,
):
    """
    Plottet die Top-n Spektren bzgl. einer Metrik (z.B. Unsicherheit, RMSE, SAM)
    und gibt Unsicherheit, RMSE und SAM im Plot-Titel aus (sofern die Arrays übergeben werden).
    """
    top_indices = np.argsort(metric_arr)[-n:][::-1]  # Top-n Indizes (absteigend)

    for rank, idx in enumerate(top_indices):
        plt.figure(figsize=(8, 4))
        x_axis = wavelength_axis if wavelength_axis is not None else np.arange(y_true.shape[1])
        plt.plot(x_axis, y_true.iloc[idx].values, label="Echtes Spektrum", color=ikv_common_colors["g"], linewidth=2)
        plt.plot(x_axis, y_pred[idx], label="Vorhersage", color=ikv_common_colors["d"], linestyle="--", linewidth=2)
        plt.xlabel("Wellenlänge [nm]" if wavelength_axis is not None else "Pixel")
        plt.ylabel("Intensität [counts]")

        # Titel zusammensetzen mit allen gewünschten Metriken
        titel_parts = [f"{metric_name} Rang {rank + 1}: Wert={metric_arr[idx]:.3f}"]
        if unsicher_arr is not None:
            titel_parts.append(f"Unsicherheit={unsicher_arr[idx]:.3f}")
        if rmse_arr is not None:
            titel_parts.append(f"RMSE={rmse_arr[idx]:.3f}")
        if sam_arr is not None:
            titel_parts.append(f"SAM={sam_arr[idx]:.3f}")

        title = " | ".join(titel_parts)

        if meta_df is not None:
            meta = meta_df.loc[y_true.index[idx]]
            meta_str = ", ".join([f"{col}: {meta[col]}" for col in meta.index if col in ["id", "exposure"]])
            title += f" | {meta_str}"

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        fname = f"{metric_name.replace(' ', '_').lower()}_top{rank + 1}_id{meta['id'] if meta_df is not None else idx}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=200)
        plt.close()
        print(f"✅ Gespeichert: {os.path.join(save_dir, fname)}")


################################Bayesian Optimization zur Maximierung der Unsicherheit#####################
# trainiert auf MC-Dropout-Daten aus Train + Test

def bayesian_sampling_with_uncertainty_all(X_trainval_np, X_test_np, unsicher_train,
                                           unsicher_test, parameter_names, used_combinations,
                                           n_points=3200, return_full_tracking=False):
    print("\n🚀 Starte Unsicherheits-basierte Bayesian Optimization (Train + Test)...")

    X_all = np.vstack([X_trainval_np, X_test_np])
    y_all = np.concatenate([unsicher_train, unsicher_test])

    print("🔧 Trainiere RF-Modell auf geschätzten Unsicherheiten (Train+Test)...")
    start = time.time()
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_model.fit(X_all, y_all)
    end = time.time()
    print(f"⏱️ RF-Training abgeschlossen in {end - start:.2f} Sekunden.")

    bounds = [
        Categorical([500, 600, 700, 800, 900, 1000], name="P_MW"),  # Leistung_W
        Categorical([3, 4, 5, 7, 8, 9], name="t_on"),  # ton_us
        Categorical([30, 42, 54, 66, 78, 90], name="t_off"),  # toff_us
        Categorical([5, 13, 21, 29, 37, 45], name="p"),  # Druck_mbar
        Categorical([20, 96, 172, 248, 324, 400], name="O2"),  # O2_sccm
        Categorical([2, 5, 9, 12, 16, 20], name="HMDSO")  # HMDSO_sccm
    ]

    from sklearn.preprocessing import MinMaxScaler
    from itertools import product
    from sklearn.metrics import pairwise_distances

    # Berechne Abstände zwischen den 64 Ecken des Parameterraums (MinMax-skalierter Raum)...
    print("\n📐 Berechne maximale Distanz zwischen den 64 Ecken des Parameterraums (MinMax-skalierter Raum)...")

    # Eckpunkte erzeugen (2^6 = 64 Kombinationen)
    bounds_numeric = [(min(b.categories), max(b.categories)) for b in bounds]
    ecken = np.array(list(product(*[(low, high) for low, high in bounds_numeric])))

    # 2. MinMaxScaler fitten (auf Bounds!)
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(ecken)

    # Eckpunkte transformieren (in Einheitswürfel)
    ecken_scaled = scaler_mm.transform(ecken)

    # Paarweise Distanzen berechnen
    ecken_dists = pairwise_distances(ecken_scaled)

    # Nur die maximale Distanz ausgeben (Raumdiagonale)
    max_distanz = np.max(ecken_dists[np.triu_indices_from(ecken_dists, k=1)])
    print(f"🔺 Maximale Distanz zwischen Eckpunkten (Raumdiagonale): {max_distanz:.4f}")

    X_all_df = pd.DataFrame(X_all, columns=parameter_names)
    X_known_scaled = scaler_mm.transform(X_all_df)

    # komplette Distanzmatrix berechnen
    dist_matrix = pairwise_distances(X_known_scaled)  # Form: (839, 839)

    # Indizes des Oberdreiecks (ohne Diagonale) entnehmen
    rows, cols = np.triu_indices_from(dist_matrix, k=1)

    # diese Indices auf dist_matrix anwenden
    upper_triangle = dist_matrix[rows, cols]

    # Histogramm der paarweisen Distanzen
    plt.figure(figsize=(8, 4))
    plt.hist(upper_triangle, bins=100, color=ikv_common_colors["g"], edgecolor=ikv_common_colors["g"])
    plt.xlabel("Paarweise Distanz (MinMaxScaler-Raum)")
    plt.ylabel("Anzahl Punktpaare")
    plt.title("Verteilung der Distanzen zwischen bekannten Kombinationen")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Verteilung der Distanzen zwischen bekannten Kombinationen.png"))
    plt.close()

    print("🔍 Quantile der paarweisen Distanzen:")
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        print(f"  {int(q * 100)}%: {np.quantile(upper_triangle, q):.4f}")

    def objective(x):
        x = np.array(x).reshape(1, -1)
        return -rf_model.predict(x)[0]

    total_candidates = int(n_points * 1)
    # Mehr Vorschläge, damit nach Filtering genug übrig bleiben

    from skopt.space import Real, Integer

    bounds = [
        Integer(500, 1000, name="P_MW"),  # Integer
        Real(3, 9, name="t_on"),  # Real (kommazahlen erlaubt)
        Integer(30, 90, name="t_off"),  # Integer
        Integer(5, 45, name="p"),  # Integer
        Integer(20, 400, name="O2"),  # Integer
        Real(2, 20, name="HMDSO")  # Real (kommazahlen erlaubt)
    ]

    print("Starte Bayesian Optimization")
    start_opt = time.time()
    res = forest_minimize(func=objective, dimensions=bounds, acq_func="EI",
                          n_calls=total_candidates, n_initial_points=20,
                          random_state=42, base_estimator="RF")
    end_opt = time.time()
    # base_estimator="RF" ist Eigenltich unnötig weil
    # der schätzer mit forest_minimzw schon ein RF ist

    print(f"⏱️ Bayesian Optimization dauerte {end_opt - start_opt:.2f} Sekunden.")

    # Erzeuge candidates DataFrame mit Ursprungsindex
    candidates = pd.DataFrame(res.x_iters, columns=parameter_names)
    candidates["predicted_uncertainty"] = [-v for v in res.func_vals]
    candidates["orig_idx"] = np.arange(len(candidates))

    # Sortiere nach Unsicherheit
    candidates_sorted = candidates.sort_values("predicted_uncertainty", ascending=False).reset_index(drop=True)

    # Debug: Prüfe Verschiebungen durch Sortierung
    verschoben = (candidates["orig_idx"].values != candidates_sorted["orig_idx"].values[:len(candidates)])
    print(f"🔍 Anzahl verschobener Kandidaten nach Sortierung: {verschoben.sum()} von {len(candidates)}")
    if verschoben.sum() > 0:
        print(pd.DataFrame({
            "vorher": candidates["orig_idx"].values[:10],
            "nachher": candidates_sorted["orig_idx"].values[:10]}))

    # Debug: Konsistenz zwischen candidates_sorted und res.x_iters
    for i in range(5):
        idx = int(candidates_sorted.iloc[i]['orig_idx'])
        print(f"Kandidat {i}:")
        print("  Aus candidates_sorted:", tuple(candidates_sorted.iloc[i][parameter_names].values))
        print("  Aus res.x_iters:", tuple(res.x_iters[idx]))
        print("---")

    # BEGINN: Analyse der minimalen Distanzen zwischen neuen und alten Punkten ---
    X_all_df = pd.DataFrame(X_all, columns=parameter_names)
    X_known_scaled = scaler_mm.transform(X_all_df)

    X_new_scaled = scaler_mm.transform(candidates_sorted[parameter_names])
    dists = pairwise_distances(X_new_scaled, X_known_scaled)
    min_dists = dists.min(axis=1)
    nearest_old_indices = dists.argmin(axis=1)
    candidates_sorted["min_dist"] = min_dists
    X_known_params = X_all_df.reset_index()  # Spalte 'index' + Parameter-Spalten

    trainval_indices = set(X_trainval.index)
    test_indices = set(X_test.index)

    analyse_rows = []
    errors = []
    for i, row in candidates_sorted.iterrows():
        new_params = tuple(row[parameter_names].values)
        nearest_idx = nearest_old_indices[i]
        old_row = X_known_params.iloc[nearest_idx]
        old_params = tuple(old_row[parameter_names].values)
        dist = min_dists[i]
        orig_idx = old_row['index']

        # Herkunft des alten Punktes bestimmen
        if orig_idx in trainval_indices:
            set_type = "trainval"
        elif orig_idx in test_indices:
            set_type = "test"
        else:
            set_type = "unknown"

        analyse_rows.append(
            dict(
                **{f"new_{p}": v for p, v in zip(parameter_names, new_params)},
                **{f"old_{p}": v for p, v in zip(parameter_names, old_params)},
                min_dist=dist,
                nearest_old_index=old_row['index'],
                predicted_uncertainty=row["predicted_uncertainty"],
                set_type=set_type,
                new_params_tuple=new_params,
                old_params_tuple=old_params

            )
        )
        if np.isclose(dist, 0, atol=1e-8) and new_params != old_params:
            print(f"❗ Nicht identische Parameter bei min_dist=0 (Index {i}):")
            print(f"   new_params: {new_params}")
            print(f"   old_params: {old_params}")
            errors.append((i, new_params, old_params))

    analyse_df = pd.DataFrame(analyse_rows)
    analyse_df.index.name = "candidate_id"
    analyse_path = os.path.join(plot_dir, "analyse_abstaende_neu_zu_alt.xlsx")
    analyse_df.to_excel(analyse_path, index=False)
    print(f"📊 Analyse-Tabelle (alle Distanzen & Parameter) exportiert: {analyse_path}")
    if errors:
        print(f"🚨 WARNUNG: {len(errors)} Fälle mit min_dist==0 aber unterschiedlichen Parametern gefunden!")
    else:
        print("✅ Bei min_dist==0 stimmen alle neuen und alten Parameter exakt überein.")
    # --- ENDE: Analyse-Block ---

    # 📊 Histogramm VOR der Filterung
    plt.figure(figsize=(8, 4))
    plt.hist(min_dists, bins=100, color=ikv_colors["g"], edgecolor=ikv_common_colors["g"])
    plt.xlabel("Minimale Distanz zu bekannten Punkten")
    plt.ylabel("Anzahl Kandidaten")
    plt.title("Distanzen neuer Kandidaten zu bekannten Punkten (vor Filterung)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Distanzen_vor_Filterung.png"))
    plt.close()

    # 💡 Manuelle Eingabe des safe_distance nach Analyse der Histogramme
    print("\n📏 Bitte gib den safe_distance an (z. B. 2.5 für moderate Exploration):")
    try:
        safe_distance = float(input("🔸 safe_distance = "))
    except:
        safe_distance = 2.5  # Fallback-Wert
        print("⚠️ Ungültige Eingabe, verwende Standardwert 2.5")

    print("🔍 Quantile der minimalen Distanzen (vor Filterung):")
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        print(f"  {int(q * 100)}%: {np.quantile(min_dists, q):.4f}")

    ########### Filterung: nur neue Kombinationen UND Mindestabstand: Mindestabstand: nehme ich nach der Masterarbeit raus#############
    filtered = []
    filtered_dists = []
    for i, row in candidates_sorted.iterrows():
        x_tuple = tuple(np.round(row[parameter_names].values, 4))
        if x_tuple in used_combinations:
            continue
        if row["min_dist"] < safe_distance:
            continue
        used_combinations.add(x_tuple)
        filtered.append(row)
        filtered_dists.append(row["min_dist"])
        if len(filtered) >= n_points:
            break

    # 📊 Histogramm NACH der Filterung
    plt.figure(figsize=(8, 4))
    plt.hist(filtered_dists, bins=100, color=ikv_colors["g"], edgecolor=ikv_common_colors["g"])
    plt.xlabel("Minimale Distanz zu bekannten Punkten")
    plt.ylabel("Anzahl gültiger Kandidaten")
    plt.title("Distanzen der Kandidaten (nach Filterung)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Distanzen_nach_Filterung.png"))
    plt.close()

    print("🔍 Quantile der minimalen Distanzen (nach Filterung):")
    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        print(f"  {int(q * 100)}%: {np.quantile(filtered_dists, q):.4f}")

    sampled_df = pd.DataFrame(filtered)

    # Wenn nach Filtering nicht genug Punkte da sind, fülle mit den nächsten besten (ohne Abstandsfilter)
    if len(sampled_df) < n_points:
        print(f"⚠️ Nur {len(sampled_df)} Punkte nach Filter. Ergänze mit nahe liegenden...")
        additional = []
        for i, row in candidates_sorted.iterrows():
            if len(filtered) + len(additional) >= n_points:
                break
            x_tuple = tuple(np.round(row[parameter_names].values, 4))
            if x_tuple in used_combinations:
                continue
            used_combinations.add(x_tuple)
            additional.append(row)
        sampled_df = pd.DataFrame(filtered + additional[:(n_points - len(filtered))])
        exclude_uncertainty = ["predicted_uncertainty"]
        sampled_df = sampled_df.drop(columns=exclude_uncertainty)

    print(f"✅ {len(sampled_df)} neue Kombinationen generiert.")

    if return_full_tracking:
        tracking_info = {
            "R2_train": r2_score(y_all, rf_model.predict(X_all)),
            "avg_predicted_uncertainty": np.mean(candidates_sorted["predicted_uncertainty"]),
            "std_predicted_uncertainty": np.std(candidates_sorted["predicted_uncertainty"]),
            "min_distance_to_known": np.min(min_dists),
            "mean_distance_to_known": np.mean(min_dists)
        }

        # 🔄 Tracking-Ergebnisse speichern
        tracking_path = os.path.join(result_dir, "tracking_info.excel")
        tracking_df = pd.DataFrame([tracking_info])  # Einzeilige Tabelle
        tracking_df.to_excel(tracking_path, index=False, float_format="%.4f")
        print(f"📋 Tracking-Information exportiert nach: {tracking_path}")

        return sampled_df, tracking_info
    return sampled_df


def export_bayesian_kombis_to_csv(kombis_df, result_dir):
    try:
        if not kombis_df.empty:
            column_map = {
                "P_MW": "High Power [W]",
                "t_on": "Pulse-On [ms]",
                "t_off": "Pulse-Off [ms]",
                "p": "Pressure [Pa]",
                "O2": "O2 [sccm]",
                "HMDSO": "HMDSO [sccm]"
            }

            export_cols = [
                "Test Point Name", "Name", "High Power [W]", "Low Power [W]", "Pulse-On [ms]", "Pulse-Off [ms]",
                "Exposure [ms]", "O2 [sccm]", "HMDSO [sccm]", "HMDSN [sccm]", "C2H2 [sccm]", "Ar [sccm]",
                "Coat Time [s]", "Purge Time [s]", "Pressure [Pa]"
            ]
            pd.set_option("display.max_rows", None)  # Alle Zeilen anzeigen
            pd.set_option("display.max_columns", None)  # Alle Spalten anzeigen
            pd.set_option("display.width", 0)  # Breite automatisch anpassen
            pd.set_option("display.max_colwidth", None)  # Inhalt der Spalten vollständig anzeigen
            print(kombis_df)
            expected_ml_columns = ["P_MW", "t_on", "t_off", "p", "O2", "HMDSO"]
            exclude_uncertainty = ["min_dist"]
            kombis_df = kombis_df.drop(columns=exclude_uncertainty)
            kombis_df = kombis_df.reset_index(drop=True)

            print("🔎 Prüfe kombis_df auf Vollständigkeit...")

            # Fehlende Spalten melden
            missing_cols = [col for col in expected_ml_columns if col not in kombis_df.columns]
            if missing_cols:
                print(f"❗ Fehlende Spalten in kombis_df: {missing_cols}")
            else:
                print("✅ Alle erwarteten Spalten sind in kombis_df vorhanden.")

            # NaN-Werte in vorhandenen Spalten prüfen
            for col in expected_ml_columns:
                if col in kombis_df.columns:
                    nan_count = kombis_df[col].isna().sum()
                    if nan_count > 0:
                        print(f"⚠️ {nan_count} NaN-Werte in Spalte '{col}'")

            export_df = pd.DataFrame(columns=export_cols)
            export_df["Test Point Name"] = [f"BAYES_{i + 1}" for i in range(len(kombis_df))]
            export_df["Name"] = [f"AutoML_Bayesian_{i + 1}" for i in range(len(kombis_df))]
            export_df["Low Power [W]"] = 0
            export_df["HMDSN [sccm]"] = 0
            export_df["C2H2 [sccm]"] = 0
            export_df["Ar [sccm]"] = 0
            export_df["Coat Time [s]"] = 120
            export_df["Purge Time [s]"] = 10

            print("📉 Übersicht fehlender Werte pro Spalte:")
            print(export_df.isna().sum())

            for ml_col, target_col in column_map.items():
                if ml_col in kombis_df.columns:
                    export_df[target_col] = kombis_df[ml_col]
                else:
                    print(f"❗ Fehlende Spalte in kombis_df: '{ml_col}', setze {target_col} auf NaN")
                    export_df[target_col] = np.nan

            export_df["Exposure [ms]"] = export_df["Pulse-On [ms]"] + export_df["Pulse-Off [ms]"]

            print("\n📋 Vollständiger Export-DataFrame:")
            print(export_df)

            export_df = export_df.fillna(0)
            # Runden
            for col in export_df.columns:
                if col in ["HMDSO [sccm]", "Pulse-On [ms]", "Exposure [ms]"]:
                    export_df[col] = export_df[col].apply(lambda x: round(x, 1))
                elif export_df[col].dtype in [float, int]:
                    export_df[col] = export_df[col].apply(lambda x: int(round(x)))

            export_path = os.path.join(result_dir, "export_bayesian_sampling.csv")
            export_df.to_csv(export_path, sep=";", index=False, float_format="%.2f", quoting=csv.QUOTE_ALL)
            print(f"📁 Exportiert: {export_path} ({len(export_df)} Kombinationen)")
    except Exception as e:
        print("⚠️ Export fehlgeschlagen:", e)


##########Erweiterung: Region-Growing + Sampling basierend auf Unsicherheitsregionen ################################

def grow_unsicherheitsregionen(X_all_scaled, uncertainty, origin, train_thresh, test_thresh, radius=1.6):
    df = pd.DataFrame(X_all_scaled, columns=[f"feature_{i}" for i in range(X_all_scaled.shape[1])])
    df["uncertainty"] = uncertainty
    df["origin"] = origin
    df["visited"] = False
    df["region"] = -1

    def is_unsicher(i):
        if origin[i] == "train":
            return uncertainty[i] > train_thresh
        else:
            return uncertainty[i] > test_thresh

    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(X_all_scaled)
    region_id = 0

    for idx in range(len(df)):
        if df.at[idx, "visited"] or not is_unsicher(idx):
            continue

        region_points = set()
        to_visit = {idx}

        while to_visit:
            current = to_visit.pop()
            if df.at[current, "visited"]:
                continue
            df.at[current, "visited"] = True
            if not is_unsicher(current):
                continue
            region_points.add(current)
            neighbors = nbrs.radius_neighbors([X_all_scaled[current]], return_distance=False)[0]
            to_visit.update(set(neighbors) - region_points)

        if region_points:
            for i in region_points:
                df.at[i, "region"] = region_id
            region_id += 1

    df_unsicher = df[df["region"] >= 0]
    param_cols = [col for col in df.columns if col.startswith("feature_")]
    bereich_df = (
        df_unsicher.groupby("region").agg({col: ["min", "max"] for col in param_cols})
    )
    bereich_df.columns = [f"{feat}_{agg}" for feat, agg in bereich_df.columns]
    bereich_df = bereich_df.reset_index()

    return df, bereich_df


def main():
    global y
    import os
    import csv
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    ############################Trainiere AutoKeras-Modelle mit Max Rückskalierung######################################
    parameter_names = ["O2", "HMDSO", "P_MW", "t_on", "t_off", "p"]  # wird später verwendet

    models, metrics_ak, histories, durations, optimizers_classes, optimizers_configs = train_autokeras_models(
        X_trainval_np, y_trainval, X_test_np, y_test, n_models=50)

    best_model = tf.keras.models.load_model(
        os.path.join(model_dir, "best_model"),
        custom_objects=ak.CUSTOM_OBJECTS
    )
    #####################################Plot für Auswirkung Belichtungszeit kleiner ton################################

    # Feature-Liste wie beim Training/Modell-Input
    feature_cols = ["P_MW","t_on","t_off","p", "O2","HMDSO"]
    wavelength_axis = wavelength_corrected  # Definiere dies vorher entsprechend
    csv_path = r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Spektren\Fertige ML CSVs\Iteration_10Spektren_1639_ohne_norm.csv"

    # Aufruf der Funktion
    plot_extreme_spectrum_from_excel_and_model(
        csv_path=r"\\ikv-live\AG-Plasma\AGPlasma\HIWIdata\Dahmen\MA\OES\Spektren\Fertige ML CSVs\Iteration_10Spektren_1639_ohne_norm.csv",
        best_model=best_model,
        feature_cols=feature_cols,
        wavelength_axis=wavelength_axis,
        plot_dir=plot_dir,
        ikv_common_colors=ikv_common_colors
    )

    plot_top_n_highest_std_spectra_from_excel(
        csv_path, best_model, feature_cols, wavelength_axis, plot_dir, ikv_common_colors, inverse_max, global_max, n=10
    )

    ####################################################################################################################
    X_all_importance = np.vstack([X_trainval_np, X_test_np])
    y_all_importance = pd.concat([y_trainval, y_test], axis=0)

    compute_and_plot_permutation_importance(
        models=models,
        X=X_all_importance,  # oder wie deine Input-Features heißen
        y=y_all_importance,  # Target, im Zweifel als Array
        feature_names=X.columns,  # Optional, sorgt aber für konsistente Achsen
        plot_dir=plot_dir,
        prefix="AutoKeras"
    )

    document_models(models, histories, durations, doc_dir=doc_dir)

    visualize_metrics(metrics_ak, "AutoKeras Modelle", "autokeras_modelle", plot_dir=plot_dir, doc_dir=doc_dir)

    ####################################Für CV Modelle Optimierer liste################################################
    # Nach dem Training:
    optimizers_classes = [model.optimizer.__class__ for model in models]
    optimizers_configs = [model.optimizer.get_config() for model in models]

    # #####################Cross-Validation##############################
    cv_models, metrics_cv, cv_histories, sam_mean_cv_list, sam_std_cv_list = cross_validate_models(
        models,
        X_trainval_np,
        y_trainval,
        X_test_np,
        y_test,
        optimizers_classes=optimizers_classes,
        optimizers_configs=optimizers_configs)

    print(metrics_cv)

    visualize_metrics(metrics_cv, "Cross-Validated Modelle (15 Stück)", "cv_1 5modelle", plot_dir=plot_dir,
                      doc_dir=doc_dir)

    save_metrics_to_excel(metrics_ak, metrics_cv, result_dir=result_dir)

    #################### Trainings- und Testmetriken mit Min-Max-Rückskalierung (inkl. Mean_SAM) (aus Autokeras)#############
    detailed_metrics = []
    for i, model in enumerate(models):
        # Train-Metriken (Normiert + Rückskaliert)
        (mse_train_norm, rmse_train_norm, mean_sam_train_norm, std_sam_train_norm,
         mse_train_global, rmse_train_global, rmse_std_train_global, r2_train_global,
         mean_sam_train_global, std_sam_train_global) = evaluate_model_max_with_sam(
            model,
            X_trainval_np,
            y_trainval,
            name="Trainingsdaten"
        )

        # Test-Metriken (Normiert + Rückskaliert)
        (mse_test_norm, rmse_test_norm, mean_sam_test_norm, std_sam_test_norm,
         mse_test_global, rmse_test_global, rmse_std_test_global, r2_test_global,
         mean_sam_test_global, std_sam_test_global) = evaluate_model_max_with_sam(
            model,
            X_test_np,
            y_test,
            name="Testdaten"
        )

        detailed_metrics.append({
            "Modell": f"AutoKeras_{i + 1}",

            # --- Train normiert
            "MSE_Train_norm": mse_train_norm,
            "RMSE_Train_norm": rmse_train_norm,
            "Mean_SAM_Train_norm": mean_sam_train_norm,
            "Std_SAM_Train_norm": std_sam_train_norm,

            # --- Train rückskaliert
            "MSE_Train_global": mse_train_global,
            "RMSE_Train_global": rmse_train_global,
            "Std_RMSE_Train_global": rmse_std_train_global,
            "R2_Train_global": r2_train_global,
            "Mean_SAM_Train_global": mean_sam_train_global,
            "Std_SAM_Train_global": std_sam_train_global,

            # --- Test normiert
            "MSE_Test_norm": mse_test_norm,
            "RMSE_Test_norm": rmse_test_norm,
            "Mean_SAM_Test_norm": mean_sam_test_norm,
            "Std_SAM_Test_norm": std_sam_test_norm,

            # --- Test rückskaliert
            "MSE_Test_global": mse_test_global,
            "RMSE_Test_global": rmse_test_global,
            "Std_RMSE_Test_global": rmse_std_test_global,
            "R2_Test_global": r2_test_global,
            "Mean_SAM_Test_global": mean_sam_test_global,
            "Std_SAM_Test_global": std_sam_test_global,
        })

    # Speichern als DataFrame/Excel
    df_detailed = pd.DataFrame(detailed_metrics)
    df_detailed.to_excel(os.path.join(result_dir, "vergleich_train_test.xlsx"), index=False)

    # ==== Balkendiagramm: Train vs. Test R² ====
    x = np.arange(len(df_detailed["Modell"]))  # Indizes für die Modelle
    bar_width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width / 2, df_detailed["R2_Train_global"], width=bar_width, color=ikv_common_colors["g"],
            label="Train R² ")
    plt.bar(x + bar_width / 2, df_detailed["R2_Test_global"], width=bar_width, color=ikv_common_colors["d"],
            label="Test R² ")
    plt.xticks(x, df_detailed["Modell"], rotation=45)
    plt.ylabel("R² Score ")
    plt.title("Train vs. Test R² Vergleich (AutoKeras Modelle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "train_vs_test_r2.png"))
    plt.close()

    ######################## Trainings- und Testmetriken Vergelich der Autokerasmodelle ENDE#####################################

    ########################################SAM Analyse########################################################

    # === SAM-Analyse + Clustering auf Testdaten und allen Daten ===
    y_pred_all = best_model.predict(X_np, verbose=0)
    y_pred_test = best_model.predict(X_test_np, verbose=0)

    # Vor dem Clustering & SAM:
    y_pred_all_re = inverse_max(y_pred_all, global_max)
    y_pred_test_re = inverse_max(y_pred_test, global_max)
    y_test_re = inverse_max(y_test, global_max)
    y_real_test_re = y_test_re.values
    y_re = inverse_max(y, global_max)

    # === Clustering & Analyse für ALLE Spektren ===
    labels_pred_all = cluster_by_sam_angle(y_pred_all_re, angle_threshold_deg=10)
    labels_real_all = cluster_by_sam_angle(y_re, angle_threshold_deg=10)

    # Vorhersage-Cluster visualisieren
    plot_cluster_spectra(y_pred_all, labels_pred_all, os.path.join(sam_plot_dir, "all_predicted_spectra"))
    # Reale Cluster visualisieren
    plot_cluster_spectra(y, labels_real_all, os.path.join(sam_plot_dir, "all_real_spectra"))

    # Parameterzuordnung zu den Clustern
    parameter_analysis(X, labels_pred_all, sam_excel_dir, suffix="all_pred")
    parameter_analysis(X, labels_real_all, sam_excel_dir, suffix="all_real")

    # Validierung Vorhersage-Cluster mit realen Spektren
    val_all = validate_clusters_with_real_sam(y_re, labels_pred_all)
    pd.DataFrame.from_dict(val_all, orient="index", columns=["SAM_mean_deg"]).to_excel(
        os.path.join(sam_excel_dir, "all_validation_predicted_with_real.xlsx")
    )
    # === Clustering & Analyse für TEST-Spektren ===
    labels_pred_test = cluster_by_sam_angle(y_pred_test_re, angle_threshold_deg=10)
    labels_real_test = cluster_by_sam_angle(y_real_test_re, angle_threshold_deg=10)

    # Vorhersage-Cluster visualisieren
    plot_cluster_spectra(y_pred_test, labels_pred_test, os.path.join(sam_plot_dir, "test_predicted_spectra"))
    # Reale Cluster visualisieren
    plot_cluster_spectra(y_test, labels_real_test, os.path.join(sam_plot_dir, "test_real_spectra"))

    # Parameterzuordnung zu den Clustern
    parameter_analysis(X_test, labels_pred_test, sam_excel_dir, suffix="test_pred")
    parameter_analysis(X_test, labels_real_test, sam_excel_dir, suffix="test_real")

    # Validierung Vorhersage-Cluster mit realen Spektren
    val_test = validate_clusters_with_real_sam(y_real_test_re, labels_pred_test)
    pd.DataFrame.from_dict(val_test, orient="index", columns=["SAM_mean_deg"]).to_excel(
        os.path.join(sam_excel_dir, "test_validation_predicted_with_real.xlsx")
    )

    # Für alle Parameter/Spektren
    result_df = compute_cluster_closest_to_middle_and_sam(
        y_real=y_re,
        y_pred=y_pred_all_re,
        cluster_labels=labels_pred_all,
        parameter_df=X,
        outdir=os.path.join("interpretation", "sam_excel", "test")
    )
    # Für test parameter und spektren
    result_df = compute_cluster_closest_to_middle_and_sam(
        y_real=y_real_test_re,
        y_pred=y_pred_test_re,
        cluster_labels=labels_pred_test,
        parameter_df=X_test,
        outdir=os.path.join("interpretation", "sam_excel", "test")
    )

    ######SAM Cluster Matching##########

    y_real = y
    y_pred_all = y_pred_all
    labels_real = labels_real_all
    labels_pred = labels_pred_all

    # Anwendung
    match_df = match_predicted_to_real_clusters(y_real, y_pred_all, labels_real, labels_pred, parameter_df=X,
                                                outdir=sam_excel_dir)
    match_df.head()

    labels_real = labels_real_test
    labels_pred = labels_pred_test

    match_df = match_predicted_to_real_clusters(y_test, y_pred_test, labels_real, labels_pred, parameter_df=X_test,
                                                outdir=sam_excel_dir)
    match_df.head()

    ############# SAM Vergleich Parameter in Cluster y_real zur Vorhersage

    prediction_consistency_by_real_clusters(
        X=X,
        y_pred=y_pred_all,
        y_real_labels=labels_real_all,
        outdir=os.path.join(sam_excel_dir, "prediction_within_real_clusters"),
        suffix="all_real"
    )

    ###########SAM CLuster Mergen + Welcher CLuster liegen nah aneinander#########################################

    # Jetzt mit den echten Intensitäten weiterarbeiten:
    combined_df, merge_df = cluster_parameter_sam_excel(
        X, labels_pred_all, y_pred_all, angle_threshold=10,
        parameter_names=["P_MW", "t_on", "t_off", "p", "O2", "HMDSO"], output_dir=sam_excel_dir
    )

    ###################################Plot Abweichung-vom-Mittelwert (Prediction vs. Real)############################

    # Nur für die Testdaten
    plot_vertical_abstand_parameterkombinationen(
        y_true=y_test_re,
        y_pred=y_pred_test_re,
        plot_dir=plot_dir,
        prefix="Test"
    )

    # Für alle daten
    plot_vertical_abstand_parameterkombinationen(
        y_true=y_re,
        y_pred=y_pred_all_re,
        plot_dir=plot_dir,
        prefix="All"
    )

    ##############################Bestimmunng der Bereiche nah mittel fern##############################################

    # Beispiel: Vorhersagen für alle Daten ("all predictions"), z.B. shape (n_samples, 3694)
    all_spectra = y_pred_all  # z.B. y_pred_all = best_model.predict(X_np)
    all_spectra_real = y.values
    X_all = X  # zugehörige Parameter, falls sie zum gleichen Index wie y_pred_all gehören

    # === 1) Referenzspektrum wählen (hier: erstes Spektrum, kann beliebig gewählt werden) ===
    reference_id = 1871
    idx_reference = df[df["id"] == reference_id].index[0]
    reference_spectrum = y_pred_all[idx_reference]

    # SAM-Winkel zu allen Vorhersagen berechnen
    sam_angles = np.array([compute_sam_angle(reference_spectrum, s) for s in all_spectra])

    # === 3) Spektrum mit größtem SAM finden ===
    idx_max = np.argmax(sam_angles)
    max_angle_spectrum = y_pred_all[idx_max]
    max_id = df.loc[idx_max, "id"]
    max_angle = sam_angles[idx_max]

    print(f"Referenz-ID: {reference_id}, Maximal-SAM-ID: {max_id}, Maximalwinkel: {max_angle:.2f}°")

    # Plot der beiden Spektren
    plt.figure(figsize=(12, 5))
    plt.plot(wavelength_corrected, reference_spectrum, label=f"Referenz (ID: {reference_id})",color=ikv_common_colors["g"], linewidth=1)
    plot_characteristic_lines_numbered()
    plt.plot(wavelength_corrected, max_angle_spectrum, label=f"Maximal SAM (ID: {max_id})",color=ikv_common_colors["d"], linestyle='--', linewidth=1)
    plt.xlabel("Wellenlänge[nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.title(
        f"Spektralvergleich (Vorhersagen): Referenz-ID {reference_id} vs. Max-SAM-ID {max_id}\nSAM-Winkel: {max_angle:.2f}°")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    vergleichsplot_path = os.path.join(pca_plot_dir,
                                       f"Vorhersage_Vergleich_Referenz_{reference_id}_MaxSAM_{max_id}.png")
    plt.savefig(vergleichsplot_path, dpi=300)
    plt.close()
    print(f"✅ Vergleichsplot gespeichert: {vergleichsplot_path}")

    # SAM-Bereiche bestimmen
    bounds = np.linspace(0, max_angle, 4)
    region_labels = np.digitize(sam_angles, bounds[1:-1])  # 0,1,2 für drei Bereiche
    region_names = ["nah", "mittel", "weit"]

    X_all = X.reset_index(drop=True)  # Stelle sicher, dass Index passt!

    # Erzeuge pro Region einen eigenen DataFrame:
    X_nah = X_all[region_labels == 0]
    X_mittel = X_all[region_labels == 1]
    X_weit = X_all[region_labels == 2]

    # Für die PDPs dann:
    X_region_dict = {
        "nah": X_nah,
        "mittel": X_mittel,
        "weit": X_weit,
        "Alle": X_all  # Für den globalen Plot
    }
    #######################Gibt die 3 Pixelbereiche wieder die in der PCA die größte Varianz zeigen#####################

    #################Loop über alle Teilbereiche##############
    pca_dict = {}  # speichert alle infos der PCA ab
    # Dictionary zum Abspeichern der Bereiche:
    ranges_per_region = {}  # z.B. "nah" → {PC0: [...], PC1: [...], PC2: [...]}, "Alle": ...

    # Loop über Teilbereiche (region_names wie ["nah", "mittel", "weit"])

    # ... bisheriger Code für "nah", "mittel", "weit" ...
    for region in range(3):
        region_label = region_names[region]
        X_region = X_region_dict[region_label]
        y_region = y[region_labels == region]
        pca, pcs = full_pca_analysis_with_all_visuals(
            y_pred_input=y_region,
            parameter_df=X_region,
            pixel_axis=wavelength_corrected,
            n_components=10,
            name=f"PCA_Vorhersagen_{region_label}_Ref{reference_id}_MaxSAM{max_id}",
            plot_dir=pca_plot_dir,
            excel_dir=pca_excel_dir
        )
        pca_dict[region_label] = pca  # im Dict speichern!

        print(f"\n🌟 Wichtigste Einzelpixel-Loadings für Bereich '{region_label}':")
        per_pc_ranges = {}

        for pc_idx in range(3):
            component = pca.components_[pc_idx]
            abs_component = np.abs(component)
            top3_indices = abs_component.argsort()[-3:][::-1]
            print(f"  PC{pc_idx + 1}:")
            for idx in top3_indices:
                wl = wavelength_corrected[idx]
                loading = component[idx]
                print(f"    EINZEL-PIXEL: Wellenlänge {wl:.1f} nm, Loading: {loading:+.3f}")

        print(f"\n📊 Signifikante Pixelbereiche für Bereich '{region_label}':")
        for pc_idx in range(3):
            component = pca.components_[pc_idx]
            ranges = get_significant_pixel_ranges(component, threshold_abs=0.05)
            ranges_sorted = sorted(
                ranges,
                key=lambda r: np.abs(component[r[0]:r[1] + 1]).sum(),
                reverse=True
            )
            top3_ranges = ranges_sorted[:3]
            per_pc_ranges[pc_idx] = top3_ranges
            for start, end in top3_ranges:
                wl_start = wavelength_corrected[start]
                wl_end = wavelength_corrected[end]
                print(f"  PC{pc_idx + 1}: Bereich Pixel {start}–{end} (Wellenlänge {wl_start:.1f}–{wl_end:.1f} nm)")
        print("-------------------------------------------------------")
        ranges_per_region[region_label] = per_pc_ranges  # Speichern für die Region!

    # ============ Jetzt: GLOBALEN BEREICH ("Alle") ergänzen =============

    # ACHTUNG: KEIN Filter auf region_labels! Hier wird das GESAMTE Set genommen:
    label_all = "Alle"
    X_region = X_region_dict[label_all]  # Das komplette Datenset

    pca, pcs = full_pca_analysis_with_all_visuals(
        y_pred_input=y_real,
        parameter_df=X_region,
        pixel_axis=wavelength_corrected,
        n_components=10,
        name=f"PCA_Vorhersagen_{label_all}_Ref{reference_id}_MaxSAM{max_id}",
        plot_dir=pca_plot_dir,
        excel_dir=pca_excel_dir
    )
    pca_dict[label_all] = pca  # Im Dict speichern!

    print(f"\n🌟 Wichtigste Einzelpixel-Loadings für Bereich '{label_all}':")
    per_pc_ranges = {}

    for pc_idx in range(3):
        component = pca.components_[pc_idx]
        abs_component = np.abs(component)
        top3_indices = abs_component.argsort()[-3:][::-1]
        print(f"  PC{pc_idx + 1}:")
        for idx in top3_indices:
            wl = wavelength_corrected[idx]
            loading = component[idx]
            print(f"    EINZEL-PIXEL: Wellenlänge {wl:.1f} nm, Loading: {loading:+.3f}")

    print(f"\n📊 Signifikante Pixelbereiche für Bereich '{label_all}':")
    for pc_idx in range(3):
        component = pca.components_[pc_idx]
        ranges = get_significant_pixel_ranges(component, threshold_abs=0.05)
        ranges_sorted = sorted(
            ranges,
            key=lambda r: np.abs(component[r[0]:r[1] + 1]).sum(),
            reverse=True
        )
        top3_ranges = ranges_sorted[:3]
        per_pc_ranges[pc_idx] = top3_ranges
        for start, end in top3_ranges:
            wl_start = wavelength_corrected[start]
            wl_end = wavelength_corrected[end]
            print(f"  PC{pc_idx + 1}: Bereich Pixel {start}–{end} (Wellenlänge {wl_start:.1f}–{wl_end:.1f} nm)")
    print("-------------------------------------------------------")
    ranges_per_region[label_all] = per_pc_ranges  # SPEICHERN unter "Alle"

    # ranges_per_region enthält jetzt für jeden Bereich ("nah", "mittel", "weit", "Alle")
        # die jeweils 3 wichtigsten Pixelbereiche pro PC!

    #############################################Eigene PDP Anwendung, weil Grafikkarten Speicher zu gering ########################################################################

    # Annahme: Alles ist bereits vorbereitet
    # - y_pred_all: Vorhersagen für alle Daten (n_samples, n_pixels)
    # - X_all: DataFrame mit allen Parametern (Index wie y_pred_all!)
    # - wavelength_corrected: Wellenlängenachse (für die Plots)
    # - region_labels: aus SAM-Binning, shape = (n_samples,)
    # - region_names: ["nah", "mittel", "weit"]
    # - ranges_per_region: Dictionary aus den PCA-Auswertungen (siehe oben)
    # - parameter_names: z.B. ["O2", "HMDSO", "P_MW", "t_on", "t_off", "p"]
    # - best_model: dein trainiertes Modell

    # === 1) DataFrames für alle Regionen erzeugen ===
    X_all = X_all.reset_index(drop=True)  # falls nicht schon gemacht!
    X_region_dict = {
        "nah": X_all[region_labels == 0].reset_index(drop=True),
        "mittel": X_all[region_labels == 1].reset_index(drop=True),
        "weit": X_all[region_labels == 2].reset_index(drop=True),
        "Alle": X_all
    }


    #Sensitivit#tsanalyse für alle Regionen eines Pixelbereichs in einem Plott
    plot_regions = ["nah", "mittel", "weit"]

    # 1. Sammle alle einzigartigen Pixelbereiche aus allen Regionen und PCs
    all_pixel_ranges = set()
    for region_label in plot_regions:
        for pc_idx in range(3):
            for pixel_range in ranges_per_region[region_label][pc_idx]:
                all_pixel_ranges.add(pixel_range)  # pixel_range ist ein Tupel (start, end)

    all_pixel_ranges = sorted(list(all_pixel_ranges))  # sortiert für konsistente Plots


    # PDPs für alle Parameter nur am Peak-Pixel pro Bereich (pro Region/PC) ===
    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        print(f"Der separierte Parameterraum hat {len(X_region)} Parameterkombinationen")
        pca = pca_dict[region_label]
        wavelength = wavelength_corrected

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges:
                # ► Peak-Pixel im Bereich bestimmen (höchstes |Loading|)
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel
                wl_peak = wavelength[peak_pixel_idx]

                peak_name = find_peak_name_for_pixel(wl_peak, tolerance=1)
                peak_str = f"({peak_name})" if peak_name else ""

                print(
                    f"Region: {region_label} | PC: {pc_idx + 1} | Bereich: {start}-{end} ({wavelength[start]:.1f}-{wavelength[end]:.1f} nm)\n"
                    f"→ Peak-Pixel: {peak_pixel_idx} (λ={wl_peak:.1f} nm), Loading={component[peak_pixel_idx]:+.3f}, Peak: {peak_name}"
                )

                plt.figure(figsize=(8, 5))
                for param in parameter_names:
                    unique_vals = np.unique(X_region[param])
                    print(
                        f"{param}: dtype={X_region[param].dtype}, unique_vals={unique_vals[:5]}... ({len(unique_vals)} Werte)")

                    # Nur noch Parameter überspringen, wenn weniger als 2 Werte oder nicht numerisch
                    if not np.issubdtype(X_region[param].dtype, np.number) or (len(unique_vals) <= 1):
                        print(f"➔ Parameter '{param}' wird übersprungen (kein numerischer Typ oder nur 1 Wert)")
                        continue
                    else:
                        param_max = X_region[param].max()
                        if param_max == 0:
                            print(f"➔ Parameter '{param}' hat nur Nullwerte – wird übersprungen!")
                            continue
                        param_values = np.linspace(X_region[param].min(), param_max, 10)
                        param_values_normed = param_values / param_max

                    mean_intensities = []
                    X_mod = X_region.copy()
                    for val in param_values:
                        X_mod[param] = val
                        preds = best_model.predict(X_mod.values, verbose=0)
                        mean_intensities.append(preds[:, peak_pixel_idx].mean())

                    if len(mean_intensities) == 0:
                        print(f"❗ Warnung: mean_intensities für {param} leer, keine Linie gezeichnet!")
                        continue

                    label = f"{param} [{X_region[param].min():.0f}-{X_region[param].max():.0f}]"
                    print(
                        f"✓ Plotte Parameter '{param}' (normiert), Peak-Pixel {peak_pixel_idx}, Wellenlänge {wl_peak:.1f}nm, Peak: {peak_name}")
                    plt.plot(param_values_normed, mean_intensities, label=label)

                plt.xlabel("Parameter (Max-Normiert)")
                plt.ylabel(f"Intensität [counts]")
                plt.title(
                    f"Sensitivität auf Peak-Pixel: λ={wl_peak:.1f}nm {peak_str}\n"
                    f"(Peakpixel aus Bereich: {wavelength[start]:.1f}-{wavelength[end]:.1f} nm)"
                )

                plt.xlim(0, 1)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                path = os.path.join(
                    PDP_alle,
                    f"PDP_{region_label}_PC{pc_idx + 1}_peakpixel_{peak_pixel_idx}_{int(wl_peak)}nm_allparams_maxnorm.png"
                )
                plt.savefig(path, dpi=300)
                plt.close()
                print(f"✅ Peak-Pixel-PDP gespeichert: {path}")

    # === 4) Optional: 2D-PDP (Parameterkombinationen) für ersten Pixelbereich einer Region/PC ===
    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        pca = pca_dict[region_label]

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges[:3]:
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel

                pdp_for_keras_model_2d(
                    model=best_model,
                    X=X_region,
                    parameter1="O2",
                    parameter2="HMDSO",
                    pixel_range=(peak_pixel_idx, peak_pixel_idx),  # Nur Peakpixel
                    pixel_source_range=(start, end),  # Urspr. Bereich
                    grid_resolution=10,
                    plot_path=os.path.join(
                        PDP_2d,
                        f"PDP2D_{region_label}_PC{pc_idx + 1}_O2_HMDSO_PeakPixel{peak_pixel_idx}.png"
                    )
                )


    #########2D: t_on t_off###############################
    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        pca = pca_dict[region_label]

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges[:3]:
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel

                pdp_for_keras_model_2d(
                    model=best_model,
                    X=X_region,
                    parameter1="t_on",
                    parameter2="t_off",
                    pixel_range=(peak_pixel_idx, peak_pixel_idx),  # Nur Peakpixel!
                    pixel_source_range=(start, end),  # Urspr. Bereich!
                    grid_resolution=10,
                    plot_path=os.path.join(
                        PDP_2d,
                        f"PDP2D_{region_label}_PC{pc_idx + 1}_O2_HMDSO_PeakPixel{peak_pixel_idx}.png"
                    )
                )

    #############2D für P_MW und t_on#################
    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        pca = pca_dict[region_label]

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges[:3]:
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel

                pdp_for_keras_model_2d(
                    model=best_model,
                    X=X_region,
                    parameter1="P_MW",
                    parameter2="t_on",
                    pixel_range=(peak_pixel_idx, peak_pixel_idx),  # Nur Peakpixel!
                    pixel_source_range=(start, end),
                    grid_resolution=10,
                    plot_path=os.path.join(
                        PDP_2d,
                        f"PDP2D_{region_label}_PC{pc_idx + 1}_P_MW & t_on_PeakPixel{peak_pixel_idx}.png"
                    )
                )

    #############2D für HMDSO und t_on#################

    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        pca = pca_dict[region_label]

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges[:3]:
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel

                pdp_for_keras_model_2d(
                    model=best_model,
                    X=X_region,
                    parameter1="HMDSO",
                    parameter2="t_on",
                    pixel_range=(peak_pixel_idx, peak_pixel_idx),  # Nur Peakpixel!
                    pixel_source_range=(start, end),
                    grid_resolution=10,
                    plot_path=os.path.join(
                        PDP_2d,
                        f"PDP2D_{region_label}_PC{pc_idx + 1}_P_MW & t_on_PeakPixel{peak_pixel_idx}.png"
                    )
                )

    ######################################PDP für Peak_pixel##########################################################

    for region_label, pc_ranges in tqdm(ranges_per_region.items()):
        X_region = X_region_dict[region_label]
        print(f"Der separierte Parameterraum hat  {len(X_region)} Parameterkombinationen")
        pca = pca_dict[region_label]  # Die passende PCA mit den Komponenten!
        wavelength = wavelength_corrected  # Wellenlängenachse

        for pc_idx, pixel_ranges in pc_ranges.items():
            component = pca.components_[pc_idx]
            for (start, end) in pixel_ranges:
                # 1) Peak-Pixel im Bereich: Pixel mit höchstem |Loading| innerhalb des Bereichs
                loading_slice = component[start:end + 1]
                peak_idx_rel = np.argmax(np.abs(loading_slice))
                peak_pixel_idx = start + peak_idx_rel
                wl_peak = wavelength[peak_pixel_idx]
                print(
                    f"  {region_label}, PC{pc_idx + 1}, Bereich {start}-{end} ({wavelength[start]:.1f}-{wavelength[end]:.1f} nm): Peak-Pixel {peak_pixel_idx} (λ={wl_peak:.1f} nm), Loading={component[peak_pixel_idx]:+.3f}")

                # 2) PDP für dieses Pixel für alle Parameter:
                for param in parameter_names:
                    unique_vals = np.unique(X_region[param])
                    if (X_region[param].dtype == "O") or (len(unique_vals) <= 10):
                        param_values = unique_vals
                    else:
                        param_min, param_max = X_region[param].min(), X_region[param].max()
                        param_values = np.linspace(param_min, param_max, 10)

                    mean_intensities = []
                    X_mod = X_region.copy()
                    for val in param_values:
                        X_mod[param] = val
                        preds = best_model.predict(X_mod.values, verbose=0)
                        mean_intensities.append(preds[:, peak_pixel_idx].mean())

                    plt.figure(figsize=(7, 4))
                    plt.plot(param_values, mean_intensities, marker="o")
                    plt.xlabel(param)
                    plt.ylabel(f"Intensität Pixel {peak_pixel_idx} (λ={wl_peak:.1f}nm)")
                    plt.title(
                        f"Sensitivitätsanalyse: {param}\n{region_label}, PC{pc_idx + 1}, Pixel {peak_pixel_idx} (λ={wl_peak:.1f}nm)")
                    plt.grid(True)
                    plt.tight_layout()
                    out_name = f"PDP_{param}_{region_label}_PC{pc_idx + 1}_Pixel{peak_pixel_idx}_{int(wl_peak)}nm.png"
                    plot_path = os.path.join(sens_dir, out_name)
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    print(f"    PDP gespeichert: {plot_path}")

    #######################################Plots der Grenzspektren der Teilbereiche##############################################
    # Finde die Indizes der Grenzwerte:
    grenz_indices = []
    for grenze in bounds:
        idx = (np.abs(sam_angles - grenze)).argmin()
        grenz_indices.append(idx)

    grenznamen = ["Referenz", "1. Grenze", "2. Grenze", "Maximal-SAM"]

    # Einzelplots für jede Grenze
    for i, idx in enumerate(grenz_indices):
        id_str = df.loc[idx, "id"]
        sam_str = f"{sam_angles[idx]:.2f}°" if i > 0 else "0.0°"
        plt.figure(figsize=(10, 4))
        plt.plot(wavelength_corrected, all_spectra_real[idx],
                 label=f"Echt ({grenznamen[i]}, ID: {id_str}, SAM: {sam_str})",color=ikv_common_colors["g"], linewidth=1)
        plt.plot(wavelength_corrected, all_spectra[idx],
                 label=f"Vorhersage ({grenznamen[i]}, ID: {id_str}, SAM: {sam_str})",color=ikv_common_colors["d"],
                 linestyle='--', linewidth=1)

        plt.xlabel("Wellenlänge [nm]")
        plt.ylabel("Intensität [a.u.]")
        plt.title(f"Grenzspekturm {grenznamen[i]} (ID: {id_str})\nSAM zum Referenz: {sam_str}")
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        pfad = os.path.join(pca_plot_dir, f"Grenzspekturm_{grenznamen[i]}_ID_{id_str}.png")
        plt.savefig(pfad, dpi=300)
        plt.close()
        print(f"✅ Plot gespeichert: {pfad}")

    # Sammelplot aller echten Grenzspektren in EINEM Plot
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(grenz_indices):
        id_str = df.loc[idx, "id"]
        sam_str = f"{sam_angles[idx]:.2f}°" if i > 0 else "0.0°"
        plt.plot(wavelength_corrected, all_spectra_real[idx],label=f"{grenznamen[i]} (ID: {id_str}, SAM: {sam_str})", linewidth=1)
    plot_characteristic_lines_numbered()
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.title("Grenzspektren (Echt)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    pfad_real = os.path.join(pca_plot_dir, "Grenzspektren_Echt_Alle.png")
    plt.savefig(pfad_real, dpi=300)
    plt.close()
    print(f"✅ Sammelplot aller echten Grenzspektren gespeichert: {pfad_real}")

    ##############Parameter der Grenzspektren extrahieren und in einer Excel abspeichern################################

    parameter_cols = [c for c in df.columns if not c.startswith("pixel_") and not c.endswith("_std")]

    # Werte der Grenzspektren extrahieren
    grenzparam_list = []
    for i, idx in enumerate(grenz_indices):
        params = df.loc[idx, parameter_cols].copy()
        params["Grenzname"] = grenznamen[i]
        params["SAM_angle"] = sam_angles[idx] if i > 0 else 0.0
        grenzparam_list.append(params)

    # DataFrame erzeugen
    grenzparam_df = pd.DataFrame(grenzparam_list)

    # Zielpfad (kannst du beliebig setzen)
    excel_out = os.path.join(pca_plot_dir, "Grenzspektren_Parameter.xlsx")

    # Speichern
    grenzparam_df.to_excel(excel_out, index=False)
    print(f"📝 Grenzspektren-Parameter in Excel gespeichert: {excel_out}")

    # Sammelplot aller Vorhersage-Grenzspektren in EINEM Plot
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(grenz_indices):
        id_str = df.loc[idx, "id"]
        sam_str = f"{sam_angles[idx]:.2f}°" if i > 0 else "0.0°"
        plt.plot(wavelength_corrected, all_spectra[idx],
                 label=f"{grenznamen[i]} (ID: {id_str}, SAM: {sam_str})", linewidth=1, linestyle="--")
    plot_characteristic_lines_numbered()
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität [a.u.]")
    plt.title("Grenzspektren (Vorhersage)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    pfad_pred = os.path.join(pca_plot_dir, "Grenzspektren_Vorhersage_Alle.png")
    plt.savefig(pfad_pred, dpi=300)
    plt.close()
    print(f"✅ Sammelplot aller Vorhersage-Grenzspektren gespeichert: {pfad_pred}")

    # Bereichszuordnung mitsprechenden Labels
    region_names = ["nah", "mittel", "weit"]
    region_labels = np.digitize(sam_angles, bounds[1:-1])
    region_labels_str = [region_names[r] for r in region_labels]

    # DataFrame mit Parametern, ID und Bereich
    zuordnung_df = df.copy()
    zuordnung_df["SAM_angle"] = sam_angles
    zuordnung_df["SAM_Bereich"] = region_labels_str

    excel_path = os.path.join(pca_excel_dir, "SAM_Bereichszuordnung.xlsx")
    zuordnung_df.to_excel(excel_path, index=False)
    print(f"✅ Bereichszuordnung als Excel gespeichert: {excel_path}")

    ##############Plotten der Regionen (nah,mittel,fern)#########################

    plot_spectra_per_region(
        zuordnung_df,
        pixel_cols=[col for col in zuordnung_df.columns if "pixel_" in col],
        wavelength_axis=wavelength_corrected,
        save_dir=save_dir,
        show_reference=True,
        reference_id=1873,
        show_grenzspektren=True
    )

    ################################Regionen (nah,mittel,fern) evaluieren#################################################################

    #Histogramm aus bestem Modell

    for i, region in enumerate(region_names):
        idx = np.where(np.array(region_labels_str) == region)[0]
        print(f"{region}: {len(idx)} Spektren")

    for i, region in enumerate(region_names):
        idx = np.where(np.array(region_labels_str) == region)[0]
        y_region = y.iloc[idx]
        y_pred_region = y_pred_all[idx]
        print(f"--- Auswertung Region: {region} ---")
        n_region = len(idx)
        mean_sam, std_sam, sam_values = evaluate_and_plot_sam(
            y_df=y_region,
            y_pred_arr=y_pred_region,
            df_meta=df_meta,
            plot_dir=plot_dir,
            doc_dir=doc_dir,
            prefix=f"{region} (n={n_region})"
        )

    ###########################################MC Dropout Anfang########################################################

    # MC Dropout Unsicherheiten berechnen
    mc_models = [mc_model_from_model(model, dropout_rate=0.5) for model in models]


    # MC Dropout für Train
    results_train = predict_with_mc_dropout(
        models=mc_models,
        X=X_trainval_np,
        y_true=y_trainval,
        repeats=100,
        plot_dir=plot_dir,
        label="Train",
        doc_dir=doc_dir,  # Ordner für .xlsx-Dateien
       )

    unsicher_train = np.mean(results_train[1], axis=(0, 2)) #stdw über alle Modelle
    print(unsicher_train)
    mean_pred_train = np.mean(results_train[0], axis=0) #Mittlere Fehler über alle Modelle pro Spektrum

    # MC Dropout für Test
    results_test = predict_with_mc_dropout(
        models=mc_models,
        X=X_test_np,
        y_true=y_test,
        repeats=100,
        plot_dir=plot_dir,
        label="Test",
        doc_dir=doc_dir,  # Ordner für .xlsx-Dateien
    )


    unsicher_test = np.mean(results_test[1], axis=(0, 2))
    print(unsicher_test)
    mean_pred_test = np.mean(results_test[0], axis=0)

    #################Vergleich vor und nach MC Dropout der 3 Metriken #######################################################

    autokeras_r2 = [m["Test_R2_global"] for m in metrics_ak]
    autokeras_rmse = [m["Test_RMSE_global"] for m in metrics_ak]
    autokeras_rmse_std = [m["Test_Std_RMSE_global"] for m in metrics_ak]
    autokeras_sam = [m["Test_Mean_SAM_global"] for m in metrics_ak]
    autokeras_sam_std = [m["Test_Std_SAM_global"] for m in metrics_ak]
    model_names = [m["Modell"] for m in metrics_ak]

    mc_rmse_list = results_test[2]
    mc_rmse_std_list = results_test[3]
    mc_r2_list = results_test[4]
    mc_sam_mean_list = results_test[5]
    mc_sam_std_list = results_test[6]

    # X-Achse
    x = np.arange(len(model_names))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(model_names) * 2.1), 5))

    # 1. RMSE (Testdaten)
    axes[0].bar(x - bar_width / 2, autokeras_rmse, yerr=autokeras_rmse_std, capsize=5,
                width=bar_width, color=ikv_common_colors["g"], label="Autokeras")
    axes[0].bar(x + bar_width / 2, mc_rmse_list, yerr=mc_rmse_std_list, capsize=5,
                width=bar_width, color=ikv_common_colors["d"], label="MC Dropout")
    axes[0].set_title("Test-RMSE")
    axes[0].set_ylabel("RMSE [counts]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)

    axes[0].grid(True, axis='y')

    # 2. R² (Testdaten)
    axes[1].bar(x - bar_width / 2, autokeras_r2, width=bar_width, color=ikv_common_colors["g"], label="Autokeras")
    axes[1].bar(x + bar_width / 2, mc_r2_list, width=bar_width, color=ikv_common_colors["d"], label="MC Dropout")
    axes[1].set_title("Test-R²")
    axes[1].set_ylabel("R²")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)

    axes[1].grid(True, axis='y')

    # 3. SAM (Testdaten)
    axes[2].bar(x - bar_width / 2, autokeras_sam, yerr=autokeras_sam_std, capsize=5,
                width=bar_width, color=ikv_common_colors["g"], label="Autokeras")
    axes[2].bar(x + bar_width / 2, mc_sam_mean_list, yerr=mc_sam_std_list, capsize=5,
                width=bar_width, color=ikv_common_colors["d"], label="MC Dropout")
    axes[2].set_title("Test-SAM")
    axes[2].set_ylabel("SAM-Winkel [°]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=45)
    axes[2].legend()
    axes[2].grid(True, axis='y')

    fig.suptitle("Vergleich Testdaten-Metriken: Autokeras vs. MC Dropout")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # === Speicherung ===
    save_path = os.path.join(plot_dir, "Vergleich_Autokeras_vs_MC_Dropout.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Vergleichsplot gespeichert: {save_path}")


    # Kalibrierungsplots für Train und testdaten
    rmse_train, rmse_test, sam_train_all, sam_test_all, rho_train_all, rho_test_all = plot_uncertainty_vs_error_both_sets(
        y_trainval, mean_pred_train, unsicher_train,
        y_test, mean_pred_test, unsicher_test,
        plot_dir=plot_dir,
        ikv_common_colors=ikv_common_colors,
        prefix="Train_Test_AllModels"
    )

    unsicher_alle = np.concatenate([unsicher_train, unsicher_test])
    rmse_alle = np.concatenate([rmse_train, rmse_test])
    print(rmse_alle.shape)

    rho_all, pval_all = compute_spearman_uncertainty_vs_error(
        unsicher_alle, rmse_alle, set_name="All", plot_dir=plot_dir, filename="unsicherheit_vs_rmse")

######################################Kalibireurng pro Modell####################################################
    spearman_scores = []

    for idx in range(len(mc_models)):
        # Für dieses Modell:
        unsicher_train = np.mean(results_train[1][idx], axis=1)  # [n_spektren,]
        mean_pred_train = results_train[0][idx]  # [n_spektren, n_pixel]
        unsicher_test = np.mean(results_test[1][idx], axis=1)  # [n_spektren,]
        mean_pred_test = results_test[0][idx]  # [n_spektren, n_pixel]

        # Kalibrierungsplot für das Modell + Spearman-Koeffizient zurückgeben!
        rmse_train, rmse_test, sam_train, sam_test, rho_train, rho_test = plot_uncertainty_vs_error_both_sets(
            y_trainval, mean_pred_train, unsicher_train,
            y_test, mean_pred_test, unsicher_test,
            plot_dir=plot_dir,
            ikv_common_colors=ikv_common_colors,
            prefix=f"Train_Test_Modell{idx + 1}"
        )
        spearman_scores.append(abs(rho_test))  # oder rho_test, je nach Wunsch

        print(f"Modell {idx + 1}: RMSE_Train={np.mean(rmse_train):.3f}, RMSE_Test={np.mean(rmse_test):.3f}, "
              f"Spearman_rho_Test={rho_test:.3f}")

    ################################Modell auswählen nach hächstem R2 nach MC Dropout udn spearman größer 0,2#########

    mc_r2_arr = np.array(mc_r2_list)
    spearman_arr = np.array(spearman_scores)
    sort_idx = np.argsort(mc_r2_arr)[::-1]  # absteigend nach R² sortiert

    best_model_idx = None
    for idx in sort_idx:
        if spearman_arr[idx] > 0.2:
            best_model_idx = idx
            break
    if best_model_idx is None:
        print("❗Kein Modell erfüllt Spearman > 0.2 – nehme Modell mit höchstem R²!")
        best_model_idx = sort_idx[0]

    print(
        f"\nBestes Modell ist Modell {best_model_idx + 1} mit R²={mc_r2_arr[best_model_idx]:.3f} und Spearman-Testwert {spearman_arr[best_model_idx]:.3f}")

###########################################################################

    best_model_mc = models[best_model_idx]

    klassisch_pred_train = best_model_mc.predict(X_trainval_np, verbose=0)
    klassisch_pred_test = best_model_mc.predict(X_test_np, verbose=0)

    unsicher_train_best = np.mean(results_train[1][best_model_idx], axis=1)
    unsicher_test_best = np.mean(results_test[1][best_model_idx], axis=1)

    # MC Dropout – sicherstes & unsicherstes Spektrum (nach Unsicherheit)
    plot_most_certain_uncertain_spectra(
        y_trainval, klassisch_pred_train, unsicher_train_best, df_trainval, plot_dir, label="AMC_Train"
    )

    plot_most_certain_uncertain_spectra(
        y_test, klassisch_pred_test, unsicher_test_best, df_test, plot_dir, label="AMC_Test"
    )

    # Best_Model-Vorhersage: Spektren mit kleinstem & größtem SAM (inkl. Unsicherheit im Titel)
    plot_extreme_sam_spectra_with_uncertainty(
        y_trainval, klassisch_pred_train, unsicher_train_best, df_trainval, plot_dir, label="BestModel_AMC_Train"
    )

    plot_extreme_sam_spectra_with_uncertainty(
        y_test, klassisch_pred_test, unsicher_test_best, df_test, plot_dir, label="BestModel-AMC_Test"
    )

    # Für alle Daten zusammen (optional, falls du das möchtest)
    best_pred_all = best_model_mc.predict(X_np, verbose=0)
    plot_extreme_sam_spectra_with_uncertainty(
        y, best_pred_all, np.concatenate([unsicher_train_best, unsicher_test_best]), df_meta, plot_dir, label="BestModel_AMC_All"
    )


    #################Plotte die 10 spektren mit dem größten RMSE, Sam und unsicherheit###############################

    #best_model_idx ist der index des Modells wo der spearman am größten ist
    unsicher_alle = np.concatenate([unsicher_train_best, unsicher_test_best])


    rmse_alle = np.sqrt(np.mean((y.values - y_pred_all) ** 2, axis=1))
    sam_alle = np.array(compute_sam_angles(y.values, y_pred_all))

    y_true_all = pd.concat([y_trainval, y_test], axis=0).reset_index(drop=True)
    y_pred_all = best_pred_all
    meta_all = pd.concat([df_trainval, df_test], axis=0).reset_index(drop=True)

    plot_top_n_spectra(
        y_true_all, y_pred_all, rmse_alle, "Größtes RMSE (aus allen Daten)", meta_all,
        wavelength_axis=wavelength_corrected, n=10, save_dir=save_dir,
        rmse_arr=rmse_alle, sam_arr=sam_alle, unsicher_arr=unsicher_alle
    )
    plot_top_n_spectra(
        y_true_all, y_pred_all, sam_alle, "Größtes SAM (aus allen Daten)", meta_all,
        wavelength_axis=wavelength_corrected, n=10, save_dir=save_dir,
        rmse_arr=rmse_alle, sam_arr=sam_alle, unsicher_arr=unsicher_alle
    )
    plot_top_n_spectra(
        y_true_all, y_pred_all, unsicher_alle, "Größte Unsicherheit (aus allen Daten)", meta_all,
        wavelength_axis=wavelength_corrected, n=10, save_dir=save_dir,
        rmse_arr=rmse_alle, sam_arr=sam_alle, unsicher_arr=unsicher_alle
    )

    ########################################################################################################################

##################Histogramm für Prediction nach bestem Modell und kleinster und größter SAM ########################
############Die anderen Histogramme sind für die Bereiche oder nach MC Dropout
    mean_sam_all, stdw_sam_all, sam_values_all = evaluate_and_plot_sam(
        y_df=y_real,
        y_pred_arr=y_pred_all,
        df_meta=df_meta,
        plot_dir=plot_dir,
        doc_dir=doc_dir,
        prefix="Keras-All-Data")

    mean_sam_all, stdw_sam_all, sam_values_all = evaluate_and_plot_sam(
        y_df=y_test,
        y_pred_arr=y_pred_test,
        df_meta=df_meta,
        plot_dir=plot_dir,
        doc_dir=doc_dir,
        prefix="Keras-Test-Data")



#########################################################################################################################
    # --- Kombinations-Export ---
    used_combinations = set(map(tuple, np.round(X_trainval_np, 4)))
    used_combinations.update(map(tuple, np.round(X_test_np, 4)))

    kombis_df = bayesian_sampling_with_uncertainty_all(
        X_trainval_np=X_trainval_np,
        X_test_np=X_test_np,
        unsicher_train=unsicher_train_best,
        unsicher_test=unsicher_test_best,
        parameter_names=list(X.columns),
        used_combinations=set(map(tuple, np.round(X_trainval_np, 4))) | set(map(tuple, np.round(X_test_np, 4))),
        n_points=800,

    )

    export_bayesian_kombis_to_csv(kombis_df, result_dir=result_dir)


if __name__ == "__main__":
    main()