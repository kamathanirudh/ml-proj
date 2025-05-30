import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference_sdk import InferenceHTTPClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2hsv

plt.rcParams.update({'font.size': 8}) 


def detect_circles_yolo(image_path, rows, cols, api_key, model_id):
    """
    Detect circles using Roboflow YOLO model, extract RGB and color info,
    plot graphs, and save results to Excel.

    Args:
        image_path (str): Path to input image.
        rows (int): Number of rows of circles expected.
        cols (int): Number of columns of circles expected.
        api_key (str): Roboflow API key.
        model_id (str): Roboflow model id.

    Returns:
        list: List of tuples containing (center, reduced_radius, points, confidence, number).
    """
    CONFIDENCE_THRESHOLD = 0.65
    REDUCTION_FACTOR = 0.80

    # Initialize Roboflow client
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)

    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return []

    # Run inference
    result = client.infer(image_path, model_id=model_id)

    # Extract detected circles with confidence filter
    detected_circles = []
    for pred in result['predictions']:
        confidence = pred['confidence']
        class_name = pred['class'].lower()

        if class_name == "cricle" and confidence >= CONFIDENCE_THRESHOLD:
            points = pred['points']
            polygon_points = np.array([(p['x'], p['y']) for p in points], dtype=np.int32).reshape((-1, 1, 2))

            (x, y), radius = cv2.minEnclosingCircle(polygon_points)
            reduced_radius = max(1, int(radius * REDUCTION_FACTOR))
            center = (int(x), int(y))
            detected_circles.append((center, reduced_radius, points, confidence))

    # Sort circles top-left to bottom-right by y, then x
    detected_circles.sort(key=lambda c: (c[0][1], c[0][0]))

    # Assign numbering
    numbered_circles = [(c[0], c[1], c[2], c[3], i+1) for i, c in enumerate(detected_circles)]

    # Show individual ROI images
    fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
    axs = axs.flatten()  # flatten in case axs is 2D

    for i, (center, radius, points, conf, number) in enumerate(numbered_circles):
        if i >= rows * cols:
            break
        ax = axs[i]
        # Extract ROI and plot on ax
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, thickness=-1)
        roi = cv2.bitwise_and(original_image, original_image, mask=mask)
        x, y = center
        roi = roi[y-radius:y+radius, x-radius:x+radius]
        ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"Circle {number}\nConf: {conf:.2f}", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Draw detections on original image
    image_with_detections = original_image.copy()
    for center, radius, points, conf, number in numbered_circles:
        polygon_points = np.array([(p['x'], p['y']) for p in points], dtype=np.int32)
        cv2.polylines(image_with_detections, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(image_with_detections, center, radius, (255, 0, 0), 2)
        cv2.putText(image_with_detections, f"{conf:.2f}", (center[0], center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image_with_detections, f"#{number}", (center[0] - 20, center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Initialize DataFrame
    df = pd.DataFrame(columns=['Circle Number', 'Average R', 'Average G', 'Average B',
                               'Normalised R', 'Normalised G', 'Normalised B',
                               'X', 'Y', 'Z', 'L', 'A', 'B'])

    # RGB to XYZ conversion matrix (sRGB)

    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    #D65
    
    '''sRGB D65 to D50 XYZ  0.4360747  0.3850649  0.1430804
                            0.2225045  0.7168786  0.0606169
                            0.0139322  0.0971045  0.7141733
        
        
        sRGB to D65 XYZ     0.4124564  0.3575761  0.1804375
                            0.2126729  0.7151522  0.0721750
                            0.0193339  0.1191920  0.9503041'''
    
    
    
    '''D65	95.047	100.000	108.883	~6500K (Daylight)
        D50	96.421	100.000	82.519	~5000K (Warmer)'''
    
    
    
    
    
    def xyz_to_lab(xyz, white_point=(95.047, 100.0, 108.883)): #D65 White Point
        # Unpack input XYZ and white point values
        X, Y, Z = xyz*100
        Xr, Yr, Zr = white_point

        # Normalize by reference white
        xr = X / Xr
        yr = Y / Yr
        zr = Z / Zr

        # Constants from CIE standard
        epsilon = 216 / 24389  # ≈ 0.008856
        kappa = 24389 / 27     # ≈ 903.3

        # Helper function as per condition
        def f(t):
            if t > epsilon:
                return t ** (1/3)
            else:
                return (kappa * t + 16) / 116

        fx = f(xr)
        fy = f(yr)
        fz = f(zr)

        # Final Lab computation
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return L, a, b




    # def avg_rgb_to_lab(avg_r, avg_g, avg_b):
    #     """
    #     Convert average R, G, B values (0-255) directly to L*a*b* using OpenCV,
    #     and rescale it to the standard CIE L*a*b* ranges.
        
    #     Returns:
    #     - L in [0, 100]
    #     - a, b approximately in [-128, 127]
    #     """
    #     rgb_pixel = np.array([[[avg_r, avg_g, avg_b]]], dtype=np.uint8)
    #     lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2Lab)[0, 0]

    #     # OpenCV returns L in [0, 255]; scale it to [0, 100]
    #     L = lab_pixel[0] * (100.0 / 255.0)

    #     # a and b are offset by 128 to fit into [0, 255]
    #     a = lab_pixel[1] - 128.0
    #     b = lab_pixel[2] - 128.0

    #     return L, a, b


    
        
        
    # Calculate average RGB, normalized RGB, convert to XYZ and LAB
    for center, radius, points, conf, number in numbered_circles:
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, thickness=-1)

        x, y = center
        roi = cv2.bitwise_and(original_image, original_image, mask=mask)
        roi = roi[y-radius:y+radius, x-radius:x+radius]
        mask_roi = mask[y-radius:y+radius, x-radius:x+radius]

                # Assume roi and mask_roi are defined and loaded properly
        circle_pixels = roi[mask_roi == 255]

        # Extract mean BGR channels (OpenCV format)
        avg_b = np.mean(circle_pixels[:, 0])
        avg_g = np.mean(circle_pixels[:, 1])
        avg_r = np.mean(circle_pixels[:, 2])
        # L, a_lab, b_lab= avg_rgb_to_lab(avg_r,avg_g,avg_b)
        # Normalize to [0, 1]
        normalised_r = avg_r / 255.0
        normalised_g = avg_g / 255.0
        normalised_b = avg_b / 255.0
        
        # Create RGB array
        rgb = np.array([normalised_r, normalised_g, normalised_b])
        
        def inverse_srgb_companding(V):
            """Apply inverse sRGB companding to each channel (V in [0,1])"""
            return np.where(V <= 0.04045,
                            V / 12.92,
                            ((V + 0.055) / 1.055) ** 2.4)

        # Apply inverse sRGB companding to get linear RGB
        linear_rgb = inverse_srgb_companding(rgb)

        # Convert linear RGB to XYZ
        xyz = np.dot(rgb_to_xyz_matrix, linear_rgb)
        X, Y, Z = xyz
        
        # def adapt_D65_to_D50(xyz):
        #     M = np.array([[ 0.9555766, -0.0230393,  0.0631636],
        #                 [-0.0282895,  1.0099416,  0.0210077],
        #                 [ 0.0122982, -0.0204830,  1.3299098]])
        #     return np.dot(M, xyz)
        
        # xyz = adapt_D65_to_D50(xyz)

        
        L, a_lab, b_lab = xyz_to_lab(xyz)

        new_row = {'Circle Number': number, 'Average R': avg_r, 'Average G': avg_g, 'Average B': avg_b,
                   'Normalised R': normalised_r, 'Normalised G': normalised_g, 'Normalised B': normalised_b,
                   'X': X, 'Y': Y, 'Z': Z, 'L': L, 'A': a_lab, 'B': b_lab}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Define concentration values (assuming uniform increments)
    conc_values = np.linspace(10, cols * 10, cols)

    # Prepare experiments data dict for plotting
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
    if rows > len(colors):
        raise ValueError("Not enough predefined colors for the number of rows")
    
    def rgb_to_hsv(row):
        r, g, b = row['Average R'] / 255.0, row['Average G'] / 255.0, row['Average B'] / 255.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Hue calculation
        if delta == 0:
            h = 0
        elif cmax == r:
            h = (60 * ((g - b) / delta)) % 360
        elif cmax == g:
            h = (60 * ((b - r) / delta + 2))
        else:  # cmax == b
            h = (60 * ((r - g) / delta + 4))

        # Saturation calculation
        s = 0 if cmax == 0 else (delta / cmax)

        # Value calculation
        v = cmax

        # Return values in expected format: H in degrees, S and V in percentage
        return pd.Series({'H': h, 'S': s * 100, 'V': v * 100})
    hsv_df = df.apply(rgb_to_hsv, axis=1)
    df = pd.concat([df, hsv_df], axis=1)

    
    experiments = {}
    for i in range(rows):
        start_idx = i * cols
        end_idx = start_idx + cols
        experiments[f"Experiment {i+1} (Circles {start_idx+1}-{end_idx})"] = {
            'color': colors[i],
            'data': df.iloc[start_idx:end_idx]
        }
        
    # Convert RGB to HSV for each row in df and add columns
    



    def plot_measurements_grid(experiments, conc_values):
        ylabels = [
            'X Value', 'Y Value', 'Z Value',
            'Average R', 'Average G', 'Average B',
            'Hue', 'Saturation', 'Value',
            'L', 'A', 'B'
        ]
        columns = [
            'X', 'Y', 'Z',
            'Average R', 'Average G', 'Average B',
            'H', 'S', 'V',
            'L', 'A', 'B'
        ]

        fig, axs = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows x 3 columns grid
        axs = axs.flatten()

        for ax, col, ylabel in zip(axs, columns, ylabels):
            for label, exp in experiments.items():
                ax.plot(conc_values, exp['data'][col], marker='o', linestyle='-', linewidth=2, color=exp['color'], label=label)
            ax.set_xlabel('Concentration')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Concentration')
            ax.legend(fontsize=8)
            ax.grid(True)

        plt.tight_layout()
        plt.show()


    plot_measurements_grid(experiments, conc_values)
    
    return numbered_circles, df

def delta_e_calc(df, rows, cols):
    total = rows * cols
    if len(df) < total:
        raise ValueError("Not enough rows in DataFrame for given grid dimensions.")
    
    # Assign column index: repeats [0, 1, ..., cols-1] for each row
    df['column index'] = [i % cols for i in range(total)]
    
    # Assign experiment: increases every 'cols' rows
    df['experiment'] = [i // cols + 1 for i in range(total)]

    # Keep only needed columns
    df_filtered = df.copy()

    # Initialize a list to store delta E for each row
    delta_e_values = []

    for exp in range(1, rows + 1):
        exp_df = df_filtered[df_filtered['experiment'] == exp]
        ref_row = exp_df[exp_df['column index'] == 0]

        if ref_row.empty:
            raise ValueError(f"Reference (column index 0) not found for experiment {exp}")

        ref_L, ref_A, ref_B = ref_row.iloc[0][['L', 'A', 'B']]

        for _, row in exp_df.iterrows():
            dL = row['L'] - ref_L
            dA = row['A'] - ref_A
            dB = row['B'] - ref_B
            delta_e = np.sqrt(dL**2 + dA**2 + dB**2)
            delta_e_values.append(delta_e)

    # Add delta E to the DataFrame
    df_filtered['Delta E'] = delta_e_values

    df_filtered.drop(['experiment', 'column index'], axis=1, inplace=True)
    return df_filtered
    

def gray_scale_conv(df):
    df['Grayscale'] = (
        0.2989 * df['Average R'] +
        0.5870 * df['Average G'] +
        0.1140 * df['Average B']
    )
    return df




def plot_wo_blank_value(df, rows, cols, concentrations):
    # Drop unnecessary columns
    df2 = df.drop(columns=['Normalised R', 'Normalised G', 'Normalised B'], errors='ignore')

    # Compute Column Index and filter out blank column
    df2["Column Index"] = (df2["Circle Number"] - 1) % cols
    df2 = df2[df2["Column Index"] != 0]

    # Final filtered dataframe
    filtered_df = df2.drop(columns=["Column Index"], errors='ignore')

    # Parameters to plot (excluding Circle Number)
    parameters = [col for col in filtered_df.columns if col != "Circle Number"]

    # ---------- 1. Line Plot Grid ----------
    max_cols = 3
    num_params = len(parameters)
    num_rows = -(-num_params // max_cols)  # Ceiling division

    fig1, axes1 = plt.subplots(num_rows, max_cols, figsize=(15, 5 * num_rows))
    axes1 = axes1.flatten()

    for i, param in enumerate(parameters):
        ax = axes1[i]

        for exp in range(rows):
            start = exp * (cols - 1)
            end = (exp + 1) * (cols - 1)
            exp_data = filtered_df.iloc[start:end]

            ax.plot(concentrations, exp_data[param], marker='o', linestyle='-', label=f"Experiment {exp+1}")

        ax.set_title(f"{param} vs. Concentration")
        ax.set_xlabel("Concentration")
        ax.set_ylabel(param)
        ax.legend()
        ax.grid(True)

    # Hide unused axes
    for j in range(i + 1, len(axes1)):
        fig1.delaxes(axes1[j])
    plt.tight_layout()
    plt.show()

    # ---------- 2. Scatter Plot Grid (Flipped Axes) ----------
    fig2, axes2 = plt.subplots(num_rows, max_cols, figsize=(15, 5 * num_rows))
    axes2 = axes2.flatten()

    for i, param in enumerate(parameters):
        ax = axes2[i]

        for exp in range(rows):
            start = exp * (cols - 1)
            end = (exp + 1) * (cols - 1)
            exp_data = filtered_df.iloc[start:end]

            ax.scatter(exp_data[param], concentrations, marker='o', label=f"Experiment {exp+1}")

        ax.set_title(f"Concentration vs. {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Concentration")
        ax.legend()
        ax.grid(True)

    for j in range(i + 1, len(axes2)):
        fig2.delaxes(axes2[j])
    plt.tight_layout()
    plt.show()

    return df2, filtered_df



def add_experiment_column(df2, total_experiments, cols_per_experiment):
    # Make a copy of df2 to avoid modifying original
    df3 = df2.copy().reset_index(drop=True)

    # Total number of data points per experiment
    rows_per_experiment = cols_per_experiment

    # Sanity check: make sure the expected number of rows is correct
    expected_rows = total_experiments * rows_per_experiment
    actual_rows = len(df3)

    if actual_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows (from {total_experiments} experiments with {rows_per_experiment} points), "
                         f"but got {actual_rows} rows in df2.")

    # Assign experiment numbers
    experiment_numbers = []
    for i in range(total_experiments):
        experiment_numbers.extend([i + 1] * rows_per_experiment)

    df3["Experiment"] = experiment_numbers

    return df3

# def prepare_correlation_analysis(df3, max_experiment_number):
#     # Step 1: Filter by max experiment number
#     df_filtered = df3[df3["Experiment"] <= max_experiment_number].copy()

#     # Step 2: Group by 'Column Index' and calculate mean
#     df4 = df_filtered.groupby("Column Index").mean(numeric_only=True).reset_index()

#     # Step 3: Drop 'Experiment' if it exists
#     df4 = df4.drop(columns=["Experiment"], errors='ignore')

#     # Step 4: Compute correlation-related stats
#     results = []

#     x = df4["Column Index"]
#     x_mean = x.mean()
#     x_var = x.var(ddof=0)

#     for col in df4.columns:
#         if col == "Column Index":
#             continue
#         y = df4[col]
#         y_mean = y.mean()
#         y_var = y.var(ddof=0)
#         y_std = np.sqrt(y_var)

#         covariance = np.mean((x - x_mean) * (y - y_mean))
#         correlation = covariance / (np.sqrt(x_var * y_var)) if x_var > 0 and y_var > 0 else np.nan

#         results.append({
#             "Parameter": col,
#             "Mean": y_mean,
#             "Variance": y_var,
#             "Std Dev": y_std,
#             "Correlation Coefficient": correlation
#         })

#     # Step 5: Sort results by absolute correlation descending
#     results_df = pd.DataFrame(results)
#     results_df["Abs Correlation"] = results_df["Correlation Coefficient"].abs()
#     results_df = results_df.sort_values(by="Abs Correlation", ascending=False).reset_index(drop=True)
#     results_df["Rank"] = results_df.index + 1

#     # Step 6: Extract ordered list of parameter names by correlation strength
#     correlation_ranking = results_df["Parameter"].tolist()

#     return df4, results_df.drop(columns=["Abs Correlation"]), correlation_ranking

def parameter_correlaton(df3):
    X = df3['Column Index'].values  # This is the concentration

    features = ['Average R', 'Average G', 'Average B', 'X', 'Y', 'Z', 'L', 'A', 'B','H','S','V',"Grayscale",'Delta E']

    manual_correlations = {}

    for feature in features:
        Y = df3[feature].values

        mean_X = np.mean(X)
        mean_Y = np.mean(Y)

        numerator = np.sum((X - mean_X) * (Y - mean_Y))
        denominator = np.sqrt(np.sum((X - mean_X)**2) * np.sum((Y - mean_Y)**2))

        r = numerator / denominator
        manual_correlations[feature] = r

    # Convert results to a DataFrame
    manual_corr_df = pd.DataFrame.from_dict(manual_correlations, orient='index', columns=['Correlation Coefficient'])
    manual_corr_df['r²'] = manual_corr_df['Correlation Coefficient'] ** 2
    manual_corr_df['Abs'] = manual_corr_df['Correlation Coefficient'].abs()
    manual_corr_df = manual_corr_df.sort_values(by='Abs', ascending=False).drop(columns='Abs')
    return manual_corr_df


def parameter_linear_regression_r2(df3):
    target = 'Column Index'
    features = ['Average R', 'Average G', 'Average B', 'X', 'Y', 'Z', 'L', 'A', 'B','H','S','V','Grayscale','Delta E']

    r2_scores = {}

    for feature in features:
        X = df3[[feature]].values  # Note: 2D array for sklearn
        y = df3[target].values

        # Fit linear regression
        model = LinearRegression().fit(X, y)

        # Calculate R²
        r2 = model.score(X, y)

        r2_scores[feature] = r2

    # Convert to DataFrame
    r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R²'])
    r2_df = r2_df.sort_values(by='R²', ascending=False)
    return r2_df


def groupwise_r2(df, conc='Column Index'):
    groups = {
        "RGB": ["Average R", "Average G", "Average B"],
        "LAB": ["L", "A", "B"],
        "XYZ": ["X", "Y", "Z"],
        "HSV": ["H", "S", "V"]
    }

    results = []
    for group_name, cols in groups.items():
        X = df[cols].values  # 3 features (e.g., R, G, B)
        y = df[conc].values  # concentration values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)  # R² value
        results.append({
            "Group": group_name,
            "R² (Linearity vs Concentration)": r2
        })

    return pd.DataFrame(results).sort_values(by="R² (Linearity vs Concentration)", ascending=False)


def groupwise_cca(df, target='Column Index'):
    """
    Compute canonical correlation between groups of variables (e.g., RGB, LAB, HSV, XYZ)
    and a single target variable using Canonical Correlation Analysis (CCA).
    
    Parameters:
    - df: pd.DataFrame, dataset containing both independent and group variables
    - target: str, name of the target column to correlate with groups

    Returns:
    - pd.DataFrame sorted by canonical correlation (descending)
    """
    
    # Define variable groups
    color_groups = {
        "RGB": ["Average R", "Average G", "Average B"],
        "LAB": ["L", "A", "B"],
        "XYZ": ["X", "Y", "Z"],
        "HSV": ["H", "S", "V"]
    }

    results = []

    for group_name, group_cols in color_groups.items():
        try:
            # Extract and scale X (group features) and Y (target)
            X = df[group_cols].values
            Y = df[[target]].values  # Keep as 2D for CCA
            
            scaler_x = StandardScaler().fit(X)
            scaler_y = StandardScaler().fit(Y)
            X_scaled = scaler_x.transform(X)
            Y_scaled = scaler_y.transform(Y)

            # Perform CCA
            cca = CCA(n_components=1)
            X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

            # Compute canonical correlation
            corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
            results.append({
                "Group": group_name,
                "Canonical Correlation": abs(corr)
            })

        except Exception as e:
            print(f"Error processing group {group_name}: {e}")
            continue

    return pd.DataFrame(results).sort_values(by="Canonical Correlation", ascending=False).reset_index(drop=True)


def groupwise_correlation_ranks(df, target='Column Index'):
    groups = {
        "RGB": ["Average R", "Average G", "Average B"],
        "LAB": ["L", "A", "B"],
        "XYZ": ["X", "Y", "Z"],
        "HSV": ["H", "S", "V"]
    }

    result = []

    for group_name, cols in groups.items():
        abs_corrs = []
        squared_corrs = []
        for col in cols:
            x = df[target].values
            y = df[col].values
            mean_x, mean_y = np.mean(x), np.mean(y)
            num = np.sum((x - mean_x) * (y - mean_y))
            den = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
            r = num / den
            abs_corrs.append(abs(r))
            squared_corrs.append(r ** 2)

        mean_abs_corr = np.mean(abs_corrs)
        mean_squared_corr = np.mean(squared_corrs)

        result.append({
            "Group": group_name,
            "Mean |r|": mean_abs_corr,
            "Mean r²": mean_squared_corr
        })

    df_result = pd.DataFrame(result)
    df_result["|r| Rank"] = df_result["Mean |r|"].rank(ascending=False).astype(int)
    df_result["r² Rank"] = df_result["Mean r²"].rank(ascending=False).astype(int)

    return df_result.sort_values(by="|r| Rank")


def groupwise_correlation_secondtry(df, target='Column Index'):
    groups = {
        "RGB": ["Average R", "Average G", "Average B"],
        "LAB": ["L", "A", "B"],
        "XYZ": ["X", "Y", "Z"],
        "HSV": ["H", "S", "V"]
    }

    group_corr = []
    for group_name, cols in groups.items():
        correlations = []
        r2_values = []
        for col in cols:
            x = df[target].values
            y = df[col].values
            mean_x, mean_y = np.mean(x), np.mean(y)
            num = np.sum((x - mean_x) * (y - mean_y))
            den = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
            r = num / den
            correlations.append(abs(r))
            r2_values.append(r**2)
        
        mean_abs_corr = np.mean(correlations)
        mean_r2 = np.mean(r2_values)

        group_corr.append({
            "Group": group_name,
            "Mean Abs Corr to Target": mean_abs_corr,
            "Mean R^2": mean_r2,
            "Feature-wise Abs Corr": dict(zip(cols, correlations)),
            "Feature-wise R^2": dict(zip(cols, r2_values))
        })

    return pd.DataFrame(group_corr).sort_values(by="Mean Abs Corr to Target", ascending=False)



if __name__ == "__main__":
    api_key = "y6u7lWMqSmjyVGQh4KJJ"
    model_id = "my-first-project-xz1fx/5"  # e.g. "yourusername/yourmodel/1"
    image_path = "test3.jpg"
    rows = 4  # number of rows in your grid
    cols = 5  # number of columns in your grid
    concentrations = [0, 0.25, 0.5, 0.75]  # For example, if cols=5

    numbered_circles, df = detect_circles_yolo(image_path, rows, cols, api_key, model_id)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    print(" ----------------------------------------------------- Data Frame:  df -----------------------------------------------------")
    print(df)
    print()
    print()
    
    df_delta_e = delta_e_calc(df, 4,5)
    print(" ----------------------------------------------------- Data Frame:  df_delta_e -----------------------------------------------------")
    print(df_delta_e)
    df=gray_scale_conv(df_delta_e)
    df2,filtered_df=plot_wo_blank_value(df, rows, cols, concentrations)
    
    print(" ----------------------------------------------------- Data Frame:  df2  -----------------------------------------------------")
    print(df2)
    print()
    print()
    
    
    
    
    print(" ----------------------------------------------------- Data Frame:  filtered_df  -----------------------------------------------------")
    print(filtered_df)
    print()
    print()
    
    
    df3 = add_experiment_column(df2, total_experiments=4, cols_per_experiment=4)
    
    print(" ----------------------------------------------------- Data Frame:  df3  -----------------------------------------------------")
    print(df3)
    print()
    print()
    
    # df4, correlation_stats, ranked_params = prepare_correlation_analysis(df3, max_experiment_number=4)

    # print(" ----------------------------------------------------- Data Frame:  df4  -----------------------------------------------------")
    # print(df4)
    # print()
    # print()
    
    # print("\nCorrelation Statistics (ranked):")
    # print(correlation_stats)
    # print()
    # print()
    
    manual_corr_df=parameter_correlaton(df3)
    print(" ----------------------------------------------------- Data Frame:  individidual_parameter_corrcoeff  -----------------------------------------------------")
    print(manual_corr_df)
    print()
    print()
    
    
    r2_df = parameter_linear_regression_r2(df3)
    print(" ----------------------------------------------------- Data Frame:  individual_parameter_r2_linearregg  -----------------------------------------------------")
    print(r2_df)
    print()
    print()
    
    
    
    groupwise_r2=groupwise_r2(df3)
    
    print(" ----------------------------------------------------- Data Frame:  groupwise_r2  -----------------------------------------------------")
    print(groupwise_r2)
    print()
    print()
    
    # Assuming df3 is already defined as your DataFrame:
    result_df = groupwise_cca(df3, target='Column Index')
    print(" ----------------------------------------------------- Data Frame:  groupwise_CCA  -----------------------------------------------------")
    print(result_df)
    print()
    print()
    
    
    print(" ----------------------------------------------------- Data Frame:  groupwise_corr_coeff  -----------------------------------------------------")
    correlation_ranks = groupwise_correlation_ranks(df3)
    print(correlation_ranks)
    print()
    print()
    
    print(" ----------------------------------------------------- Data Frame:  groupwise_corr_coeff_second_try -----------------------------------------------------")
    result = groupwise_correlation_secondtry(df3, target='Column Index')
    print(result)
    print()
    print()
