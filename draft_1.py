import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference_sdk import InferenceHTTPClient
from sklearn.linear_model import LinearRegression



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

    # Reference white D65
    X_n, Y_n, Z_n = 95.047, 100.000, 108.883

    def xyz_to_lab(X, Y, Z):
        X /= X_n
        Y /= Y_n
        Z /= Z_n
        def f(t):
            return np.where(t > 0.008856, t**(1/3), (7.787 * t) + (16 / 116))
        fx, fy, fz = f(X), f(Y), f(Z)
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return L, a, b

    # Calculate average RGB, normalized RGB, convert to XYZ and LAB
    for center, radius, points, conf, number in numbered_circles:
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, thickness=-1)

        x, y = center
        roi = cv2.bitwise_and(original_image, original_image, mask=mask)
        roi = roi[y-radius:y+radius, x-radius:x+radius]
        mask_roi = mask[y-radius:y+radius, x-radius:x+radius]

        circle_pixels = roi[mask_roi == 255]
        avg_b = np.mean(circle_pixels[:, 0])
        avg_g = np.mean(circle_pixels[:, 1])
        avg_r = np.mean(circle_pixels[:, 2])

        normalised_r = avg_r / 255.0
        normalised_g = avg_g / 255.0
        normalised_b = avg_b / 255.0

        rgb = np.array([normalised_r, normalised_g, normalised_b])
        xyz = np.dot(rgb_to_xyz_matrix, rgb)
        X, Y, Z = xyz
        L, a_lab, b_lab = xyz_to_lab(X, Y, Z)

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
        r, g, b = row['Average R'], row['Average G'], row['Average B']
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
        return pd.Series({'H': hsv[0], 'S': hsv[1], 'V': hsv[2]})   
    
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

import pandas as pd

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

def prepare_correlation_analysis(df3, max_experiment_number):
    # Step 1: Filter by max experiment number
    df_filtered = df3[df3["Experiment"] <= max_experiment_number].copy()

    # Step 2: Group by 'Column Index' and calculate mean
    df4 = df_filtered.groupby("Column Index").mean(numeric_only=True).reset_index()

    # Step 3: Drop 'Experiment' if it exists
    df4 = df4.drop(columns=["Experiment"], errors='ignore')

    # Step 4: Compute correlation-related stats
    results = []

    x = df4["Column Index"]
    x_mean = x.mean()
    x_var = x.var(ddof=0)

    for col in df4.columns:
        if col == "Column Index":
            continue
        y = df4[col]
        y_mean = y.mean()
        y_var = y.var(ddof=0)
        y_std = np.sqrt(y_var)

        covariance = np.mean((x - x_mean) * (y - y_mean))
        correlation = covariance / (np.sqrt(x_var * y_var)) if x_var > 0 and y_var > 0 else np.nan

        results.append({
            "Parameter": col,
            "Mean": y_mean,
            "Variance": y_var,
            "Std Dev": y_std,
            "Correlation Coefficient": correlation
        })

    # Step 5: Sort results by absolute correlation descending
    results_df = pd.DataFrame(results)
    results_df["Abs Correlation"] = results_df["Correlation Coefficient"].abs()
    results_df = results_df.sort_values(by="Abs Correlation", ascending=False).reset_index(drop=True)
    results_df["Rank"] = results_df.index + 1

    # Step 6: Extract ordered list of parameter names by correlation strength
    correlation_ranking = results_df["Parameter"].tolist()

    return df4, results_df.drop(columns=["Abs Correlation"]), correlation_ranking

def groupwise_correlation(df4):
    import numpy as np
    
    groups = {
        "RGB": ["Average R", "Average G", "Average B"],
        "LAB": ["L", "A", "B"],
        "XYZ": ["X", "Y", "Z"],
        "HSV": ["H", "S", "V"]
    }

    group_corr = []
    for group_name, cols in groups.items():
        # Calculate correlation matrix for the group columns
        corr_matrix = df4[cols].corr()
        # Example: mean absolute correlation between columns as a proxy
        mean_corr = (np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])).mean()
        group_corr.append({
            "Group": group_name,
            "Mean Absolute Correlation": mean_corr
        })

    return pd.DataFrame(group_corr).sort_values(by="Mean Absolute Correlation", ascending=False)

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
    
    df4, correlation_stats, ranked_params = prepare_correlation_analysis(df3, max_experiment_number=4)

    print(" ----------------------------------------------------- Data Frame:  df4  -----------------------------------------------------")
    print(df4)
    print()
    print()
    
    print("\nCorrelation Statistics (ranked):")
    print(correlation_stats)
    print()
    print()
    
    groupwise_results = groupwise_correlation(df4)
    print(groupwise_results)