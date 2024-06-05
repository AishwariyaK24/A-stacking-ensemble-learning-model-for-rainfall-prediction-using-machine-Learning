import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from PIL import Image, ImageTk

# Load trained models
rf_model = joblib.load('rf_model.pkl')
catboost_model = joblib.load('catboost_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lgbm_model = joblib.load('lgbm_model.pkl')
gb_model = joblib.load('meta_model.pkl')

def predict():
    try:
        # Get user inputs
        temp = float(entry_temp.get())
        dewp = float(entry_dewp.get())
        visib = float(entry_visib.get())
        wdsp = float(entry_wdsp.get())
        mxspd = float(entry_mxspd.get())
        max_temp = float(entry_max_temp.get())
        min_temp = float(entry_min_temp.get())
        day = int(day_entry.get())
        month = int(month_entry.get())
        year = int(year_entry.get())
        
        # Make predictions using the loaded models
        features = np.array([[temp, dewp, visib, wdsp, mxspd, max_temp, min_temp, day, month, year]])
        rf_pred = rf_model.predict(features)
        catboost_pred = catboost_model.predict(features)
        xgb_pred = xgb_model.predict(features)
        lgbm_pred = lgbm_model.predict(features)
        
        # Combine base model predictions
        base_model_predictions = np.column_stack((rf_pred, catboost_pred, xgb_pred, lgbm_pred))
        
        # Use meta-model to make final prediction
        final_pred = gb_model.predict(base_model_predictions)[0]
        
        # Format final prediction to display one decimal point
        formatted_pred = f"{final_pred:.1f}"
        
        # Display formatted prediction
        prediction_label.config(text=f"Final Rainfall Prediction: {formatted_pred} mm")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create GUI window
root = tk.Tk()
root.title("Rainfall Prediction")

# Set the window size to 800x600 pixels
root.geometry("1400x1100")

# Load and resize the background image
bg_image = Image.open("rain.jpg")
bg_image = bg_image.resize((1400, 1100), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label with the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Define a custom font for the title
title_font = ('TIMES NEW ROMAN', 24, 'bold')

# Label for the title
title_label = tk.Label(root, text="Rainfall Prediction", font=title_font)
title_label.place(relx=0.5, rely=0.03, anchor='center')

# Define a custom font for labels
label_font = ('TIMES NEW ROMAN', 16, 'bold')

# Create input labels and entry fields for each feature with increased width and font size
entry_width = 15
entry_font = ('TIMES NEW ROMAN', 16,'bold')

tk.Label(root, text="Temperature:", font=label_font).place(relx=0.2, rely=0.3, anchor='e')
entry_temp = tk.Entry(root, width=entry_width, font=entry_font)
entry_temp.place(relx=0.3, rely=0.3, anchor='w')

tk.Label(root, text="Dew Point:", font=label_font).place(relx=0.2, rely=0.4, anchor='e')
entry_dewp = tk.Entry(root, width=entry_width, font=entry_font)
entry_dewp.place(relx=0.3, rely=0.4, anchor='w')

tk.Label(root, text="Visibility:", font=label_font).place(relx=0.2, rely=0.5, anchor='e')
entry_visib = tk.Entry(root, width=entry_width, font=entry_font)
entry_visib.place(relx=0.3, rely=0.5, anchor='w')

tk.Label(root, text="Wind Speed:", font=label_font).place(relx=0.2, rely=0.6, anchor='e')
entry_wdsp = tk.Entry(root, width=entry_width, font=entry_font)
entry_wdsp.place(relx=0.3, rely=0.6, anchor='w')

tk.Label(root, text="Max Wind Speed:", font=label_font).place(relx=0.2, rely=0.7, anchor='e')
entry_mxspd = tk.Entry(root, width=entry_width, font=entry_font)
entry_mxspd.place(relx=0.3, rely=0.7, anchor='w')

tk.Label(root, text="Max Temperature:", font=label_font).place(relx=0.2, rely=0.8, anchor='e')
entry_max_temp = tk.Entry(root, width=entry_width, font=entry_font)
entry_max_temp.place(relx=0.3, rely=0.8, anchor='w')

tk.Label(root, text="Min Temperature:", font=label_font).place(relx=0.2, rely=0.9, anchor='e')
entry_min_temp = tk.Entry(root, width=entry_width, font=entry_font)
entry_min_temp.place(relx=0.3, rely=0.9, anchor='w')

tk.Label(root, text="Day:", font=label_font).place(relx=0.2, rely=0.1, anchor='e')
day_entry = tk.Entry(root, width=entry_width, font=entry_font)
day_entry.place(relx=0.3, rely=0.1, anchor='w')

tk.Label(root, text="Month:", font=label_font).place(relx=0.2, rely=0.16, anchor='e')
month_entry = tk.Entry(root, width=entry_width, font=entry_font)
month_entry.place(relx=0.3, rely=0.16, anchor='w')

tk.Label(root, text="Year:", font=label_font).place(relx=0.2, rely=0.22, anchor='e')
year_entry = tk.Entry(root, width=entry_width, font=entry_font)
year_entry.place(relx=0.3, rely=0.22, anchor='w')

# Create a label for displaying predictions
prediction_label = tk.Label(root, font=label_font)
prediction_label.place(relx=0.7, rely=0.9, anchor='center')

# Create a custom font for the button
button_font = ('TIMES NEW ROMAN', 16, 'bold')

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict, font=button_font)
predict_button.place(relx=0.7, rely=0.8, anchor='center')

root.mainloop()
