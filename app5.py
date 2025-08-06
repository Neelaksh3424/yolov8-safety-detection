import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import os
import glob
import cv2
import tempfile

# --------- Custom Class Names ---------
CUSTOM_NAMES = ['FireExtinguisher', 'ToolBox', 'OxygenTank']

# --------- Theme Colors and Fonts ---------
BG_MAIN = "#232949"
BG_PANEL = "#29304D"
ACCENT = "#5AC8FA"
TEXT_MAIN = "#FFFFFF"
HEADER_FONT = ("Montserrat", 18, "bold")
BODY_FONT = ("Montserrat", 12)
BTN_FONT = ("Montserrat", 12)
FOOTER_FONT = ("Montserrat", 9, "italic")

# --------- Model Load ---------
model = YOLO(r"best.pt")   # Update path as needed

# --------- App Window ---------
root = tk.Tk()
root.title("ASVA | Space Object Detector")
root.geometry("700x650")
root.configure(bg=BG_MAIN)

# --------- Header ---------
header = tk.Frame(root, bg=BG_MAIN)
header.pack(fill="x", pady=(25, 12), padx=22)
tk.Label(header, text="ASVA Space Object Detector", font=HEADER_FONT, fg=ACCENT, bg=BG_MAIN).pack(anchor="w")
tk.Label(header, text="Real-time detection of critical space station tools.", font=BODY_FONT, fg="#A7BFFC", bg=BG_MAIN).pack(anchor="w")

# --------- Image Preview Area ---------
img_panel = tk.Frame(root, bg=BG_PANEL, bd=0, relief=tk.RIDGE)
img_panel.pack(pady=10, padx=24, ipadx=3, ipady=3)
img_label = tk.Label(img_panel, bg=BG_PANEL, bd=0)
img_label.pack(padx=6, pady=7)

# --------- Detection Output ---------
result_panel = tk.Frame(root, bg=BG_MAIN)
result_panel.pack(pady=14, padx=30, fill="x")
tk.Label(result_panel, text="Detection Results", font=("Montserrat", 13, "bold"), fg=ACCENT, bg=BG_MAIN).pack(anchor="w", pady=(5, 3))
output_text = tk.Text(result_panel, height=5, width=54, bg=BG_PANEL, fg=TEXT_MAIN, font=BODY_FONT, bd=0, relief=tk.FLAT, highlightthickness=0)
output_text.pack()
output_text.config(state="disabled")

# --------- Camera State ---------
live_running = False
cap_live = None

# --------- Utility Functions ---------
def clear_previous_outputs(folder="runs/detect/predict"):
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

def display_result_image(result_path):
    try:
        img = Image.open(result_path)
        img = ImageOps.pad(img, (430, 310), color=BG_PANEL)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
    except Exception:
        output_text.insert(tk.END, "Error loading image.\n")
        output_text.config(state="disabled")

def handle_detection_results(results):
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        class_indices = [int(x) for x in results[0].boxes.cls]
        detected = []
        for i in class_indices:
            if 0 <= i < len(CUSTOM_NAMES):
                detected.append(CUSTOM_NAMES[i])
            else:
                detected.append(f"Unknown({i})")
        summary = {cls: detected.count(cls) for cls in CUSTOM_NAMES}
        for key in summary:
            if summary[key]:
                output_text.insert(tk.END, f"• {key}: {summary[key]}\n")
    else:
        output_text.insert(tk.END, "No objects detected in image.\n")
    output_text.config(state="disabled")

# --------- Detect from File ---------
def detect_image():
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    img_label.config(image=None)
    clear_previous_outputs()

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        output_text.config(state="disabled")
        return

    img = Image.open(file_path).convert("RGB")
    temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(temp_img.name, format="JPEG")
    temp_img.close()

    results = model.predict(
        source=temp_img.name,
        save=True,
        conf=0.25,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )
    save_dir = results[0].save_dir
    detected_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not detected_images:
        output_text.insert(tk.END, "No result image found.\n")
        output_text.config(state="disabled")
        btn_detect.config(text="Select Next Image")
        btn_camera.config(text="Capture Again")
        return

    result_path = os.path.join(save_dir, detected_images[0])
    display_result_image(result_path)
    handle_detection_results(results)
    btn_detect.config(text="Select Next Image")
    btn_camera.config(text="Capture Again")

# --------- Single Frame Camera Capture ---------

def capture_from_camera():
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    img_label.config(image=None)
    clear_previous_outputs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        output_text.insert(tk.END, "Could not open webcam.\n")
        output_text.config(state="disabled")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        output_text.insert(tk.END, "Failed to capture image from camera.\n")
        output_text.config(state="disabled")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(frame_rgb).save(temp_img.name, format="JPEG")
    temp_img.close()

    results = model.predict(
        source=temp_img.name,
        save=True,
        conf=0.25,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )
    save_dir = results[0].save_dir
    detected_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not detected_images:
        output_text.insert(tk.END, "No result image found.\n")
        output_text.config(state="disabled")
        btn_detect.config(text="Select Next Image")
        btn_camera.config(text="Capture Again")
        return

    result_path = os.path.join(save_dir, detected_images[0])
    display_result_image(result_path)
    handle_detection_results(results)
    btn_detect.config(text="Select Next Image")
    btn_camera.config(text="Capture Again")

# --------- Live Video Feed Detection ---------

def live_video_feed():
    global live_running, cap_live
    if not live_running:
        output_text.config(state="normal")
        output_text.delete(1.0, tk.END)
        img_label.config(image=None)
        clear_previous_outputs()
        cap_live = cv2.VideoCapture(0)
        if not cap_live.isOpened():
            output_text.insert(tk.END, "Could not open webcam.\n")
            output_text.config(state="disabled")
            cap_live = None
            return
        live_running = True
        btn_live.config(text="Stop Live Video Feed Detection")
        update_live_feed()
    else:
        stop_live_feed()

def stop_live_feed():
    global live_running, cap_live
    live_running = False
    btn_live.config(text="Live Video Feed Detection")
    if cap_live:
        cap_live.release()
        cap_live = None
    img_label.config(image=None)
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    output_text.config(state="disabled")

def update_live_feed():
    global live_running, cap_live
    if not live_running or cap_live is None:
        return
    ret, frame = cap_live.read()
    if not ret:
        stop_live_feed()
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, conf=0.25)
    annotated = results[0].plot()
    img = Image.fromarray(annotated)
    img = ImageOps.pad(img, (430, 310), color=BG_PANEL)
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk
    handle_detection_results(results)
    img_label.after(30, update_live_feed)

# --------- Buttons Frame ---------
btn_frame = tk.Frame(root, bg=BG_MAIN)
btn_frame.pack(pady=18)

btn_detect = tk.Button(
    btn_frame,
    text="Select Image and Detect",
    command=detect_image,
    font=BTN_FONT,
    bg=ACCENT,
    fg="#232323",
    activebackground="#499dd1",
    activeforeground="#fff",
    bd=0,
    relief=tk.FLAT,
    padx=10, pady=6, cursor="hand2"
)
btn_detect.pack(side="left", padx=10)

btn_camera = tk.Button(
    btn_frame,
    text="Capture with Camera",
    command=capture_from_camera,
    font=BTN_FONT,
    bg=ACCENT,
    fg="#232323",
    activebackground="#499dd1",
    activeforeground="#fff",
    bd=0,
    relief=tk.FLAT,
    padx=10, pady=6, cursor="hand2"
)
btn_camera.pack(side="left", padx=10)

btn_live = tk.Button(
    btn_frame,
    text="Live Video Feed Detection",
    command=live_video_feed,
    font=BTN_FONT,
    bg=ACCENT,
    fg="#232323",
    activebackground="#499dd1",
    activeforeground="#fff",
    bd=0,
    relief=tk.FLAT,
    padx=10, pady=6, cursor="hand2"
)
btn_live.pack(side="left", padx=10)

# --------- Optional: Stop Camera on Close ---------
def on_close():
    global cap_live, live_running
    stop_live_feed()
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_close)

# --------- Footer ---------
footer = tk.Frame(root, bg=BG_MAIN)
footer.pack(side="bottom", fill="x")
tk.Label(
    footer,
    text="Team ASVA  |  Space Station Object Detection  |  Hackathon 2025",
    font=FOOTER_FONT,
    fg="#A7BFFC",
    bg=BG_MAIN,
    anchor="e"
).pack(side="right", padx=10, pady=6)

root.mainloop()