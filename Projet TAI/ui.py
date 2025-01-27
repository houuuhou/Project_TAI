import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from sift import load_dataset_images, find_most_similar_image
#from sift_homography import load_dataset_images, find_most_similar_image
from ttkbootstrap import Style
from ttkbootstrap.constants import *

class TrafficSignMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Matcher")
        self.root.geometry("900x600")  # Set default window size
        self.center_window(900, 600)
        self.root.configure(bg="#f4f4f4")  # Set background color

        # Apply theme
        style = Style(theme="flatly")  # Choose from various themes like "darkly", "solar", "superhero", etc.
        style.configure("TButton", font=("Helvetica", 12), padding=6)
        style.configure("TLabel", font=("Helvetica", 12), background="#f4f4f4", padding=10)

        # Dataset and query image
        self.dataset_images = []
        self.image_names = []
        self.query_image = None

        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        # Frames for better organization
        top_frame = ttk.Frame(self.root, padding=15)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Add a heading at the top
        heading_label = ttk.Label(
            top_frame, 
            text="Traffic Sign Matcher 🚦",  # Add an emoji for flair if you like!
            font=("Helvetica", 20, "bold"),  # Larger and bold font
            anchor="center"  # Center-align text
        )
        heading_label.pack(fill=tk.X, pady=10)

        canvas_frame = ttk.Frame(self.root, padding=10)
        canvas_frame.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)

        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # Buttons
        self.upload_query_btn = ttk.Button(button_frame, text="Upload Query Image", command=self.upload_query_image)
        self.upload_query_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.load_dataset_btn = ttk.Button(button_frame, text="Load Dataset", command=self.load_dataset)
        self.load_dataset_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.find_match_btn = ttk.Button(button_frame, text="Find Match", command=self.find_match)
        self.find_match_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for Images
        self.query_canvas = ttk.Label(canvas_frame, text="Query Image", background="#e6e6e6", anchor="center")
        self.query_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.match_canvas = ttk.Label(canvas_frame, text="Best Match", background="#e6e6e6", anchor="center")
        self.match_canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def load_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_images, self.image_names = load_dataset_images(folder)
            messagebox.showinfo("Success", f"Loaded {len(self.dataset_images)} images.")

    def upload_query_image(self):
        file = filedialog.askopenfilename(title="Select Query Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file:
            self.query_image = cv2.imread(file)
            img = self.resize_image_to_canvas(self.query_image)
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

            # Update query canvas with the image
            self.query_canvas.config(image=img_tk, text="")  # Remove width/height
            self.query_canvas.image = img_tk  # Keep a reference to prevent garbage collection


    def find_match(self):
        if self.query_image is None or not self.dataset_images:
            messagebox.showwarning("Warning", "Load both query image and dataset first!")
            return

        try:
            best_image, best_image_name, best_keypoints, best_matches, query_keypoints = find_most_similar_image(
                self.query_image, self.dataset_images, self.image_names
            )

            if best_image is not None and best_matches:
                # Resize and display the matched image
                img = self.resize_image_to_canvas(best_image)
                img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.match_canvas.config(image=img_tk, text="")  # Update image and clear text
                self.match_canvas.image = img_tk  # Keep a reference to avoid garbage collection

                # Optional: Visualize matches 
                result_image = cv2.drawMatches(
                    self.query_image, query_keypoints, best_image, best_keypoints, 
                    best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(result_image)
                result_pil.show()  # Open visualization in a separate viewer
            else:
                messagebox.showinfo("Result", "No matching image found.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def resize_image_to_canvas(self, img, max_size=(400, 300)):
        """Resize image to fit within the given max size while maintaining aspect ratio."""
        h, w = img.shape[:2]
        max_width, max_height = max_size

        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    def center_window(self, width, height):
        """Center the window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignMatcherApp(root)
    root.mainloop()
