from tkinter import BooleanVar, Checkbutton, Tk, Label, Button, Entry, Toplevel
from PIL import ImageGrab, Image, ImageTk
import time
import os
import pandas as pd

from Sam import remove_background_sam

# Define paths
output_folder = r"C:\Users\user\Desktop"
csv_file = r"C:\Users\user\Desktop\Pokemon-Smash-or-Pass\pokemon_data.csv"

current_image = None
current_image_name = None
foreground_point = None  # Global



# Initialize DataFrame
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=['filename', 'score'])

# Screenshot function using Toplevel instead of new Tk()
def take_screenshot():
    capture_result = {}

    overlay = Toplevel(root)
    overlay.attributes('-fullscreen', True)
    overlay.attributes('-alpha', 0.3)
    overlay.config(bg='black')

    def on_mouse_down(event):
        capture_result['start_x'], capture_result['start_y'] = event.x, event.y

    def on_mouse_up(event):
        end_x, end_y = event.x, event.y
        overlay.destroy()

        x1 = min(capture_result['start_x'], end_x)
        y1 = min(capture_result['start_y'], end_y)
        x2 = max(capture_result['start_x'], end_x)
        y2 = max(capture_result['start_y'], end_y)

        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        timestamp = int(time.time())
        image_name = f"screenshot_{timestamp}.png"

        global current_image, current_image_name
        current_image = img
        current_image_name = image_name

        show_image(img)


    overlay.bind("<ButtonPress-1>", on_mouse_down)
    overlay.bind("<ButtonRelease-1>", on_mouse_up)

def save_score(image_name, score):
    global df, current_image
    path = os.path.join(output_folder, image_name)
    current_image.save(path)

    df.loc[len(df)] = {'Name': image_name, 'Smash Normalized': score}
    print(df.tail())
    df.to_csv(csv_file, index=False)
    print(f"Image and score for {image_name} saved.")

def show_image(image_obj):
    global current_image, current_image_name, foreground_point

    current_image = image_obj
    image = image_obj.copy()

    # Use background removal if checkbox is active
    if show_transparent.get():
        image = remove_background_sam(
            image,
            foreground_point=foreground_point,
            invert=invert_mask.get()
        )

    image = image.resize((300, 300))
    img = ImageTk.PhotoImage(image)

    img_label.config(image=img)
    img_label.image = img  # Keep reference



def submit_score():
    try:
        score = float(score_entry.get())
        if 0 <= score <= 10:
            save_score(current_image_name, score)
            score_entry.delete(0, 'end')
        else:
            print("Score must be between 0 and 10.")
    except ValueError:
        print("Invalid input. Enter a number.")




def on_image_click(event):
    global foreground_point
    if current_image:
        original_w, original_h = current_image.size
        x = int(event.x * original_w / 300)
        y = int(event.y * original_h / 300)
        foreground_point = [x, y]
        print(f"Foreground point set to: {foreground_point}")
        show_image(current_image)



# GUI setup
root = Tk()
root.title("Pokemon Smash or Pass Tool")
root.geometry("400x500")


# Toggle for transparent background
show_transparent = BooleanVar(value=False)
toggle_button = Checkbutton(root, text="Remove Background", variable=show_transparent, command=lambda: show_image(current_image))
toggle_button.pack(pady=5)

invert_mask = BooleanVar(value=False)
invert_checkbox = Checkbutton(root, text="Invert Mask", variable=invert_mask, command=lambda: show_image(current_image))
invert_checkbox.pack(pady=5)

img_label = Label(root)
img_label.pack()
img_label.bind("<Button-1>", on_image_click)  # <== This is the correct spot

score_entry = Entry(root)
submit_button = Button(root, text="Submit Score")

screenshot_button = Button(root, text="Take Screenshot", command=take_screenshot)
screenshot_button.pack(pady=10)

score_entry = Entry(root)
score_entry.pack(pady=10)
score_entry.bind('<Return>', lambda event: submit_score())

submit_button = Button(root, text="Submit Score", command=submit_score)
submit_button.pack(pady=10)

root.mainloop()