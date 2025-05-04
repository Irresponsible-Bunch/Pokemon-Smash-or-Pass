from tkinter import Tk, Label, Button, Entry, Toplevel
from PIL import ImageGrab, Image, ImageTk
import time
import os
import pandas as pd

# Define paths
output_folder = r"C:\Users\user\Desktop"
csv_file = r"C:\Users\user\Desktop\Pokemon-Smash-or-Pass\pokemon_data.csv"

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
        path = os.path.join(output_folder, f"screenshot_{timestamp}.png")
        img.save(path)
        capture_result['path'] = path
        show_image(path,f"screenshot_{timestamp}.png")

    overlay.bind("<ButtonPress-1>", on_mouse_down)
    overlay.bind("<ButtonRelease-1>", on_mouse_up)

# Save score to CSV
def save_score(image_path, score):
    global df
    df.loc[len(df)] = {'Name': image_path, 'score': score}
    print(df.tail())
    # df.to_csv(csv_file, index=False)
    print(f"Score for {image_path} saved.")

# Display image and prompt for score
def show_image(image_path,image_name):
    image = Image.open(image_path)
    image = image.resize((300, 300))
    img = ImageTk.PhotoImage(image)

    img_label.config(image=img)
    img_label.image = img  # keep reference

    def submit_score():
        try:
            print(score_entry)
            score = float(score_entry.get())
            if 0 <= score <= 10:
                save_score(image_name, score)
                score_entry.delete(0, 'end')
            else:
                print("Score must be between 0 and 10.")
        except ValueError:
            print("Invalid input. Enter a number.")

    # Bind Enter key to submit_score
    score_entry.bind('<Return>', lambda event: submit_score())

    submit_button.config(command=submit_score)
    score_entry.pack(pady=10)
    submit_button.pack(pady=10)


# GUI setup
root = Tk()
root.title("Pokemon Smash or Pass Tool")
root.geometry("400x400")

img_label = Label(root)
img_label.pack()

score_entry = Entry(root)
submit_button = Button(root, text="Submit Score")

screenshot_button = Button(root, text="Take Screenshot", command=take_screenshot)
screenshot_button.pack(pady=10)

root.mainloop()