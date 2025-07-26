import tkinter as tk
from threading import Thread
import time

def train_model(update_callback, epochs):
    for epoch in range(epochs):
        time.sleep(1)  # simulate training time
        update_callback(f"Epoch {epoch+1} finished")
    update_callback("Training complete!")

def start_training():
    try:
        epochs = int(epoch_entry.get())
    except ValueError:
        status_text.insert(tk.END, "Please enter a valid integer for epochs\n")
        return
    
    train_button.config(state=tk.DISABLED)
    
    def run_training():
        train_model(update_status, epochs)
        train_button.config(state=tk.NORMAL)
    
    Thread(target=run_training).start()

def update_status(message):
    status_text.insert(tk.END, message + "\n")
    status_text.see(tk.END)

root = tk.Tk()
root.title("CIFAR-10 Classifier")

# Input for epochs
tk.Label(root, text="Epochs:").pack()
epoch_entry = tk.Entry(root)
epoch_entry.pack()
epoch_entry.insert(0, "10")  # default value

# Start Training button
train_button = tk.Button(root, text="Start Training", command=start_training)
train_button.pack(pady=10)

# Status text box
status_text = tk.Text(root, height=15, width=50)
status_text.pack(padx=10, pady=10)

root.mainloop()
