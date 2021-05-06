from tkinter import *
from PIL import ImageTk, Image  
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk()
root.title('Dummy Program')

# Add a frame
frame = LabelFrame(root, text= "This is my frame...", padx=100, pady=15)
frame.pack(padx=100, pady=10)

b = Button(frame, text="Click or Don't Click", padx=40)
b.pack()

# run the root
root.mainloop()