from tkinter import *
from PIL import ImageTk, Image  
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk()
root.title('Dummy Program')

# Add a frame

r = IntVar()
r.set("2")

def clicked(value):
    myLabel = Label(root, text=value)
    myLabel.pack()

Radiobutton(root, text="Option 1", variable=r, value=1, command= lambda: clicked(r.get())).pack()
Radiobutton(root, text="Option 2", variable=r, value=2, command= lambda: clicked(r.get())).pack()

myButton = Button(root, text="clickme!", command= lambda: clicked(r.get()))
myButton.pack()

myLabel = Label(root, text=r.get())
myLabel.pack()
# run the root
root.mainloop()