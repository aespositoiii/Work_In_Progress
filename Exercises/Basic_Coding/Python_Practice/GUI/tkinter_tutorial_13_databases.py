from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Dummy Program')
root.geometry("200x200")

clicked = StringVar(value='Select a Day')

def show():
    if clicked.get() == 'Select a Day':
        Label(root, text="Make a Selection!").pack()
    else:
        Label(root, text=clicked.get()).pack()

options = [
    "Monday", 
    "Tuesday", 
    "Wednesday", 
    "Thursday", 
    "Friday", 
    "Saturday", 
    "Sunday"
]

drop = OptionMenu(root, clicked, *options)
drop.pack()

myButton = Button(root, text="Show Selection", command=show)
myButton.pack()

# run the root
root.mainloop()