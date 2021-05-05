from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Dummy Program')
root.geometry("200x200")

def show():
    myLabel = Label(root, text=var.get()).pack()

var=StringVar(value='off')

box = Checkbutton(root, text="Check the box", variable=var, onvalue='on', offvalue='off')
box.pack()

myLabel = Label(root, text=var.get())

myButton = Button(root, text="show selection", command=show)
myButton.pack()

# run the root
root.mainloop()