from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Dummy Program')

def open():

    top = Toplevel()
    top.title('OOOOoooooo, Another Window')

    Button(top, text="close Window", command=top.destroy).pack()
  
# New Window



a_label = Label(root, text="This is the Root Window")
a_label.grid(row=0, column=0)

button = Button(root, text="open new window", command=open)
button.grid(row=1, column=0)



# run the root
root.mainloop()