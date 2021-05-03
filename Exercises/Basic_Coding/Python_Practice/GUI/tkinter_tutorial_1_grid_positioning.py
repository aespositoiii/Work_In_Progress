from tkinter import *

# create the root
root = Tk()

# create a label
myLabel1 = Label(root, text="Hello World!")
myLabel2 = Label(root, text="balogna sales: 1500 units")
myLabel3 = Label(root, text="balogna sales: 1500 units")
myLabel4 = Label(root, text="balogna sales: 1500 units")
myLabel5 = Label(root, text="balogna sales: 1500 units")

# push the label onto the window
'''myLabel.pack()'''

# push the label onto the window in grid format
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=1)
myLabel3.grid(row=2, column=2)
myLabel4.grid(row=3, column=3)
myLabel5.grid(row=4, column=2)

# run the root
root.mainloop()