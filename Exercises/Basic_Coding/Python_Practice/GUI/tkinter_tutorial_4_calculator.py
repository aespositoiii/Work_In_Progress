from tkinter import *
# from tkmacosx import Button - allows background color and other features not available on macosx


# create the root
root = Tk()
root.title('Calculator')


d = Entry(root, width=40, borderwidth=5)
e = Entry(root, width=40, borderwidth=5)
d.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
e.grid(row=1, column=0, columnspan=4, padx=10, pady=10)



try:
    f_num = f_num
except:
    f_num = 0.
    op = 'clear'
    d.insert(0, f_num)

# Button Actions

def operate(operation, last, current):
    if (operation == 'clear') | (op == 'eq'):
        return current
    elif operation == 'add':
        return last + current
    elif operation == 'sub':
        return last - current
    elif operation == 'mult':
        return last * current
    elif operation == 'div':
        return last / current

def button_concat(number):
    current = e.get()
    e.delete(0, END)
    e.insert(0, str(current) + str(number))
    return

def button_add():
    global f_num
    global op
    current = e.get()
    e.delete(0, END)
    f_num = operate(op, int(f_num), int(current))
    d.delete(0,END)
    d.insert(0, f_num)
    op = 'add'
    return

def button_sub():
    global f_num
    global op
    current = e.get()
    e.delete(0, END)
    f_num = operate(op, int(f_num), int(current))
    d.delete(0,END)
    d.insert(0, f_num)
    op = 'sub'
    return

def button_mult():
    global f_num
    global op
    current = e.get()
    e.delete(0, END)
    f_num = operate(op, int(f_num), int(current))
    d.delete(0,END)
    d.insert(0, f_num)
    op = 'mult'
    return

def button_div():
    global f_num
    global op
    current = e.get()
    e.delete(0, END)
    f_num = operate(op, int(f_num), int(current))
    d.delete(0,END)
    d.insert(0, f_num)
    op = 'div'
    return

def button_eq():
    global f_num
    global op
    current = e.get()
    e.delete(0, END)
    f_num = operate(op, int(f_num), int(current))
    d.delete(0,END)
    d.insert(0, f_num)
    op = 'eq'
    
    return

def button_clear():
    global f_num
    global op
    e.delete(0, END)
    d.delete(0,END)
    f_num = 0
    d.insert(0, f_num)
    op = 'clear'
    return

# Define Number Buttons

button_1 = Button(root, text='1', padx=40, pady=30, command=lambda: button_concat(1))
button_2 = Button(root, text='2', padx=40, pady=30, command=lambda: button_concat(2))
button_3 = Button(root, text='3', padx=40, pady=30, command=lambda: button_concat(3))
button_4 = Button(root, text='4', padx=40, pady=30, command=lambda: button_concat(4))
button_5 = Button(root, text='5', padx=40, pady=30, command=lambda: button_concat(5))
button_6 = Button(root, text='6', padx=40, pady=30, command=lambda: button_concat(6))
button_7 = Button(root, text='7', padx=40, pady=30, command=lambda: button_concat(7))
button_8 = Button(root, text='8', padx=40, pady=30, command=lambda: button_concat(8))
button_9 = Button(root, text='9', padx=40, pady=30, command=lambda: button_concat(9))
button_0 = Button(root, text='0', padx=40, pady=30, command=lambda: button_concat(0))

# Define function buttons

button_clear = Button(root, text='C', padx=40, pady=30, command=button_clear)
button_add = Button(root, text='+', padx=40, pady=30, command=button_add)
button_sub = Button(root, text='-', padx=40, pady=30, command=button_sub)
button_mult = Button(root, text='*', padx=40, pady=30, command=button_mult)
button_div = Button(root, text='/', padx=40, pady=30, command=button_div)
button_eq = Button(root, text='=', padx=40, pady=30, command=button_eq)

# place buttons in grid

button_7.grid(row=2, column=0)
button_8.grid(row=2, column=1)
button_9.grid(row=2, column=2)
button_div.grid(row=2, column=3)

button_4.grid(row=3, column=0)
button_5.grid(row=3, column=1)
button_6.grid(row=3, column=2)
button_mult.grid(row=3, column=3)

button_1.grid(row=4, column=0)
button_2.grid(row=4, column=1)
button_3.grid(row=4, column=2)
button_sub.grid(row=4, column=3)

button_clear.grid(row=5, column=0)
button_0.grid(row=5, column=1)
button_eq.grid(row=5, column=2)
button_add.grid(row=5, column=3)

'''# create a label
myLabel1 = Label(root, text="Hello World!")
myLabel2 = Label(root, text="balogna sales: 1500 units")
myLabel3 = Label(root, text="balogna sales: 1500 units")
myLabel4 = Label(root, text="balogna sales: 1500 units")
myLabel5 = Label(root, text="balogna sales: 1500 units")

# push the label onto the window
myLabel.pack()

# push the label onto the window in grid format
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=1)
myLabel3.grid(row=2, column=2)
myLabel4.grid(row=3, column=3)
myLabel5.grid(row=4, column=2)
# Entry Widget

e = Entry(root, width=50, bg='grey', borderwidth=12)
e.grid(row=2, column=1)
e.insert(0, 'Default Text')
# make a button


def myClick():
    sometext = Label(root, text=e.get())
    sometext.grid(row=2, column=2)


myButton1 = Button(root, text="BUTTon", state=DISABLED)
myButton1.grid(row=0, column=0)

myButton2 = Button(root, text="Anotha one", padx=5, pady=15, highlightbackground='#000000', fg='red')
myButton2.grid(row=0, column=1)

myButton3 = Button(root, text="Anotha, notha one", command=myClick)
myButton3.grid(row=1, column=2)'''




# run the root
root.mainloop()