from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
from tkinter import filedialog
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from log_gabor.log_gabor import log_gabor
# from tkmacosx import Button - allows background color and other features not available on macosx



# create the root
root = Tk() 
root.title('Image Processing Application')
root.geometry("200x400")

# Function def to open a new window

def proc_dev_interface():

    proc_dev_win.geometry("800x1000")

    # Build the process options
    
    def process_options():
        
        global parameter_list
        global process_ops_frame
        global process_ops_dict

        try:
            process_ops_frame.place_forget()
            plt.close()
        except:
            pass
        
        
        process_ops_frame = LabelFrame(proc_dev_win, text = (process_select.get() + " Process Options"))
        process_ops_frame.place(x=410, width=370, y=40, height=400)

        # ( ... , "process NAME" : [ 'PARAMETER NAME', ' - PARAMETER DESCRIPTION', SCALE MIN, SCALE MAX, RESOLUTION])

        process_ops_dict =   {   "Gabor": [
                                            ['ksize', ' - kernel size', 3, 100, 1, 'scale'], 
                                            ['sigma', ' - standard dev.', 1, 5, .1, 'scale'], 
                                            ['theta', ' - orientation angle', 0, 2, .05, 'scale'], 
                                            ['lambda', ' - wavelength', 1, 10, 1, 'scale'], 
                                            ['gamma', ' - aspect ratio', 0.05, 1, .01, 'scale'], 
                                            ['psi', ' - phase offset', 0, 2, .25, 'scale']],

                                "Log_Gabor": [
                                            ['f0', ' - mean frequency', 1, imgG.shape[1]//4, 1, 'scale'],
                                            ['sigmaOnf', ' - standard deviation of filter', 0.005, 1, .005, 'scale'],
                                            ['angle', ' - filter angle', 0., 1.95, .05, 'scale'],
                                            ['thetaSigma', ' - standard deviation of the angle', 0.1, 3, .05, 'scale']],

                                "Gauss": [
                                            ['size', ' - kernel size', 3, 100, 2, 'scale']],
                                
                                "Canny": [
                                            ['max', ' - strong edge threshold', 1, 255, 1, 'scale'],
                                            ['min', ' - weak edge threshold', 0, 254, 1, 'scale']],
                                
                                "Laplace": [
                                            ['size', ' - kernal size', 3, 31, 2, 'scale']],

                                "Math": [   ['Arithmetic', 'Compare', 'Logs_and_Exponents'],

                                            {
                                            'Arithmetic': ['Add', 'Subtract', 'Multiply', 'Divide'],
                                            'Compare': ['Max', 'Min'],
                                            'Logs_and_Exponents': ['(im)^n', 'e^(im)]', 'log(im)']
                                            }
                                ]
                                
                            }
        parameter_list = process_ops_dict[process_select.get()]

        if process_select.get() == "Math":
                def math_routine():
                    math_routine_frame = LabelFrame(process_ops_frame, text = (math_type.get() + " Options"))
                    math_routine_frame.place(relx=.05, rely=.45, width=330, height=200)


                    

                math_type = StringVar()
                math_type.set('None')
                math_option = process_ops_dict["Math"][0]
                for i in range(len(math_option)):
                    Radiobutton( process_ops_frame, text=math_option[i], variable=math_type, command=math_routine, value=process_ops_dict["Math"][0][i]).place(relx=.1, rely = (.1 + i * .1))

        else:
            for parameters in parameter_list:
                if parameters[-1] == 'scale':
                    exec( "global {}_scale".format(parameters[0]), globals())
                    exec( "{}_scale = Scale(process_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                    exec( "{}_scale.pack()".format(parameters[0]))
        
        def preview_result():
            global parameter_values
            image = imgG
            parameter_values = [[]] * len(parameter_list)
            for i in range(len(parameter_list)):
                if parameters[-1] == 'scale':
                    exec('parameter_values[{}] = {}_scale.get()'.format(i,parameter_list[i][0]), globals())
            
            if process_select.get() == "Gabor":
                try:
                    plt.close()
                except:
                    pass

                kernel = cv.getGaborKernel((parameter_values[0], parameter_values[0]), parameter_values[1], parameter_values[2] * np.pi, parameter_values[3] * np.pi, parameter_values[4], parameter_values[5] * np.pi, ktype=cv.CV_32F)
                
                fig = plt.figure('Kernel Preview')
                plt.title('Kernel')
                plt.imshow(kernel)
                plt.show(block=False)
                
                mask_preview = cv.filter2D(imgG, cv.CV_32F, kernel)
                cv.imshow('Result', mask_preview)
                cv.waitKey(1)

            elif process_select.get() == "Log_Gabor":

                result, LG = log_gabor(image, parameter_values[0], parameter_values[1], parameter_values[2], parameter_values[3])
                                
                cv.imshow('Mask Preview', LG)
                cv.imshow('Result', result)
                cv.waitKey(1)

            elif process_select.get() == "Gauss":

                result = cv.GaussianBlur(image, (parameter_values[0], parameter_values[0]), 0)
                                
                cv.imshow('Result', result)
                cv.waitKey(1)

            elif process_select.get() == "Canny":

                result = cv.Canny(image, parameter_values[1], parameter_values[0])
                                
                cv.imshow('Result', result)
                cv.waitKey(1)

            elif process_select.get() == "Laplace":
                im = cv.GaussianBlur(image, (3, 3), 0)
                result = cv.Laplacian(im, cv.CV_16S, ksize=parameter_values[0])
                                
                cv.imshow('Result', result)
                cv.waitKey(1)



        #Button(process_ops_frame, text="Preview Mask", padx= 10, pady=10, command=preview_mask).pack()  
        Button(proc_dev_win, text="Preview Result", padx= 10, pady=10, command=preview_result).place(x= 410, width=100, y = 460, height= 30)

    # Build the process selection frame
    process_frame = LabelFrame(proc_dev_win, text = "Process Selection")
    process_frame.place(x=20, width=370, y=40, height=500) 
    processes = [ "Gabor",
                "Log_Gabor",
                "Gauss",
                "Canny",
                "Laplace",
                "Math",
                "Thresholding"]

    process_select = StringVar()
    process_select.set("None")
    
    for i in range(len(processes)):
        Radiobutton( process_frame, text=processes[i], variable=process_select, command=process_options, value=processes[i]).place(relx=.1, rely=(.05 + .8 *( i / (len(processes)-1))))
    
    
    

    


# Open the file for process development

def open_proc_dev_file():

    global img
    global imgG
    global images
    global image_names

    root.filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select a File", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))
    img = cv.imread(root.filename, cv.IMREAD_COLOR)
    

    cv.imshow('Base Image', img)
    cv.waitKey(1)

    images = np.zeros(img.shape[0:2], dtype='float32')
    images = images[np.newaxis,:,:]
    print(images.shape)
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgG = imgG.astype('float32')
    imgG = cv.normalize(imgG, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    images[0] = imgG
    image_names = ['Base_Image_Norm']
    processing_summary = {
                            image_names[0] : 
    }
    cv.imshow('norm grey', images[0])
    proc_dev_interface()

    proc_dev_button.place_forget()

# Open the winodw for process development

def open_proc_dev():
    

    global proc_dev_win

    dev_im_proc_button["state"] = "disabled"
    batch_proc_button["state"] = "disabled"

    proc_dev_win = Toplevel()
    proc_dev_win.geometry("300x40")
    proc_dev_win.title("Image Processing Development")

    global proc_dev_button
    proc_dev_button = Button(proc_dev_win, padx=5, pady=5, text="Select Base Image", command=open_proc_dev_file)
    proc_dev_button.place(x=10, width =280, y=10, height=20)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to leave process development?"):
            if messagebox.askyesno("Save Work?", "Would you like to save your progress?"):
                print("Save Progress")
            proc_dev_win.destroy()
            cv.destroyWindow("Base Image")
            dev_im_proc_button["state"] = "normal"
            batch_proc_button["state"] = "normal"


    proc_dev_win.protocol("WM_DELETE_WINDOW", on_closing)

# Open the batch processing window
def open_batch_proc():

    batch_proc_win = Toplevel()
    batch_proc_win.title("Batch Processing")



# Set up the action buttons on the main window

dev_im_proc_button = Button(root, padx=20, pady=20, text="Develop Image Process", command=open_proc_dev)
dev_im_proc_button.place(relx=.1, relwidth=.8, rely=.3, relheight=.2)

batch_proc_button = Button(root, padx=20, pady=20, text="Batch Processing", command=open_batch_proc)
batch_proc_button.place(relx=.1, relwidth=.8, rely=.55, relheight=.2)

quit_button = Button(root, padx=20, pady=20, text="Quit", command=root.quit)
quit_button.place(relx=.1, relwidth=.8, rely=.8, relheight=.15)

def on_closing():
    if messagebox.askokcancel("Quit", "Are you sure you would like to quit?"):
        print("Save Progress")
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# run the root
root.mainloop()