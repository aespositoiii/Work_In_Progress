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
        global image_source
        global math_type
        global thresh_select
        global threshold_option
        global norm_select
        global norm_option

        try:
            process_ops_frame.place_forget()
            plt.close()
        except:
            pass
        
        
        process_ops_frame = LabelFrame(proc_dev_win, text = (process_select.get() + " Process Options"))
        process_ops_frame.place(x=410, width=370, y=40, height=600)
        image_select_src = StringVar(value=image_names[0])
        Label(process_ops_frame, text='Source Image Selection:', anchor='w').place(x=20, y=10)
        image_source = OptionMenu(process_ops_frame, image_select_src, *image_names)
        image_source.place(x=200, y=10)

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
                                            }],
                                
                                "Thresholding":
                                        [   ['Binary', 'Inverse_Binary', 'Truncated', 'To_Zero', 'Inverse_To_Zero', 'Adaptive_Thresh_Mean_C', 'Adaptive_Thresh_Gaussian_C', 'Otsu_Bin', 'Triangle_Bin'],
                                            
                                            [   ['Threshold', ' - boundary value', 0, 255, 1],
                                                ['Maximum_Value', ' - binary value', 0,255,1],
                                                ['Block_Size', ' - area for threshold averaging', 0,255,1],
                                                ['Constant_Offset', ' - binary offset', 0,255,1]]],

                                "Normalize":
                                        [   ['Norm_Inf', 'Norm_L1', 'Norm_L2', 'Norm_L2_Square', 'Norm_Hamming', 'Norm_Hamming2', 'Norm_Relative', 'Norm_Min_Max'],
                                            
                                            [   ['Alpha', ' - lower bound or norm value', 0, 255, 1],
                                                ['beta', ' - upper bound or N/A', 0,255,1]]]
                                
                            }
        

        filter_list = [ "Gabor",
                        "Log_Gabor",
                        "Gauss",
                        "Canny",
                        "Laplace"]

        if process_select.get() in filter_list:
            parameter_list = process_ops_dict[process_select.get()]
            for parameters in parameter_list:
                exec( "global {}_scale".format(parameters[0]), globals())
                exec( "{}_scale = Scale(process_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                exec( "{}_scale.place(x=10, y=40+{}*60)".format(parameters[0], parameter_list.index(parameters)))


        elif process_select.get() == "Math":
                def math_routine():

                    global math_routine_frame
                    global constant
                    global image_source2

                    try:
                        math_routine_frame.pack_forget()
                    except:
                        pass
                    math_routine_frame = LabelFrame(process_ops_frame, text = (math_type.get() + " Options"), height=20, width=20)
                    math_routine_frame.place(x=10, y=150, height=420, width=340)

                    

                    math_operation = StringVar()
                    math_operation.set('None')
                    operations = process_ops_dict["Math"][1][math_type.get()]

                    for i in range(len(operations)):
                        Radiobutton(math_routine_frame, text=operations[i], variable=math_operation, anchor='w', value=operations[i]).place(x=20, y=10+i*30)
                    
                    if math_type.get() == 'Logs_and_Exponents':
                        pass

                    elif (math_type.get() == 'Arithmetic') | (math_type.get() == 'Compare'):
                        def use_constant():
                            if constant.get() == True:
                                second_source_label.place_forget()
                                image_source2.place_forget()
                                math_constant.place(x=20, y=50+len(operations)*30)

                            elif constant.get() == False:
                                second_source_label.place(x=20, y=50+len(operations)*30)
                                image_source2.place(x=200, y=50+len(operations)*30)
                                math_constant.place_forget()
                        
                        second_source_label = Label(math_routine_frame, text='Secondary Source Selection:', anchor='w')
                        second_source_label.place(x=20, y=50+len(operations)*30)
                        image_source2 = OptionMenu(math_routine_frame, image_select_src, *image_names)
                        image_source2.place(x=200, y=50+len(operations)*30)
                        
                        math_constant = Scale(math_routine_frame, label='Constant Operator', from_=.01, to=1, resolution=.01, orient=HORIZONTAL, length=300)
                        #math_constant.place(x=20, y=90+len(operations)*30)

                        constant = BooleanVar()
                        constant.set(False)
                        is_constant = Checkbutton(math_routine_frame, text="Use Constant?", variable=constant, command=use_constant, onvalue=True, offvalue=False)
                        is_constant.place(x=20, y=20+len(operations)*30)

                math_type = StringVar()
                math_type.set('None')
                math_option = process_ops_dict["Math"][0]
                
                for i in range(len(math_option)):
                    Radiobutton(process_ops_frame, text=math_option[i], variable=math_type, command=math_routine, anchor='w', value = math_option[i]).place(x=20, y=40+i*30)

                print(math_type.get())

        elif process_select.get() == "Thresholding":
            
            def thresh_routine():
                global thresh_ops_frame
                global thresh_select

                try:
                    thresh_ops_frame.place_forget()
                    plt.close()
                except:
                    pass
                
                thresh_ops_frame = LabelFrame(process_ops_frame, text = (process_select.get() + " Process Options"))
                thresh_ops_frame.place(x=10, y=70+len(threshold_option)*30, width=340, height=230)

                standard_thresh = ['Binary', 'Inverse_Binary', 'Truncated', 'To_Zero', 'Inverse_To_Zero', 'Otsu_Bin', 'Triangle_Bin']
                adapt_thresh = ['Adaptive_Thresh_Mean_C', 'Adaptive_Thresh_Gaussian_C']

                if thresh_select.get() in standard_thresh:
                    parameter_list = process_ops_dict["Thresholding"][1][0:2]
                    for parameters in parameter_list:
                        exec( "global {}_scale".format(parameters[0]), globals())
                        exec( "{}_scale = Scale(thresh_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                        exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))
                
                elif thresh_select.get() in adapt_thresh:
                    parameter_list = process_ops_dict["Thresholding"][1][1:4]
                    for parameters in parameter_list:
                        exec( "global {}_scale".format(parameters[0]), globals())
                        exec( "{}_scale = Scale(thresh_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                        exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))

            threshold_option = process_ops_dict["Thresholding"][0]

            
            thresh_select = StringVar()
            thresh_select.set('None')

            for i in range(len(threshold_option)):
                Radiobutton( process_ops_frame, text=threshold_option[i], variable=thresh_select, command=thresh_routine, anchor='w', value = threshold_option[i]).place(x=20, y=40+i*30)
        

        elif process_select.get() == "Normalize":
            
            def norm_routine():
                global norm_ops_frame
                global norm_select

                try:
                    norm_ops_frame.place_forget()
                    plt.close()
                except:
                    pass
                
                norm_ops_frame = LabelFrame(process_ops_frame, text = (process_select.get() + " Process Options"))
                norm_ops_frame.place(x=10, y=70+len(norm_option)*30, width=340, height=230)

                parameter_list = process_ops_dict["Normalize"][1]
                for parameters in parameter_list:
                    exec( "global {}_scale".format(parameters[0]), globals())
                    exec( "{}_scale = Scale(norm_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                    exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))
            
            norm_option = process_ops_dict["Normalize"][0]
            
            norm_select = StringVar()
            norm_select.set('None')

            for i in range(len(norm_option)):
                Radiobutton( process_ops_frame, text=norm_option[i], variable=norm_select, command=norm_routine, anchor='w', value = norm_option[i]).place(x=20, y=40+i*30)
    
        def preview_result():
            global parameter_values
            global math_type
            global result
            
            use_parameters = ["Gabor", "Log_Gabor", "Gauss", "Canny", "Laplace", "Thresholding", "Normalize"]

            parameter_values = [[]] * len(parameter_list)

            if process_select.get() in use_parameters:
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
                
                result = cv.filter2D(images[image_names.index(image_select_src.get())], cv.CV_32F, kernel)
                cv.imshow('Result', result)
                cv.waitKey(1)


            elif process_select.get() == "Log_Gabor":

                result, LG = log_gabor(images[image_names.index(image_select_src.get())], parameter_values[0], parameter_values[1], parameter_values[2], parameter_values[3])
                                
                cv.imshow('Mask Preview', LG)
                cv.imshow('Result', result)
                cv.waitKey(1)


            elif process_select.get() == "Gauss":

                result = cv.GaussianBlur(images[image_names.index(image_select_src.get())], (parameter_values[0], parameter_values[0]), 0)
                                
                cv.imshow('Result', result)
                cv.waitKey(1)
                

            elif process_select.get() == "Canny":

                result = cv.Canny(images[image_names.index(image_select_src.get())], parameter_values[1], parameter_values[0])
                                
                cv.imshow('Result', result)
                cv.waitKey(1)
                

            elif process_select.get() == "Laplace":
                im = cv.GaussianBlur(images[image_names.index(image_select_src.get())], (3, 3), 0)
                result = cv.Laplacian(im, cv.CV_16S, ksize=parameter_values[0])
                
                cv.imshow('Result', result)
                cv.waitKey(1)


            elif process_select.get() == "Math":
                print(math_type.get)
                #if math_type.get == ''
            
        def apply_result():
            global mask_count
            global image_source
            global result

            print(process_select.get())

            if process_select.get() in [ "Gabor", "Log_Gabor", "Gauss", "Canny", "Laplace"]:
                image_names.append('Mask' + str(mask_count))
                print(image_names)

                result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

                images.append(result)

                cv.imshow(image_names[-1], images[-1])

                process_summary.append( {   'name'      : 'Mask' + str(mask_count),
                                            'source'    : image_select_src.get(),
                                            'image'     : result,
                                            'process'   : process_select.get(),
                                            'parameters': parameter_values
                                            })

                mask_count += 1

            image_source.place_forget()
            image_source = OptionMenu(process_ops_frame, image_select_src, *image_names)
            image_source.place(x=200, y=10)

        Button(proc_dev_win, text="Preview Result", padx= 10, pady=10, command=preview_result).place(x= 410, width=100, y = 660, height= 30)
        Button(proc_dev_win, text="Apply Result", padx= 10, pady=10, command=apply_result).place(x= 410, width=100, y = 700, height= 30)
 


    # Build the process selection frame
    process_frame = LabelFrame(proc_dev_win, text = "Process Selection")
    process_frame.place(x=20, width=370, y=40, height=700) 
    processes = [ "Gabor",
                "Log_Gabor",
                "Gauss",
                "Canny",
                "Laplace",
                "Math",
                "Thresholding",
                "Normalize"]

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
    global process_summary
    global mask_count

    root.filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select a File", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))
    img = cv.imread(root.filename, cv.IMREAD_COLOR)
    

    cv.imshow('Base_Image', img)
    cv.waitKey(1)    

    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #imgG = imgG.astype('float32')
    imgG = cv.normalize(imgG, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    images = list([imgG])
    image_names = ['Base_Image']
    #print(type(image_names))
    
    mask_count = 1

    process_summary = [
                        {   'name'      : 'Base_Image_Gray',
                            'source'    : 'Base_Image',
                            'image'     : imgG,
                            'process'   : 'Convert_to_Gray'}
                    ]
                            

    cv.imshow('Base_Image_Gray', images[0])
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
            cv.destroyAllWindows()
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