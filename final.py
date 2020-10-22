# Implementation of Model without Scikit + Iterative Approach.
# Added three normalization options.

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import sys
import os

X = pd.read_csv("Fish.csv")

gui = Tk()
gui.title("Fish Weight Estimation")
gui.geometry('600x1080')

# creating label for dataset
l1 = Label(gui, text="Dataset:", font=("Open Sans", 18))
l1.grid(row=0, column=0, sticky="e")
l2 = Label(gui, text="FISH", font=("Open Sans",18))
l2.grid(row=0, column=1, sticky="w")

# creating label for features
l3 = Label(gui, text="Features:", font=("Open Sans",18))
l3.grid(row=2, column=0, sticky="e")
lb3 = Listbox(gui, height = 6, width = 10, font=("Open Sans",12))
lb3.insert(1, "1. Weight")
lb3.insert(2, "2. Length1")
lb3.insert(3, "3. Length2")
lb3.insert(4, "4. Length3")
lb3.insert(5, "5. Height")
lb3.insert(6, "6. Width")
lb3.grid(row=2, column=1, sticky="w")

# creating labels column
l4 = Label(gui, text="Label:", font=("Open Sans",18))
l4.grid(row=3, column=0, sticky="e")

# creating drop-down list for column names
col_label = StringVar()
col_label.set("Label")
drop_label = OptionMenu(gui, col_label, "Weight", "Length1", "Length2", "Length3", "Height", "Width")
drop_label.grid(row=3, column=1, sticky="w")

# creating labels column
l5 = Label(gui, text="Normalization:", font=("Open Sans",18))
l5.grid(row=4, column=0, sticky="e")

# creating drop-down list for normaliztion type
norm_label = StringVar()
norm_label.set("Type")
dro_label = OptionMenu(gui, norm_label, "Standardization", "Scaling", "Min-Max")
dro_label.grid(row=4, column=1, sticky="w")

# taking input of split ratio
l6 = Label(gui, text="Split Ratio:", font=("Open Sans",18))
l6.grid(row=5, column=0, sticky="e")
e = Entry(gui, width = "5", font=("Open Sans",18))
e.grid(row=5, column=1, sticky="w")
la6 = Label(gui, text="(Enter a value between 0.1 and 1)", font=("Open Sans",12))
la6.grid(row=5, column=2, sticky="w") 

def main(y_col, norm, split_val):

    global X
    # Which Column is output
    y_var = X.columns[y_col]
    #print("The label is : ",y_var)
    
    # Split Dataset into samples and labels
    y = X.iloc[:,y_col]
    X = X.drop(y_var, axis=1)
    tot_sam, feat = X.shape
    #print("Total Size Of Dataset : ",X.shape)
    
    # Assign numbers to classes/names of fish
    zax, X.iloc[:,0] = np.unique(X.iloc[:,0], return_inverse=True)
    
    # Print Assigned Values of classes and
    print("Changing String(Species) to Intergers...")
    for i in range(zax.shape[0]):
        print(zax[i] , " = " , i)
        
    # Print Number of Fishes in each class
    #print(X['Species'].value_counts())
    
    # How much do you want to split as Training and Testing Samples
    num_sam = int(tot_sam * split_val)

    # Shuffle Training Samples and Labels
    X = np.array(X)
    y = np.array(y)
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]

    # Create Training Dataset And Print Size
    global Xtr
    global ytr
    Xtr = X[:num_sam,:]
    ytr = np.array(y[:num_sam])
    #print("Size of Training Samples : ",Xtr.shape)
    #print("Size of Training Labels  : ",ytr.shape)

    # Create Testing Dataset And Print Size
    global Xte
    global yte
    Xte = np.array(X[num_sam:,:])
    yte = np.array(y[num_sam:])
    #print("Size of Testing Samples  : ",Xte.shape)
    #print("Size of Testing Labels   : ",yte.shape)

    # to create number of training samples box
    frame = LabelFrame(gui, text="Size of Samples", font=("Open Sans",14), padx=50, pady=10)
    frame.grid(row=7, column=0, columnspan=4)
    example = Label(frame, text=
                    "Size of Training Samples : " + str(Xtr.shape)+"\n"+
                    "Size of Training Labels  : " + str(ytr.shape)+"\n"+
                    "Size of Testing Samples  : " + str(Xte.shape)+"\n"+
                    "Size of Testing Labels   : " + str(yte.shape))
    example.pack()

    r,c = Xtr.shape
    rte,cte = Xte.shape

    # Type of Normalization
    if norm == 1:
        # Standardization
        # Normalize the Training data.
        for i in range(c):
            mx = sum(Xtr[:,i])
            mean_x = (mx/r);
            sd_x = np.sqrt(sum(pow(Xtr[:,i] - mean_x,2))/c)
            Xtr[:,i] = (Xtr[:,i]-mean_x)/sd_x
            
        # Normalize the Testing data.
        for i in range(cte):
            mx = sum(Xte[:,i])
            mean_x = (mx/rte);
            sd_x = np.sqrt(sum(pow(Xte[:,i] - mean_x,2))/cte)
            Xte[:,i] = (Xte[:,i]-mean_x)/sd_x

    elif norm == 2:
        # Scaling
        # Normalize the Training data.
        for i in range(c):
            max_val = np.max(Xtr[:,i])
            Xtr[:,i] = Xtr[:,i]/max_val
            
        # Normalize the Testing data.
        for i in range(cte):
            max_val = np.max(Xte[:,i])
            Xte[:,i] = (Xte[:,i])/max_val

    elif norm == 3:
        # Min-Max
        # Normalize the Training data.
        for i in range(c):
            max_val = np.max(Xtr[:,i])
            min_val = np.min(Xtr[:,i])
            Xtr[:,i] = (Xtr[:,i] - min_val)/(max_val - min_val)
            
        # Normalize the Testing data.
        for i in range(cte):
            max_val = np.max(Xte[:,i])
            min_val = np.min(Xte[:,i])
            Xte[:,i] = (Xte[:,i] - min_val)/(max_val - min_val)

    #Calculate Prediction Of Training Dataset
    def training(w):        
        y_pred = Xtr.dot(w)

        #Calculate Performance of the Model.

        #print("Root Mean Squared Error")
        #print(np.sqrt(np.mean(ytr - y_pred)**2))

        #Printing to GUI
        l9 = Label(gui, text="Training:", font=("Opens Sans",16))
        l9.grid(row=12, column=0, sticky = "w")

        l7 = Label(gui, text="Root Mean Squared Error:" + str(np.sqrt(np.mean(ytr - y_pred)**2)), font=("Open Sans",14))
        l7.grid(row=13, column=0, columnspan=5, sticky = "w")

        def graph1():
            #Plot the Graph: Prediction vs Original
            c = [i for i in range(0,num_sam,1)]
            plt.plot(c, ytr,color = 'Blue')
            plt.plot(c, y_pred,color = 'red')
            plt.title('Train(Blue) vs pred(Red)')
            plt.show()

        #button is created and function is called
        b2 = Button(gui, text="Graph", width = 20, fg = "White", bg = "#747575", command=lambda: graph1())
        b2.grid(row=14, column=0, columnspan=2)       


    #Calculate Prediction Of Testing Dataset
    def testing(w):
        y_predt = Xte.dot(w)

        #Calculate Performance of the Model.

        #print("Root Mean Squared Error")
        #print(np.sqrt(np.mean(yte - y_predt)**2))

        #Printing to GUI
        l10 = Label(gui, text="Testing:", font=("Opens Sans",16))
        l10.grid(row=15, column=0, sticky = "w")
        
        l8 = Label(gui, text="Root Mean Squared Error:" + str(np.sqrt(np.mean(yte - y_predt)**2)), font=("Open Sans",14))
        l8.grid(row=16, column=0, columnspan=5, sticky = "w")

        def graph2():
            #Plot the Graph: Prediction vs Original
            
            c = [i for i in range(0,tot_sam-num_sam,1)]
            plt.plot(c, yte,color = 'Blue')
            plt.plot(c, y_predt,color = 'red')
            plt.title('Test(Blue) vs pred(Red)')
            plt.show()

        #button is created and function is called
        b3 = Button(gui, text="Graph", width = 20, fg = "White", bg = "#747575", command=lambda: graph2())
        b3.grid(row=17, column=0, columnspan=2)

    #Optimized weights
    def UOP():
       # Augment Data with ones.
        global Xtr
        global Xte
        Xtr = np.hstack((Xtr,np.ones((r,1))))
        Xte = np.hstack((Xte,np.ones((rte,1))))
        
        # Optimization: Weights
        global w
        w = np.linalg.pinv(Xtr).dot(ytr);
        
        training(w)
        testing(w)
          

    #iterations
    def iterations():

        def it(lr, niter):
            w = np.zeros(c) # weights
            b = 0           # bias
            n = r       

            global Xtr
            global Xte

            for i in range(niter):
                yptr = np.dot(Xtr,w) + b
                dw = (1/n) * np.dot(Xtr.T , (yptr-ytr))
                db = (1/n) * np.sum(yptr-ytr)
                w = w - lr*dw
                b = b - lr*db
                
            w = np.append(w,b)
        
            # Augment Data with ones.
            Xtr = np.hstack((Xtr,np.ones((r,1))))
            Xte = np.hstack((Xte,np.ones((rte,1))))

            training(w)
            testing(w)

        # taking input for lr try 0.1
        l11 = Label(gui, text="Learning Rate:", font=("Open Sans",14))
        l11.grid(row=9, column=0, sticky="e")
        e1 = Entry(gui, width = "5", font=("Open Sans",14))
        e1.grid(row=9, column=1, sticky="w")

        # taking input for Number of Iterations try 1000
        l11 = Label(gui, text="Number of Iterations:", font=("Open Sans",14))
        l11.grid(row=10, column=0, sticky="e")
        e2 = Entry(gui, width = "5", font=("Open Sans",14))
        e2.grid(row=10, column=1, sticky="w")

        b4 = Button(gui, text="Submit", width = 20, fg = "White", bg = "#747575", command=lambda: it(float(e1.get()), int(e2.get())))
        b4.grid(row=11, column=0, columnspan=2)


    #Radio button to select UOP or iterations
    ra = IntVar()
    Radiobutton(gui, text="Use Optimized Weights", variable=ra, value=1, command=lambda: UOP()).grid(row=8, column=0)
    Radiobutton(gui, text="Use Gradient Descent", variable=ra, value=2, command=lambda: iterations()).grid(row=8, column=1)
    
def convert(a, b, c):
    if a=="Weight":
        y_col = 1
    elif a=="Length1":
        y_col = 2
    elif a=="Length2":
        y_col = 3
    elif a=="Length3":
        y_col = 4
    elif a=="Height":
        y_col = 5
    elif a=="Width":
        y_col = 6

    split_val = float(c)

    if b=="Standardization":
        norm = 1
    elif b=="Scaling":
        norm = 2
    elif b=="Min-Max":
        norm = 3
    
    #print("values are",y_col, norm, split_val)
    main(y_col, norm, split_val)

#button is created and function is called
b1 = Button(gui, text="Submit", width = 20, fg = "White", bg = "#747575", command=lambda: convert(col_label.get(),norm_label.get(), e.get()))
b1.grid(row=6, column=0, columnspan=2)

#Reset button
def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)
b5 = Button(gui, text="Reset", width = 20, fg = "White", bg = "#747575", command=restart_program)
b5.grid(row=18, column=0, columnspan=2)

gui.mainloop()
