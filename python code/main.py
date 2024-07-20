import pickle
import os
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
from PIL import Image, ImageDraw
import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = 1, 1, 1
        self.clf = LinearSVC()
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None
        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!", parent=msg)
        if os.path.exists(self.proj_name):
            with open(os.path.join(self.proj_name, f"{self.proj_name}_data.pickle"), "rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            self.clf = data['clf']
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

            os.makedirs(self.proj_name, exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class1), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class2), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class3), exist_ok=True)

    def init_gui(self):
        WIDTH, HEIGHT = 500, 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"NeuralNine Drawing Classifier Alpha v0.2 - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.image1)

        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure([0, 1, 2], weight=1)

        Button(btn_frame, text=self.class1, command=lambda: self.save(1)).grid(row=0, column=0, sticky=W+E)
        Button(btn_frame, text=self.class2, command=lambda: self.save(2)).grid(row=0, column=1, sticky=W+E)
        Button(btn_frame, text=self.class3, command=lambda: self.save(3)).grid(row=0, column=2, sticky=W+E)
        Button(btn_frame, text="Brush-", command=self.brushminus).grid(row=1, column=0, sticky=W+E)
        Button(btn_frame, text="Clear", command=self.clear).grid(row=1, column=1, sticky=W+E)
        Button(btn_frame, text="Brush+", command=self.brushplus).grid(row=1, column=2, sticky=W+E)
        Button(btn_frame, text="Train Model", command=self.train_model).grid(row=2, column=0, sticky=W+E)
        Button(btn_frame, text="Save Model", command=self.save_model).grid(row=2, column=1, sticky=W+E)
        Button(btn_frame, text="Load Model", command=self.load_model).grid(row=2, column=2, sticky=W+E)
        Button(btn_frame, text="Change Model", command=self.rotate_model).grid(row=3, column=0, sticky=W+E)
        Button(btn_frame, text="Predict", command=self.predict).grid(row=3, column=1, sticky=W+E)
        Button(btn_frame, text="Save Everything", command=self.save_everything).grid(row=3, column=2, sticky=W+E)

        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W+E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, class_num):
        self.image1.save("temp.png")
        img = Image.open("temp.png")
        img.thumbnail((50, 50), Image.Resampling.LANCZOS)

        if class_num == 1:
            img.save(os.path.join(self.proj_name, self.class1, f"{self.class1_counter}.png"), "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(os.path.join(self.proj_name, self.class2, f"{self.class2_counter}.png"), "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(os.path.join(self.proj_name, self.class3, f"{self.class3_counter}.png"), "PNG")
            self.class3_counter += 1

        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def train_model(self):
        img_list = []
        class_list = []

        try:
            for x in range(1, self.class1_counter):
                img = cv.imread(os.path.join(self.proj_name, self.class1, f"{x}.png"))[:, :, 0]
                img = img.reshape(2500)
                img_list.append(img)
                class_list.append(1)

            for x in range(1, self.class2_counter):
                img = cv.imread(os.path.join(self.proj_name, self.class2, f"{x}.png"))[:, :, 0]
                img = img.reshape(2500)
                img_list.append(img)
                class_list.append(2)

            for x in range(1, self.class3_counter):
                img = cv.imread(os.path.join(self.proj_name, self.class3, f"{x}.png"))[:, :, 0]
                img = img.reshape(2500)
                img_list.append(img)
                class_list.append(3)

            img_list = np.array(img_list)
            class_list = np.array(class_list)

            self.clf.fit(img_list, class_list)
            tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully trained!", parent=self.root)
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"An error occurred during training: {e}", parent=self.root)

    def predict(self):
        self.image1.save("temp.png")
        img = Image.open("temp.png")
        img.thumbnail((50, 50), Image.Resampling.LANCZOS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        
      
        review = tkinter.messagebox.askquestion("Review Prediction", f"The drawing is probably a {self.class1 if prediction[0] == 1 else (self.class2 if prediction[0] == 2 else self.class3)}. Was this prediction correct?", parent=self.root)
        
      
        if review == 'yes':
            tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Great! Glad the prediction was correct.", parent=self.root)
        elif review == 'no':
       
            correct_class = simpledialog.askstring("Correct Class", "What is the correct class?", parent=self.root)
            if correct_class == self.class1:
                img.save(os.path.join(self.proj_name, self.class1, f"{self.class1_counter}.png"), "PNG")
                self.class1_counter += 1
            elif correct_class == self.class2:
                img.save(os.path.join(self.proj_name, self.class2, f"{self.class2_counter}.png"), "PNG")
                self.class2_counter += 1
            elif correct_class == self.class3:
                img.save(os.path.join(self.proj_name, self.class3, f"{self.class3_counter}.png"), "PNG")
                self.class3_counter += 1
            else:
                tkinter.messagebox.showwarning("Warning", "Invalid class name provided. Please try again.", parent=self.root)
        
        self.clear()

    def rotate_model(self):
        model_types = [LinearSVC, KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GaussianNB]
        current_model_index = model_types.index(type(self.clf))
        next_model_index = (current_model_index + 1) % len(model_types)
        self.clf = model_types[next_model_index]()
        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump(self.clf, f)
            tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as f:
                self.clf = pickle.load(f)
            tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully loaded!", parent=self.root)

    def save_everything(self):
        data = {"c1": self.class1, "c2": self.class2, "c3": self.class3, "c1c": self.class1_counter,
                "c2c": self.class2_counter, "c3c": self.class3_counter, "clf": self.clf, "pname": self.proj_name}
        with open(os.path.join(self.proj_name, f"{self.proj_name}_data.pickle"), "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Project successfully saved!", parent=self.root)

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

DrawingClassifier()
