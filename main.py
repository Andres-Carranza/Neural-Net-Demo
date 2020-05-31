from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import model
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

class graphics:

    def __init__(self):
        global window
        global canvas
        global mouse_pressed
        global canvas_image
        global draw
        global status_data

        mouse_pressed = False

        window = tk.Tk()

        menu = tk.Frame()
        menu.pack()

        clear = tk.Button(master=menu,text="Clear")
        clear.grid(row=0,column=0,sticky='w')
        clear.bind("<Button-1>",self.clear)

        capture = tk.Button(master=menu,text = "Capture")
        capture.grid(row=0,column=1,sticky='e')
        capture.bind("<Button-1>", self.capture)

        status = tk.Frame()
        status.pack(fill=tk.X )

        status_label = tk.Label(master=status,text="Status: ")
        status_label.grid(row=0,column=0,sticky='w')

        status_data = tk.Label(master=status,text="Starting program...")
        status_data.grid(row=0,column=1, padx=5)

        canvas = tk.Canvas(master=window, width=280, height=280,bg='black')
        canvas.pack()

        canvas_image = Image.new("L", (280, 280), color="black")
        draw = ImageDraw.Draw(canvas_image)

        canvas.bind("<Button-1>", self.mouse_clicked)
        canvas.bind("<ButtonRelease-1>", self.mouse_released)
        canvas.bind('<Motion>', self.motion)


    def mouse_clicked(self,event):
        global mouse_pressed
        global canvas
        global draw

        mouse_pressed= True


        x, y = event.x, event.y
        if mouse_pressed:
            canvas.create_oval(x, y, x + 30, y + 30, fill ="white", outline='white')
            draw.ellipse([x, y, x + 30, y + 30], fill ="white",outline='white')


    def mouse_released(self, event):
        global mouse_pressed

        mouse_pressed = False


    def motion(self,event):
        global mouse_pressed
        global canvas
        global draw

        x, y = event.x, event.y
        if mouse_pressed:
            canvas.create_oval(x, y, x + 30, y + 30, fill ="white", outline='white')
            draw.ellipse([x, y, x + 30, y + 30], fill ="white",outline='white')

    def clear(self,event):
        global canvas
        global canvas_image
        global draw

        canvas.delete("all")

        canvas_image = Image.new("L", (280, 280), color="black")
        draw = ImageDraw.Draw(canvas_image)


    def capture(self, event):
        global canvas_image
        global my_model
        global status_data

        pixels = np.asarray(canvas_image.resize((28, 28)))
        #canvas_image.save("capture.png")

        print(type(pixels))
        print(pixels.shape)

        pixels = np.array([pixels/255])

        print(type(pixels))
        print(pixels.shape)

        status_data['text'] ="Making prediction..."

        prediction = my_model.predict([pixels], batch_size =1)

        print(pixels[0])

        print(type(prediction))
        print(prediction.shape)
        print(prediction[0][1])

        prediction = prediction[0]

        num, acc = self.find_max(predictions=prediction)
        status_data['text'] ="Prediction: "+str(num) + " Confidence: "+ str(acc)

    def find_max(self,predictions):
        max = 0
        maxi = 0
        for index, value in enumerate(predictions):
            if value > max:
                max = value
                maxi = index
        return maxi,max

    def start_loop(self):
        global window
        window.mainloop()


    def setup_model(self,model):
        global status_data
        global my_model

        status_data['text'] ="Loading data..."

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        print(type(x_train))
        print(x_train.shape)
        print(type(x_train[0]))
        print(x_train[0].shape)

        x_train_normalized = x_train / 255
        x_test_normalized = x_test / 255

        status_data['text'] = "Training model..."

        # The following variables are the hyperparameters.
        learning_rate = 0.003
        epochs = 50
        batch_size = 4000
        validation_split = 0.2

        # Establish the model's topography.
        my_model = model.create_model(learning_rate)

        # Train the model on the normalized training set.
        epochs, hist = model.train_model(my_model, x_train_normalized, y_train,
                                         epochs, batch_size, validation_split)

        # Plot a graph of the metric vs. epochs.
        list_of_metrics_to_plot = ['accuracy']
        model.plot_curve(epochs, hist, list_of_metrics_to_plot)

        # Evaluate against the test set.
        print("\n Evaluate the new model against the test set:")
        my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

        status_data['text'] ="Model trained."


graphics = graphics()

graphics.setup_model(model=model())
graphics.start_loop()






