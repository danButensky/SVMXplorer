import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from svm_model import SVM_Model

class SVMPlot:
    def __init__(self):
        """
        Initialize the SVMPlot class.
        """
        # Initialize data for class P and class N
        self.blue_data = np.array([[2.0, 5], [3, 4], [4, 2], [5, 1]])  # data of class P
        self.green_data = np.array([[6.0, 10], [7, 5], [8, 6], [9, 8]])  # data of class N
        
        # Initialize the SVM model with default parameters
        self.rbf_model = SVM_Model(kernel='linear', degree=3)
        
        # Create a figure and axes for the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Initialize instance variables for UI elements
        self.clear_button = None
        self.boundary = 0
        self.minLim = 0
        self.maxLim = 10
        self.h = .02 # mesh step size

        # Create scatter plots for class P and class N
        self.blue_scatter = self.ax.scatter([],[], color='blue', marker='s')
        self.green_scatter = self.ax.scatter([],[], color='green', marker='^')

        # Create a button to clear the points
        clear_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.clear_button = widgets.Button(clear_button_ax, 'Clear')

        # Create UI elements for selecting SVM parameters
        kernel_options = ['linear', 'poly', 'rbf']
        radio_ax  = plt.axes([0.55, 0.7, 0.1, 0.2])
        self.kernel_buttons  = widgets.RadioButtons(radio_ax , kernel_options)

        slider_ax = plt.axes([0.55, 0.6, 0.4, 0.05])
        self.C_slider = widgets.Slider(slider_ax, 'C:', 0.1, 10, valinit=1, valstep=0.1)

        slider_ax = plt.axes([0.55, 0.5, 0.4, 0.05])
        self.Coef0_slider = widgets.Slider(slider_ax, 'Coef0:', 0, 100, valinit=0, valstep=1)

        slider_ax = plt.axes([0.55, 0.4, 0.4, 0.05])
        self.Degree_slider = widgets.Slider(slider_ax, 'Degree:', 0, 12, valinit=3, valstep=1)

        gamma_options = ['scale', 'auto']
        radio_ax  = plt.axes([0.57, 0.15, 0.1, 0.2])
        self.gamma_buttons  = widgets.RadioButtons(radio_ax , gamma_options)

        slider_ax = plt.axes([0.7, 0.3, 0.2, 0.05])
        self.Gamma_slider = widgets.Slider(slider_ax, 'Gamma:', 0.1, 10, valinit=0.1, valstep=0.1)

    def onclick(self, event):
        """
        Event handler for mouse clicks on the plot.

        Parameters:
            event (matplotlib.backend_bases.MouseEvent): Mouse event object.
        """
        # Check if the event occurred on the plot
        if event.inaxes == self.ax:
            # Get the clicked coordinates
            x = event.xdata
            y = event.ydata

            if x is not None and y is not None:
                if event.button == 1:
                    self.add_point('blue', x, y)
                else:
                    self.add_point('green', x, y)

    def clear_points(self, event):
        """
        Event handler for clearing the points.

        Parameters:
            event (matplotlib.backend_bases.MouseEvent): Mouse event object.
        """
        # Clear the coordinate lists for class P and class N
        self.blue_data = np.array([[3, 3]])
        self.green_data = np.array([[8, 8]])

        # Remove the existing scatter plots
        self.blue_scatter.remove()
        self.green_scatter.remove()

        # Create empty scatter plots for class P and class N
        self.blue_scatter = self.ax.scatter([],[], color='blue', marker='s')
        self.green_scatter = self.ax.scatter([],[], color='green', marker='^')

        # Reset the SVM parameters
        self.C_slider.set_val(self.rbf_model.c)
        self.Coef0_slider.set_val(self.rbf_model.coef0)
        self.Degree_slider.set_val(self.rbf_model.degree)

        # Redraw the plot
        self.update()

    def add_point(self, label, x, y):
        """
        Add a data point to the specified class.

        Parameters:
            label (str): Label of the class ('blue' for class P, 'green' for class N).
            x (float): X-coordinate of the data point.
            y (float): Y-coordinate of the data point.
        """
        if label == 'green':
            self.green_data = np.concatenate((self.green_data, [[x,y]]))
        else:
            self.blue_data = np.concatenate((self.blue_data, [[x,y]]))
        self.update()

    def update(self):
        """
        Update the plot and SVM model based on the data points and parameters.
        """
        # Update scatter plots for class P and class N
        blue_points = np.hsplit(self.blue_data, 2)
        green_points = np.hsplit(self.green_data, 2)

        self.blue_scatter.set_offsets(np.c_[blue_points[0],blue_points[1]])
        self.green_scatter.set_offsets(np.c_[green_points[0], green_points[1]])

        # Prepare the training data for SVM model
        x_train = np.append(self.blue_data, self.green_data, axis=0)
        y_train = np.append(np.zeros(len(self.blue_data), dtype=int), np.ones(len(self.green_data), dtype=int), axis=0)

        # Fit the SVM model to the training data
        self.rbf_model.fit_model(x_train, y_train)

        self.fig.canvas.draw_idle()

        # Draw the decision boundary
        self.draw_decision_boundary()
        plt.pause(0.0001)

    def draw_decision_boundary(self):
        """
        Draw the decision boundary on the plot.
        """
        # Remove the old decision boundary if it exists
        if not self.boundary == 0:
            for b in self.boundary.collections:
                b.remove()

        # Create a mesh grid to plot the decision boundary
        xx, yy = np.meshgrid(np.arange(self.minLim, self.maxLim, self.h),
                             np.arange(self.minLim, self.maxLim, self.h))

        Z = self.rbf_model.predict_model(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        self.boundary = self.ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    def select_kernel(self, label):
        """
        Select the kernel type for the SVM model.

        Parameters:
            label (str): Label of the selected kernel type.
        """
        self.rbf_model.kernel = label
        self.update()

    def select_gamma(self, label):
        """
        Select the gamma value for the SVM model.

        Parameters:
            label (str): Label of the selected gamma value.
        """
        self.rbf_model.gamma = label
        self.update()

    def update_GammaSlider(self, val):
        """
        Update the gamma value using the Gamma slider.

        Parameters:
            val (float): New value of the Gamma slider.
        """
        for circle in self.gamma_buttons.circles:
            circle.set_facecolor('white')

        # Get the current value of the slider
        new_gamma = self.Gamma_slider.val

        # Update model and plot
        self.rbf_model.gamma = new_gamma

        self.update()

    def update_slider(self, val):
        """
        Update the SVM parameters using the sliders.

        Parameters:
            val (float): New value of the slider.
        """
        # Get the current value of the sliders
        new_coef0 = self.Coef0_slider.val
        new_C = self.C_slider.val
        new_degree = self.Degree_slider.val

        # Update model and plot
        self.rbf_model.coef0 = new_coef0
        self.rbf_model.c = new_C
        self.rbf_model.degree = new_degree
        self.update() 

    def run_plot(self):
        """
        Run the SVM plot application.
        """
        # Connect the onclick event to the plot
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Connect the button click event handler function to the clear button
        self.clear_button.on_clicked(self.clear_points)

        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.5, wspace=0.2)

        # Connect the kernel selection event handler function to the kernel buttons
        self.kernel_buttons.on_clicked(self.select_kernel)

        title = "Gamma:"
        title_ax = self.fig.add_axes([0.48, 0.3, 0.1, 0.1], frame_on=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=10)

        # Connect the gamma selection event handler function to the gamma buttons
        self.gamma_buttons.on_clicked(self.select_gamma)

        # Connect the slider event handler functions to the sliders
        self.C_slider.on_changed(self.update_slider)
        self.Coef0_slider.on_changed(self.update_slider)
        self.Degree_slider.on_changed(self.update_slider)
        self.Gamma_slider.on_changed(self.update_GammaSlider)

        self.update()

        # Show the plot in a separate window
        plt.show()