import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.playback import play

class AudioVisualizerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Visualizer")

        # Create the canvas to display the spectrogram
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # Add a button to load and play the audio file
        self.load_button = tk.Button(self.master, text="Load and Play Audio", command=self.load_audio)
        self.load_button.pack(side=tk.TOP, padx=10, pady=10)

    # Function to load the audio file, play the audio, and display the spectrogram
    def load_audio(self):
        # Load the audio file using pydub
        audio_data = AudioSegment.from_file("C:/Users/Archi/pyt/ROBOTS/DressDown.mp3", format="mp3")

        # Extract the sample rate and convert the audio data to a NumPy array
        sample_rate = audio_data.frame_rate
        audio_array = np.array(audio_data.get_array_of_samples(), dtype=np.float32) / 32767.0

        # Compute the spectrogram using a Fourier transform
        frequencies, times, spectrogram = signal.spectrogram(audio_array, sample_rate)

        # Plot the spectrogram
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = ax.pcolormesh(times, frequencies, np.log(spectrogram + 1e-9))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        self.canvas.draw()

        # Define a function to update the spectrogram
        def update_spectrogram():
            # Replace with your own audio reading code
            audio_data = AudioSegment.from_file("C:/Users/Archi/pyt/ROBOTS/DressDown.mp3", format="mp3")

            # Convert the audio data to a NumPy array
            audio_array = np.array(audio_data.get_array_of_samples(), dtype=np.float32) / 32767.0

            # Update the spectrogram
            frequencies, times, spectrogram = signal.spectrogram(audio_array, sample_rate)
            img.set_array(np.log(spectrogram))
            self.canvas.draw()

            # Play the audio
            play(audio_data)

            # Call this function again in 50 milliseconds
            self.master.after(50, update_spectrogram)

        # Call the update function to start updating the spectrogram
        self.master.after(50, update_spectrogram)

        # Play the audio
        play(audio_data)


# Create the main window and run the GUI
root = tk.Tk()
app = AudioVisualizerGUI(root)
root.mainloop()
