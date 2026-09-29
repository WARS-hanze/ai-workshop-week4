# Gesture Iteration Workshop

In this workshop, you will iterate on a hand gesture recognition system. 
The system is based on Google's MediaPipe, and runs locally on your laptop 
(no data leaves your laptop). 

This system can theoretically recognize rock, paper & scissors, but it's 
missing training data! You will iteratively improve the system and expand on it.

## Getting Started 

### Setting up the environment

In order to get started, several things are needed:

 - `anaconda` (visit [anaconda.com/download](https://www.anaconda.com/download) and download the latest version)

Anaconda is a powerful environment / package management tool that is widely used in order to manage working with various (mostly python) packages. We will be using it to install our python packages and other dependencies we need to the MediaPipe app.

Now, after installing anaconda, run the program `anaconda powershell prompt`. The resulting screen should display something similar to this: `(base) PS C:\Users\Steven Warmelink>`. In this terminal, run the following commands in succession:

0. Get the repository

Download [this repository](https://github.com/WARS-hanze/ai-workshop-week4/tree/master) (either clone the repository or download and extract the zip). Run anaconda powershell prompt in the folder that contains the repository files, or start anaconda powershell prompt and move there using `cd'. 

1. Create the conda environment and install the required packages
```
conda env create -f environment.yml
```

2. Activate the conda environment
```
conda activate ai-week4-workshop
```

3. Install mediapipe.
```
python -m pip install mediapipe==0.10.21
```

4. Run the python app. 
```
python app.py --session round0
```

**The `--session round0' part is the name you can give to a session. If you later start a run with the same session name, it uses the data you stored during that run. Otherwise, it starts blank (with no training data).**

## Using the application
A window opens showing your webcam feed with hand landmarks drawn on top.

**Key bindings** (click the window first so it has focus):

| Key | Action |
|---|---|
| `t` | Toggle **Train** / **Test** mode |
| `r` / `p` / `s` | (Train mode) add the current hand pose as rock / paper / scissors |
| `u` | Undo the last added sample |
| `c` | Clear all samples in this session |
| `w` | Save now (it also auto-saves on quit) |
| `q` or `Esc` | Quit |
| `n` | Normalize (toggle on/off) | 

### Round workflow

Each round is a separate save file under `sessions/`, so you can compare
rounds and never lose earlier data:

```
python app.py --session round0     # baseline
```

## FAQ
- **My webcam doesn't work!**: close other apps that
  might be using it (Zoom, Teams, browser tabs with camera access). On
  Windows, if it still fails, try editing `app.py` and changing
  `cv2.VideoCapture(0)` to `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.
- **I get warnings about inference_feedback_manager, and log messages on startup!** Harmless warnings, feel free to ignore
