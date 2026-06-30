# LipRead Studio

This repository now includes a small local inference app:

- a FastAPI backend that loads the existing LipNet weights once and exposes `/predict`
- a React frontend that uploads a video and displays the transcription

# LipNet: End-to-End Sentence-level Lipreading
Keras implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599).

![LipNet performing prediction (subtitle alignment only for visualization)](assets/lipreading.gif)

## Results
|       Scenario          | Epoch |  CER  |  WER  |  BLEU |
|:-----------------------:|:-----:|:-----:|:-----:|:-----:|
|  Unseen speakers [C]    |  N/A  |  N/A  |  N/A  |  N/A  |
|    Unseen speakers      |  178  |  6.19%  |  14.19%  |  88.21%  |
| Overlapped speakers [C] |  N/A  |  N/A  |  N/A  |  N/A  |
|   Overlapped speakers   |  368  |  1.56%  |  3.38%  |  96.93%  |

**Notes**:

- [C] means using curriculum learning.
- N/A means either the training is in progress or haven't been performed.
- Your contribution in sharing the results of this model is highly appreciated :)

## Dependencies
* Keras 2.0+
* Tensorflow 1.0+
* PIP (for package installation)

Plus several other libraries listed on `setup.py`

## Usage
To use the model, first you need to clone the repository:
```
git clone https://github.com/rizkiarm/LipNet
```
Then you can install the package:
```
cd LipNet/
pip install -e .
```
**Note:** if you don't want to use CUDA, you need to edit the ``setup.py`` and change ``tensorflow-gpu`` to ``tensorflow``

You're done!

## Quick Test Video

Use the bundled sample clip at [evaluation/samples/id2_vcd_swwp2s.mpg](evaluation/samples/id2_vcd_swwp2s.mpg) for a quick test. It is already in the repo, so you can copy it into any folder you want to upload from.

If you want another known-good sample, the `evaluation/samples/GRID/` folder also contains additional clips.

## Run The Demo

On Windows PowerShell, start both services with:

```powershell
.\start.ps1
```

Add `-OpenBrowser` if you want the frontend to open automatically.

The launcher starts the backend and frontend in hidden background windows, then opens the browser to the site.

After that, open the frontend at `http://127.0.0.1:5173` and check the API at `http://127.0.0.1:8000/health`.

## Local Share Guide (No Docker)

If you want to share this project with teammates without Docker, include the repository (Git or ZIP) and instruct them to run the provided start scripts and weight-download scripts.

- `start.ps1` — existing Windows launcher (hidden windows).
- `start.sh` — Unix launcher (creates `.venv`, installs requirements, starts backend & frontend).
- `download_weights.sh` / `download_weights.ps1` — helper scripts to download model weights into `evaluation/models/`.

Quick commands:

Windows (PowerShell):

```powershell
.\start.ps1 -OpenBrowser
```

Unix (bash):

```bash
chmod +x start.sh
./start.sh
```

Download weights (example):

```bash
./download_weights.sh https://example.com/overlapped-weights368.h5
```

Sample test:
- Upload `evaluation/samples/id2_vcd_swwp2s.mpg` to the frontend.
- Expected transcript: `set white in c two soon`

Logs: `backend.log`, `frontend.log`. Health check: `http://127.0.0.1:8000/health`.

Here is some ideas on what you can do next:
* Modify the package and make some improvements to it.
* Train the model using predefined training scenarios.
* Make your own training scenarios.
* Use [pre-trained weights](https://github.com/rizkiarm/LipNet/tree/master/evaluation/models) to do lipreading.
* Go crazy and experiment on other dataset! by changing some hyperparameters or modify the model.

## Dataset
Source

This project uses a subset of the GRID Audiovisual Speech Corpus — a large, publicly available dataset of speakers reciting short sentences to camera at 25 fps.
Subset Selection
Rather than using the full corpus, we selected a focused subset for word-level classification:
PropertyValueTask12-class word classificationWordsagain, bin, blue, green, lay, now, place, please, red, set, soon, whiteSpeakerss1, s2, s14, s26. Clips per word per speaker 15 Total clips 720
Preprocessing
Raw .mpg video files were processed using MediaPipe Face Mesh to detect and crop the lip region from each frame. Each clip was standardized to 20 frames (with ±5 context frames around the word boundary) and saved as a NumPy array of shape (20, 96, 96) — 20 grayscale frames at 96×96 pixels.
Clip filenames encode the speaker, video ID, and frame range, e.g.:
s1_bwag6p_f021-026.npy  →  speaker s1, video bwag6p, frames 21–26
Directory Structure
data/
└── train/
    ├── again/
    ├── bin/
    ├── blue/
    ├── green/
    ├── lay/
    ├── now/
    ├── place/
    ├── please/
    ├── red/
    ├── set/
    ├── soon/
    └── white/
Label Mapping :
Labels are integer-encoded (see labels.json):
json{"again": 0, "bin": 1, "blue": 2, "green": 3, "lay": 4, "now": 5,
 "place": 6, "please": 7, "red": 8, "set": 9, "soon": 10, "white": 11}

## Pre-trained weights
For those of you who are having difficulties in training the model (or just want to see the end results), you can download and use the weights provided here: https://github.com/rizkiarm/LipNet/tree/master/evaluation/models. 

More detail on saving and loading weights can be found in [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

## Training
There are five different training scenarios that are (going to be) available:

### Prerequisites
1. Download all video (normal) and align from the GRID Corpus website.
2. Extracts all the videos and aligns.
3. Create ``datasets`` folder on each training scenario folder.
4. Create ``align`` folder inside the ``datasets`` folder.
5. All current ``train.py`` expect the videos to be in the form of 100x50px mouthcrop image frames.
You can change this by adding ``vtype = "face"`` and ``face_predictor_path`` (which can be found in ``evaluation/models``) in the instantiation of ``Generator`` inside the ``train.py``
6. The other way would be to extract the mouthcrop image using ``scripts/extract_mouth_batch.py`` (usage can be found inside the script).
7. Create symlink from each ``training/*/datasets/align`` to your align folder.
8. You can change the training parameters by modifying ``train.py`` inside its respective scenarios.

### Random split (Unmaintained)
Create symlink from ``training/random_split/datasets/video`` to your video dataset folder (which contains ``s*`` directory).

Train the model using the following command:
```
./train random_split [GPUs (optional)]
```

**Note:** You can change the validation split value by modifying the ``val_split`` argument inside the ``train.py``.
### Unseen speakers
Create the following folder:
* ``training/unseen_speakers/datasets/train``
* ``training/unseen_speakers/datasets/val``

Then, create symlink from ``training/unseen_speakers/datasets/[train|val]/s*`` to your selection of ``s*`` inside of the video dataset folder.

The paper used ``s1``, ``s2``, ``s20``, and ``s22`` for evaluation and the remainder for training.

Train the model using the following command:
```
./train unseen_speakers [GPUs (optional)]
```
### Unseen speakers with curriculum learning
The same way you do unseen speakers.

**Note:** You can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

```
./train unseen_speakers_curriculum [GPUs (optional)]
```

### Overlapped Speakers
Run the preparation script:
```
python prepare.py [Path to video dataset] [Path to align dataset] [Number of samples]
```
**Notes:**
- ``[Path to video dataset]`` should be a folder with structure: ``/s{i}/[video]``
- ``[Path to align dataset]`` should be a folder with structure: ``/[align].align``
- ``[Number of samples]`` should be less than or equal to ``min(len(ls '/s{i}/*'))``

Then run training for each speaker:
```
python training/overlapped_speakers/train.py s{i}
```

### Overlapped Speakers with curriculum learning
Copy the ``prepare.py`` from ``overlapped_speakers`` folder to ``overlapped_speakers_curriculum`` folder, 
and run it as previously described in overlapped speakers training explanation.

Then run training for each speaker:
```
python training/overlapped_speakers_curriculum/train.py s{i}
```
**Note:** As always, you can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

## Evaluation
To evaluate and visualize the trained model on a single video / image frames, you can execute the following command:
```
./predict [path to weight] [path to video]
```
**Example:**
```
./predict evaluation/models/overlapped-weights368.h5 evaluation/samples/id2_vcd_swwp2s.mpg
```
## Work in Progress
This is a work in progress. Errors are to be expected.
If you found some errors in terms of implementation please report them by submitting issue(s) or making PR(s). Thanks!

**Some todos:**
- [X] Use ~~Stanford-CTC~~ Tensorflow CTC beam search
- [X] Auto spelling correction
- [X] Overlapped speakers (and its curriculum) training
- [ ] Integrate language model for beam search
- [ ] RGB normalization over the dataset.
- [X] Validate CTC implementation in training.
- [ ] Proper documentation
- [ ] Unit tests
- [X] (Maybe) better curriculum learning.
- [ ] (Maybe) some proper scripts to do dataset stuff.

## License
MIT License
