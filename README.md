# Sign Language to Speech

A webcam app that recognises American Sign Language (ASL) letters and speaks them aloud. This prototype was a finalist at Negotium.

**Demo:** [demo.mp4](demo.mp4)

## How it works

**Training (`sign_to_speech.py`)**
- Fine-tunes a ResNet-18, pretrained on ImageNet, on the Kaggle *ASL Alphabet* dataset.
- Augments the images with flips, small rotations, colour jitter and random crops, then trains for 10 epochs with Adam.
- Saves the weights to `asl_cnn_model.pth` and the class names to `asl_classes.txt`.

**Live prediction (`predict_speak2.py`)**
- Classifies every webcam frame and draws the predicted letter on screen.
- When the same letter is predicted 10 frames in a row, it speaks the letter with Google Text-to-Speech. A 2-second cooldown stops it repeating itself.

The letters **J** and **Z** are left out. They're signed with movement, which a model that looks at one frame at a time can't see. Leaving them out also kept training light enough for a laptop.

## Run it

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle. Unzip it so the training images sit in `asl_alphabet/asl_alphabet_train/<letter>/`. To match my setup, delete the `J` and `Z` folders.
3. Train the model (a GPU helps a lot):
   ```bash
   python sign_to_speech.py
   ```
4. Run the live demo. Press `q` to quit. Speech needs an internet connection, because gTTS uses Google's service.
   ```bash
   python predict_speak2.py
   ```

## Limitations and next steps

- **No held-out accuracy yet.** I judged the model by testing it live. The training script doesn't yet measure accuracy on images it hasn't seen. Adding a validation split is the next step.
- **Studio photos vs. a real webcam.** The dataset's photos have consistent lighting and backgrounds, so accuracy drops in other conditions. More varied training data would help.
- **Testing was limited by my own signing.** I'm still learning ASL, which made it hard to check every letter properly.
- **Motion letters** (J and Z) would need a model that looks at sequences of frames.

## What I learned

This was my first computer-vision project. I learned how transfer learning lets a pretrained network adapt to a new task with modest compute. I also learned that the preprocessing at prediction time has to match training exactly. An earlier version skipped the normalisation step during live prediction, which quietly made it less accurate. I built this with help from ChatGPT.
