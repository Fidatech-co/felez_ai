# felez_ai

## Screen extraction

```bash
python screen_detector.py path/to/photo.jpg --output path/to/screen.png
```

If you omit `--output` the script writes `<photo>_screen.png` next to the original.

The saved screen crops are contrast-enhanced black & white images resized to 800x600 px so they can feed directly into downstream ML models.

## Number reader

Use EasyOCR to read gauge values from a processed screen:

```bash
python number_reader.py path/to/<photo>_screen.png --json
```

Add `--gpu` if your system has CUDA available; otherwise it runs on CPU.
