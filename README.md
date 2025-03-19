# Handwritten Text Recognition with TrOCR

This project is a Streamlit-based web application that recognizes handwritten text from images using Microsoft's TrOCR model.

## Features
- Upload an image file or provide an image URL
- Recognize handwritten text using the TrOCR model
- Display the uploaded image and extracted text

## Installation
To run this application, follow these steps:

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit application with:
```sh
streamlit run app.py
```

## Dependencies
This project requires the following Python libraries:
- `streamlit`
- `transformers`
- `PIL` (Pillow)
- `requests`
- `torch`

To install them, run:
```sh
pip install streamlit transformers pillow requests torch
```

## Model Details
This application uses the `microsoft/trocr-base-handwritten` model from the Hugging Face Transformers library. It processes images and extracts handwritten text using an encoder-decoder architecture.

## Example
You can test the app using the following example image:
```
https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg
```
Simply paste the URL into the input field or upload your own handwritten image.

## License
This project is licensed under the MIT License.

