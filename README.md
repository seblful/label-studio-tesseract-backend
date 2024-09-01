# Interactive bounding boxes OCR using Tesseract

Use an OCR engine for interactive ML-assisted labeling, facilitating faster 
annotation for layout detection, classification, and recognition
models.

Tested against Label Studio 1.13.1.

## Setup process

Before you begin:
* Ensure git is installed
* Ensure Docker Compose is installed.


### 1. Install Label Studio

Launch Label Studio. You can follow the guide from the [official documentation](https://labelstud.io/guide/install.html) or use the following commands:


If you're using local file serving, be sure to [get a copy of the API token](https://labelstud.io/guide/user_account#Access-token) from
Label Studio to connect the model.

### 2. Create a Label Studio project

Create a new project for Tesseract OCR. In the project **Settings** set up the **Labeling Interface**.

### 3. Install Tesseract OCR

Download the Label Studio Tesseract backend repository.
   ```
   git clone https://github.com/seblful/label-studio-tesseract-backend.git
   cd label-studio-tesseract-backend
   ```

Configure parameters in `.env` file:

   ```
   LABEL_STUDIO_HOST=<IPv4 Address (check your ipconfig)>
   LABEL_STUDIO_ACCESS_TOKEN=<optional token for local file access>
   ```

### 4. Start the Tesseract and MinIO servers

   ```
   docker compose up
   ```

### 5. Upload tasks

   Upload images directly to Label Studio using the Label Studio interface.


### 6. Add model in project settings

From the project settings, select the **Model** page and click [**Connect Model**](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).

   Add the URL `http://localghost:9090` and save the model as an ML backend.

### 7. Label in interactive mode

To use this functionality, activate **Auto-Annotation** and use the `Autodetect` rectangle for drawing boxes
