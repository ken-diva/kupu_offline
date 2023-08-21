# Kupu Web Module

This project provides a web-based interface for automatic bone scan segmentation using machine learning and annotation tools for manual segmentation.

## Installation (Google Cloud Run)

<!-- docker build -t kupuwebmodule . 
docker tag kupuwebmodule gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule 
docker push gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule 
gcloud run deploy kupuwebmodule --image gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule -->

<!-- Download the **.pt** model from [here](https://drive.google.com/file/d/1IyIrCrXVGr_SwnqKClzv62y2g8QnUZis/view?usp=sharing)
  - Image and mask sample can be downloaded [here](https://drive.google.com/drive/folders/128bpdCCcgq91C7l0k3ZeSInM71rWXmw0?usp=sharing) (optional)
- Create folder named _model_ and then move **.pt** model there -->

### Requirements

[Google Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy) has to be enabled to deploy the application. The following steps can be run in the [Google Cloud Shell](https://cloud.google.com/shell/docs/quickstart) or locally using the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

### Build and Deploy

1. Clone this repository and navigate to the root directory of the project.

2. Download the **.pt** model from [here](https://drive.google.com/file/d/1IyIrCrXVGr_SwnqKClzv62y2g8QnUZis/view?usp=sharing) and move it to the _model_ folder.

3. Build the Docker image and push it to the Google Cloud Container Registry:

    ```bash
    docker build -t kupuwebmodule .
    docker tag kupuwebmodule gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule
    docker push gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule
    ```

4. Deploy the application to Google Cloud Run:

    ```bash
    gcloud run deploy kupuwebmodule --image gcr.io/{{GCLOUD-PROJECT-NAME}}/kupuwebmodule
    ```

5. After the deployment is complete, you will be provided with a URL to access the application. You can also find this URL by navigating to the [Cloud Run](https://console.cloud.google.com/run) page in the Google Cloud Console.

## Installation (Local)

### Requirements

- [Python 3.10.9](https://www.python.org/downloads/release/python-3109/)
- [Docker](https://docs.docker.com/get-docker/)

### Setup

1. Clone this repository and navigate to the root directory of the project.

2. Download the **.pt** model from [here](https://drive.google.com/file/d/1IyIrCrXVGr_SwnqKClzv62y2g8QnUZis/view?usp=sharing) and move it to the _model_ folder.

3. Create a virtual environment and activate it:

    ```bash
    pip install virtualenv # if not already installed
    virtualenv venv
    source venv/bin/activate
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Build the Docker image:

    ```bash
    docker build -t kupuwebmodule .
    ```

### Run

1. Run the Docker image:

    ```bash
    docker run -p 8080:8080 kupuwebmodule
    ```

2. Navigate to [http://localhost:8080](http://localhost:8080) in your browser.
