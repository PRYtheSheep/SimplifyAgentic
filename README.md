# SimplyfyNext IGNITE Agentic 2025
By Team Ninja Turtle University <br>
Members:
- Melvin
- Ruo-Yang
- Dominic
## Installation
1. Create a new conda envrionment with the envrionment.yml with `conda env create -f .../envrionment.yml`. This will create a new envrionment named "agentic".
2. Manually install whisper with `pip install openai-whisper`. This also installs pytorch without cuda by default. Install pytorch with cuda if you wish to run whisper on your local GPU. 
4. Ensure ffmpeg is installed on device. This is needed for whisper to run.
5. Create a `.env` file with the following fields:
## Run
1. Open conda command prompt and activate the agentic envrioment `conda activate agentic`
2. Run the streamlit app with `Streamlit run ".../path/to/app/app.py"`. This will open a web browser to the streamlit interface.
3. Select an input media, enter in the context and run
