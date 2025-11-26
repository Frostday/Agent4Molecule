# Agent4Molecule


## Frontend setup and execution instructions
  1. Have streamlit and py3dmol installed (pip install streamlit py3dmol).
  2. Set up a chat_history/ folder in frontend/ . This will contain all the local chat sessions with the agent
  3. Before running the frontend, export GEMINI_API_KEY with a gemini api key. Do not hardcode and push this token
  4. To run the frontend, run the following from the frontend/ folder. The port 8050 can be changed as needed.
  ```
  streamlit run chat_interface.py --server.port=8050 --server.address=0.0.0.0
  ```

## Run

```python client.py enzygen_server.py

python client.py heme_binder_server.py
python client.py docking_server.py
python client.py gromacs_server.py
```


export GEMINI_API_KEY=AIzaSyBAPm9hDxFiM0kSGGdCQDvQtA7l2RjD6d4