# Agent4Molecule


## Frontend initial setup  
  1. Clone the repository to the system/server. Our existing MCP server submit jobs to the PSC server.
  2. Install the required packages to run the agent and frontend code: ```pip install -r requirements.txt```
  3. Modify line 21 and update USER_DIR to a directory where all agent states and chat histories can be saved.
  4. In src/mcp_agent/, modify all paths at the top of enzygen_server.py,heme_binder_server.py, and ppdiff_server.py to point to the right package locations on your system.
  5. In src/mcp_agent/utils, modify all paths in each .py file to be valid on your system.
  6. Make sure you have a valid gemini API key before running the UI.
  7. Follow the instructions to set up the environment for enzygen and heme binder.

## Running the Frontend
  1. Modify line 121 of src/chat_interface.py with the path of the MCP server you wish to run the agent with. Currently, it is set up with enzygen_server.py.
  2. Before running the frontend, export GEMINI_API_KEY with your key. Do not hardcode and push this token to github.
  3. To run the frontend, run the following. The port can be changed from 8050 as needed.
  ```
  streamlit run src/chat_interface.py --server.port=8050 --server.address=0.0.0.0
  streamlit run src/chat_interface.py --server.port=<PORT> --server.address=0.0.0.0
  ```
  The UI can now be accessed at http://localhost:8050/, or http://localhost:<PORT>/.

  4. If the UI is launched from a compute node instead of a login node, perform the following steps:
     a. Request the Interact GPU and take note of the compute node number. On the PSC system, this can range from v001 to v024.
     b. Export GEMINI_API_KEY with your key on the compute node's environment.
     c. Run the following command with a port such as 8050: ```streamlit run src/chat_interface.py --server.port=<GPU_PORT> --server.address=0.0.0.0 ```
     d. On your local terminal, set up another connection to the remote server using port forwarding with a new local port. For example, if using PSC, a compute node of v007, and a local port of 9001, the command would be
     ```
     ssh -L 9001:v007:8050 username@bridges2.psc.edu
     ssh -L LOCAL_PORT:GPU_NO:GPU_PORT username@server_address
     ```
     e. The UI is now accessed at http://localhost:9001/, or http://localhost:<LOCAL_PORT>/.

## Adding custom MCP servers to the UI  
To make an MCP server work our UI setup, we recommend adding descriptions for each function and parameter. Specifically, each non-internal function (those decorated with @mcp.tool()) must return a dictionary with the following key:value pairs:
  - "status": <string>
      - "successful": The function successfully finished.
      - "error": There was some error in the code execution. Use this flag when an exception is caught in a try-catch block.
  - "answer": <string>
      - Verbose answer of what output was produced. Include the file path and any other pertinent information. This field is ultimately displayed on the UI side.
  - "visualize": <string>
      - "none": No visualization expected for this function
      - "plot": The frontend uses st.image() to render the file in the "file_path" key.
      - "molecule" : The frontend uses molviewspec to render the file in the "file_path" key.
      - "docking": The frontend adds a ligand to the molecular visualization.
  - "message_render": <string>
      - "text": Use regular st.markdown() to display the text in "answer".
      - "table": If the "answer" key is in some columnar format, use st.table() to display the result.
  - "file_path": <string> 
      - A single or list of files to be visualized. Use "none" if there is nothing to visualize.

For examples, refer to the existing MCP servers in mcp_agent (enzygen_server.py, heme_binder_server.py, ppdiff_server.py).



