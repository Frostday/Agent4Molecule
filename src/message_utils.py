# render_utils.py
import streamlit as st
import molviewspec as mvs
import py3Dmol
import pandas as pd
from io import StringIO
from visualization_panel import add_visualization
import ast

VIZ_RAND = 0
NUMS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
def render_message(role: str, content):

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    
    elif role == "tool":
        with st.chat_message("assistant"):
            label_preview = content["answer"].strip().replace("\n", " ")
            if len(label_preview) > 30:
                label_preview = label_preview[:30] + "…"
                with st.expander(label_preview):
      
                    if content.get("message_render") == "table":
                        data_str = content["answer"].split("\n\n", 1)[1]


                        df = pd.read_csv(StringIO(data_str))

                        st.dataframe(df) 
                    else:
                        st.text(f"{content['answer']}")  
            
        # Case 1: visualization of molecule

            if isinstance(content, dict) and content.get("visualize") != "none":
                if content.get("visualize") == 'plot':
                    try:
                        st.image(content["file_path"], use_container_width=True)
                    except Exception as e:
                        st.markdown('No image to show')
                else:
                 
                    if content["file_path"][0] == '[':
                        paths = ast.literal_eval(content["file_path"])
                        with st.expander("Molecule Viewer", expanded=False):
                            


                            choice = st.selectbox(
                                "Choose pdb:",
                                paths
                                )
                            _render_molecule({"file_path": choice})
                    else:
                        _render_molecule(content)
                
                return
    
    elif role == "assistant" or role == "model":

    # ---- ASSISTANT or TOOL ----
        with st.chat_message("assistant"):

        # Case 1: visualization of molecule
            if isinstance(content, dict) and content.get("visualize") != "none" and "file_path" in content:
                if content.get("visualize") == 'plot':
                    st.image(content["file_path"], use_container_width=True)
                else:
         
                    if content["file_path"][0] == '[':
                        paths = ast.literal_eval(content["file_path"])
                        with st.expander("Molecule Viewer", expanded=False):

        # --- File selector inside expander ---
                            choice = st.selectbox(
                                "Choose pdb:",
                                paths
                                )
                            _render_molecule({"file_path": choice})
                    else:
                        _render_molecule(content)
    
                return

        # Case 2: table-like data
            elif isinstance(content, dict) and "table" in content:
                data_str = content["answer"].split("\n\n", 1)[1]
                df = pd.read_csv(StringIO(data_str))
                st.dataframe(df)
                return

        # Case 3: tool response with JSON
            elif isinstance(content, dict) and "answer" in content:
                st.text(content["answer"])
                return

        # Case 4: plain text or markdown
            else:
                if isinstance(content,dict):
                    st.markdown(content['content'])
                else:
                    st.markdown(str(content))


def _render_molecule(content):
    """
    Renders molecule visualization from pdb_content or file_path.
    Chooses between molviewspec or py3Dmol.
    """

    try:
        # --- Option 1: molviewspec rendering ---

        with open(content['file_path'], 'r') as f:
                pdb_data = f.read()

        sanitized_name = "test_name"
        data_dict = {}
        data_dict[sanitized_name] = pdb_data.encode('utf-8')
        builder = mvs.create_builder()
        structure = builder.download(url="test_name").parse(format="pdb").model_structure()
        structure.component(selector="polymer").representation(type="cartoon").color(
        color="lightblue")

        if content['visualize'] == 'docking':
            structure.component(selector="ligand").representation(
                type="cartoon"
                        ).color(color="orange")
        snapshot = builder.get_snapshot(
                title=f"{sanitized_name} Viewer",
                description=f"Displaying uploaded PDB: {sanitized_name}"
                )
        metadata = mvs.GlobalMetadata(description="Uploaded PDB Viewer")
        states_data = mvs.States(snapshots=[snapshot], metadata=metadata)
        mvs.molstar_streamlit(state=states_data,data=data_dict, molstar_version='5.0.0-dev.6' ) 
       
    except Exception as e:
        # --- Fallback: py3Dmol viewer ---
        print(e)
        # view = py3Dmol.view(width=800, height=600)
        # pdb_data = content.get("pdb_content", "")
        # view.addModel(pdb_data, "pdb")
        # view.setStyle({}, {"cartoon": {"color": "lightblue"}})
        # view.zoomTo()
        # st.components.v1.html(view._make_html(), height=600, width=800)
        
