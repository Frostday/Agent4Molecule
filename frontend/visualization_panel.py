# visualization_panel.py
import streamlit as st
import py3Dmol
from Bio.PDB import PDBParser
import molviewspec as mvs

def init_visualization_state():
    """Ensure visualizations list exists in session state."""
    if "visualizations" not in st.session_state:
        st.session_state.visualizations = []


def add_visualization(viz_data: dict):
    """Add a new visualization to the state list."""
    init_visualization_state()
    st.session_state.visualizations.append(viz_data)


def remove_visualization(index: int):
    """Remove visualization by index."""
    if 0 <= index < len(st.session_state.visualizations):
        st.session_state.visualizations.pop(index)


def render_visualization_panel():
    """Render the sidebar or right pane with all current visualizations."""
    init_visualization_state()


    if len(st.session_state.visualizations) == 0:
        st.sidebar.info("No visualizations yet.")
        return
    

    for i, viz in enumerate(st.session_state.visualizations):
        # with st.container():
        with st.sidebar.expander(f"{viz.get('name', f'Visualization {i+1}')}", expanded=False):
            render_viz(viz)
            if st.button("Remove", key=f"remove_viz_{i}"):
                remove_visualization(i)
                st.rerun()
            # st.markdown(f"**{viz['title']}**")
            # st.image(viz["url"], use_column_width=True)
            # if st.button(f"🗑 Remove", key=f"remove_{i}"):
            #     st.session_state.visualizations.pop(i)
            #     st.rerun()

    # st.sidebar.markdown("### 🧬 Visualizations")
    # for i, viz in enumerate(st.session_state.visualizations):
    #     with st.sidebar.expander(f"{viz.get('name', f'Visualization {i+1}')}", expanded=False):
    #         render_viz(viz)
    #         if st.button("Remove", key=f"remove_viz_{i}"):
    #             remove_visualization(i)
    #             st.rerun()




def render_viz(content):
    """Decide which visualization library to use."""
    if content["visualize"] in ["docking","molecule"]:
        try:
            # --- Option 1: molviewspec rendering ---
            # if isinstance(content["file_path"], list): #fix
            #     file_path= content['file_path'][-1]
            # else:
            #      file_path= content['file_path']
            import ast

            file_path = st.literal_eval(content["file_path"])[-1]
            with open(file_path, 'r') as f:
                    pdb_data = f.read()

            print("file_path", file_path)
            sanitized_name = file_path.split("/")[-1]
            data_dict = {}
            data_dict[sanitized_name] = pdb_data.encode('utf-8')
            builder = mvs.create_builder()



            structure = builder.download(url=sanitized_name).parse(format="pdb").model_structure()
            whole = structure.component(selector="polymer").representation(type="cartoon").color(
            color="lightblue")

            

            if content['visualize'] == 'docking':
                structure.component(selector="ligand").representation(
                    type="cartoon"
                            ).color(color="orange")



            else:

                parser = PDBParser(QUIET=True)
                bio_structure = parser.get_structure("protein", content["file_path"])
                chain_residues = {}

                for model in bio_structure:
                    for chain in model:
                        residues = [residue.get_id()[1] for residue in chain if residue.get_id()[0] == " "]
                        chain_residues[chain.id] = residues
        

                all_chains = list(chain_residues.keys())
                selected_chains = st.multiselect(
                    "Select Chain(s):",
                    options=all_chains,
                    default=all_chains[:1] if all_chains else [],
                    help="Choose one or more chains from the structure."
                )

                selected_ranges = {}
                # --- Residue range sliders per chain ---
                for chain_id in selected_chains:
                    residues = chain_residues[chain_id]
                    if residues:
                        min_res, max_res = min(residues), max(residues)
                        selected_range = st.slider(
                        f"Select residue range for chain {chain_id}",
                        min_value=min_res,
                        max_value=max_res,
                        value=(min_res, min_res + min(50, max_res - min_res)),  # show first 50 by default
                        step=1,
                    )
                    selected_ranges[chain_id] = selected_range




            # selectors = []

            # for model in bio_structure:
            #     for chain in model:
            #         print(f"Chain {chain.id}")
            #         for residue in chain:
            #             res_name = residue.get_resname()   # 3-letter residue name, e.g., "ALA"
            #             res_id = residue.get_id()[1]       # residue number
            #             selectors.append(f"{chain.id}:{res_id}")
            #             st.write(chain,res_id)
            
            # my_selector = ",".join(selectors[:10])
            # st.write(my_selector)
    
                key = 'A'
            
            # st.write(selected_ranges[key][0])
            
                whole.color(color="red",selector=mvs.ComponentExpression(label_asym_id=selected_chains[0],
                                                                        beg_label_seq_id=selected_ranges[key][0], end_label_seq_id= selected_ranges[key][1]))


            snapshot = builder.get_snapshot(
            title=f"{sanitized_name} Viewer",
            description=f"Displaying uploaded PDB: {sanitized_name}"
                    )
            metadata = mvs.GlobalMetadata(description="Uploaded PDB Viewer")
            states_data = mvs.States(snapshots=[snapshot], metadata=metadata)

            mvs.molstar_streamlit(state=states_data,data=data_dict, molstar_version='5.0.0-dev.6')
        
        except Exception:
            # --- Fallback: py3Dmol viewer ---
            view = py3Dmol.view(width=800, height=600)
            pdb_data = content.get("pdb_content", "")
            view.addModel(pdb_data, "pdb")
            view.setStyle({}, {"cartoon": {"color": "lightblue"}})
            view.zoomTo()
            st.components.v1.html(view._make_html(), height=600, width=800)
    else:
        st.image(content["file_path"], use_container_width=True)


