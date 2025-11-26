import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import sys
import json

from google import genai
from google.genai.types import GenerateContentConfig, Content, Part
from agent_utils import content_to_dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompts import SYSTEM_MESSAGE

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = "gemini-2.0-flash"

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        try:
            await self.session.initialize()
        except Exception as e:
            print("Initialization failed:", e)
            raise

        # List available tools
        response = await self.session.list_tools()
        for tool in response.tools:
            print(f"\nTool Name: {tool.name}")
            print(f"Description: {tool.description}")
        
        tools = response.tools

    async def process_query(self, query: str) -> str:
        messages = [
            genai.types.Content( 
                role='user',
                parts=[
                    genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
                ]
            )
        ]

        # Get available tools from MCP
        response = await self.session.list_tools()
        available_tools = [
            genai.types.Tool(
                function_declarations=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            k: v
                            for k, v in tool.inputSchema.items()
                            if k not in ["additionalProperties", "$schema"]
                        },
                    }
                ]
            )
            for tool in response.tools
        ]

        while True:
            response = self._client.models.generate_content(
                model=self.llm,
                contents=messages,
                config=GenerateContentConfig(
                    tools=available_tools
                )
            )
            messages.append(response.candidates[0].content)

            os.makedirs("outputs", exist_ok=True)
            with open("outputs/chat_history_temp.json", "w") as f:
                json.dump([content_to_dict(m) for m in messages], f, indent=2)

            flag = True
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_id = part.function_call.id
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args
                    print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    flag = False
                    messages.append(
                        genai.types.Content(
                            role='tool',
                            parts=[
                                genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": str(tool_result.content[0].text)}
                                )
                            ]
                        )
                    )
            if flag:
                break
            # print(messages)
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/eval_46112_docking.json", "w") as f:
            json.dump([content_to_dict(m) for m in messages], f, indent=2)
        # os.system("rm -f outputs/chat_history_temp.json")
        
        return messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        # query = input("\nQuery: ").strip()
        # query = "Design an enzyme that functions as an adenylate-processing protein, acting like a cyclase to transform ATP into 3’,5’-cyclic AMP while releasing pyrophosphate. The enzyme should resemble known adenylylcyclases in structure and activity, and be capable of catalyzing the formation of cyclic AMP as a signaling molecule. Try docking to make sure the generated structure has good binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/substrate_ligand.pdb."
        # query = "Design an enzyme that functions as a homoserine dehydrogenase, catalyzing the NAD⁺- or NADP⁺-dependent oxidation of L-homoserine to L-aspartate semialdehyde, a key step in the aspartate-derived amino acid biosynthetic pathway. The enzyme should resemble known homoserine dehydrogenases in structure, featuring a Rossmann-fold domain for cofactor binding and conserved catalytic residues that facilitate hydride transfer and stabilization of the carbinol intermediate. Ensure the designed structure maintains stereospecificity for L-homoserine.Convert the mol to pdb if necessary. Try docking to verify that the generated enzyme shows favorable binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.1.1.3/substrate_ligand.mol."
        # query = "Design an enzyme that functions as an acetaldehyde dehydrogenase, catalyzing the NAD⁺-dependent oxidation of acetaldehyde to acetyl-CoA. The enzyme should resemble known acylating aldehyde dehydrogenases (such as Ada, DmpF, or BphJ) that form a thiohemiacetal intermediate with coenzyme A before transferring the acyl group to generate acetyl-CoA. Include conserved cysteine and histidine residues at the active site to mediate nucleophilic attack and hydride transfer. Ensure the structure contains a Rossmann-fold domain for NAD⁺ binding and a well-defined CoA-binding channel. Try docking to confirm that the generated enzyme exhibits strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.2.1.10/substrate_ligand.mol."
        # query = "Design an enzyme that functions as a nicotinamide N-methyltransferase, catalyzing transfer of a methyl group from S-adenosyl-L-methionine (SAM) to nicotinamide to form 1-methylnicotinamide and S-adenosyl-L-homocysteine (SAH). The enzyme should resemble known NNMTs, featuring a Rossmann-like SAM-binding pocket and catalytic residues that position nicotinamide for N-methyl transfer. Maintain specificity for nicotinamide over related pyridine substrates and ensure an enclosed hydrophobic channel for SAM/SAH exchange. Try docking to verify that the generated structure shows favorable binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.1.1.1/substrate_ligand.mol."
        # query = "Design an enzyme that catalyzes the CoA-dependent cleavage of (5S)-5-amino-3-oxohexanoate by acetyl-CoA to yield (3S)-3-aminobutanoyl-CoA and acetoacetate. The structure should resemble known kce homologs, featuring a cysteine-based catalytic center and a CoA-binding tunnel for acyl transfer. Ensure stereospecific recognition of the (5S) substrate and stabilization of the β-keto intermediate. Try docking to confirm strong binding affinity to/jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.3.1.247/substrate_ligand.mol."
        # query = "Design an enzyme that functions as an AMP phosphorylase, catalyzing phosphorolysis of nucleoside 5′-monophosphates to yield the free base and α-D-ribose 1,5-bisphosphate. The primary reaction to support is: AMP + phosphate → adenine + α-D-ribose 1,5-bisphosphate (with optional activity toward CMP and UMP). The enzyme should resemble known deoA/AMPpase pentosyltransferases in fold and active-site architecture, with a phosphate-binding pocket and residues positioned to stabilize an oxocarbenium-like transition state during ribosyl transfer to Pi at C1. Maintain specificity for ribo NMPs (5′-phosphate recognized) and enforce base-binding geometry that tolerates purine/pyrimidine substrates while excluding deoxynucleotides. Try docking to confirm favorable binding for the intended NMP substrate and phosphate within the catalytic pocket using /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.4.2.57/substrate_ligand.mol."
        # query = "Design a Zn²⁺-dependent metallocarboxypeptidase that cleaves C-terminal residues from peptides with broad specificity, similar to carboxypeptidase T. The enzyme should recognize both aromatic and basic side chains while being hindered when the penultimate residue is proline, and it should contain a catalytic Glu as the general base plus a Zn²⁺-binding motif (His–Glu–His) forming the core of the active site. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.4.17.18/substrate_ligand.mol."
        # query = "Design an ATP-dependent kinase that phosphorylates the bacterial signaling molecule autoinducer-2 (AI-2), similar to the enzyme LsrK. The protein should contain an N-terminal substrate-binding domain and a C-terminal ATP-binding domain characteristic of sugar kinases, using a hinge motion to align ATP and AI-2 for phosphate transfer. Include residues for Mg²⁺ coordination and stabilization of the phosphoryl-transfer transition state. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.7.1.189/substrate_ligand.mol."
        # query = "Design an NAD⁺-dependent oxidoreductase that catalyzes the interconversion between 3alpha-hydroxy and 3-keto (or 17beta-hydroxy and 17-keto) steroid forms, similar to 3alpha(17beta)-hydroxysteroid dehydrogenase. The enzyme should adopt a classical short-chain dehydrogenase/reductase (SDR) fold with a conserved Tyr-Lys-Ser catalytic triad and a Rossmann-fold NAD⁺-binding pocket. Ensure the active site positions the steroid substrate for precise hydride transfer and maintains stereospecificity at the 3alpha or 17beta position. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.1.1.239/substrate_ligand.mol."
        # query = "Design an NAD⁺-dependent oxidoreductase that catalyzes the reversible conversion between 3β-hydroxy- and 3-keto-steroids, similar to 3β-hydroxysteroid dehydrogenase (3-KSR, HSD17B7/ERG27). The enzyme should adopt the short-chain dehydrogenase/reductase (SDR) fold with a conserved Tyr-Lys-Ser catalytic triad and a Rossmann-fold NAD⁺-binding pocket. Ensure stereospecific hydride transfer at the 3β position of the steroid substrate and proper alignment of the A-ring for oxidation/reduction. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.1.1.270/substrate_ligand.mol."
        # query = "Design an NADP-dependent 7β-hydroxysteroid dehydrogenase that catalyzes the reversible oxidation of 7β-hydroxysteroids to 7-ketosteroids. The enzyme should resemble known 7β-hydroxysteroid dehydrogenases (NADP⁺) in fold and mechanism, featuring a Rossmann-like NADP⁺-binding domain that recognizes the 2′-phosphate group and a conserved Tyr–Lys–Ser catalytic triad for stereospecific hydride transfer at the C7β position. Ensure the active site accommodates bile-acid–like substrates and maintains strict 7β selectivity. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.1.1.201/substrate_ligand.mol."
        # query = "Design an RNA pseudouridine synthase that specifically converts uridine2604 to pseudouridine in 23S rRNA, similar to RluF (YjbC). The enzyme should adopt the canonical pseudouridine synthase fold with a catalytic Asp residue acting as a nucleophile to facilitate base rotation and C–C bond formation at the C5 position of uridine. Include key RNA-recognition loops for stem–loop binding and proper positioning of U2604 within the catalytic pocket. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/5.4.99.21/substrate_ligand.mol"
        # query = "Design a serine protease that functions like coagulation factor IXa, the activated form of Christmas factor. The enzyme should cleave peptide bonds in factor X to generate factor Xa, featuring a classical trypsin-like fold with the catalytic triad His–Asp–Ser. Include a calcium-binding site and surface loops for interaction with factor VIIIa and phospholipid membranes to ensure correct complex formation during coagulation. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.4.21.22/substrate_ligand.mol."
        # query = "Design a carbon-oxygen lyase that catalyzes the reversible hydration/dehydration of fumarate ↔ (S)-malate, similar to fumarate hydratase (fumarase; L-malate hydro-lyase). The enzyme should enforce stereospecific addition/elimination of water across the C=C bond to yield L-malate, using a catalytic acid–base pair to activate water and stabilize the carbanion-like intermediate. Mimic the canonical fumarase active-site channel that recognizes the dicarboxylate motif and orients C2–C3 for reaction. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/4.2.1.2/substrate_ligand.mol."
        # query = "Design a carbohydrate esterase that removes O-acetyl groups from xylan, similar to acetylxylan esterase. The enzyme should adopt an α/β-hydrolase or SGNH-hydrolase architecture with a catalytic Ser–His–Asp triad and an oxyanion hole to stabilize the acyl-enzyme intermediate, releasing acetate. Include a substrate channel that recognizes the xylan backbone and accommodates 2-O and 3-O acetylation patterns on xylose residues. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.1.1.72/substrate_ligand.mol."
        # query = "Design a PLP-dependent aminotransferase that converts LL-2,6-diaminopimelate and 2-oxoglutarate into tetrahydrodipicolinate, L-glutamate, and H₂O, similar to LL-diaminopimelate aminotransferase (DapL). The enzyme should feature a Lys–PLP Schiff base, a dicarboxylate-binding pocket that enforces the LL stereochemistry, and gating loops that position the β,γ-unsaturated intermediate for cyclization to tetrahydrodipicolinate. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.6.1.83/substrate_ligand.mol."
        # query = "Design a serine hydrolase that deacylates lysophospholipids (e.g., lyso-PC) to release a free fatty acid and the corresponding glycerophosphodiester, similar to lysophospholipase (aka lecithinase B / phospholipase B). The enzyme should use an α/β-hydrolase (or SGNH) fold with a catalytic Ser–His–Asp triad and an oxyanion hole, and feature a lipid-binding tunnel that recognizes the glycerol-3-phosphate backbone and the phosphocholine headgroup while accommodating both sn-1 and sn-2 lysophospholipid isomers. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.1.1.5/substrate_ligand.mol."
        # query = "Design a nucleoside-triphosphate:AMP phosphotransferase that functions like adenylate kinase isozyme 3 (a GTP:AMP phosphotransferase). The enzyme should catalyze NTP + AMP ⇌ NDP + ADP, use a P-loop NTP-binding fold with a flexible “lid” domain to align substrates, and require Mg²⁺ for phosphoryl transfer. Include binding pockets for both AMP and diverse ribonucleoside triphosphates (e.g., GTP) and enforce transition-state stabilization during phosphate transfer. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.7.4.10/substrate_ligand.mol."
        # query = "Design a divalent-cation–dependent apyrase (ATP diphosphohydrolase) that sequentially hydrolyzes nucleoside tri- and diphosphates (e.g., ATP → ADP → AMP + Pi), similar to soluble NTPDase/apyrase enzymes. The protein should include a Ca²⁺/Mg²⁺-coordinated active site with the hallmark apyrase-conserved regions (ACR motifs) that position the β- and γ-phosphates for in-line attack and stabilize the transition state. Ensure a nucleotide pocket that recognizes the ribose/base while accommodating different NTPs/NDPs and supports processive two-step hydrolysis. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.6.1.5/substrate_ligand.mol."
        # query = "Design a carbon-oxygen lyase that catalyzes the dehydration of D-erythro-1-(imidazol-4-yl)glycerol-3-phosphate (IGP) to imidazole acetol phosphate (IAP), similar to imidazoleglycerol-phosphate dehydratase (IGPD). The enzyme should feature a metal-assisted active site (e.g., Mn²⁺) with catalytic acid/base residues to promote enolization and water elimination, and a binding pocket that recognizes both the imidazole ring and the 3-phosphate group for proper orientation. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/4.2.1.19/substrate_ligand.mol."
        # query = "Design a glycosyltransferase that autocatalytically attaches glucose residues from UDP-glucose to a specific tyrosine residue on itself to form the short α-1,4-glucan primer that initiates glycogen synthesis, similar to glycogenin glucosyltransferase. The enzyme should position UDP-glucose and the target Tyr side chain for in-line transfer, stabilize the leaving UDP, and support processive addition of several glucose units. Include a binding pocket that recognizes the glucosyl moiety and orients the growing α-1,4-linked chain. Use the /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.4.1.186/substrate.inchi I give you to verify enzyme using esp score. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.4.1.186/substrate_ligand.mol."
        # query = "Design a glycosyltransferase that transfers a glucuronic acid residue from UDP-glucuronate onto a 3-β-D-galactosyl-4-β-D-galactosyl-O-β-D-xylosyl-protein linker region of a proteoglycan, similar to galactosylgalactosylxylosylprotein 3-β-glucuronosyltransferase (glucuronosyltransferase I). The enzyme should recognize the tetrasaccharide linker attached to the core protein and catalyze formation of a β-D-glucuronosyl-(1→3)-galactose linkage, with a binding pocket that accommodates UDP-glucuronate and the growing glycan chain. Use the /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.4.1.135/substrate.smi I give you to verify enzyme using esp score. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/2.4.1.135/substrate_ligand.mol."
        # query = "Design a pyrimidine-5'-nucleotide nucleosidase that hydrolyzes the N-glycosidic bond of pyrimidine 5'-mononucleotides to yield the free pyrimidine base and ribose 5-phosphate, similar to pyrimidine nucleotide N-ribosidase (Pyr5N). The enzyme should recognize CMP/UMP-like substrates, positioning the pyrimidine ring and ribose for efficient cleavage of the C1′–N glycosidic bond and stabilization of the oxocarbenium-like transition state. Include a binding pocket that discriminates pyrimidine nucleotides from purines while tolerating different 5'-phosphate environments. Use the /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.2.2.10/substrate.smi I give you to verify enzyme using esp score. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/3.2.2.10/substrate_ligand.mol."
        query = "Design a metal-dependent lyase that converts 2-phospho-4-(cytidine 5'-diphospho)-2-C-methyl-D-erythritol into 2-C-methyl-D-erythritol 2,4-cyclodiphosphate with release of CMP, similar to 2-C-methyl-D-erythritol 2,4-cyclodiphosphate synthase (MECDP synthase). The enzyme should recognize the CDP-methylerythritol substrate, coordinate divalent cations to bind the diphosphate groups, and position catalytic residues to promote intramolecular cyclization of the diphosphate while cleaving the CMP leaving group. Use the /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/4.6.1.12/substrate.smi I give you to verify enzyme using esp score. Try docking to verify strong binding affinity to /jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/4.6.1.12/substrate_ligand.mol."
        
        message_history = await self.process_query(query)
        # print(message_history)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())