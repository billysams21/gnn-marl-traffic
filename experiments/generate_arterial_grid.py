"""
Generate an Arterial+Collector PKJI-inspired network for SUMO.
Layout:
         N1      N2      N3
         |       |       |
W_C1 -- C1 ---- C2 ---- C3 -- E_C3   (Kolektor Utara, 1 lane, Y=150)
         |       |       |
W_A0 == A1 ==== A2 ==== A3 == E_A4   (Arteri Utama, 2 lane, Y=0)
         |       |       |
W_B1 -- B1 ---- B2 ---- B3 -- E_B3   (Kolektor Selatan, 1 lane, Y=-150)
         |       |       |
         S1      S2      S3

Lengths:
- X1 to X2 (Col 1 to Col 2): 450m
- X2 to X3 (Col 2 to Col 3): 350m
- Vertical Connectors: 150m
- All Entry/Exit edges: 350m
"""

import argparse
import os

def create_nod_xml(filepath):
    nodes = [
        # Nodes
        ('<node id="A1" x="0" y="0" type="traffic_light"/>'),
        ('<node id="A2" x="450" y="0" type="traffic_light"/>'),
        ('<node id="A3" x="800" y="0" type="traffic_light"/>'),
        
        ('<node id="C1" x="0" y="150" type="traffic_light"/>'),
        ('<node id="C2" x="450" y="150" type="traffic_light"/>'),
        ('<node id="C3" x="800" y="150" type="traffic_light"/>'),
        
        ('<node id="B1" x="0" y="-150" type="traffic_light"/>'),
        ('<node id="B2" x="450" y="-150" type="traffic_light"/>'),
        ('<node id="B3" x="800" y="-150" type="traffic_light"/>'),

        # Boundaries - Entry/Exits (350m offset from outer nodes)
        ('<node id="W_A0" x="-350" y="0" type="priority"/>'),
        ('<node id="E_A4" x="1150" y="0" type="priority"/>'),
        
        ('<node id="W_C1" x="-350" y="150" type="priority"/>'),
        ('<node id="E_C3" x="1150" y="150" type="priority"/>'),
        
        ('<node id="W_B1" x="-350" y="-150" type="priority"/>'),
        ('<node id="E_B3" x="1150" y="-150" type="priority"/>'),
        
        ('<node id="N1" x="0" y="500" type="priority"/>'),
        ('<node id="N2" x="450" y="500" type="priority"/>'),
        ('<node id="N3" x="800" y="500" type="priority"/>'),
        
        ('<node id="S1" x="0" y="-500" type="priority"/>'),
        ('<node id="S2" x="450" y="-500" type="priority"/>'),
        ('<node id="S3" x="800" y="-500" type="priority"/>'),
    ]
    with open(filepath, 'w') as f:
        f.write('<nodes>\n')
        f.write('\n'.join('    ' + n for n in nodes))
        f.write('\n</nodes>\n')

def create_edg_xml(filepath):
    edges = []
    
    def add_bidir(id1, id2, numLanes):
        edges.append(f'<edge id="{id1}{id2}" from="{id1}" to="{id2}" numLanes="{numLanes}" />')
        edges.append(f'<edge id="{id2}{id1}" from="{id2}" to="{id1}" numLanes="{numLanes}" />')

    # Arteri (2 lanes)
    add_bidir("W_A0", "A1", 2)
    add_bidir("A1", "A2", 2)
    add_bidir("A2", "A3", 2)
    add_bidir("A3", "E_A4", 2)

    # Kolektor Utara (1 lane)
    add_bidir("W_C1", "C1", 1)
    add_bidir("C1", "C2", 1)
    add_bidir("C2", "C3", 1)
    add_bidir("C3", "E_C3", 1)
    
    # Kolektor Selatan (1 lane)
    add_bidir("W_B1", "B1", 1)
    add_bidir("B1", "B2", 1)
    add_bidir("B2", "B3", 1)
    add_bidir("B3", "E_B3", 1)

    # Vertical connectors (1 lane)
    add_bidir("C1", "A1", 1)
    add_bidir("C2", "A2", 1)
    add_bidir("C3", "A3", 1)
    
    add_bidir("A1", "B1", 1)
    add_bidir("A2", "B2", 1)
    add_bidir("A3", "B3", 1)

    # North/South Entry Exits (1 lane)
    add_bidir("N1", "C1", 1)
    add_bidir("N2", "C2", 1)
    add_bidir("N3", "C3", 1)
    
    add_bidir("B1", "S1", 1)
    add_bidir("B2", "S2", 1)
    add_bidir("B3", "S3", 1)

    with open(filepath, 'w') as f:
        f.write('<edges>\n')
        f.write('\n'.join('    ' + e for e in edges))
        f.write('\n</edges>\n')

if __name__ == "__main__":
    out_dir = "data/networks/arterial_3x3"
    os.makedirs(out_dir, exist_ok=True)
    nod_file = os.path.join(out_dir, "arterial.nod.xml")
    edg_file = os.path.join(out_dir, "arterial.edg.xml")
    
    create_nod_xml(nod_file)
    create_edg_xml(edg_file)
    print(f"Generated {nod_file} and {edg_file}")
