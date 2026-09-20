#!/usr/bin/env python3
"""Replace every ASCII-art block in Chapter 6 with a native Mermaid diagram.

The generated Markdown renders the diagrams directly in GitHub, VS Code,
Obsidian and most modern Markdown viewers -- no image files required.

Run with:
    /opt/miniconda3/bin/python3 build_mermaid_chapter6.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from render_chapter6_blocks import (
    CAPTION_OVERRIDES,
    LANGUAGE_TAGS,
    SOURCE_MD,
    normalize_fences,
    parse_blocks,
)

OUTPUT_MD = SOURCE_MD.with_name(SOURCE_MD.stem + "_mermaid.md")

# One Mermaid definition per ASCII diagram, in document order.
MERMAID_DIAGRAMS: dict[int, str] = {
    1: """flowchart LR
    subgraph TRAD["Traditional Irrigation"]
        direction TB
        T1["Fixed Timer / Manual Check"] --> T2["Uniform Field-Wide Flooding"]
        T2 --> T3["High Evaporation & Runoff<br/>40%–60% Water Loss"]
    end
    subgraph PA["Precision AI Irrigation"]
        direction TB
        P1["IoT Sensors & Remote Sensing"] --> P2["Spatial Evapotranspiration AI"]
        P2 --> P3["Variable Rate Drip / Pivot<br/>90%+ Water Efficiency"]
    end
    classDef bad fill:#fdecea,stroke:#c0392b,stroke-width:2px,color:#7b241c;
    classDef good fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class T1,T2,T3 bad;
    class P1,P2,P3 good;""",
    2: """flowchart TD
    S["Soil<br/>Ψ_soil = −0.01 to −0.03 MPa<br/>Field Capacity"] -->|Capillary Uptake| R["Root / Stem<br/>Ψ_root = −0.3 to −0.8 MPa"]
    R -->|Xylem Transport| L["Leaf<br/>Ψ_leaf = −1.0 to −2.5 MPa"]
    L -->|Transpiration| A["Atmosphere<br/>Ψ_atm = −10 to −100 MPa<br/>Very Negative"]
    classDef wet fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef dry fill:#ffe6d6,stroke:#e07b39,stroke-width:2px,color:#7a3b12;
    class S,R wet;
    class L,A dry;""",
    3: """flowchart TD
    ET["Total Evapotranspiration (ET)"] --> E["Evaporation (E)<br/>Soil Surface"]
    ET --> T["Transpiration (T)<br/>Stomatal Pore"]
    classDef root fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef leaf fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    classDef air fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    class ET root;
    class E air;
    class T leaf;""",
    4: """xychart-beta
    title "Crop Coefficient (Kc) Across Growth Stages"
    x-axis ["Initial", "Development", "Mid-Season", "Late-Season"]
    y-axis "Crop Coefficient (Kc)" 0 --> 1.2
    line [0.40, 0.75, 1.15, 0.55]""",
    5: """flowchart TD
    A["Saturation θ_sat — 0 kPa<br/>All pore spaces filled"] --> B["Gravitational Water<br/>Drains away in 1–3 days"]
    B --> C["Field Capacity θ_FC — −10 to −33 kPa<br/>Upper limit of PAW"]
    C --> D["Permanent Wilting Point θ_PWP — −1500 kPa<br/>Lower limit of PAW"]
    D --> E["Air Space / Unavailable Water"]
    C -. "Plant Available Water  PAW = θ_FC − θ_PWP" .- D
    classDef sat fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef avail fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    classDef dry fill:#ffe6d6,stroke:#e07b39,stroke-width:2px,color:#7a3b12;
    class A,B sat;
    class C,D avail;
    class E dry;""",
    6: """flowchart TD
    subgraph SRC["Multi-Source Data Ingestion Engine"]
        direction LR
        G["Ground In-Situ IoT<br/>- Frequency Domain<br/>- TDR / FDR Probes<br/>- Microclimate Node"]
        A["Aerial / Drone<br/>- Thermal Infrared<br/>- Multispectral<br/>- High Spatial Resolution"]
        S["Satellite Remote Sensing<br/>- Sentinel-2 (MSI)<br/>- Landsat 8/9 (TIR)<br/>- High Revisit Rate"]
    end
    G --> P
    A --> P
    S --> P
    P["Data Preprocessing &amp; Alignment<br/>- Spatial Resampling<br/>- Cloud / Shadow Masking<br/>- Kalman Filter Clean"]
    P --> I["Integrated Spatial-Temporal Grid"]
    classDef src fill:#eef2f7,stroke:#5b6b7c,stroke-width:2px,color:#22303c;
    classDef core fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    class G,A,S src;
    class P,I core;""",
    7: """flowchart TD
    MC["Microcontroller + LoRaWAN Transmitter"] --> SF["Soil Surface"]
    SF --> D1["10 cm — FDR Sensor 1<br/>Topsoil Evaporation Zone"]
    D1 --> D2["30 cm — FDR Sensor 2<br/>Active Root Uptake Zone"]
    D2 --> D3["60 cm — FDR Sensor 3<br/>Deep Root Boundary"]
    D3 --> D4["90 cm — FDR Sensor 4<br/>Percolation Loss Monitor"]
    classDef head fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef sensor fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    class MC,SF head;
    class D1,D2,D3,D4 sensor;""",
    8: """flowchart LR
    A["CWSI = 0.0<br/>No Stress<br/>Tc &lt; Ta<br/>Stomata Open &amp; Transpiring"] --> B["CWSI = 0.5<br/>Moderate Stress"]
    B --> C["CWSI = 1.0<br/>Maximum Stress<br/>Tc ≫ Ta<br/>Stomata Closed / Water Deficit"]
    classDef cool fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    classDef mid fill:#fff4d6,stroke:#d4a017,stroke-width:2px,color:#7a5b00;
    classDef hot fill:#fdecea,stroke:#c0392b,stroke-width:2px,color:#7b241c;
    class A cool;
    class B mid;
    class C hot;""",
    9: """flowchart TD
    ROOT["AI Predictive Architecture"]
    ROOT --> P["Physics-Informed ML (PINN)<br/>- Integrates Richards' Equation<br/>- Mass Conservation Guarantees"]
    ROOT --> T["Temporal Models<br/>LSTM / Temporal Transformers<br/>- Long-range θ(t+k) Forecast<br/>- Seasonal &amp; Weather Memory"]
    P --> H["Hybrid Output<br/>Spatiotemporal Volumetric Moisture<br/>&amp; Dynamic Irrigation Recommendations"]
    T --> H
    classDef root fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef branch fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef out fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class ROOT root;
    class P,T branch;
    class H out;""",
    10: """flowchart TD
    I1["Soil Moisture Probe"] --> NN["Neural Network<br/>Predicted θ(z, t)"]
    I2["Weather Forecast"] --> NN
    I3["Irrigation Log"] --> NN
    NN --> DL["Data Loss L_data"]
    NN --> PH["Richards' Equation<br/>Mass Conservation Check"]
    PH --> PL["Physics Loss L_physics"]
    DL --> BP["Backpropagate Combined Loss<br/>L = L_data + λ · L_physics"]
    PL --> BP
    BP -.->|Update Weights| NN
    classDef input fill:#eef2f7,stroke:#5b6b7c,stroke-width:2px,color:#22303c;
    classDef net fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef loss fill:#ffe6d6,stroke:#e07b39,stroke-width:2px,color:#7a3b12;
    classDef trainer fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class I1,I2,I3 input;
    class NN net;
    class DL,PH,PL loss;
    class BP trainer;""",
    11: """flowchart LR
    subgraph DS["Data Sources"]
        direction TB
        A1["IoT Probes"]
        A2["Satellite Maps"]
        A3["Weather Forecast"]
    end
    DS --> AI["AI Decision Engine<br/>Modern RL Agent / PINN<br/>Action: Select Valve<br/>Duty Cycles &amp; Rates"]
    AI --> ACT1["Variable Rate Pivot"]
    AI --> ACT2["Drip Solenoid Valves"]
    ACT1 -.->|Feedback: Soil Moisture Delta &amp; CWSI| AI
    ACT2 -.->|Feedback: Soil Moisture Delta &amp; CWSI| AI
    classDef src fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef engine fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef act fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class A1,A2,A3 src;
    class AI engine;
    class ACT1,ACT2 act;""",
    12: """flowchart TB
    subgraph R1["Field Row 1"]
        direction LR
        A1["Zone A1: 12 mm<br/>Sandy Loam Soil"]
        A2["Zone A2: 5 mm<br/>Clay Heavy Soil"]
    end
    subgraph R2["Field Row 2"]
        direction LR
        B1["Zone B1: 0 mm<br/>Saturated Depression"]
        B2["Zone B2: 18 mm<br/>Shallow Rooting Zone"]
    end
    classDef wet fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef mid fill:#fff4d6,stroke:#d4a017,stroke-width:2px,color:#7a5b00;
    classDef dry fill:#fdecea,stroke:#c0392b,stroke-width:2px,color:#7b241c;
    class A1 wet;
    class A2 mid;
    class B1 dry;
    class B2 wet;""",
    13: """flowchart TD
    A["Sensing Nodes (16 Probes)<br/>Sentinel-2 &amp; UAV Thermal Imagery"] --> B["AI Ensemble<br/>XGBoost ET0 + LSTM Soil Moisture Model"]
    B --> C["Zone-Specific VRI Duty Cycle Controller<br/>Center Pivot"]
    classDef sense fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef brain fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef actuate fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class A sense;
    class B brain;
    class C actuate;""",
    14: """flowchart TD
    A["Dendrometers + Thermal Cameras<br/>+ Matric Potential Sensors"] --> B["PINN + Reinforcement Learning Agent<br/>Target Ψ_leaf = −1.2 to −1.4 MPa"]
    B --> C["Automated Drip Manifold<br/>Solenoid Actuation Systems"]
    classDef sense fill:#dceeff,stroke:#2f7fd1,stroke-width:2px,color:#0d3f70;
    classDef brain fill:#e8e0ff,stroke:#6b4fd1,stroke-width:2px,color:#2f2170;
    classDef actuate fill:#d9f2e6,stroke:#1e8449,stroke-width:2px,color:#145a32;
    class A sense;
    class B brain;
    class C actuate;""",
    15: """mindmap
  root((Challenges &amp; Bottlenecks))
    Sensor Drift
      Biofouling in Soil
      Calibration Decay
    Model Generalization
      Spatial Transferability
      Concept and Data Drift
    Infrastructure Costs
      High Initial CAPEX
      LoRaWAN / 5G Setup""",
    16: """mindmap
  root((Future Horizons))
    LEO Satellite IoT
      Direct-to-Cell
      Continuous Global Field Telemetry
    Edge Quantum-AI
      High-Dimensional Multi-Objective Ops
      Decisions in Seconds
    Autonomous Robotics
      Micro-Drip Repair
      Underground Soil Sampling Drones""",
}


def main() -> int:
    if not SOURCE_MD.exists():
        raise SystemExit(f"Source document not found: {SOURCE_MD}")

    lines, notes = normalize_fences(SOURCE_MD.read_text(encoding="utf-8").split("\n"))
    for note in notes:
        print(f"  repair: {note}")

    output: list[str] = []
    figure_number = 0

    for kind, payload in parse_blocks(lines):
        if kind == "text":
            output.append(payload)
            continue

        _, info, body = payload
        if info.lower() in LANGUAGE_TAGS:
            output.append("\n".join(["```" + info, *body, "```"]))
            continue

        figure_number += 1
        diagram = MERMAID_DIAGRAMS.get(figure_number)
        if diagram is None:
            output.append("\n".join(["```", *body, "```"]))
            continue

        output.append("```mermaid")
        output.append(diagram)
        output.append("```")

        caption = CAPTION_OVERRIDES.get(figure_number)
        if caption:
            caption = re.sub(r"\s+", " ", caption).strip()
            output.append("")
            output.append(f"*Figure {figure_number}. {caption}.*")
        print(f"  diagram {figure_number}: {diagram.splitlines()[0]}")

    OUTPUT_MD.write_text("\n".join(output), encoding="utf-8")

    expected = max(MERMAID_DIAGRAMS)
    if figure_number != expected:
        print(
            f"\nWARNING: source produced {figure_number} diagram blocks "
            f"but {expected} Mermaid definitions are provided.",
            file=sys.stderr,
        )

    print(f"\nDiagrams written: {figure_number}")
    print(f"Output file     : {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
