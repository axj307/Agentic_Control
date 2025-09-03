# Agentic Control System Architecture Diagram

## Overview

This document provides the specifications for creating comprehensive flowcharts and diagrams to illustrate the agentic control system architecture, integrating LangGraph reasoning, GRPO training, and MCP tools for advisor presentation.

## Diagram 1: High-Level System Architecture

### Description
Shows the overall system with three main components: LangGraph Agent, GRPO Training Pipeline, and Control Environment, connected through the ART framework.

### Mermaid Code
```mermaid
graph TB
    subgraph "Agentic Control System"
        subgraph "LangGraph Reasoning Layer"
            LG[LangGraph Agent]
            SA[State Analyzer]
            TP[Trajectory Planner] 
            CC[Control Calculator]
            SV[Safety Verifier]
            
            LG --> SA
            SA --> TP
            TP --> CC
            CC --> SV
        end
        
        subgraph "MCP Tools Layer"
            MCPTools[MCP Server]
            PhysTools[Physics Tools]
            CtrlTools[Control Tools]
            SafeTools[Safety Tools]
            
            MCPTools --> PhysTools
            MCPTools --> CtrlTools
            MCPTools --> SafeTools
        end
        
        subgraph "GRPO Training Pipeline"
            TrainLoop[Training Loop]
            TrajGen[Trajectory Generation]
            RewardCalc[Reward Calculation]
            ModelUpdate[Model Update]
            
            TrainLoop --> TrajGen
            TrajGen --> RewardCalc
            RewardCalc --> ModelUpdate
            ModelUpdate --> TrainLoop
        end
        
        subgraph "Control Environment"
            DI[Double Integrator]
            State[System State]
            Action[Control Action]
            Reward[Reward Signal]
            
            DI --> State
            State --> Action
            Action --> DI
            DI --> Reward
        end
    end
    
    %% Connections between layers
    SV --> Action
    Action --> State
    State --> SA
    Reward --> RewardCalc
    TrajGen --> LG
    PhysTools --> CC
    CtrlTools --> CC
    SafeTools --> SV
    
    %% Styling
    classDef reasoning fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef training fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef environment fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef mcp fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class LG,SA,TP,CC,SV reasoning
    class TrainLoop,TrajGen,RewardCalc,ModelUpdate training
    class DI,State,Action,Reward environment
    class MCPTools,PhysTools,CtrlTools,SafeTools mcp
```

## Diagram 2: Detailed Agentic Loop Flow

### Description
Shows the step-by-step flow of the agentic control loop, from state observation to action execution.

### Mermaid Code
```mermaid
flowchart TD
    Start([Start Episode]) --> ObsState[Observe System State]
    
    ObsState --> LangGraph{LangGraph Agent}
    
    LangGraph --> StateAnalysis[State Analysis Node]
    StateAnalysis --> |"Position, Velocity, Target"| TrajectoryPlan[Trajectory Planning Node]
    TrajectoryPlan --> |"Waypoints, Path"| ControlCalc[Control Calculation Node]
    ControlCalc --> |"Control Command"| SafetyCheck[Safety Verification Node]
    
    SafetyCheck --> SafetyDecision{Safety Check}
    SafetyDecision -->|Safe| ExecuteAction[Execute Control Action]
    SafetyDecision -->|Unsafe| ModifyAction[Modify Action for Safety]
    ModifyAction --> ExecuteAction
    
    ExecuteAction --> Environment[Double Integrator Environment]
    Environment --> UpdateState[Update System State]
    UpdateState --> CalcReward[Calculate Reward]
    
    CalcReward --> EpisodeCheck{Episode Complete?}
    EpisodeCheck -->|No| ObsState
    EpisodeCheck -->|Yes| StoreTrajectory[Store Trajectory for Training]
    
    StoreTrajectory --> GRPOTraining[GRPO Training Update]
    GRPOTraining --> ModelImprovement[Model Improvement]
    ModelImprovement --> NextEpisode[Next Episode]
    NextEpisode --> Start
    
    %% MCP Tools Integration
    subgraph "MCP Tools"
        MCPStateAnalyzer[State Analyzer Tool]
        MCPController[LQR Controller Tool]
        MCPSafety[Safety Checker Tool]
        MCPPhysics[Physics Simulator Tool]
    end
    
    StateAnalysis -.->|Uses| MCPStateAnalyzer
    ControlCalc -.->|Uses| MCPController
    SafetyCheck -.->|Uses| MCPSafety
    TrajectoryPlan -.->|Uses| MCPPhysics
    
    %% Styling
    classDef startEnd fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px
    classDef process fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef langgraph fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef mcp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class Start,NextEpisode startEnd
    class ObsState,ExecuteAction,Environment,UpdateState,CalcReward,StoreTrajectory,GRPOTraining,ModelImprovement process
    class SafetyDecision,EpisodeCheck decision
    class LangGraph,StateAnalysis,TrajectoryPlan,ControlCalc,SafetyCheck langgraph
    class MCPStateAnalyzer,MCPController,MCPSafety,MCPPhysics mcp
```

## Diagram 3: GRPO Training Pipeline Detail

### Description
Detailed view of the GRPO training process showing how trajectories are collected, rewards calculated, and model updated.

### Mermaid Code
```mermaid
graph TD
    subgraph "GRPO Training Pipeline"
        subgraph "Data Collection Phase"
            InitModel[Initialize Model]
            CollectTraj[Collect Trajectories]
            RunEpisodes[Run Control Episodes]
            StoreData[Store Episode Data]
            
            InitModel --> CollectTraj
            CollectTraj --> RunEpisodes
            RunEpisodes --> StoreData
        end
        
        subgraph "Reward Processing"
            CalcRewards[Calculate Rewards]
            RULERScore[RULER Scoring]
            PhysicsReward[Physics-based Rewards]
            TaskReward[Task Completion Rewards]
            
            CalcRewards --> RULERScore
            CalcRewards --> PhysicsReward
            CalcRewards --> TaskReward
        end
        
        subgraph "Policy Optimization"
            AdvantageEst[Advantage Estimation]
            PolicyGrad[Policy Gradient Calculation]
            ModelUpdate[Model Parameter Update]
            Checkpoint[Save Checkpoint]
            
            AdvantageEst --> PolicyGrad
            PolicyGrad --> ModelUpdate
            ModelUpdate --> Checkpoint
        end
        
        subgraph "Evaluation"
            EvalPhase[Evaluation Phase]
            MetricsCalc[Performance Metrics]
            Convergence[Convergence Check]
            NextIteration[Next Training Iteration]
            
            EvalPhase --> MetricsCalc
            MetricsCalc --> Convergence
            Convergence --> NextIteration
        end
    end
    
    %% Flow connections
    StoreData --> CalcRewards
    RULERScore --> AdvantageEst
    PhysicsReward --> AdvantageEst
    TaskReward --> AdvantageEst
    Checkpoint --> EvalPhase
    NextIteration --> CollectTraj
    
    %% External connections
    ART[ART Framework] --> InitModel
    LangGraphAgent[LangGraph Agent] --> RunEpisodes
    ControlEnv[Control Environment] --> RunEpisodes
    
    %% Styling
    classDef collection fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef reward fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef optimization fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef evaluation fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef external fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px
    
    class InitModel,CollectTraj,RunEpisodes,StoreData collection
    class CalcRewards,RULERScore,PhysicsReward,TaskReward reward
    class AdvantageEst,PolicyGrad,ModelUpdate,Checkpoint optimization
    class EvalPhase,MetricsCalc,Convergence,NextIteration evaluation
    class ART,LangGraphAgent,ControlEnv external
```

## Diagram 4: MCP Tools Integration Architecture

### Description
Shows how MCP (Model Context Protocol) tools integrate with the agentic system to provide physics-informed reasoning.

### Mermaid Code
```mermaid
graph TB
    subgraph "MCP Tools Architecture"
        subgraph "MCP Server Layer"
            MCPServer[MCP Server]
            ToolRegistry[Tool Registry]
            ToolExecutor[Tool Executor]
            
            MCPServer --> ToolRegistry
            ToolRegistry --> ToolExecutor
        end
        
        subgraph "Physics-Informed Tools"
            StateAnalyzer[State Analyzer Tool]
            LyapunovTool[Lyapunov Stability Tool]
            LQRTool[LQR Controller Tool]
            MCPTool[MPC Predictor Tool]
            
            StateAnalyzer --> |"Stability Analysis"| LyapunovTool
            LQRTool --> |"Optimal Gains"| MCPTool
        end
        
        subgraph "Safety & Verification Tools"
            SafetyChecker[Safety Checker Tool]
            ConstraintVerifier[Constraint Verifier]
            BarrierFunction[Control Barrier Function]
            
            SafetyChecker --> ConstraintVerifier
            ConstraintVerifier --> BarrierFunction
        end
        
        subgraph "Scenario Generation Tools"
            ScenarioGen[Scenario Generator]
            AdaptiveGen[Adaptive Scenario Tool]
            DifficultyScaler[Difficulty Scaler]
            
            ScenarioGen --> AdaptiveGen
            AdaptiveGen --> DifficultyScaler
        end
    end
    
    subgraph "Agent Integration"
        LangGraphNodes[LangGraph Nodes]
        ToolCalls[Tool Function Calls]
        ResponseParser[Response Parser]
        ActionSynthesis[Action Synthesis]
        
        LangGraphNodes --> ToolCalls
        ToolCalls --> ResponseParser
        ResponseParser --> ActionSynthesis
    end
    
    %% Connections
    ToolExecutor --> StateAnalyzer
    ToolExecutor --> SafetyChecker
    ToolExecutor --> ScenarioGen
    ToolCalls --> MCPServer
    MCPServer --> ToolExecutor
    
    %% Control Loop Integration
    SystemState[System State] --> LangGraphNodes
    ActionSynthesis --> ControlAction[Control Action]
    ControlAction --> Environment[Environment]
    Environment --> SystemState
    
    %% Styling
    classDef mcp fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef physics fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef safety fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef scenario fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef agent fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef system fill:#f5f5f5,stroke:#616161,stroke-width:2px
    
    class MCPServer,ToolRegistry,ToolExecutor mcp
    class StateAnalyzer,LyapunovTool,LQRTool,MCPTool physics
    class SafetyChecker,ConstraintVerifier,BarrierFunction safety
    class ScenarioGen,AdaptiveGen,DifficultyScaler scenario
    class LangGraphNodes,ToolCalls,ResponseParser,ActionSynthesis agent
    class SystemState,ControlAction,Environment system
```

## Code to Generate Diagrams

I'll create Python scripts to generate these diagrams using various tools:

### Method 1: Using Mermaid (Recommended)
```python
# File: generate_diagrams.py
import os
import subprocess
from pathlib import Path

def generate_mermaid_diagram(mermaid_code, output_file, title):
    """Generate diagram from Mermaid code"""
    
    # Create mermaid file
    mmd_file = f"{output_file}.mmd"
    with open(mmd_file, 'w') as f:
        f.write(mermaid_code)
    
    # Generate PNG using mermaid-cli
    try:
        subprocess.run([
            'mmdc', 
            '-i', mmd_file,
            '-o', f"{output_file}.png",
            '-t', 'default',
            '--width', '1200',
            '--height', '800'
        ], check=True)
        
        print(f"✅ Generated {output_file}.png")
        
        # Clean up mermaid file
        os.remove(mmd_file)
        
    except subprocess.CalledProcessError:
        print(f"❌ Failed to generate {output_file}. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
    except FileNotFoundError:
        print("❌ mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli")

# Diagram specifications (insert the Mermaid code from above)
diagrams = {
    "system_architecture": {
        "title": "High-Level System Architecture",
        "code": """[Insert Mermaid code for Diagram 1]"""
    },
    "agentic_loop": {
        "title": "Detailed Agentic Loop Flow", 
        "code": """[Insert Mermaid code for Diagram 2]"""
    },
    "grpo_pipeline": {
        "title": "GRPO Training Pipeline",
        "code": """[Insert Mermaid code for Diagram 3]"""
    },
    "mcp_architecture": {
        "title": "MCP Tools Integration",
        "code": """[Insert Mermaid code for Diagram 4]"""
    }
}

if __name__ == "__main__":
    output_dir = Path("docs/diagrams")
    output_dir.mkdir(exist_ok=True)
    
    for name, spec in diagrams.items():
        output_path = output_dir / name
        generate_mermaid_diagram(spec["code"], str(output_path), spec["title"])
```

### Method 2: Using Graphviz (Alternative)
```python
# File: generate_graphviz_diagrams.py
import graphviz

def create_system_overview():
    """Create high-level system overview using Graphviz"""
    
    dot = graphviz.Digraph('agentic_control_system')
    dot.attr(rankdir='TB', size='12,8')
    
    # Define node styles
    dot.attr('node', shape='box', style='rounded,filled')
    
    # LangGraph Layer
    with dot.subgraph(name='cluster_langgraph') as lg:
        lg.attr(label='LangGraph Reasoning Layer', color='blue')
        lg.node('agent', 'LangGraph Agent', fillcolor='lightblue')
        lg.node('state_analyzer', 'State Analyzer', fillcolor='lightblue')
        lg.node('trajectory_planner', 'Trajectory Planner', fillcolor='lightblue')
        lg.node('control_calc', 'Control Calculator', fillcolor='lightblue')
        lg.node('safety_verifier', 'Safety Verifier', fillcolor='lightblue')
    
    # MCP Tools Layer
    with dot.subgraph(name='cluster_mcp') as mcp:
        mcp.attr(label='MCP Tools Layer', color='orange')
        mcp.node('mcp_server', 'MCP Server', fillcolor='lightyellow')
        mcp.node('physics_tools', 'Physics Tools', fillcolor='lightyellow')
        mcp.node('control_tools', 'Control Tools', fillcolor='lightyellow')
        mcp.node('safety_tools', 'Safety Tools', fillcolor='lightyellow')
    
    # GRPO Training
    with dot.subgraph(name='cluster_training') as train:
        train.attr(label='GRPO Training Pipeline', color='purple')
        train.node('train_loop', 'Training Loop', fillcolor='plum')
        train.node('traj_gen', 'Trajectory Generation', fillcolor='plum')
        train.node('reward_calc', 'Reward Calculation', fillcolor='plum')
        train.node('model_update', 'Model Update', fillcolor='plum')
    
    # Control Environment
    with dot.subgraph(name='cluster_env') as env:
        env.attr(label='Control Environment', color='green')
        env.node('double_integrator', 'Double Integrator', fillcolor='lightgreen')
        env.node('system_state', 'System State', fillcolor='lightgreen')
        env.node('control_action', 'Control Action', fillcolor='lightgreen')
        env.node('reward_signal', 'Reward Signal', fillcolor='lightgreen')
    
    # Add edges
    dot.edge('agent', 'state_analyzer')
    dot.edge('state_analyzer', 'trajectory_planner') 
    dot.edge('trajectory_planner', 'control_calc')
    dot.edge('control_calc', 'safety_verifier')
    dot.edge('safety_verifier', 'control_action')
    
    # Cross-layer connections
    dot.edge('physics_tools', 'control_calc', style='dashed')
    dot.edge('safety_tools', 'safety_verifier', style='dashed')
    dot.edge('traj_gen', 'agent', style='dashed')
    dot.edge('reward_signal', 'reward_calc', style='dashed')
    
    return dot

if __name__ == "__main__":
    # Generate system overview
    system_diagram = create_system_overview()
    system_diagram.render('docs/diagrams/system_overview_graphviz', format='png', cleanup=True)
    print("✅ Generated system overview diagram")
```

## Installation Instructions

To generate the diagrams, you'll need to install the required tools:

### For Mermaid Diagrams (Recommended):
```bash
# Install Node.js and mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Or using conda
conda install -c conda-forge nodejs
npm install -g @mermaid-js/mermaid-cli
```

### For Graphviz Diagrams:
```bash
# Install graphviz
conda install graphviz python-graphviz

# Or using pip
pip install graphviz
```

## Presentation-Ready Versions

I'll also create presentation-ready versions with:
- High-resolution outputs (300 DPI)
- Professional color schemes
- Clear typography for projectors
- Multiple formats (PNG, PDF, SVG)

Would you like me to:
1. Generate the actual diagram files using Mermaid?
2. Create additional specialized diagrams (e.g., training timeline, performance comparison)?
3. Add more detailed technical annotations for specific components?

The flowcharts clearly show how LangGraph provides the reasoning layer, GRPO handles the learning optimization, and MCP tools provide physics-informed capabilities - all working together in the agentic control loop!