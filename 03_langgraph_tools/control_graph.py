"""
Control Graph for Tool-Augmented LLM Control

This module implements a LangGraph-based control system that uses
physics-aware tools to make better control decisions.

The graph workflow:
1. observe_state - Get current observation
2. analyze_errors - Use ErrorAnalyzerTool
3. plan_action - Use PIDCalculatorTool or TrajectoryPlannerTool
4. verify_safety - Use SafetyVerifierTool
5. execute_action - Output final control

Usage:
    graph = build_control_graph()
    result = graph.invoke({"position": 0.5, "velocity": 0.0, "target_pos": 0.0})
"""

import json
from typing import Dict, List, Optional, Any, TypedDict
import numpy as np

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except (ImportError, TypeError) as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️  LangGraph issue: {e}. Using mock implementation.")
    
    # Create placeholder classes for development
    class StateGraph:
        def __init__(self):
            self.nodes = {}
            self.edges = {}
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self):
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph):
            self.graph = graph
        
        def invoke(self, state):
            # Use physics tools directly instead of mock
            try:
                # Get current state
                pos = state.get("position", 0.0)
                vel = state.get("velocity", 0.0) 
                target_pos = state.get("target_pos", 0.0)
                target_vel = state.get("target_vel", 0.0)
                
                # Use your physics tools
                from control_tools import analyze_errors, select_control_strategy, calculate_lqr_control
                
                # Step 1: Analyze errors
                error_analysis = analyze_errors(pos, vel, target_pos, target_vel)
                pos_error = error_analysis['position_error']
                vel_error = error_analysis['velocity_error']
                
                # Step 2: Select strategy
                strategy = select_control_strategy(pos_error, vel_error)
                
                # Step 3: Calculate control
                if strategy['strategy'] == 'lqr_optimal':
                    control_result = calculate_lqr_control(pos, vel, target_pos, target_vel)
                    action = control_result['control_action']
                    confidence = 0.9
                    reasoning = f"LQR control: {control_result['reasoning']}"
                else:
                    # Fallback to PID
                    from control_tools import calculate_pid_control
                    control_result = calculate_pid_control(pos_error, vel_error)
                    action = control_result['control_action']
                    confidence = control_result['confidence']
                    reasoning = f"PID control: {control_result['reasoning']}"
                
                return {
                    "action": action, 
                    "confidence": confidence, 
                    "reasoning": reasoning,
                    "strategy_used": strategy['strategy'],
                    "tool_calls": [f"analyze_errors", f"select_strategy", strategy['strategy']],
                    "messages": [],
                    "error_analysis": error_analysis,
                    "strategy_selection": strategy,
                    "control_plan": control_result,
                    "safety_check": {"safe": True}
                }
                
            except Exception as e:
                # Fallback to safe action if tools fail
                return {
                    "action": 0.0, 
                    "confidence": 0.1, 
                    "reasoning": f"Tool error: {e}",
                    "tool_calls": [],
                    "messages": [],
                    "error_analysis": {},
                    "strategy_selection": {},
                    "control_plan": {},
                    "safety_check": {"safe": False}
                }
    
    END = "END"
    
    class BaseMessage:
        def __init__(self, content):
            self.content = content
    
    class HumanMessage(BaseMessage):
        pass
    
    class AIMessage(BaseMessage):
        pass

from control_tools import get_control_tools, analyze_errors, select_control_strategy, calculate_pid_control, calculate_lqr_control, plan_trajectory, verify_safety


class ControlState(TypedDict):
    """State for the control graph"""
    # Input state
    position: float
    velocity: float
    target_pos: float
    target_vel: float
    step: int
    
    # Analysis results
    error_analysis: Optional[Dict]
    strategy_selection: Optional[Dict]
    control_plan: Optional[Dict]
    trajectory_plan: Optional[Dict]
    safety_check: Optional[Dict]
    
    # Output
    action: float
    confidence: float
    reasoning: str
    tool_calls: List[Dict]
    
    # Messages for LLM integration
    messages: List[BaseMessage]


def observe_state(state: ControlState) -> ControlState:
    """
    Initial observation and state setup
    """
    # Create initial observation message
    pos = state["position"]
    vel = state["velocity"]
    target_pos = state["target_pos"]
    target_vel = state.get("target_vel", 0.0)
    step = state.get("step", 0)
    
    observation = f"""SPACECRAFT CONTROL OBSERVATION - Step {step}

Current State:
- Position: {pos:.3f} m
- Velocity: {vel:.3f} m/s

Target State:
- Position: {target_pos:.3f} m  
- Velocity: {target_vel:.3f} m/s

Mission: Use physics-aware tools to determine optimal control action.
Available tools: analyze_errors, select_control_strategy, calculate_pid_control, calculate_lqr_control, plan_trajectory, verify_safety
"""
    
    state["messages"] = [HumanMessage(content=observation)]
    state["tool_calls"] = []
    state["step"] = step
    
    return state


def analyze_errors_node(state: ControlState) -> ControlState:
    """
    Node that calls the error analysis tool
    """
    pos = state["position"]
    vel = state["velocity"]
    target_pos = state["target_pos"]
    target_vel = state.get("target_vel", 0.0)
    
    # Call error analysis tool
    if LANGGRAPH_AVAILABLE:
        try:
            error_result = analyze_errors.invoke({
                "position": pos,
                "velocity": vel, 
                "target_pos": target_pos,
                "target_vel": target_vel
            })
        except:
            # Fallback to direct call
            error_result = analyze_errors(pos, vel, target_pos, target_vel)
    else:
        error_result = analyze_errors(pos, vel, target_pos, target_vel)
    
    state["error_analysis"] = error_result
    
    # Add tool call to history
    tool_call = {
        "tool": "analyze_errors",
        "inputs": {"position": pos, "velocity": vel, "target_pos": target_pos, "target_vel": target_vel},
        "outputs": error_result
    }
    state["tool_calls"].append(tool_call)
    
    # Add message about analysis
    analysis_msg = f"""ERROR ANALYSIS COMPLETE:
- Position Error: {error_result['position_error']:.3f} m
- Velocity Error: {error_result['velocity_error']:.3f} m/s  
- Control Phase: {error_result['phase']}
- Urgency: {error_result['urgency']}
- Stability: {error_result['stability']}
- Recommendations: {error_result['recommendations'][0] if error_result['recommendations'] else 'None'}
"""
    
    state["messages"].append(AIMessage(content=analysis_msg))
    
    return state


def select_strategy_node(state: ControlState) -> ControlState:
    """
    Node that selects the optimal control strategy based on error analysis
    """
    error_analysis = state["error_analysis"]
    pos_error = error_analysis["position_error"]
    vel_error = error_analysis["velocity_error"]
    
    # Call strategy selection tool
    if LANGGRAPH_AVAILABLE:
        try:
            strategy_result = select_control_strategy.invoke({
                "pos_error": pos_error,
                "vel_error": vel_error
            })
        except:
            strategy_result = select_control_strategy(pos_error, vel_error)
    else:
        strategy_result = select_control_strategy(pos_error, vel_error)
    
    state["strategy_selection"] = strategy_result
    
    tool_call = {
        "tool": "select_control_strategy",
        "inputs": {"pos_error": pos_error, "vel_error": vel_error},
        "outputs": strategy_result
    }
    state["tool_calls"].append(tool_call)
    
    strategy_msg = f"""STRATEGY SELECTION:
- Selected Strategy: {strategy_result['strategy']}
- Error Magnitude: {strategy_result['error_magnitude']:.3f}
- Recommended Gains: {strategy_result['recommended_gains']}
- Reasoning: {strategy_result['reasoning']}
"""
    
    state["messages"].append(AIMessage(content=strategy_msg))
    return state


def plan_action_node(state: ControlState) -> ControlState:
    """
    Node that plans the control action using the selected strategy
    """
    error_analysis = state["error_analysis"]
    strategy_selection = state["strategy_selection"]
    pos_error = error_analysis["position_error"]
    vel_error = error_analysis["velocity_error"]
    phase = error_analysis["phase"]
    urgency = error_analysis["urgency"]
    selected_strategy = strategy_selection["strategy"]
    
    # Choose control method based on strategy selection
    if selected_strategy == "lqr_optimal":
        # Use LQR for large errors
        pos = state["position"]
        vel = state["velocity"]
        target_pos = state["target_pos"]
        target_vel = state.get("target_vel", 0.0)
        
        if LANGGRAPH_AVAILABLE:
            try:
                lqr_result = calculate_lqr_control.invoke({
                    "current_pos": pos,
                    "current_vel": vel,
                    "target_pos": target_pos,
                    "target_vel": target_vel
                })
            except:
                lqr_result = calculate_lqr_control(pos, vel, target_pos, target_vel)
        else:
            lqr_result = calculate_lqr_control(pos, vel, target_pos, target_vel)
        
        state["control_plan"] = {
            "control_action": lqr_result["control_action"],
            "confidence": 0.95 if not lqr_result["saturated"] else 0.85,
            "reasoning": lqr_result["reasoning"],
            "components": {"lqr_based": True, "gains": lqr_result["lqr_gains"]}
        }
        
        tool_call = {
            "tool": "calculate_lqr_control",
            "inputs": {"current_pos": pos, "current_vel": vel, "target_pos": target_pos, "target_vel": target_vel},
            "outputs": lqr_result
        }
        state["tool_calls"].append(tool_call)
        
        plan_msg = f"""LQR OPTIMAL CONTROL:
- Recommended Force: {lqr_result['control_action']:.3f}
- LQR Gains: K1={lqr_result['lqr_gains'][0]:.3f}, K2={lqr_result['lqr_gains'][1]:.3f}
- Saturated: {lqr_result['saturated']}
- Reasoning: {lqr_result['reasoning']}
"""
        
    elif selected_strategy == "adaptive_pid" or selected_strategy == "standard_pid":
        # Use adaptive PID with recommended gains
        recommended_gains = strategy_selection["recommended_gains"]
        kp = recommended_gains["kp"]
        kd = recommended_gains["kd"]
        
        if LANGGRAPH_AVAILABLE:
            try:
                pid_result = calculate_pid_control.invoke({
                    "pos_error": pos_error,
                    "vel_error": vel_error,
                    "kp": kp,
                    "kd": kd,
                    "adaptive": False  # Use specified gains
                })
            except:
                pid_result = calculate_pid_control(pos_error, vel_error, kp=kp, kd=kd, adaptive=False)
        else:
            pid_result = calculate_pid_control(pos_error, vel_error, kp=kp, kd=kd, adaptive=False)
        
        state["control_plan"] = pid_result
        
        tool_call = {
            "tool": "calculate_pid_control",
            "inputs": {"pos_error": pos_error, "vel_error": vel_error, "kp": kp, "kd": kd},
            "outputs": pid_result
        }
        state["tool_calls"].append(tool_call)
        
        plan_msg = f"""ADAPTIVE PID CONTROL:
- Recommended Force: {pid_result['control_action']:.3f}
- Strategy: {pid_result['strategy_used']}
- P-term: {pid_result['components']['p_term']:.3f}
- D-term: {pid_result['components']['d_term']:.3f}
- Confidence: {pid_result['confidence']:.3f}
- Reasoning: {pid_result['reasoning']}
"""
        
    elif phase == "fine_tune" or urgency == "low":
        # Fallback to standard PID for fine control
        # Use standard PID for fine control
        if LANGGRAPH_AVAILABLE:
            try:
                pid_result = calculate_pid_control.invoke({
                    "pos_error": pos_error,
                    "vel_error": vel_error,
                    "adaptive": True
                })
            except:
                pid_result = calculate_pid_control(pos_error, vel_error, adaptive=True)
        else:
            pid_result = calculate_pid_control(pos_error, vel_error, adaptive=True)
        
        state["control_plan"] = pid_result
        
        tool_call = {
            "tool": "calculate_pid_control", 
            "inputs": {"pos_error": pos_error, "vel_error": vel_error, "adaptive": True},
            "outputs": pid_result
        }
        state["tool_calls"].append(tool_call)
        
        plan_msg = f"""ADAPTIVE PID CONTROL (FALLBACK):
- Recommended Force: {pid_result['control_action']:.3f}
- Strategy: {pid_result['strategy_used']}
- P-term: {pid_result['components']['p_term']:.3f}
- D-term: {pid_result['components']['d_term']:.3f}
- Confidence: {pid_result['confidence']:.3f}
- Reasoning: {pid_result['reasoning']}
"""
        
    else:
        # Use trajectory planning for more complex situations
        pos = state["position"]
        vel = state["velocity"]
        target_pos = state["target_pos"]
        target_vel = state.get("target_vel", 0.0)
        
        if LANGGRAPH_AVAILABLE:
            try:
                traj_result = plan_trajectory.invoke({
                    "current_pos": pos,
                    "current_vel": vel,
                    "target_pos": target_pos,
                    "target_vel": target_vel
                })
            except:
                traj_result = plan_trajectory(pos, vel, target_pos, target_vel)
        else:
            traj_result = plan_trajectory(pos, vel, target_pos, target_vel)
        
        state["trajectory_plan"] = traj_result
        
        # Extract immediate action from trajectory plan
        if traj_result["control_sequence"]:
            immediate_action = traj_result["control_sequence"][0][0]  # First action
            confidence = 0.8 if traj_result["feasible"] else 0.6
        else:
            immediate_action = 0.0
            confidence = 0.3
        
        # Create control plan in same format as PID
        control_plan = {
            "control_action": immediate_action,
            "confidence": confidence,
            "reasoning": f"Trajectory planning: {traj_result['strategy']}",
            "components": {"trajectory_based": True}
        }
        state["control_plan"] = control_plan
        
        tool_call = {
            "tool": "plan_trajectory",
            "inputs": {"current_pos": pos, "current_vel": vel, "target_pos": target_pos, "target_vel": target_vel},
            "outputs": traj_result
        }
        state["tool_calls"].append(tool_call)
        
        plan_msg = f"""TRAJECTORY PLAN:
- Strategy: {traj_result['strategy']}
- Immediate Action: {immediate_action:.3f}
- Estimated Time: {traj_result['estimated_time']}s
- Feasible: {traj_result['feasible']}
"""
    
    state["messages"].append(AIMessage(content=plan_msg))
    
    return state


def verify_safety_node(state: ControlState) -> ControlState:
    """
    Node that verifies the safety of the planned action
    """
    control_plan = state["control_plan"]
    planned_action = control_plan["control_action"]
    pos = state["position"]
    vel = state["velocity"]
    
    # Call safety verification tool
    if LANGGRAPH_AVAILABLE:
        try:
            safety_result = verify_safety.invoke({
                "control_action": planned_action,
                "current_pos": pos,
                "current_vel": vel
            })
        except:
            safety_result = verify_safety(planned_action, pos, vel)
    else:
        safety_result = verify_safety(planned_action, pos, vel)
    
    state["safety_check"] = safety_result
    
    tool_call = {
        "tool": "verify_safety",
        "inputs": {"control_action": planned_action, "current_pos": pos, "current_vel": vel},
        "outputs": safety_result
    }
    state["tool_calls"].append(tool_call)
    
    # Use the adjusted action from safety check
    final_action = safety_result["adjusted_action"]
    action_changed = safety_result["action_changed"]
    
    safety_msg = f"""SAFETY VERIFICATION:
- Original Action: {planned_action:.3f}
- Final Action: {final_action:.3f}
- Action Changed: {action_changed}
- Safe: {safety_result['safe']}
- Warnings: {', '.join(safety_result['warnings']) if safety_result['warnings'] else 'None'}
"""
    
    state["messages"].append(AIMessage(content=safety_msg))
    
    return state


def execute_action_node(state: ControlState) -> ControlState:
    """
    Final node that outputs the control action
    """
    safety_check = state["safety_check"]
    control_plan = state["control_plan"]
    error_analysis = state["error_analysis"]
    
    # Get final action from safety check
    final_action = safety_check["adjusted_action"]
    
    # Calculate overall confidence
    base_confidence = control_plan["confidence"]
    if safety_check["action_changed"]:
        overall_confidence = base_confidence * 0.9  # Slight reduction if action was changed
    else:
        overall_confidence = base_confidence
    
    # Create comprehensive reasoning
    reasoning_parts = [
        f"Phase: {error_analysis['phase']}",
        f"Urgency: {error_analysis['urgency']}",
        control_plan["reasoning"],
    ]
    
    if safety_check["warnings"]:
        reasoning_parts.append(f"Safety warnings: {', '.join(safety_check['warnings'])}")
    
    if safety_check["action_changed"]:
        reasoning_parts.append("Action adjusted for safety")
    
    comprehensive_reasoning = " | ".join(reasoning_parts)
    
    # Set final outputs
    state["action"] = final_action
    state["confidence"] = round(overall_confidence, 3)
    state["reasoning"] = comprehensive_reasoning
    
    # Add final execution message
    exec_msg = f"""CONTROL EXECUTION:
- Final Action: {final_action:.3f}
- Confidence: {overall_confidence:.3f}
- Reasoning: {comprehensive_reasoning}

Tool calls completed: {len(state['tool_calls'])}
"""
    
    state["messages"].append(AIMessage(content=exec_msg))
    
    return state


def build_control_graph() -> Any:
    """
    Build and compile the control graph workflow
    """
    # Create the graph
    if LANGGRAPH_AVAILABLE:
        try:
            workflow = StateGraph(ControlState)
        except TypeError:
            # Try newer API
            workflow = StateGraph(state_schema=ControlState)
    else:
        workflow = StateGraph()
    
    # Add nodes
    workflow.add_node("observe_state", observe_state)
    workflow.add_node("analyze_errors", analyze_errors_node)
    workflow.add_node("select_strategy", select_strategy_node)
    workflow.add_node("plan_action", plan_action_node)
    workflow.add_node("verify_safety", verify_safety_node) 
    workflow.add_node("execute_action", execute_action_node)
    
    # Add edges to define the workflow
    workflow.add_edge("observe_state", "analyze_errors")
    workflow.add_edge("analyze_errors", "select_strategy")
    workflow.add_edge("select_strategy", "plan_action")
    workflow.add_edge("plan_action", "verify_safety")
    workflow.add_edge("verify_safety", "execute_action")
    workflow.add_edge("execute_action", END)
    
    # Set entry point
    workflow.set_entry_point("observe_state")
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph


class ToolAugmentedController:
    """
    Controller that uses the LangGraph workflow for tool-augmented control
    """
    
    def __init__(self):
        self.graph = build_control_graph()
        self.step_count = 0
        
    def get_action(self, position: float, velocity: float, 
                   target_pos: float, target_vel: float = 0.0) -> Dict:
        """
        Get control action using tool-augmented reasoning
        """
        self.step_count += 1
        
        # Create initial state
        initial_state = {
            "position": position,
            "velocity": velocity,
            "target_pos": target_pos, 
            "target_vel": target_vel,
            "step": self.step_count,
            "error_analysis": None,
            "strategy_selection": None,
            "control_plan": None,
            "trajectory_plan": None,
            "safety_check": None,
            "action": 0.0,
            "confidence": 0.0,
            "reasoning": "",
            "tool_calls": [],
            "messages": []
        }
        
        # Execute the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "action": final_state["action"],
                "confidence": final_state["confidence"],
                "reasoning": final_state["reasoning"],
                "step": self.step_count,
                "tool_calls": final_state["tool_calls"],
                "messages": [msg.content for msg in final_state["messages"]],
                "error_analysis": final_state.get("error_analysis", {}),
                "strategy_selection": final_state.get("strategy_selection", {}),
                "control_plan": final_state.get("control_plan", {}),
                "safety_check": final_state.get("safety_check", {})
            }
            
        except Exception as e:
            return {
                "action": 0.0,
                "confidence": 0.0,
                "reasoning": f"Graph execution failed: {e}",
                "step": self.step_count,
                "tool_calls": [],
                "messages": [],
                "error": str(e)
            }


def test_control_graph():
    """Test the control graph with various scenarios"""
    print("🧪 Testing Control Graph")
    print("=" * 50)
    
    controller = ToolAugmentedController()
    
    # Test scenarios
    test_cases = [
        {"name": "Close positioning", "pos": 0.1, "vel": 0.0, "target": 0.0},
        {"name": "Far positioning", "pos": 0.8, "vel": 0.0, "target": 0.0},
        {"name": "High velocity", "pos": 0.0, "vel": 0.5, "target": 0.0},
        {"name": "Moving away", "pos": 0.2, "vel": 0.3, "target": 0.0},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 Test {i}: {test_case['name']}")
        print(f"   State: pos={test_case['pos']}, vel={test_case['vel']}")
        print(f"   Target: {test_case['target']}")
        
        result = controller.get_action(
            position=test_case['pos'],
            velocity=test_case['vel'],
            target_pos=test_case['target'],
            target_vel=0.0
        )
        
        print(f"   Action: {result['action']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Tools used: {len(result['tool_calls'])}")
        
        # Show tool chain
        for j, tool_call in enumerate(result['tool_calls']):
            print(f"     {j+1}. {tool_call['tool']}")
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Success")
    
    print("\n🔄 Full Workflow Example:")
    print("-" * 30)
    
    # Detailed example with one case
    detailed_result = controller.get_action(0.5, 0.0, 0.0, 0.0)
    
    if 'messages' in detailed_result:
        for i, message in enumerate(detailed_result['messages']):
            print(f"{i+1}. {message.split(':')[0]}:")
            print(f"   {message.split(':', 1)[1].strip()[:100]}...")
    
    print(f"\n📊 Final Result:")
    print(f"   Action: {detailed_result['action']}")
    print(f"   Confidence: {detailed_result['confidence']}")
    print(f"   Reasoning: {detailed_result['reasoning']}")
    
    print("\n✅ Control graph testing completed!")


if __name__ == "__main__":
    test_control_graph()