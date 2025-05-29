import time
from dotenv import load_dotenv, find_dotenv
import os 
# Load environment
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#langchain utils import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from pydantic import BaseModel, Field

class NaoActionAgent:
    def __init__(self, model_name: str = "gpt-4o", embedding_model: str = "text-embedding-3-large"):

        # Initialize LLM and embeddings
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize nao_qibullet
        self.nao = __import__('nao_pybullet', fromlist=['Nao']).Nao()

        # Schemas
        class MoveArgs(BaseModel):
            x: float = Field(..., description="Forward displacement in meters")
            y: float = Field(..., description="Lateral displacement in meters")
            theta: float = Field(..., description="Rotation (yaw) in radians")
        class WalkArgs(BaseModel):
            x: float   = Field(..., description="Forward distance (m)")
            y: float   = Field(..., description="Lateral distance (m)")
            theta: float = Field(..., description="Yaw rotation (rad)")
        
        # Wrap nao's action methods as tools
        self.tools = [
            Tool.from_function(
                self.nao.capture_image,
                name="capture_image",
                description="Captures Image from top or bottom camera",
            ),
            Tool.from_function(
                self.nao.stream_video,
                name="stream_video",
                description="Streams video from top or bottom camera",
            ),
            Tool.from_function(
                self.nao.speak,
                name="speak",
                description="text to voice output",
            ),
            Tool.from_function(
                self.nao.wave,
                name="wave",
                description="Waves by right or left hand",
            ),
            StructuredTool.from_function(
                self.nao.stand,
                name="stand",
                description="Goes to Stand posture.",
                args_schema=None
            ),
            StructuredTool.from_function(
                self.nao.sit,
                name="sit",
                description="Goes to Sit posture.",
                args_schema=None
            ),
            StructuredTool.from_function(
                self.nao.crouch,
                name="crouch",
                description="Goes to Crouch posture.",
                args_schema=None
            ),
            StructuredTool.from_function(
                self.nao.move,
                name="move",
                description="Move to Specific position",
                args_schema=MoveArgs
            ),
            Tool.from_function(
                self.nao.nod_head,
                name="nod_head",
                description="Nods head up_down or right_left",
            ),
            StructuredTool.from_function(
                self.nao.rest,
                name="rest",
                description="Goes to Rest posture.",
                args_schema=None
            ),
            Tool.from_function(
                self.nao.turn_head,
                name="turn_head",
                description="Turns head right or left",
            ),
            Tool.from_function(
                self.nao.gaze_head,
                name="gaze_head",
                description="Gazes head up or down",
            ),
            Tool.from_function(
                self.nao.raise_arms,
                name="raise_arms",
                description="Raises arms both or left or right",
            ),
            Tool.from_function(
                self.nao.handshake,
                name="handshake",
                description="Handshake with or right or left hand",
            ),
            StructuredTool.from_function(
            func=self.nao.walk,
            name="walk",
            description="Make the robot walk by (x, y, theta).",
            args_schema=WalkArgs
            ),
            StructuredTool.from_function(
                self.nao.come_back_home,
                name="come_back_home",
                description="Gets back to home position.",
                args_schema=None
            ),
            StructuredTool.from_function(
                self.nao.reset_nao_pose,
                name="reset_nao_pose",
                description="Resets Robot's posture",
                args_schema=None
            ),
            StructuredTool.from_function(
                self.nao.shutdown,
                name="shutdown",
                description="Stops Simulation.",
                args_schema=None
            ),
        ]
        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)

           # Setup prompt
        _SYSTEM_TEMPLATE = """You are a part of a Nao robot. You help to perform different actions.
        User's response will go to a Memory agent and after nessesary memory saving and retrieval, Memory agent's response will come to you and you will perform actions according to the memory agent's instructions.
        Available action tools [capture_image, stream_video, speak, wave, stand, sit, crouch, rest, move, nod_head, turn_head, gaze_head, raise_arms, walk, handshake, come_back_home, reset_nao_pose, shutdown]
        Instructions for generating actions:
        1. Read the Memory agent's instruction carefully and plan how will you perform actions step by step. Then perform tool calls. Remind that you are geeting messages form the Memory Agent. Not from user. Your response will go to user.
        2. Communicate humanly. Perform necessary gesture in your communication e.g wave hand after saying greetings.
        3. Call tools untill your communiction is successful with user.
        4. After performing all necessary actions, say conclusions and call reset_nao_pose to reset robot posture.

        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_TEMPLATE),
            ("placeholder", "{messages}"),
        ])

        # Build conversation graph
        class State(MessagesState):
            pass

        def action_agent(state: State, config: RunnableConfig) -> State:
            username = config["configurable"].get("user_id")
            date = config["configurable"].get("thread_id")
            bound = self.prompt | self.model_with_tools
            prediction = bound.invoke({
                "username": username,
                "date": date,
                "messages": state["messages"],
            })
            return {"messages": [prediction]}

        def route_tools(state: State):
            return "tools" if state["messages"][-1].tool_calls else END

        # Instantiate state graph
        builder = StateGraph(State)
        builder.add_node("action_agent", action_agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "action_agent")
        builder.add_conditional_edges("action_agent", route_tools, ["tools", END])
        builder.add_edge("tools", "action_agent")

        self.graph = builder.compile(checkpointer=MemorySaver())
    
    def pretty_print_stream_chunk(self, chunk):
        for node_name, updates in chunk.items():
            print(f"Update from node: {node_name}")
            if "messages" in updates:
                updates["messages"][-1].pretty_print()
            else:
                print(updates)
            print()
    

    def chat_cli(self):
        username = input("Enter your username: ").strip()
        date = time.strftime("%Y-%m-%d", time.localtime())
        config = {"configurable": {"user_id": username, "thread_id": date}}

        for chunk in self.graph.stream({"messages": "Hi"}, config=config):
            self.pretty_print_stream_chunk(chunk)
        while True:
            user_input = input(f"\nTalk with the robot (or type 'stop' to end): ")
            if user_input.strip().lower() == "stop":
                break
            msgs = [HumanMessage(content=user_input)]
            for chunk in self.graph.stream({"messages": msgs}, config=config):
                self.pretty_print_stream_chunk(chunk)

# Run CLI
if __name__ == "__main__":

    agent = NaoActionAgent()
    agent.chat_cli()