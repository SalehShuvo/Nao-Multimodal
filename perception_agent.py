from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig


import os, time
from dotenv import load_dotenv, find_dotenv
# Load environment
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class PerceptionAgent:
    def __init__(self, model_name: str = "gpt-4o", embedding_model: str = "text-embedding-3-large"):
        # Initialize LLM and embeddings
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model=embedding_model)
        
        from camera import capture_and_save_image
        from face_analysis import analyze_face

        # Wrap the two agents’ top‐level run_once methods as tools:
        self.tools = [
            StructuredTool.from_function(
                capture_and_save_image,
                name="capture_and_save_image",
                description="Capture one frame from the specified camera and save it to disk",
                args_schema=None
            ),
                StructuredTool.from_function(
                analyze_face,
                name="analyze_face",
                description="Analyze face of user and returns a dict containing gender, and emotion. Empty strings for undetected attributes",
                args_schema=None
            ),

        ]

        system_prompt = """
        You are a Perception Agent. You will get an image and questions related to the scene. Your task is to answer those questions.
        If you get empty string during face analysis, capture another image.
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])
        self.model_with_tools = self.model.bind_tools(self.tools)

        # And wire it all into a StateGraph so that tool calls loop back
        class State(MessagesState): pass

        def perception_agent(state: State, config: RunnableConfig) -> State:
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

        builder = StateGraph(State)
        builder.add_node("perception_agent", perception_agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "perception_agent")
        builder.add_conditional_edges("perception_agent", route_tools, ["tools", END])
        builder.add_edge("tools", "perception_agent")

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
    
    
    def run_once(self, user_input):
        """
        Run a single turn through the perception-agent graph.
        user_input: either a plain string or a list of HumanMessage
        returns: final text response (str)
        """
        # normalize into list of HumanMessage
        if isinstance(user_input, str):
            messages = [HumanMessage(content=user_input)]
        else:
            messages = user_input

        # build config with a user_id and thread_id (date)
        config = {
            "configurable": {
                "user_id": "planner",  # or whatever identifier you like
                "thread_id": time.strftime("%Y-%m-%d", time.localtime())
            }
        }

        reply = None
        # stream through the graph
        for chunk in self.graph.stream({"messages": messages}, config=config):
            # look for the planner‐agent node output without further tool_calls
            if "perception_agent" in chunk:
                node_updates = chunk["perception_agent"]
                msg_obj = node_updates["messages"][-1]
                # if it's not asking for another tool call, capture it
                if not msg_obj.tool_calls:
                    reply = msg_obj.content
        # print(reply)
        return reply


if __name__ == "__main__":
    perception=PerceptionAgent()
    perception.chat_cli()