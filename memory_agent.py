import os
import platform
import time
import uuid
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

# Load environment
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Platform clear command
ios_name = platform.system()
CLEAR_CMD = 'cls' if ios_name == 'Windows' else 'clear'

class MemoryAgent:
    def __init__(
        self,
        db_dir: str = "./memory_db",
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
        max_search_results: int = 1,
    ):
        # Initialize LLM and embeddings
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize memory stores
        self.mem = __import__('memory', fromlist=['Memory']).Memory(db_dir=db_dir)

        # Wrap memory methods as tools
        self.tools = [
            Tool.from_function(
                self.mem.save_semantic_memory,
                name="save_semantic_memory",
                description="Save a fact to semantic memory",
            ),
            Tool.from_function(
                self.mem.save_episodic_memory,
                name="save_episodic_memory",
                description="Save a user preference to episodic memory",
            ),
            Tool.from_function(
                self.mem.save_procedural_memory,
                name="save_procedural_memory",
                description="Save a procedure to procedural memory",
            ),
            Tool.from_function(
                self.mem.search_semantic_memory,
                name="search_semantic_memory",
                description="Retrieve relevant semantic memories",
            ),
            Tool.from_function(
                self.mem.search_episodic_memory,
                name="search_episodic_memory",
                description="Retrieve relevant episodic memories",
            ),
            Tool.from_function(
                self.mem.search_procedural_memory,
                name="search_procedural_memory",
                description="Retrieve relevant procedural memories",
            ),
            Tool.from_function(
                self.mem.get_full_long_term_memory,
                name="get_full_long_term_memory",
                description="List all stored memories",
            ),
            TavilySearchResults(max_results=max_search_results),
        ]

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)


        # Setup prompt
        _SYSTEM_TEMPLATE = """You are a memory agent of a Nao Robot. The robot have long term memory and action capabilities. 
                    You work to store memories and retrieve them. Your response will then be passed to an action agent. So by your response, you will comunicate with that action agent.

                    User's name is {username}. Today is {date}.
                    Memories are saved with conversation date.
                    There are three types of memories: semantic, episodic, procedural.
                    memory_type: "semantic"
                    for storing:
                    - facts about the user. Like username, address, personal preference like favourite color, food etc
                    - Personal information like the institution user is studying, company he is doing job etc.
                    memory_type: "episodic"
                    for storing:
                    - User's preference of your response, for example: You elaborate much about a topic but user wants it brief. Then you store this preference of user in episodic memory
                    - User-specific adaptation: Adjust your explanation according to user's expertise level. Store information in "episodic" memory about user's ability to learn so that you can generate response accordingly.
                    memory_type: "procedural"
                    for storing:
                    - Procedure of any action or work explained by the user.

                    Engage with the user naturally, as a trusted colleague or friend.
                    There's no need to explicitly mention your memory capabilities.
                    Instead, seamlessly incorporate your understanding of the user
                    into your responses. Be attentive to subtle cues and underlying
                    emotions. Adapt your communication style to match the user's
                    preferences and current emotional state. Use tools to persist
                    information you want to retain in the next conversation. If you
                    do call tools, all text preceding the tool call is an internal
                    message. Respond AFTER calling the tool, once you have
                    confirmation that the tool completed successfully.

                    Here are your instructions for reasoning about the user's messages:
                    1. Actively use memory tools [save_semantic_memory, save_episodic_memory, save_procedural_memory, search_semantic_memory, search_episodic_memory,search_procedural_memory, search_web, get_full_long_term_memory]
                    2. Always search for relevant memory before generating response.
                    use [search_semantic_memory, search_episodic_memory, search_procedural_memory] tools before generating response to recall rellevant memories from long term memories.
                    3. If you do not get info from your memory search, call get_ful_long_term_memory to get all the memories
                    4. Before saving a memory, search for memories if the memory already exists in there.
                    5. After all necessary memory calls, generate response for the Action Agent with instructions how to interact with the user.
                        The Action Agent have action tools [capture_image, stream_video, speak, wave, stand, sit, crouch, rest, move, nod_head, turn_head, gaze_head, raise_arms, walk, handshake, come_back_home, reset_nao_pose, shutdown]
                        e.g. If user asks "what is my favorite color?" your response for the action agent will be "user's favourite color is blue, say the user that user's favourite color is blue" 
                        to instruct the Action Agent to make it say something, for example to recite a poem, Response with 'say - that poems line'
                        Don't generate response like you are talking with the user. Remind that you are not talking with the user, you are generating instructions for the Action Agent using Acition Agent's available tools.


                    \n\n"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_TEMPLATE),
            ("placeholder", "{messages}"),
        ])

        # Build conversation graph
        class State(MessagesState):
            pass

        def memory_agent(state: State, config: RunnableConfig) -> State:
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
        builder.add_node("memory_agent", memory_agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "memory_agent")
        builder.add_conditional_edges("memory_agent", route_tools, ["tools", END])
        builder.add_edge("tools", "memory_agent")

        self.graph = builder.compile(checkpointer=MemorySaver())

    def pretty_print_stream_chunk(self, chunk):
        for node_name, updates in chunk.items():
            print(f"Update from node: {node_name}")
            if "messages" in updates:
                updates["messages"][-1].pretty_print()
            else:
                print(updates)
            print()


    def get_memory_response(self, chunk):
        if "memory_agent" in chunk:
            node_updates = chunk["memory_agent"]
            msg = node_updates["messages"][-1].content
            if msg:
                return msg


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
                #self.get_memory_response(chunk)

# Run CLI
if __name__ == "__main__":
    agent = MemoryAgent()
    agent.chat_cli()