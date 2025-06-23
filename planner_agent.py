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

class PlannerAgent:
    def __init__(self, model_name: str = "gpt-4o", embedding_model: str = "text-embedding-3-large"):
        # Initialize LLM and embeddings
        self.model = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model=embedding_model)
        
        self.memory_agent = __import__('memory_agent', fromlist=['MemoryAgent']).MemoryAgent()
        self.action_agent = __import__('nao_action_agent', fromlist=['NaoActionAgent']).NaoActionAgent()
        from face_analysis import analyze_face

        # Wrap the two agents’ top‐level run_once methods as tools:
        self.tools = [
            Tool.from_function(
                self.memory_agent.run_once,
                name="memory_agent",
                description="Use this to read from or write to long‐term memory"
            ),
            Tool.from_function(
                self.action_agent.run_once,
                name="action_agent",
                description="Use this to perform physical or vocal actions"
            ),
                StructuredTool.from_function(
                analyze_face,
                name="analyze_face",
                description="Analyze face of user and returns emotion of user",
                args_schema=None
            ),
        ]

        # Build a simple “planner” prompt
        system_prompt = """
    You are the Planner of a LLM Integrated Nao Robot. The Robot have long term memories (semantic, episodic, procedurals) and physical action capabilities.
    The robot can also find out user's gender and emotion.

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

    action_agent has available action tools [speak, search_web, wave, stand, sit, crouch, rest, move, nod_head, turn_head, gaze_head, raise_arms, walk, handshake, come_back_home, reset_nao_pose, shutdown]
    
    you also have an extra tool [analyze_face] to get user's emotion

     When the user says something, decide whether you need to:
    - call the memory_agent (e.g. to look up or store info from memory)
    - call the action_agent (e.g. to perform various actions)
    - call analyze_face to get user's emotion
    - always use action agent to say anything. (e.g. if user wants to sing you waka waka song, message to action agent: sing waka waka song).
    User's name is {username}. Today is {date}.
    
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

    Remind that you are communicating with memory and action agent. Do not pass message to these agent as you are communicating with the user. Always command those agents. Do not use, Hello {username} how are you doing? Instead of this, Use, say: Hello {username} how are you doing?

If you call a tool, return exactly the tool call JSON; afterwards the tool's output will come back to you and you can plan the next step.
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])
        self.model_with_tools = self.model.bind_tools(self.tools)

        # And wire it all into a StateGraph so that tool calls loop back
        class State(MessagesState): pass

        def planner_agent(state: State, config: RunnableConfig) -> State:
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
        builder.add_node("planner_agent", planner_agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "planner_agent")
        builder.add_conditional_edges("planner_agent", route_tools, ["tools", END])
        builder.add_edge("tools", "planner_agent")

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


if __name__ == "__main__":
    planner=PlannerAgent()
    planner.chat_cli()