import time
from langchain_core.messages import HumanMessage

from memory_agent import MemoryAgent
from nao_action_agent import NaoActionAgent



if __name__ == "__main__":
    mem = MemoryAgent()
    action = NaoActionAgent()
    username = input("Enter your username: ").strip()
    date = time.strftime("%Y-%m-%d", time.localtime())
    config = {"configurable": {"user_id": username, "thread_id": date}}

    for chunk in action.graph.stream({"messages": "Hi"}, config=config):
        action.pretty_print_stream_chunk(chunk)
    while True:
        user_input = input(f"\nTalk with the robot (or type 'stop' to end): ")
        if user_input.strip().lower() == "stop":
            break
        msgs = [HumanMessage(content=user_input)]

        # memory save and retrieval
        for chunk in mem.graph.stream({"messages": msgs}, config=config):
            mem.pretty_print_stream_chunk(chunk)
            memory_msgs = mem.get_memory_response(chunk)

        # print action responses
        for chunk in action.graph.stream({"messages": memory_msgs}, config=config):
            action.pretty_print_stream_chunk(chunk)