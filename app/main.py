from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
import asyncio


MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "2"}}


async def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)

        while True:
            user_input = input("You: ")

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            for event in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values"
            ):
                if "messages" in event:
                    event["messages"][-1].pretty_print()


asyncio.run(main())
