from agent import Agent
import sys
import os
import inspect

print(f"Agent file: {inspect.getfile(Agent)}")
print(f"Agent attributes: {dir(Agent)}")

try:
    print("Agent source:")
    print(inspect.getsource(Agent))
except Exception as e:
    print(f"Could not get source: {e}")

try:
    a = Agent(None, None, None, None, None, None, None)
    if hasattr(a, 'play_one'):
        print("Agent has play_one")
    else:
        print("Agent DOES NOT have play_one")
except Exception as e:
    print(f"Error creating agent: {e}")
