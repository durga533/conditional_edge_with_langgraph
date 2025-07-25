from langgraph.graph import MessageGraph,END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env vars
load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


write_prompt = ChatPromptTemplate.from_template("Write a LinkedIn post about the {topic}")

writer_chain = write_prompt | model

reviewer_prompt = ChatPromptTemplate.from_template("Review the LinkedIn post {post} and suggest improvements")

reviewer_chain = reviewer_prompt | model


graph = MessageGraph()

def writer_node(state):
  topic = state[-1].content
  answer = writer_chain.invoke({"topic":topic})
  return state + [AIMessage(content=answer.content)]


def reviewer_node(state):
  post = state[-1].content
  review = reviewer_chain.invoke({"post":post})
  return state + [AIMessage(content=review.content)]


graph.add_node("Writer", writer_node)
graph.add_node("Reviewer", reviewer_node)
graph.set_entry_point("Writer")



def condition_node(state):
  if len(state) >=5:
    return END
  else:
    return "Reviewer"


graph.add_conditional_edges("Writer", condition_node)
graph.add_edge("Reviewer", "Writer")

Agent = graph.compile()

result = Agent.invoke([HumanMessage(content = "Write a post about Agentic AI")])

for msg in result:
    print(msg.content)

display(Image(Agent.get_graph().draw_mermaid_png()))