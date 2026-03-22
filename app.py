import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

GROQ_API_KEY = "your-groq-api-key-here"

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)

class TicketState(TypedDict):
    ticket: str
    category: str
    priority: str
    reply: str

def classify_ticket(state: TicketState) -> TicketState:
    response = llm.invoke([
        SystemMessage(content="You are a support ticket classifier. Classify the ticket into one category: Billing, Technical, General, Urgent. Reply with ONLY the category name."),
        HumanMessage(content=state["ticket"])
    ])
    state["category"] = response.content.strip()
    return state

def assign_priority(state: TicketState) -> TicketState:
    response = llm.invoke([
        SystemMessage(content="You are a priority assignment agent. Based on the ticket and category, assign priority: High, Medium, or Low. Reply with ONLY the priority level."),
        HumanMessage(content=f"Ticket: {state['ticket']}\nCategory: {state['category']}")
    ])
    state["priority"] = response.content.strip()
    return state

def draft_reply(state: TicketState) -> TicketState:
    response = llm.invoke([
        SystemMessage(content="You are a customer support agent. Write a professional, empathetic reply to the support ticket. Keep it concise - 3-4 sentences max."),
        HumanMessage(content=f"Ticket: {state['ticket']}\nCategory: {state['category']}\nPriority: {state['priority']}")
    ])
    state["reply"] = response.content.strip()
    return state

def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("prioritize", assign_priority)
    graph.add_node("draft_reply", draft_reply)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "prioritize")
    graph.add_edge("prioritize", "draft_reply")
    graph.add_edge("draft_reply", END)
    return graph.compile()

agent = build_graph()

st.set_page_config(page_title="AI Ticket Triage Agent", page_icon="🎫")
st.title("🎫 AI Ticket Triage Agent")
st.caption("Powered by LangGraph + Groq LLaMA 3 — Automatic classification, prioritization & reply drafting")

st.markdown("### Paste a support ticket below")

examples = [
    "My payment was charged twice and I need a refund immediately!",
    "How do I reset my password?",
    "The app keeps crashing when I try to upload files. This is urgent!",
    "I want to know about your premium plans.",
]

selected = st.selectbox("Or pick an example ticket:", ["-- Select --"] + examples)
ticket_input = st.text_area("Support Ticket:", value=selected if selected != "-- Select --" else "", height=120)

if st.button("Process Ticket", type="primary"):
    if ticket_input.strip():
        with st.spinner("AI agents working..."):
            result = agent.invoke({
                "ticket": ticket_input,
                "category": "",
                "priority": "",
                "reply": ""
            })

        col1, col2 = st.columns(2)
        with col1:
            color = {"Urgent": "🔴", "Billing": "🟡", "Technical": "🔵", "General": "🟢"}
            emoji = color.get(result["category"], "⚪")
            st.metric("Category", f"{emoji} {result['category']}")
        with col2:
            p_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            p_emoji = p_color.get(result["priority"], "⚪")
            st.metric("Priority", f"{p_emoji} {result['priority']}")

        st.markdown("### Drafted Reply")
        st.info(result["reply"])

        st.markdown("### Agent Pipeline")
        st.success("Step 1: Classified → Step 2: Priority Assigned → Step 3: Reply Drafted")
    else:
        st.warning("Please enter a support ticket!")
