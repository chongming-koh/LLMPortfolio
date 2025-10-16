# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:26:23 2025

@author: Koh Chong Ming
"""

import os
import requests
import gradio as gr
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import time
import json

#Find the key file

os.chdir("C:\\Users\\vital\\PythonStuff\\keys")
cwd = os.getcwd() 

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

llama_8b_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_70b_model ="meta-llama/Llama-3.3-70B-Instruct"
gemma_9b_model = "google/gemma-2-9b-it-fast"
Qwen2_5_72B_model = "Qwen/Qwen2.5-72B-Instruct"
DeepSeek_V33024 ="deepseek-ai/DeepSeek-V3-0324"
openai_20b = "openai/gpt-oss-20b"
Hermes_4_70B_model ="NousResearch/Hermes-4-70B"

system_prompt = "You are a helpful assistant for an airline called FlightAI. \
    Answer general questions normally. \
    ONLY call get_ticket_price if the user explicitly asks about ticket prices or destinations. \
    Call book_tickets if the user asks to book or purchase tickets. \
    Call convert_currency if the user asks to convert prices or currencies. \
    Do NOT call any tools for unrelated questions like greetings, names, or general chat. \
    Keep answers short and courteous."

print(system_prompt)

# Ticket price data in SGD
ticket_prices = {"london": 1040, "paris": 1170, "tokyo": 1820, "berlin": 650}

# Currency conversion rates
conversion_rates = {"USD": 1.30, "EUR": 1.50}

# ---------------- Functions ----------------

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    price = ticket_prices.get(city)
    if price:
        return {"destination_city": city, "price_sgd": price}
    else:
        return {"error": "Unknown destination."}

def format_currency(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    return f"{value:,.2f}"

def book_tickets(destination_city, number_of_tickets):
    print(f"Tool book_tickets called for {destination_city} ({number_of_tickets} tickets)")
    city = destination_city.lower()
    if city not in ticket_prices:
        return {"error": "Destination not available."}
    total_price = ticket_prices[city] * number_of_tickets

    # Simplified plain text receipt with formatted currency
    receipt = (
        f"\n✈️ FlightAI Booking Receipt\n"
        f"---------------------------------\n"
        f"Destination: {city.title()}\n"
        f"Tickets Purchased: {number_of_tickets}\n"
        f"Price per Ticket: ${format_currency(ticket_prices[city])} SGD\n"
        f"Total Amount: ${format_currency(total_price)} SGD\n"
        f"---------------------------------\n"
        f"Thank you for choosing FlightAI. Have a pleasant journey!"
    )

    return {
        "destination_city": city,
        "tickets": number_of_tickets,
        "total_price_sgd": total_price,
        "receipt": receipt
    }


def convert_currency(amount, from_currency):
    print(f"Tool convert_currency called: {amount} {from_currency} -> SGD")
    try:
        amount = float(amount)  # ensure numeric conversion
    except ValueError:
        return {"error": "Amount must be a number."}

    rate = conversion_rates.get(from_currency.upper())
    if rate:
        converted = amount * rate
        receipt = (
            f"\n💱 Currency Conversion Receipt\n"
            f"---------------------------------\n"
            f"Amount: {format_currency(amount)} {from_currency.upper()}\n"
            f"Conversion Rate: 1 {from_currency.upper()} = {rate} SGD\n"
            f"Converted Amount: {format_currency(converted)} SGD\n"
            f"---------------------------------\n"
            f"Thank you for using FlightAI Currency Converter."
        )
        return {
            "amount_sgd": round(converted, 2),
            "rate": rate,
            "receipt": receipt
        }
    else:
        return {"error": "Unsupported currency."}

# ---------------- Tool definitions ----------------

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to a destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {"type": "string", "description": "City for the ticket inquiry."},
        },
        "required": ["destination_city"]
    }
}

book_function = {
    "name": "book_tickets",
    "description": "Book tickets for a specific destination and number of passengers.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {"type": "string", "description": "City for booking."},
            "number_of_tickets": {"type": "integer", "description": "Number of tickets to purchase."},
        },
        "required": ["destination_city", "number_of_tickets"]
    }
}

convert_function = {
    "name": "convert_currency",
    "description": "Convert USD or EUR amount into SGD using predefined rates.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount to convert."},
            "from_currency": {"type": "string", "description": "Currency code: USD or EUR."},
        },
        "required": ["amount", "from_currency"]
    }
}

# Register all tools
tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": book_function},
    {"type": "function", "function": convert_function}
]

# ---------------- Chat Logic ----------------
# This fn will be called if the 1st response from LLM contains "tool_calls". Then it will invoke this fn
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if func_name == "get_ticket_price":
        result = get_ticket_price(**args)
    elif func_name == "book_tickets":
        result = book_tickets(**args)
    elif func_name == "convert_currency":
        result = convert_currency(**args)
    else:
        result = {"error": "Unknown tool called."}

    response = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    }
    return response


def chat(message, history,
         system_prompt=system_prompt,
         max_tokens=1048,
         client=nebius_client,
         model=llama_70b_model,
         temperature=0.7):

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools
    )

    print("🗣️ Initial reply:", completion.choices[0].message.content)

    if completion.choices[0].finish_reason == "tool_calls":
        message = completion.choices[0].message
        print("🤖 Model decided to call a tool!")
        tool_response = handle_tool_call(message)
        messages.append(message)
        messages.append(tool_response)
        #Make a second call to the model,  asking it to continue the conversation using the function’s result in JSON
        completion = client.chat.completions.create(model=model, messages=messages)
        print("🗣️ Final assistant reply:", completion.choices[0].message.content)

    return completion.choices[0].message.content

# ---------------- Custom Baby Blue UI ----------------

custom_css = """
body {
    background-color: #b3daff; /* baby blue */
}

.gradio-container {
    background-color: #b3daff !important;
    font-family: 'Segoe UI', Arial, sans-serif;
}

.message {
    border-radius: 16px !important;
    padding: 10px 14px !important;
    font-size: 16px;
    line-height: 1.5;
}

.user {
    background-color: #e6f3ff !important;
    color: #003366 !important;
}

.assistant {
    background-color: #f0f8ff !important;
    color: #002244 !important;
    border: 1px solid #cce0ff;
}

footer, .footer {
    background-color: #b3daff !important;
}

button, .gr-button {
    background-color: #80c1ff !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    transition: 0.3s;
}

button:hover, .gr-button:hover {
    background-color: #66b3ff !important;
}
"""

gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🛫 FlightAI Customer Support",
    description="Ask about flights, ticket prices, or make bookings in a friendly baby-blue chat window ✈️",
    theme="soft",
    css=custom_css,
).launch(debug=True, inbrowser=True)
