import streamlit as st
import pywhatkit as kit
from datetime import datetime, timedelta

# Streamlit sidebar button
if st.sidebar.button("Emergency"):
    # Get the current time
    now = datetime.now()

    # Set the time delay in seconds (e.g., 60 seconds)
    delay_seconds = 30

    # Calculate the send time by adding the delay to the current time
    send_time = now + timedelta(seconds=delay_seconds)

    # Format the send time as HH:MM
    send_time_str = send_time.strftime("%H:%M")
    print(send_time_str)

    # Get the recipient's phone number (replace with the desired number)
    recipient_phone_number = "+919395555272"

    # Prompt for the message to send
    message = st.text_input("Enter the message to send:")

    if message:
        # Schedule the WhatsApp message
        kit.sendwhatmsg(recipient_phone_number, message, send_time_str)
        st.success(f"WhatsApp message scheduled for {send_time_str}: '{message}'")

