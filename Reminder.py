import streamlit as st
from datetime import datetime, time, timedelta
from winotify import Notification


def main():
    st.title("Windows Notifications App")

    num_notifications = st.number_input("Enter the number of notifications:", min_value=1, value=1, step=1)

    notifications = []

    for i in range(num_notifications):
        st.write(f"Notification {i + 1}")
        notification_time = st.time_input(f"Time for Notification {i + 1}")
        message = st.text_input(f"Message for Notification {i + 1}", f"Take Tablet {i + 1}")

        notifications.append({"time": notification_time, "message": message})

    if st.button("Set Notifications"):
        for notification in notifications:
            notification_time = notification["time"]
            message = notification["message"]

            # Calculate time until next notification
            current_datetime = datetime.now()
            target_datetime = datetime.combine(current_datetime.date(), notification_time)

            if target_datetime <= current_datetime:
                target_datetime += timedelta(days=1)

            time_until_notification = (target_datetime - current_datetime).total_seconds()

            st.write(f"Notification set for {notification_time.strftime('%I:%M %p')} with message: {message}")

            # Schedule the notification
            toast = Notification(app_id="Notification Alert",
                                 title="Take Tablet",
                                 msg=message,
                                 duration="long",
                                 icon=r"C:\Users\kumar\OneDrive\Pictures\antman1_edited.jpg")
            toast.add_actions(label="Click Me", launch="https://google.com")

            toast.show()


if __name__ == "__main__":
    main()
