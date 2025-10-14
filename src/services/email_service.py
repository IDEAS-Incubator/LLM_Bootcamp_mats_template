import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path


class EmailService:
    def __init__(self):
        load_dotenv()

        # Try to get from environment variables first
        self.sender_email = os.getenv("EMAIL_SENDER")
        self.password = os.getenv("EMAIL_PASSWORD")

        print(f"Email service initialized with: {self.sender_email}")

        # Validate email configuration
        if self.password == "YOUR_APP_PASSWORD_HERE":
            print("Please update the email password in src/services/email_service.py")
            print(
                "💡 Replace 'YOUR_APP_PASSWORD_HERE' with your actual Gmail App Password"
            )
            print("   To get an App Password:")
            print("   1. Go to Google Account Security")
            print("   2. Enable 2-Factor Authentication")
            print("   3. Generate an App Password for 'Mail'")
            return

        # Load receiver emails from config
        config_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.receiver_emails = config["notifications"]["email"]["receiver_emails"]

    def send_signal_email(self, team: str, signal_info: dict):
        """Send email notification for trading signals"""

        # Check if email service is properly configured
        if (
            not hasattr(self, "sender_email")
            or not self.sender_email
            or not self.password
        ):
            print("Email service not properly configured!")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.receiver_emails)
            msg["Subject"] = (
                f"Trading Signal Alert - {team} - {signal_info['signal'].upper()}"
            )

            # Create email body
            body = f"""
Trading Signal Alert for {team}

Signal Type: {signal_info['signal'].upper()}
Price: {signal_info['price']:.2f}
Strategy: {signal_info['strategy'].upper()}
Timestamp: {signal_info['timestamp']}

Additional Information:
"""

            if signal_info["strategy"] == "bollinger":
                body += f"""
Upper Band: {signal_info['upper_band']:.2f}
Lower Band: {signal_info['lower_band']:.2f}
SMA: {signal_info['sma']:.2f}
"""
            else:  # DCA strategy
                body += f"""
Distance from SMA: {signal_info['distance']:.2f}%
Volatility: {signal_info['volatility']:.4f}
SMA: {signal_info['sma']:.2f}
"""

            msg.attach(MIMEText(body, "plain"))

            print(f"Attempting to send email to: {', '.join(self.receiver_emails)}")

            # Send email
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.send_message(msg)

            print("Email sent successfully!")
            return True

        except smtplib.SMTPAuthenticationError as e:
            print(f"Authentication failed: {str(e)}")
            print("💡 For Gmail, you need to:")
            print("   1. Enable 2-factor authentication")
            print("   2. Generate an App Password")
            print("   3. Use the App Password in your .env file")
            return False
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
