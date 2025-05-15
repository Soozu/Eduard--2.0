import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from datetime import datetime
import logging
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('email_service')

# Email configuration
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'christianpacifico20@gmail.com'
EMAIL_PASSWORD = 'deoqsfkuocmczvzk'
EMAIL_FROM = 'WeTravel@gmail.com'

def send_ticket_email(recipient_email, ticket_id, travel_data):
    """
    Send an email to the user with their ticket details
    
    Args:
        recipient_email (str): User's email address
        ticket_id (str): The generated ticket ID
        travel_data (dict): The travel planning data
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        logger.info(f"Preparing to send email for ticket {ticket_id} to {recipient_email}")
        
        # Create message container
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Your Travel Ticket: {ticket_id}'
        msg['From'] = EMAIL_FROM
        msg['To'] = recipient_email
        
        # Create HTML email content
        html_content = generate_email_template(ticket_id, travel_data)
        
        # Attach HTML content
        part = MIMEText(html_content, 'html')
        msg.attach(part)
        
        logger.info("Connecting to SMTP server...")
        # Connect to SMTP server
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.ehlo()  # Can help with connection issues
        server.starttls(context=ssl.create_default_context())  # Secure the connection
        server.ehlo()  # Re-identify ourselves over TLS connection
        
        logger.info("Logging into email account...")
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        
        logger.info("Sending email...")
        # Send email
        server.sendmail(EMAIL_FROM, recipient_email, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Authentication error with email provider: {e}")
        logger.error("Please check your email username and password/app password")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return False

def generate_email_template(ticket_id, travel_data):
    """
    Generate HTML email content with travel details
    
    Args:
        ticket_id (str): The ticket ID
        travel_data (dict): The travel planning data
    
    Returns:
        str: HTML content for the email
    """
    # Get current date for footer
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract travel data
    destination = travel_data.get('destination', 'Not specified')
    travel_dates = travel_data.get('travel_dates', 'Not specified')
    travelers = travel_data.get('travelers', 1)
    budget = travel_data.get('budget', 'Not specified')
    
    # Get status information
    status = travel_data.get('status', 'active')
    status_message = travel_data.get('status_message', '')
    
    # Status styling
    status_colors = {
        'active': '#3498db',      # Blue
        'confirmed': '#2ecc71',   # Green
        'cancelled': '#e74c3c',   # Red
        'completed': '#9b59b6',   # Purple
        'pending': '#f39c12'      # Orange
    }
    
    status_color = status_colors.get(status, '#3498db')
    
    # Format budget to include PHP symbol if it's a number
    if isinstance(budget, (int, float)):
        budget = f"₱{budget:,.2f}"
    elif isinstance(budget, str) and budget.isdigit():
        budget = f"₱{float(budget):,.2f}"
    
    # Generate itinerary HTML
    itinerary_html = ""
    if 'itinerary' in travel_data and travel_data['itinerary']:
        for day in travel_data['itinerary']:
            day_num = day.get('day', 0)
            places = day.get('places', [])
            meals = day.get('meals', [])
            transportation = day.get('transportation', 'Not specified')
            estimated_cost = day.get('estimated_cost', 'Not specified')
            
            itinerary_html += f"""
            <div style="margin-bottom: 30px; border-left: 4px solid #3498db; padding-left: 15px;">
                <h3 style="color: #3498db; margin-bottom: 10px;">Day {day_num}</h3>
                <div style="margin-bottom: 15px;">
                    <h4 style="color: #2c3e50; margin-bottom: 8px; margin-top: 0;">Places to Visit</h4>
                    <ul style="list-style-type: none; padding-left: 0;">
            """
            
            for place in places:
                place_name = place.get('name', 'Unknown')
                category = place.get('category', '')
                description = place.get('description', '')
                # Get rating to display as stars
                rating = place.get('rating', 0)
                star_rating = '★' * int(rating) + '☆' * (5 - int(rating)) if rating else ''
                
                itinerary_html += f"""
                    <li style="margin-bottom: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
                        <strong style="font-size: 16px;">{place_name}</strong> 
                        <span style="color: #777; font-style: italic;">({category})</span>
                        {f'<div style="color: #f39c12; letter-spacing: 2px; margin: 5px 0;">{star_rating}</div>' if star_rating else ''}
                        {f'<div style="color: #555; font-size: 14px; margin-top: 5px;">{description[:100]}{"..." if len(description) > 100 else ""}</div>' if description else ''}
                    </li>
                """
            
            itinerary_html += """
                    </ul>
                </div>
            """
            
            # Add meals section
            if meals:
                itinerary_html += """
                <div style="margin-bottom: 15px;">
                    <h4 style="color: #2c3e50; margin-bottom: 8px; margin-top: 0;">Meals</h4>
                    <ul style="list-style-type: none; padding-left: 0;">
                """
                
                for meal in meals:
                    meal_type = meal.get('type', '')
                    suggestion = meal.get('suggestion', '')
                    
                    itinerary_html += f"""
                        <li style="margin-bottom: 5px;">
                            <strong>{meal_type}:</strong> {suggestion}
                        </li>
                    """
                
                itinerary_html += """
                    </ul>
                </div>
                """
            
            # Add transportation section
            itinerary_html += f"""
                <div style="margin-bottom: 15px;">
                    <h4 style="color: #2c3e50; margin-bottom: 8px; margin-top: 0;">Transportation</h4>
                    <p style="margin: 0; color: #555;">{transportation}</p>
                </div>
            """
            
            # Add estimated cost section
            itinerary_html += f"""
                <div style="margin-bottom: 5px;">
                    <h4 style="color: #2c3e50; margin-bottom: 8px; margin-top: 0;">Estimated Cost</h4>
                    <p style="margin: 0; color: #555; font-weight: bold;">{estimated_cost}</p>
                </div>
            """
            
            itinerary_html += """
            </div>
            """
    else:
        itinerary_html = "<p>No itinerary details available.</p>"
    
    # Status message section
    status_message_html = ""
    if status_message:
        status_message_html = f"""
        <div style="background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 20px; border-left: 5px solid {status_color};">
            <p style="font-size: 16px; margin: 0;">{status_message}</p>
        </div>
        """
    
    # HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your Travel Ticket</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; border-top: 5px solid {status_color};">
            <h1 style="color: #3498db; margin-top: 0;">Your Travel Ticket</h1>
            <p style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">Ticket ID: <span style="color: #e74c3c;">{ticket_id}</span></p>
            <p style="color: #777; font-style: italic;">Keep this ID to track your travel plans</p>
            <div style="background-color: {status_color}; color: white; display: inline-block; padding: 5px 10px; border-radius: 3px; font-weight: bold; text-transform: uppercase; font-size: 12px; margin-top: 10px;">
                {status}
            </div>
        </div>
        
        {status_message_html}
        
        <div style="background-color: #fff; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0;">Trip Details</h2>
            
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; font-weight: bold; width: 40%;">Destination:</td>
                    <td style="padding: 8px 0;">{destination}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Travel Dates:</td>
                    <td style="padding: 8px 0;">{travel_dates}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Number of Travelers:</td>
                    <td style="padding: 8px 0;">{travelers}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Budget:</td>
                    <td style="padding: 8px 0;">{budget}</td>
                </tr>
            </table>
        </div>
        
        <div style="background-color: #fff; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0;">Your Itinerary</h2>
            {itinerary_html}
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #777; font-size: 14px;">
            <p>Thank you for choosing our service.</p>
            <p>If you have any questions or need assistance, please reply to this email or visit our website.</p>
            <p>{current_date}</p>
        </div>
    </body>
    </html>
    """
    
    return html 