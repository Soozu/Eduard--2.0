# WerTigo Travel Planning System

A modern travel planning and ticket tracking system with AI-powered recommendations.

## Installation

This project uses modern Python packaging with `pyproject.toml`. To install:

### Development Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/wertigo.git
   cd wertigo
   ```

2. Install the package in development mode:
   ```
   pip install -e .
   ```

   Or with development dependencies:
   ```
   pip install -e ".[dev]"
   ```

### Database Setup

1. Make sure you have MySQL installed and running.

2. Configure your database settings in `utils/db_config.py`.

3. Run the database setup script:
   ```
   python utils/create_tables.py
   ```

## Running the Application

Start the server:
```
python utils/run_server.py
```

Open the client application by navigating to the `Client` directory and opening `index.html` in your web browser.

## Project Structure

- `utils/` - Backend Python code and API
- `Client/` - Frontend HTML, CSS, and JavaScript
- `pyproject.toml` - Project dependencies and metadata

## Dependencies

All dependencies are specified in the `pyproject.toml` file and will be automatically installed when using pip.

## Database Schema

The application uses a MySQL database with the following main tables:
- `destinations` - Information about travel destinations
- `tickets` - User ticket information including itineraries 